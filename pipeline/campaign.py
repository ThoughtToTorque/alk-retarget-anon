"""Phase-4 Campaign A runner: paired multi-method evaluation.

For each (task, seed, tier) the scene pair is generated ONCE (demo scene =
the fixed feasible demo scene, copied from the canonical easy-tier pair;
TARGET scene re-randomized under the tier's placement sampler,
baselines.difficulty).  Every method is then evaluated on the SAME pair
(paired design, docs/CODE_MAP.md statistics protocol):

  * ours_full / ours_noreg / ours_nocorr  -- pipeline.retarget_runner.run_pair
    with the registration/correction flags.
  * registry baselines (baselines.runner_hooks REGISTRY) -- run_method()
    produces T_map; execution goes through the SAME waypoint path as the
    pipeline (retarget_waypoint_dicts on the demo waypoints, demo gripper
    schedule, simtasks.motion.execute_waypoints, object-centric success
    checker).  MOKA additionally anchors on its own marked grasp point
    (demo_grasp/target_grasp aux output); no other method receives the
    pipeline's grasp-point correction (that is OUR component, quantified by
    the ours_nocorr ablation).

Per-rollout JSON: <out-root>/<task>/<seed>/rollout_<method>.json -- the
basename matches stats/aggregate.py's rollout_*.json glob and the record
carries the keys its tidy rows read (task/seed/method/success/grasped/
rot_err_deg/trans_err_m/chamfer_*/failure_stage/conditioning[scalar]/...).

Resumable: existing rollout files are skipped, so a crashed process (rare
robosuite segfaults) can simply be restarted.

Usage (one process per task; env instance reused across seeds/methods):

  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m pipeline.campaign \
      --task pour --tier medium --seed-start 1000 --seed-end 1059
  python -m pipeline.campaign --task pour --report-feasibility \
      --seed-start 1000 --seed-end 1009

Render size: --cam-size (DEFAULT 256 = envs.DEFAULT_CAM_SIZE, unchanged) sets
the square capture resolution.  Scene geometry is independent of it (placement
is drawn from the global numpy RNG seeded per pair in envs.reset_with_seed),
so the same seeds give the same scenes at any size; a NON-default size needs
its own --data-root, because captures of different resolutions must not be
mixed inside one pair (enforced by check_pair_res).  Campaign H
(results/campaign_h) is the resolution-sensitivity sweep built on this flag:

  python -m pipeline.campaign --task pour --cam-size 512 \
      --data-root data_medium_res512 \
      --out-root results/campaign_h/res512
"""
import argparse
import json
import os
import shutil
import time
import traceback

import numpy as np

from simtasks import capture, envs, motion, scene_pairs
from pipeline import oracle
from pipeline import retarget_runner as rr
from baselines import common as bcommon
from baselines import difficulty
from baselines import runner_hooks as hooks

SIMBENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EASY_DATA_ROOT = scene_pairs.DATA_ROOT

# tier -> default scene-pair data root (hard shares medium PLACEMENT; its
# sensor noise is applied at perception time, not baked into captures)
TIER_DATA_ROOT = {
    "easy": EASY_DATA_ROOT,
    "medium": os.path.join(SIMBENCH, "data_medium"),
    "hard": os.path.join(SIMBENCH, "data_medium"),
}
DEFAULT_OUT_ROOT = os.path.join(SIMBENCH, "results", "campaign_a")

OURS_METHODS = [
    ("ours_full", {"registration": True, "correction": True}),
    ("ours_noreg", {"registration": False, "correction": True}),
    ("ours_nocorr", {"registration": True, "correction": False}),
]
REGISTRY_METHODS = ["icp", "icp_centroid", "moka_oracle", "rekep",
                    "kp_fps4", "kp_random4", "kp_pca2", "kp_dense_init"]
ALL_METHODS = [n for n, _ in OURS_METHODS] + REGISTRY_METHODS

# Campaign F -- ALK point-subset variants of the closed-form alignment (which
# ALK points enter the Procrustes sum; alkbench.alk.ALK_SUBSETS).  NOT part of
# ALL_METHODS, so the default method list of every earlier campaign is
# unchanged; request them explicitly with --methods.
#   alk4       == ours_full (alk_subset=None -> the untouched default path, so
#                 its rollout JSONs are byte-comparable to campaign_a's
#                 ours_full apart from the method/variant names and time_s)
#   alk3_c134  == the earlier three-point construction, sum over {C1, C3, C4}
#   *_adaptive == same subset with the conditioning-adaptive registration
#                 flag (paper Section 3.7), trigger read on the SOLVED subset
ALK_SUBSET_METHODS = [
    ("alk4", {"registration": True, "correction": True, "alk_subset": None}),
    ("alk3_c134", {"registration": True, "correction": True,
                   "alk_subset": "alk3_c134"}),
    ("alk3_c123", {"registration": True, "correction": True,
                   "alk_subset": "alk3_c123"}),
    ("alk2_c12", {"registration": True, "correction": True,
                  "alk_subset": "alk2_c12"}),
    ("alk4_adaptive", {"registration": True, "correction": True,
                       "alk_subset": None, "adaptive": True}),
    ("alk3_c134_adaptive", {"registration": True, "correction": True,
                            "alk_subset": "alk3_c134", "adaptive": True}),
]
OURS_KW = dict(OURS_METHODS + ALK_SUBSET_METHODS)
KNOWN_METHODS = ALL_METHODS + [n for n, _ in ALK_SUBSET_METHODS]


# ---------------------------------------------------------------------------
# tier plumbing
# ---------------------------------------------------------------------------

# robosuite NutAssembly default per-nut ranges (nut_assembly.py); the two
# nuts' handles cannot coexist inside one widened box (RandomizationError
# from a flat sampler), so the tier override goes into a composite sampler
# that widens ONLY the target nut and keeps the other nut -- cleared right
# after reset in single_object_mode 2 anyway -- at its default sliver.
NUT_DEFAULT_RANGES = {
    "SquareNut": {"x_range": [-0.115, -0.11], "y_range": [0.11, 0.225]},
    "RoundNut": {"x_range": [-0.115, -0.11], "y_range": [-0.225, -0.11]},
}


def _nut_composite_sampler(task, ov):
    from robosuite.utils.placement_samplers import (
        SequentialCompositeSampler, UniformRandomSampler)
    target = envs.TASKS[task].target_instance
    ref = np.asarray(ov["reference_pos"], dtype=np.float64)
    comp = SequentialCompositeSampler(name="ObjectSampler")
    for nut in ("SquareNut", "RoundNut"):  # default order (RNG determinism)
        if nut == target:
            kw = {"x_range": list(ov["x_range"]),
                  "y_range": list(ov["y_range"]),
                  "rotation": ov.get("rotation"),
                  "z_offset": float(ov.get("z_offset", 0.02))}
        else:
            kw = {"x_range": list(NUT_DEFAULT_RANGES[nut]["x_range"]),
                  "y_range": list(NUT_DEFAULT_RANGES[nut]["y_range"]),
                  "rotation": None, "z_offset": 0.02}
        comp.append_sampler(UniformRandomSampler(
            name="%sSampler" % nut, rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True, reference_pos=ref, **kw))
    return comp


def make_tier_env(task, tier, cam_size=envs.DEFAULT_CAM_SIZE):
    """Env whose placement sampler follows the tier override (None = keep the
    task's current/easy sampler).

    `cam_size` (DEFAULT envs.DEFAULT_CAM_SIZE = 256 -- existing behavior
    unchanged) is the square RENDER size of the scene captures only: it
    changes nothing about the scene itself (placement is drawn from the
    global numpy RNG seeded by envs.reset_with_seed, independently of the
    camera resolution) and nothing about execution.  Campaign H
    (resolution sensitivity) runs the same seeds at 384/512.
    """
    ov = difficulty.sampler_override(tier, task)
    if ov is None:
        return envs.make_env(task, camera_size=cam_size)
    if task in ("nut_loosen", "cap_twist"):
        return envs.make_env(task, camera_size=cam_size,
                             placement_initializer=_nut_composite_sampler(
                                 task, ov))
    from robosuite.utils.placement_samplers import UniformRandomSampler
    kw = dict(ov)
    kw["reference_pos"] = np.asarray(kw["reference_pos"], dtype=np.float64)
    sampler = UniformRandomSampler(
        name="CampaignSampler",
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        **kw)
    return envs.make_env(task, camera_size=cam_size,
                         placement_initializer=sampler)


def ensure_demo_assets(task, data_root):
    """The demo recording (fixed feasible demo scene + keyframes/waypoints)
    is tier-invariant; copy it once from the canonical easy-tier data root.

    If the canonical recording does not exist yet (fresh checkout: no scene
    data is shipped, everything is regenerated from seeds), it is RECORDED
    first, in the canonical env at the fixed demo seed -- the same thing
    pipeline.retarget_runner.run_pair does for a single rollout.
    """
    dst = os.path.join(data_root, task, str(rr.DEMO_PAIR_SEED), "demo")
    if os.path.exists(os.path.join(dst, "waypoints.json")):
        return dst
    src = os.path.join(EASY_DATA_ROOT, task, str(rr.DEMO_PAIR_SEED), "demo")
    if not os.path.exists(os.path.join(src, "waypoints.json")):
        print("[%s] recording the canonical demonstration into %s"
              % (task, src), flush=True)
        rr.ensure_demo(task, data_root=EASY_DATA_ROOT)
    if not os.path.exists(os.path.join(src, "waypoints.json")):
        raise RuntimeError("canonical demo recording missing: %s" % src)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp_copy"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    os.rename(tmp, dst)
    return dst


def capture_cam_size(scene_dir):
    """Square render size of a saved capture (max over its cameras), or None
    if the capture does not exist."""
    mp = os.path.join(scene_dir, "meta.json")
    if not os.path.exists(mp):
        return None
    with open(mp) as f:
        meta = json.load(f)
    sizes = set()
    for cm in meta["cameras"].values():
        sizes.add((int(cm["height"]), int(cm["width"])))
    return max(max(s) for s in sizes) if sizes else None


def ensure_demo_scene_res(task, data_root, cam_size):
    """Make the canonical seed-0 demo SCENE capture match `cam_size`.

    `ensure_demo_assets` copies the whole tier-invariant demo recording from
    the canonical easy-tier data root, whose scene capture is rendered at
    256.  Campaign H needs the demo scene rendered at the SAME resolution as
    the target scene (the pipeline builds the demo ALK from it), so when the
    requested size differs the capture is re-rendered from the fixed demo
    seed -- keyframes.json / waypoints.json / trajectory.npz (the scripted
    expert's TCP trajectory: resolution-independent) are untouched.
    No-op when the capture already has the requested size, so the default
    256 path never re-renders anything.

    The re-render uses the CANONICAL env (`envs.make_env`, i.e. the demo
    scene's own placement sampler), NOT the tier env: the demonstration
    scene is tier-invariant by construction (`ensure_demo_assets` copies it
    from the easy-tier data root) and the tier env's widened sampler would
    draw a DIFFERENT demo object pose from the same seed, which would change
    the scene instead of only its resolution.  Verified by
    `stats.campaign_h_tables.pose_match` (demo AND target ground-truth poses
    bit-identical across data roots).
    """
    scene_dir = os.path.join(data_root, task, str(rr.DEMO_PAIR_SEED),
                             "demo", "scene")
    have = capture_cam_size(scene_dir)
    if have == cam_size:
        return False
    env0 = envs.make_env(task, camera_size=cam_size)
    try:
        envs.reset_with_seed(env0, scene_pairs.DEMO_SEEDS[task])
        capture.capture_scene(env0, scene_dir,
                              extra_state=envs.TASKS[task].extra_state(env0))
    finally:
        env0.close()
    print("[%s] re-rendered demo scene capture at %dpx (was %s) with the "
          "canonical demo-scene sampler" % (task, cam_size, have), flush=True)
    return True


def check_pair_res(task, seed, data_root, cam_size):
    """Guard: every capture the methods read must be at the requested render
    size (catches a data root whose captures were generated at another size,
    which would silently mix resolutions across the demo/target pair)."""
    for rel in (os.path.join("demo", "scene"), os.path.join("target", "scene")):
        d = os.path.join(data_root, task, str(seed), rel)
        got = capture_cam_size(d)
        if got != cam_size:
            raise RuntimeError(
                "capture %s is %s px but --cam-size is %d px; use a separate "
                "--data-root per render size" % (d, got, cam_size))


def result_path(out_root, task, seed, method):
    return os.path.join(out_root, task, str(seed), "rollout_%s.json" % method)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rr._json_safe(obj), f, indent=2)
    os.rename(tmp, path)


# ---------------------------------------------------------------------------
# per-method rollouts
# ---------------------------------------------------------------------------

def run_ours_variant(name, task, seed, env, data_root, tier, adaptive=False):
    kw = dict(OURS_KW[name])
    # a method whose entry pins `adaptive` (Campaign F *_adaptive) wins over
    # the run-wide --adaptive flag
    adaptive = bool(kw.pop("adaptive", adaptive))
    r = rr.run_pair(task, seed, env=env, data_root=data_root, save=False,
                    adaptive=adaptive, **kw)
    r["method"] = name
    r["variant"] = name
    r["tier"] = tier
    orc = r.get("oracle")
    if orc and orc.get("demo_alk") is not None:
        alk = np.asarray(orc["demo_alk"])
        subset = kw.get("alk_subset")
        # the scalar `conditioning` column is the conditioning of the
        # configuration the method actually SOLVES (== the full ALK unless a
        # point subset is in play); the full-ALK numbers stay available too
        cond = bcommon.conditioning(alk)
        if subset:
            cond_full, cond = cond, bcommon.conditioning(
                rr.select_alk_subset(alk, subset))
            r["conditioning_detail_alk4"] = cond_full
        r["conditioning"] = cond["sigma23"]           # scalar for aggregate
        r["conditioning_detail"] = cond
    return r


def run_registry_method(name, task, seed, pair, demo, demo_cap, target_cap,
                        ctx, env, tier):
    t0 = time.time()
    inst = pair["target_instance"]
    demo_pose = pair["demo_object_poses"][inst]
    target_pose = pair["target_object_poses"][inst]
    result = {"task": task, "seed": seed, "method": name, "variant": name,
              "tier": tier, "success": False, "failure_stage": None,
              "T_gt": np.asarray(ctx.T_gt).tolist()}
    try:
        mres = hooks.run_method(name, demo_cap, target_cap, ctx)
    except Exception as e:  # method could not produce a T_map -> failure
        traceback.print_exc()
        result["failure_stage"] = "method_error"
        result["error"] = repr(e)
        result["time_s"] = round(time.time() - t0, 1)
        return result

    T_map = np.asarray(mres["T_map"], dtype=np.float64)
    rot_err, trans_err = hooks.map_errors(T_map, ctx.T_gt, demo_pose["pos"],
                                          target_pose["pos"])
    result.update({
        "T_map": T_map.tolist(),
        "rot_err_deg": rot_err,
        "trans_err_m": trans_err,
        "method_time_s": mres.get("time_s"),
    })
    for kk in ("chamfer_before_m", "chamfer_after_m"):
        if kk in mres:
            result[kk] = mres[kk]
    cond = mres.get("conditioning")
    if isinstance(cond, dict):  # scalar top-level for stats.aggregate
        result["conditioning"] = cond.get("sigma23")
        result["conditioning_detail"] = cond
    elif cond is not None:
        result["conditioning"] = float(cond)
    result["aux"] = {k: v for k, v in mres.items()
                     if k not in ("method", "T_map", "time_s",
                                  "chamfer_before_m", "chamfer_after_m",
                                  "conditioning")}

    p_d = ctx.percept(demo_cap, "demo")      # cached across methods
    p_t = ctx.percept(target_cap, "target")
    result["camera"] = p_t["camera"]
    result["perception"] = {
        "demo": {"sanity_ok": p_d["sanity_ok"],
                 "centroid_err_m": p_d["centroid_err_m"]},
        "target": {"sanity_ok": p_t["sanity_ok"],
                   "centroid_err_m": p_t["centroid_err_m"]},
    }

    # ---- shared execution path (same as pipeline.retarget_runner) ----------
    if name.startswith("moka"):
        # MOKA's marked anchor is part of the method: hang the demo template
        # on it (T_map is translation-only; anchors follow its aux output).
        wps = rr.retarget_waypoint_dicts(
            demo["waypoints"], T_map,
            demo_grasp=np.asarray(mres["demo_grasp"], dtype=np.float64),
            target_grasp=np.asarray(mres["target_grasp"], dtype=np.float64))
    else:
        wps = rr.retarget_waypoint_dicts(demo["waypoints"], T_map)
    envs.reset_with_seed(env, pair["target_seed"])
    exec_res = motion.execute_waypoints(env, wps,
                                        grasp_check=rr._grasp_check(task))
    success = bool(envs.success_checker(task)(env))
    result.update({
        "grasped": exec_res["grasped"],
        "waypoints_converged": exec_res["converged"],
        "n_exec_steps": exec_res["n_steps"],
        "success": success,
    })
    if task == "pour":  # orientation-aware criterion diagnostics (pour only)
        result["pour_orientation"] = rr._json_safe(
            envs.pour_orientation_trace(env))
    if not success:  # same one-level failure-stage guess as the pipeline
        if not (p_d["sanity_ok"] and p_t["sanity_ok"]):
            stage = "perception"
        elif trans_err > 0.05 or rot_err > 20.0:
            stage = "mapping"
        elif exec_res["grasped"] is False:
            stage = "grasp"
        else:
            stage = "post_action"
        result["failure_stage"] = stage
    result["time_s"] = round(time.time() - t0, 1)
    return result


# ---------------------------------------------------------------------------
# task loop
# ---------------------------------------------------------------------------

def run_task(task, seeds, methods, tier, data_root, out_root, adaptive=False,
             cam_size=envs.DEFAULT_CAM_SIZE):
    ensure_demo_assets(task, data_root)
    # demo scene first (own canonical env, closed again), then the tier env
    ensure_demo_scene_res(task, data_root, cam_size)
    env = make_tier_env(task, tier, cam_size=cam_size)
    demo = None
    demo_cap = None
    n_done = 0
    try:
        for seed in seeds:
            todo = [m for m in methods
                    if not os.path.exists(result_path(out_root, task, seed, m))]
            if not todo:
                continue
            pair = rr.ensure_pair(task, seed, env=env, data_root=data_root)
            check_pair_res(task, seed, data_root, cam_size)
            if demo is None:
                demo = rr.load_demo(task, data_root)
                demo_cap = bcommon.load_capture(
                    os.path.join(rr.demo_dir_for(task, data_root), "scene"))
            target_cap = bcommon.load_capture(
                os.path.join(data_root, task, str(seed), "target", "scene"))
            inst = pair["target_instance"]
            T_gt = oracle.gt_relative_transform(
                pair["demo_object_poses"][inst],
                pair["target_object_poses"][inst])
            # one context per pair: perception cached across registry methods;
            # per-pair seeds make random4 an average over draws across seeds
            ctx = bcommon.MethodContext(
                task=task, T_gt=T_gt, keyframes=demo["keyframes"],
                waypoints=demo["waypoints"], sigma_px=0.0, noise_seed=seed,
                random4_seed=seed,
                sensor_noise=difficulty.sensor_noise_params(tier))
            for name in todo:
                t0 = time.time()
                try:
                    if name in OURS_KW:
                        r = run_ours_variant(name, task, seed, env,
                                             data_root, tier,
                                             adaptive=adaptive)
                    else:
                        r = run_registry_method(name, task, seed, pair, demo,
                                                demo_cap, target_cap, ctx,
                                                env, tier)
                except Exception as e:
                    traceback.print_exc()
                    r = {"task": task, "seed": seed, "method": name,
                         "variant": name, "tier": tier, "success": False,
                         "failure_stage": "exception", "error": repr(e),
                         "time_s": round(time.time() - t0, 1)}
                _write_json(result_path(out_root, task, seed, name), r)
                n_done += 1
                print("[%s %d %-13s] success=%s stage=%-12s rot=%s trans=%s (%.1fs)"
                      % (task, seed, name, r.get("success"),
                         str(r.get("failure_stage")),
                         ("%.1fdeg" % r["rot_err_deg"])
                         if r.get("rot_err_deg") is not None else "-",
                         ("%.1fmm" % (1e3 * r["trans_err_m"]))
                         if r.get("trans_err_m") is not None else "-",
                         time.time() - t0), flush=True)
    finally:
        env.close()
    return n_done


# ---------------------------------------------------------------------------
# medium-tier feasibility report
# ---------------------------------------------------------------------------

def feasibility_report(task, seeds, data_root, out_root):
    """Summarize pair validity + ours_full outcomes for the given seeds.

    Validity = pair generated without error, settled target-object pose has a
    sane height, and perception passed its centroid sanity check."""
    rows = []
    for seed in seeds:
        row = {"seed": seed, "pair_ok": False, "obj_pos": None,
               "success": None, "failure_stage": None, "sanity_ok": None,
               "grasped": None, "converged": None}
        pj = os.path.join(data_root, task, str(seed), "pair.json")
        if os.path.exists(pj):
            with open(pj) as f:
                pair = json.load(f)
            pose = pair["target_object_poses"][pair["target_instance"]]
            row["pair_ok"] = True
            row["obj_pos"] = pose["pos"]
        rp = result_path(out_root, task, seed, "ours_full")
        if os.path.exists(rp):
            with open(rp) as f:
                r = json.load(f)
            row["success"] = r.get("success")
            row["failure_stage"] = r.get("failure_stage")
            perc = r.get("perception") or {}
            tgt = perc.get("target") or {}
            row["sanity_ok"] = tgt.get("sanity_ok")
            row["grasped"] = r.get("grasped")
            conv = r.get("waypoints_converged")
            row["converged"] = (all(conv) if isinstance(conv, list) else conv)
            row["rot_err_deg"] = r.get("rot_err_deg")
            row["trans_err_m"] = r.get("trans_err_m")
        rows.append(row)
    n_pair = sum(1 for r in rows if r["pair_ok"])
    n_succ = sum(1 for r in rows if r["success"])
    summary = {
        "task": task,
        "seeds": list(seeds),
        "n_pairs_ok": n_pair,
        "n_success_ours_full": n_succ,
        "n": len(rows),
        "z_values": [r["obj_pos"][2] for r in rows if r["obj_pos"]],
        "rows": rows,
    }
    out = os.path.join(out_root, "feasibility_%s.json" % task)
    _write_json(out, summary)
    print(json.dumps({k: summary[k] for k in
                      ("task", "n_pairs_ok", "n_success_ours_full", "n")},
                     indent=2))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=sorted(envs.TASKS.keys()))
    p.add_argument("--tier", default="medium",
                   choices=sorted(difficulty.TIERS.keys()))
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--seed-end", type=int, default=1059,
                   help="inclusive")
    p.add_argument("--methods", nargs="*", default=ALL_METHODS,
                   help="default: the Campaign-A method set; the Campaign-F "
                        "ALK point-subset variants (%s) must be requested "
                        "explicitly"
                        % ", ".join(n for n, _ in ALK_SUBSET_METHODS))
    p.add_argument("--data-root", default=None)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--report-feasibility", action="store_true",
                   help="only summarize existing pairs + ours_full rollouts")
    p.add_argument("--cam-size", type=int, default=envs.DEFAULT_CAM_SIZE,
                   help="square render size of the scene captures (DEFAULT "
                        "%d = unchanged).  Scene GEOMETRY is independent of "
                        "it (placement RNG is seeded per pair), so the same "
                        "seeds give the same scenes at any size; use a "
                        "SEPARATE --data-root per size (Campaign H)"
                        % envs.DEFAULT_CAM_SIZE)
    p.add_argument("--adaptive", action="store_true",
                   help="DEFAULT OFF: conditioning-adaptive registration + "
                        "auto slender depth prior for the ours_* methods "
                        "(pipeline.retarget_runner adaptive=True); use a "
                        "dedicated --out-root so baseline rollouts are not "
                        "mixed")
    args = p.parse_args(argv)

    data_root = args.data_root or TIER_DATA_ROOT[args.tier]
    seeds = list(range(args.seed_start, args.seed_end + 1))
    for m in args.methods:
        if m not in KNOWN_METHODS:
            raise SystemExit("unknown method %r (have %s)"
                             % (m, KNOWN_METHODS))
    if args.report_feasibility:
        feasibility_report(args.task, seeds, data_root, args.out_root)
        return 0
    t0 = time.time()
    n = run_task(args.task, seeds, args.methods, args.tier, data_root,
                 args.out_root, adaptive=args.adaptive,
                 cam_size=args.cam_size)
    print("done: %s, %d new rollouts in %.1f min"
          % (args.task, n, (time.time() - t0) / 60.0), flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
