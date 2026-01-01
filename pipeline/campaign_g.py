"""Campaign G runner: UNIFORM vs SELECTIVE registration-bound widening.

Resolves an apparent contradiction between the authors' original real-robot
paper and our conditioning-adaptive registration result, in simulation only:

  original paper (hyperparameter sensitivity): "Registration bounds
  dtheta_max = 15 deg, dt_max = 20 mm cover typical keypoint errors; wider
  bounds provided no improvement."

  Campaign A-adaptive: widening the rotation bound to +-90 deg ABOUT THE ALK
  PRINCIPAL AXIS when the demo ALK is ill conditioned takes pour from
  33/60 to 51/60 (McNemar p = 4e-5).

Hypothesis under test: "wider bounds" in the sensitivity study means UNIFORM
widening (all three rotation axes and the translation together), which
enlarges the search box isotropically and admits Chamfer-equivalent wrong
poses -> no net gain; SELECTIVE widening along the single provably
unobservable direction is what helps.

Protocol (identical to Campaign A so results are directly comparable):
5 tasks x 60 paired seeds {1000..1059}, medium tier (`data_medium/` scene
pairs), oracle discrete answers, grasp-point correction ON, method
`ours_full`.  Only the registration bounds change between settings.

Settings (`SETTINGS`):
  base15_20    +-15 deg / +-20 mm            production default; REUSED from
                                             results/campaign_a (--link-base)
  u30_40       +-30 deg / +-40 mm  uniform, plain (base step sizes)
  u45_60       +-45 deg / +-60 mm  uniform, plain
  u90_60       +-90 deg / +-60 mm  uniform, plain
  u90_20       +-90 deg / +-20 mm  uniform rotation only (translation at base)
  u90_60_c2f   +-90 deg / +-60 mm  uniform, coarse-to-fine (15 deg coarse
                                   rotation step = the adaptive path's coarse
                                   step, so the ONLY difference from
                                   `sel90_nodp` is WHICH DoF are widened)
  sel90_nodp   selective +-90 deg about the ALK principal axis, depth prior
                                   OFF (isolates the widening geometry)
  adaptive     the shipped conditioning-adaptive path (selective widening +
                                   auto slender depth prior); REUSED from
                                   results/campaign_a_adaptive where present

Nothing outside this file and `alkbench/registration_uniform.py` is
modified: the widened variants are injected by temporarily rebinding
`pipeline.retarget_runner.bounded_registration` (and
`pipeline.oracle.solve`, to capture the demo ALK the selective variant
needs) for the duration of one rollout.  The default code path is untouched,
so a concurrent agent editing `retarget_runner` / `alk` / `registration`
cannot conflict with this campaign.

Output: <out-root>/<task>/<setting>/<seed>/rollout_ours_full.json
(resumable: existing files are skipped).

Usage:
  MUJOCO_GL=egl python -m pipeline.campaign_g --link-base --link-adaptive
  MUJOCO_GL=egl python -m pipeline.campaign_g --verify-reuse --tasks pour
  MUJOCO_GL=egl python -m pipeline.campaign_g --tasks pour \
      --settings u30_40 u45_60 u90_60 u90_20 u90_60_c2f
  MUJOCO_GL=egl python -m pipeline.campaign_g --bench-registration --tasks pour
"""
import argparse
import contextlib
import json
import os
import time
import traceback

import numpy as np

from alkbench import registration_uniform as ru
from alkbench.registration import adaptive_registration
from simtasks import capture, envs
from pipeline import campaign as camp
from pipeline import oracle, perception
from pipeline import retarget_runner as rr

SIMBENCH = camp.SIMBENCH
DEFAULT_OUT_ROOT = os.path.join(SIMBENCH, "results", "campaign_g")
CAMPAIGN_A = os.path.join(SIMBENCH, "results", "campaign_a")
CAMPAIGN_A_ADAPTIVE = os.path.join(SIMBENCH, "results", "campaign_a_adaptive")

TASKS = ["pour", "nut_loosen", "cap_twist", "rim_grasp", "box_open"]
METHOD = "ours_full"

SETTINGS = {
    "base15_20": {"kind": "base"},
    "u30_40": {"kind": "uniform", "theta_max_deg": 30.0, "t_max_mm": 40.0,
               "mode": "plain"},
    "u45_60": {"kind": "uniform", "theta_max_deg": 45.0, "t_max_mm": 60.0,
               "mode": "plain"},
    "u90_60": {"kind": "uniform", "theta_max_deg": 90.0, "t_max_mm": 60.0,
               "mode": "plain"},
    "u90_20": {"kind": "uniform", "theta_max_deg": 90.0, "t_max_mm": 20.0,
               "mode": "plain"},
    "u90_60_c2f": {"kind": "uniform", "theta_max_deg": 90.0,
                   "t_max_mm": 60.0, "mode": "c2f"},
    "sel90_nodp": {"kind": "selective", "axis_bound_deg": 90.0},
    "adaptive": {"kind": "adaptive"},
}
DEFAULT_SETTINGS = ["u30_40", "u45_60", "u90_60", "u90_20", "u90_60_c2f"]

# rollout fields that must match bit-for-bit when a rollout is re-run
# (everything except wall-clock timings, which are not reproducible)
DETERMINISTIC_FIELDS = [
    "success", "grasped", "n_exec_steps", "waypoints_converged",
    "failure_stage", "rot_err_deg", "trans_err_m", "rot_err_init_deg",
    "trans_err_init_m", "chamfer_before_m", "chamfer_after_m", "T_map",
    "T_init", "T_gt", "target_grasp_used", "grasp_correction_norm_m",
]


def result_path(out_root, task, setting, seed, method=METHOD):
    return os.path.join(out_root, task, setting, str(seed),
                        "rollout_%s.json" % method)


# ---------------------------------------------------------------------------
# registration injection (additive: nothing on disk is edited)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def registration_override(make_fn):
    """Temporarily replace the registration call used by `run_pair`.

    `make_fn(holder)` returns the replacement for
    `pipeline.retarget_runner.bounded_registration`.  `holder` is a dict the
    replacement may read/write; `pipeline.oracle.solve` is wrapped so that
    `holder["orc"]` holds the oracle answers of the current rollout (the
    selective variant needs the demo ALK, which `bounded_registration`'s
    signature does not carry).  Restored on exit, including on exception.
    """
    holder = {}
    orig_solve = oracle.solve
    orig_reg = rr.bounded_registration

    def solve_capture(*a, **kw):
        orc = orig_solve(*a, **kw)
        holder["orc"] = orc
        return orc

    oracle.solve = solve_capture
    rr.bounded_registration = make_fn(holder)
    try:
        yield holder
    finally:
        oracle.solve = orig_solve
        rr.bounded_registration = orig_reg


@contextlib.contextmanager
def adaptive_instrumented():
    """Time + count Chamfer evals of the shipped adaptive registration."""
    holder = {}
    orig = rr.adaptive_registration

    def wrapper(*a, **kw):
        t0 = time.time()
        with ru.count_chamfer() as counter:
            out = orig(*a, **kw)
        holder["reg_record"] = {"reg_time_s": float(time.time() - t0),
                                "n_chamfer_evals": int(counter["n"])}
        return out

    rr.adaptive_registration = wrapper
    try:
        yield holder
    finally:
        rr.adaptive_registration = orig


def _uniform_factory(cfg):
    def make_fn(holder):
        def fn(src, tgt, T_init, **kw):
            out = ru.uniform_widen_registration(
                src, tgt, T_init,
                theta_max_deg=cfg["theta_max_deg"],
                t_max_mm=cfg["t_max_mm"], mode=cfg["mode"])
            holder["reg_record"] = {
                "uniform_widen": out["uniform_widen"],
                "reg_time_s": out["reg_time_s"],
                "n_chamfer_evals": out["n_chamfer_evals"],
                "chamfer_coarse_m": out.get("chamfer_coarse"),
            }
            return out
        return fn
    return make_fn


def _selective_factory(cfg):
    """Selective widening WITHOUT the auto slender depth prior.

    `run_pair(adaptive=False)` keeps `oracle.solve(depth_consistent=False)`,
    so this isolates the geometry of the widening (one axis, +-90 deg) from
    the second ingredient of the shipped adaptive path.
    """
    def make_fn(holder):
        def fn(src, tgt, T_init, **kw):
            alk_demo = np.asarray(holder["orc"]["demo_alk"], dtype=np.float64)
            t0 = time.time()
            with ru.count_chamfer() as counter:
                out = adaptive_registration(
                    src, tgt, T_init, alk_demo,
                    axis_bound_deg=cfg.get("axis_bound_deg", 90.0))
            ab = out["adaptive"]
            holder["reg_record"] = {
                "reg_time_s": float(time.time() - t0),
                "n_chamfer_evals": int(counter["n"]),
                "selective": {
                    "adaptive_triggered": bool(ab["adaptive_triggered"]),
                    "axis_rot_bound_deg": ab["axis_rot_bound_deg"],
                    "coarse_axis_theta_deg": out["coarse_axis_theta_deg"],
                    "sigma_ratio": ab["sigma_ratio"],
                    "widened_axis_world": (None if out["axis_world"] is None
                                           else np.asarray(
                                               out["axis_world"]).tolist()),
                },
            }
            return out
        return fn
    return make_fn


# ---------------------------------------------------------------------------
# one rollout
# ---------------------------------------------------------------------------

def run_setting(task, seed, setting, env, data_root, tier):
    cfg = SETTINGS[setting]
    kind = cfg["kind"]
    if kind == "base":
        r = camp.run_ours_variant(METHOD, task, seed, env, data_root, tier)
        rec = {}
    elif kind == "adaptive":
        with adaptive_instrumented() as holder:
            r = camp.run_ours_variant(METHOD, task, seed, env, data_root,
                                      tier, adaptive=True)
        rec = holder.get("reg_record", {})
    else:
        factory = (_uniform_factory(cfg) if kind == "uniform"
                   else _selective_factory(cfg))
        with registration_override(factory) as holder:
            r = camp.run_ours_variant(METHOD, task, seed, env, data_root, tier)
        rec = holder.get("reg_record", {})
    r["setting"] = setting
    r["setting_kind"] = kind
    r["campaign"] = "g"
    r["registration_detail"] = rr._json_safe(rec)
    for k in ("reg_time_s", "n_chamfer_evals"):
        if k in rec:
            r[k] = rec[k]
    return r


def run_task(task, seeds, settings, tier, data_root, out_root):
    camp.ensure_demo_assets(task, data_root)
    env = camp.make_tier_env(task, tier)
    n_done = 0
    try:
        for seed in seeds:
            todo = [s for s in settings
                    if not os.path.exists(result_path(out_root, task, s, seed))]
            if not todo:
                continue
            rr.ensure_pair(task, seed, env=env, data_root=data_root)
            for setting in todo:
                t0 = time.time()
                try:
                    r = run_setting(task, seed, setting, env, data_root, tier)
                except Exception as e:
                    traceback.print_exc()
                    r = {"task": task, "seed": seed, "method": METHOD,
                         "variant": METHOD, "tier": tier, "setting": setting,
                         "campaign": "g", "success": False,
                         "failure_stage": "exception", "error": repr(e),
                         "time_s": round(time.time() - t0, 1)}
                camp._write_json(result_path(out_root, task, setting, seed), r)
                n_done += 1
                print("[%s %d %-11s] success=%s stage=%-12s rot=%s trans=%s "
                      "reg=%s (%.1fs)"
                      % (task, seed, setting, r.get("success"),
                         str(r.get("failure_stage")),
                         ("%.1fdeg" % r["rot_err_deg"])
                         if r.get("rot_err_deg") is not None else "-",
                         ("%.1fmm" % (1e3 * r["trans_err_m"]))
                         if r.get("trans_err_m") is not None else "-",
                         ("%.0fms/%dev" % (1e3 * r["reg_time_s"],
                                           r.get("n_chamfer_evals", -1)))
                         if r.get("reg_time_s") is not None else "-",
                         time.time() - t0), flush=True)
    finally:
        env.close()
    return n_done


# ---------------------------------------------------------------------------
# reuse: symlink Campaign A / A-adaptive rollouts into the campaign_g tree
# ---------------------------------------------------------------------------

def link_reuse(out_root, tasks, seeds, src_root, setting, label):
    n_linked = n_missing = 0
    for task in tasks:
        for seed in seeds:
            src = os.path.join(src_root, task, str(seed),
                               "rollout_%s.json" % METHOD)
            dst = result_path(out_root, task, setting, seed)
            if os.path.exists(dst) or os.path.islink(dst):
                continue
            if not os.path.exists(src):
                n_missing += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
            n_linked += 1
    print("%s: linked %d, missing %d" % (label, n_linked, n_missing),
          flush=True)
    return n_linked, n_missing


# ---------------------------------------------------------------------------
# reuse validity: re-run a few seeds and compare bit-exactly
# ---------------------------------------------------------------------------

def verify_reuse(tasks, seeds, tier, data_root, out_root):
    """Re-run `ours_full` at the production defaults and compare to
    results/campaign_a on the deterministic fields (all but wall clock)."""
    scratch = os.path.join(out_root, "_verify")
    report = {"tier": tier, "seeds": list(seeds), "tasks": {}}
    for task in tasks:
        camp.ensure_demo_assets(task, data_root)
        env = camp.make_tier_env(task, tier)
        rows = []
        try:
            for seed in seeds:
                rr.ensure_pair(task, seed, env=env, data_root=data_root)
                path = os.path.join(scratch, task, str(seed),
                                    "rollout_%s.json" % METHOD)
                if os.path.exists(path):
                    with open(path) as f:
                        new = json.load(f)
                else:
                    new = run_setting(task, seed, "base15_20", env, data_root,
                                      tier)
                    camp._write_json(path, new)
                ref_path = os.path.join(CAMPAIGN_A, task, str(seed),
                                        "rollout_%s.json" % METHOD)
                with open(ref_path) as f:
                    ref = json.load(f)
                diffs = [k for k in DETERMINISTIC_FIELDS
                         if json.dumps(ref.get(k), sort_keys=True)
                         != json.dumps(new.get(k), sort_keys=True)]
                rows.append({"seed": seed, "identical": not diffs,
                             "diff_fields": diffs,
                             "success_ref": ref.get("success"),
                             "success_new": new.get("success"),
                             "rot_ref": ref.get("rot_err_deg"),
                             "rot_new": new.get("rot_err_deg")})
                print("[verify %s %d] identical=%s %s"
                      % (task, seed, not diffs, diffs), flush=True)
        finally:
            env.close()
        report["tasks"][task] = rows
    report["all_identical"] = all(r["identical"] for rows in
                                 report["tasks"].values() for r in rows)
    camp._write_json(os.path.join(out_root, "verify_reuse.json"), report)
    print(json.dumps({"all_identical": report["all_identical"]}, indent=2))
    return report


# ---------------------------------------------------------------------------
# registration-only cost benchmark (no simulation execution)
# ---------------------------------------------------------------------------

def bench_registration(tasks, seeds, data_root, out_root, settings=None):
    """Wall-clock + Chamfer-eval cost of ONE registration call per setting.

    Perception and the oracle run once per (task, seed); each setting's
    registration is then timed on the identical inputs, so the numbers
    isolate the search cost of the bound (the rest of a rollout -- rendering,
    MuJoCo execution -- is setting independent).
    """
    from alkbench import procrustes
    settings = settings or ["base15_20", "u30_40", "u45_60", "u90_20",
                            "u90_60", "u90_60_c2f", "sel90_nodp", "adaptive"]
    rows = []
    for task in tasks:
        demo = rr.load_demo(task, data_root)
        cam = perception.pick_camera(demo["scene"], task)
        p_demo = perception.perceive(demo["scene"], task, k=8, seed=0,
                                     camera=cam)
        pre = [kf for kf in demo["keyframes"]
               if kf["name"] == "pre_grasp"][0]
        for seed in seeds:
            pair_dir = os.path.join(data_root, task, str(seed))
            if not os.path.exists(os.path.join(pair_dir, "pair.json")):
                continue
            with open(os.path.join(pair_dir, "pair.json")) as f:
                pair = json.load(f)
            tgt_scene = capture.load_capture(os.path.join(pair_dir, "target",
                                                          "scene"))
            p_tgt = perception.perceive(tgt_scene, task, k=8, seed=0,
                                        camera=cam)
            inst = pair["target_instance"]
            T_gt = oracle.gt_relative_transform(
                pair["demo_object_poses"][inst],
                pair["target_object_poses"][inst])
            for setting in settings:
                cfg = SETTINGS[setting]
                dc = "auto" if cfg["kind"] == "adaptive" else False
                try:
                    orc = oracle.solve(p_demo, p_tgt, T_gt,
                                       np.asarray(pre["tcp_pos"]), seed=0,
                                       depth_consistent=dc)
                except ValueError:
                    continue
                T0 = procrustes(orc["demo_alk"], orc["target_alk"])
                src = p_demo["cands"].points3d
                tgt = p_tgt["cands"].points3d
                t0 = time.time()
                with ru.count_chamfer() as counter:
                    if cfg["kind"] in ("adaptive", "selective"):
                        out = adaptive_registration(
                            src, tgt, T0, np.asarray(orc["demo_alk"]),
                            axis_bound_deg=cfg.get("axis_bound_deg", 90.0))
                    elif cfg["kind"] == "base":
                        out = ru.uniform_widen_registration(src, tgt, T0)
                    else:
                        out = ru.uniform_widen_registration(
                            src, tgt, T0,
                            theta_max_deg=cfg["theta_max_deg"],
                            t_max_mm=cfg["t_max_mm"], mode=cfg["mode"])
                dt = time.time() - t0
                rot, trans = rr.map_errors(out["T"], T_gt,
                                           pair["demo_object_poses"][inst]["pos"],
                                           pair["target_object_poses"][inst]["pos"])
                rows.append({"task": task, "seed": seed, "setting": setting,
                             "reg_time_s": dt,
                             "n_chamfer_evals": int(counter["n"]),
                             "n_src": int(np.asarray(src).shape[0]),
                             "n_tgt": int(np.asarray(tgt).shape[0]),
                             "chamfer_init": float(out["chamfer_init"]),
                             "chamfer_final": float(out["chamfer_final"]),
                             "rot_err_deg": rot, "trans_err_m": trans})
            print("[bench %s %d] done" % (task, seed), flush=True)
    out_path = os.path.join(out_root, "bench_registration.json")
    camp._write_json(out_path, {"rows": rows})
    print("wrote %s (%d rows)" % (out_path, len(rows)))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="*", default=TASKS, choices=TASKS)
    p.add_argument("--settings", nargs="*", default=DEFAULT_SETTINGS,
                   choices=sorted(SETTINGS))
    p.add_argument("--tier", default="medium")
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--seed-end", type=int, default=1059, help="inclusive")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--link-base", action="store_true",
                   help="symlink results/campaign_a ours_full rollouts as the "
                        "base15_20 setting (no re-execution)")
    p.add_argument("--link-adaptive", action="store_true",
                   help="symlink existing results/campaign_a_adaptive "
                        "ours_full rollouts as the adaptive setting")
    p.add_argument("--verify-reuse", action="store_true",
                   help="re-run the default path on --verify-seeds and compare "
                        "bit-exactly against results/campaign_a")
    p.add_argument("--verify-seeds", type=int, nargs="*",
                   default=[1000, 1001, 1002])
    p.add_argument("--bench-registration", action="store_true",
                   help="time one registration call per setting on cached "
                        "scene pairs (no MuJoCo execution)")
    args = p.parse_args(argv)

    data_root = args.data_root or camp.TIER_DATA_ROOT[args.tier]
    seeds = list(range(args.seed_start, args.seed_end + 1))
    os.makedirs(args.out_root, exist_ok=True)

    if args.link_base:
        link_reuse(args.out_root, args.tasks, seeds, CAMPAIGN_A,
                   "base15_20", "base15_20 <- campaign_a")
    if args.link_adaptive:
        link_reuse(args.out_root, args.tasks, seeds, CAMPAIGN_A_ADAPTIVE,
                   "adaptive", "adaptive <- campaign_a_adaptive")
    if args.verify_reuse:
        verify_reuse(args.tasks, args.verify_seeds, args.tier, data_root,
                     args.out_root)
    if args.bench_registration:
        bench_registration(args.tasks, seeds, data_root, args.out_root)
    if args.link_base or args.link_adaptive or args.verify_reuse \
            or args.bench_registration:
        return 0

    t0 = time.time()
    total = 0
    for task in args.tasks:
        total += run_task(task, seeds, args.settings, args.tier, data_root,
                          args.out_root)
    print("done: %s, %d new rollouts in %.1f min"
          % (",".join(args.tasks), total, (time.time() - t0) / 60.0),
          flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
