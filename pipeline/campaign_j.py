"""Campaign J -- deployable mark-based baseline: MOKA (real VLM) vs ours.

Motivation
----------
Every published-baseline column in Campaign A is oracle-driven, and the only
baseline that was given a REAL VLM (`moka_vlm`, Campaign D / I) was given an
interface MOKA does not use: free-form pixel-coordinate regression.  MOKA's
actual interface is MARK-BASED VISUAL PROMPTING -- candidate points are
annotated on the image and the VLM SELECTS among them -- which is the same
interface class our pipeline uses.  Campaign J therefore re-implements MOKA
faithfully (`baselines/moka_marks.py`) and runs it head-to-head against our
pipeline driven by the SAME real VLM, on the SAME scenes, seeds, executor and
success checkers.

Arms
----
  moka_real         MOKA, mark selection by the real VLM, motion mapping in
                    the configuration frozen on non-eval data.
  moka_real_strict  identical VLM selections, MOKA-verbatim motion mapping
                    (30 antipodal proposals + snap the anchor to the closest
                    proposal).  No extra VLM queries.
  moka_real_wp      identical VLM selections, and MOKA's TILE channel is used
                    for the free-space pre-contact waypoint.  No extra queries.
  moka_marks_oracle MOKA's pipeline with ORACLE mark selection -- the ceiling
                    of the mark-based representation, isolating selection
                    error from candidate-generation and mapping error.
  ours_vlm          our pipeline (ALK -> Procrustes -> bounded registration)
                    with the same real VLM answering phi1/phi2/phi3.
References `ours_full` (oracle discrete) and `moka_oracle` (Campaign A's
translation-only oracle-pixel MOKA) are read from Campaign A / A2.

Grid: 5 tasks x 60 paired seeds {1000..1059}, medium tier.
  * `box_open` uses data_medium_a2/ and results/campaign_a2/ (Campaign A2
    narrowed its medium-tier yaw window -> new scene pairs).
  * `pour`     uses data_medium/ (scenes unchanged) and results/campaign_a2/
    (Campaign A2 replaced its success criterion with the orientation-aware
    one, which is live in simtasks/envs.py).
  * the other three use data_medium/ and results/campaign_a/.

Phases (all resumable; existing JSONs are skipped)
  freeze   -- markup rendering + prompt/config selection on NON-EVAL scenes
              (easy tier data/, seeds 1..5, and the demo scenes).  Writes
              results/campaign_j/freeze.json.  NOTHING after this is tuned.
  plan     -- MOKA high-level reasoning + demo-side mark selection, one query
              each per task (cached).
  select   -- per (task, seed): MOKA's target-side mark-selection query and
              our discrete phi query.  No simulator needed.
  execute  -- rollouts for all arms from the cached selections.  MUJOCO_GL=egl.
  report   -- results/tables/campaign_j.{md,tex} + FINDINGS inputs.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

SIMBENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SIMBENCH not in sys.path:
    sys.path.insert(0, SIMBENCH)

from alkbench import (alk_from_candidates, procrustes,          # noqa: E402
                      bounded_registration, transform_points,
                      rotation_angle_deg)
from baselines import common as bc                              # noqa: E402
from baselines import moka_marks as mm                          # noqa: E402
from pipeline import oracle                                     # noqa: E402
from stats import tests as st                                   # noqa: E402

OUT_ROOT = os.path.join(SIMBENCH, "results", "campaign_j")
TABLES_DIR = os.path.join(SIMBENCH, "results", "tables")
EASY_DATA_ROOT = os.path.join(SIMBENCH, "data")

TASKS_J = ("nut_loosen", "cap_twist", "rim_grasp", "pour", "box_open")
SEEDS_J = tuple(range(1000, 1060))
DEV_SEEDS = (1, 2, 3, 4, 5)          # easy tier, disjoint from the eval seeds
DEMO_PAIR_SEED = 0
K = 8
KMEANS_SEED = 0

DATA_ROOT_FOR = {t: os.path.join(SIMBENCH, "data_medium") for t in TASKS_J}
DATA_ROOT_FOR["box_open"] = os.path.join(SIMBENCH, "data_medium_a2")
REF_ROOT_FOR = {t: os.path.join(SIMBENCH, "results", "campaign_a")
                for t in TASKS_J}
REF_ROOT_FOR["pour"] = os.path.join(SIMBENCH, "results", "campaign_a2")
REF_ROOT_FOR["box_open"] = os.path.join(SIMBENCH, "results", "campaign_a2")

MOKA_ARMS = ("moka_real", "moka_real_strict", "moka_real_wp",
             "moka_marks_oracle", "moka_real_ourcands")
ALL_ARMS = MOKA_ARMS + ("ours_vlm",)

# ---------------------------------------------------------------------------
# FROZEN configuration (set by the `freeze` phase on non-eval data; see
# results/campaign_j/freeze.json and docs/REPRODUCE.md)
# ---------------------------------------------------------------------------

MOKA_CFG = {
    # FROZEN on non-eval scenes (results/campaign_j/freeze.json):
    #   motion mapping  snap="both", 30 antipodal proposals  (MOKA's own
    #     budget; snapping BOTH anchors to their closest proposal cancels the
    #     sampler's systematic top-surface offset and gave the lowest
    #     translation error of the six mapping variants swept);
    #   images  gridded full scene + zoom crop + demo reference;
    #   chain-of-thought OFF (MOKA Table VII asks for it, but on dev scenes it
    #     did not improve the selection and roughly quadrupled latency).
    "moka_real": dict(snap="both", n_proposals=30, demo_image=True,
                      zoom_image=True, cot=False),
    # MOKA verbatim on the anchor: snap only the TARGET anchor, 30 proposals.
    "moka_real_strict": dict(snap="target", n_proposals=30, demo_image=True,
                             zoom_image=True, cot=False),
    "moka_real_wp": dict(snap="both", n_proposals=30, demo_image=True,
                         zoom_image=True, cot=False),
    "moka_marks_oracle": dict(snap="both", n_proposals=30),
    # candidate-generation ablation: MOKA's interface and motion mapping, but
    # marked on OUR k-means candidate pool instead of MOKA's contour-FPS
    # points.  Separates "the interface is weaker" from "the CANDIDATES are
    # weaker".  Needs its own VLM query (different marks) -> own cache file.
    "moka_real_ourcands": dict(snap="both", n_proposals=30, demo_image=True,
                               zoom_image=True, cot=False,
                               candidates="kmeans"),
}

# our discrete prompts: the four Campaign-D prompts VERBATIM (already frozen
# there on non-eval data) plus a `pour` prompt frozen in this campaign's
# freeze phase.  The image-anchored rule for pour follows the measured
# agentview convention (world +x -> image DOWN, world +y -> image RIGHT), so
# the oracle's AXIS_REF = +x endpoint is the LOWER marker in the image.
POUR_PROMPT = (
    "grasp the elongated block across its long axis. The object is a single "
    "rectangular block lying on the table; its long axis runs between its "
    "two short end faces. phi1 = the marker closest to ONE end of the long "
    "axis; phi2 = the marker closest to the OTHER end of the long axis. The "
    "two markers must be the pair that is FARTHEST APART along the block. "
    "Choose as phi1 the end that is further to the LEFT in the image (if "
    "they are equally far left, the one nearer the bottom of the image). "
    "Always answer phi3 = 0.")


def task_prompts():
    from pipeline.campaign_d import TASK_PROMPTS
    p = dict(TASK_PROMPTS)
    p["pour"] = POUR_PROMPT
    return p


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def _json_safe(x):
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json_safe(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(_json_safe(obj), f, indent=1)
    os.replace(path + ".tmp", path)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def data_root(task):
    return DATA_ROOT_FOR[task]


def pair_ctx(task, seed, root=None, tier_noise=None):
    """(demo_capture, target_capture, ctx, pair) for one eval pair."""
    root = root or data_root(task)
    pdir = os.path.join(root, task, str(seed))
    pair = _read_json(os.path.join(pdir, "pair.json"))
    ddir = os.path.join(root, task, str(DEMO_PAIR_SEED), "demo")
    demo_cap = bc.load_capture(os.path.join(ddir, "scene"))
    tgt_cap = bc.load_capture(os.path.join(pdir, "target", "scene"))
    kf = _read_json(os.path.join(ddir, "keyframes.json"))
    wp = _read_json(os.path.join(ddir, "waypoints.json"))
    inst = pair["target_instance"]
    T_gt = oracle.gt_relative_transform(pair["demo_object_poses"][inst],
                                        pair["target_object_poses"][inst])
    ctx = bc.MethodContext(task=task, T_gt=T_gt, keyframes=kf, waypoints=wp,
                           k=K, kmeans_seed=KMEANS_SEED, sigma_px=0.0,
                           noise_seed=seed, random4_seed=seed,
                           sensor_noise=tier_noise)
    return demo_cap, tgt_cap, ctx, pair


def map_errors(T_est, T_gt, demo_pos, target_pos):
    T_est = np.asarray(T_est, dtype=np.float64)
    T_gt = np.asarray(T_gt, dtype=np.float64)
    rot = float(rotation_angle_deg(T_est[:3, :3] @ T_gt[:3, :3].T))
    mapped = transform_points(T_est, np.asarray(demo_pos, np.float64)[None])[0]
    return rot, float(np.linalg.norm(mapped
                                     - np.asarray(target_pos, np.float64)))


# ---------------------------------------------------------------------------
# phase: freeze  (NON-EVAL scenes only)
# ---------------------------------------------------------------------------

def _dev_ctx(task, seed):
    return pair_ctx(task, seed, root=EASY_DATA_ROOT)


def freeze(tasks=TASKS_J, seeds=DEV_SEEDS, out_root=OUT_ROOT):
    """Choose (a) MOKA's motion-mapping configuration, (b) whether MOKA's
    low-level query gets the demo reference image, (c) our `pour` discrete
    prompt -- all on NON-EVAL easy-tier scenes.  Writes freeze.json and one
    markup PNG per task for the record."""
    from pipeline.campaign_d import make_recording_solver, query_choice
    import baselines.moka_marks as M

    os.makedirs(os.path.join(out_root, "markups"), exist_ok=True)
    rec = {"dev_data_root": EASY_DATA_ROOT, "dev_seeds": list(seeds),
           "tasks": list(tasks)}

    # ---- (a) MOKA motion-mapping sweep, oracle selection (no VLM) ----------
    sweep = []
    for snap in ("none", "target", "both"):
        for nprop in (30, M.N_GRASP_PROPOSALS):
            rows = []
            for task in tasks:
                for seed in seeds:
                    try:
                        d, t, ctx, pair = _dev_ctx(task, seed)
                    except (IOError, OSError):
                        continue
                    r = M.run_moka_marks_oracle(d, t, ctx, snap=snap,
                                                n_proposals=nprop)
                    inst = pair["target_instance"]
                    rot, tr = map_errors(r["T_map"], ctx.T_gt,
                                         pair["demo_object_poses"][inst]["pos"],
                                         pair["target_object_poses"][inst]["pos"])
                    rows.append({"task": task, "seed": seed, "rot_deg": rot,
                                 "trans_m": tr,
                                 "anchor_err_m": r["anchor_err_m"],
                                 "best_mark_err_m":
                                     r["best_possible_mark_err_m"]})
            sweep.append({"snap": snap, "n_proposals": nprop,
                          "median_rot_deg": float(np.median(
                              [x["rot_deg"] for x in rows])),
                          "median_trans_mm": 1e3 * float(np.median(
                              [x["trans_m"] for x in rows])),
                          "median_anchor_mm": 1e3 * float(np.median(
                              [x["anchor_err_m"] for x in rows])),
                          "n": len(rows)})
            print("[freeze mapping] snap=%-6s nprop=%3d  rot %.1f deg  "
                  "trans %.0f mm  anchor %.0f mm (n=%d)"
                  % (snap, nprop, sweep[-1]["median_rot_deg"],
                     sweep[-1]["median_trans_mm"],
                     sweep[-1]["median_anchor_mm"], sweep[-1]["n"]),
                  flush=True)
    rec["moka_mapping_sweep"] = sweep
    best = min(sweep, key=lambda s: (s["median_trans_mm"],
                                     s["median_rot_deg"]))
    rec["moka_mapping_chosen"] = {"snap": best["snap"],
                                  "n_proposals": best["n_proposals"]}

    # ---- markup renders (for the record / visual check) --------------------
    for task in tasks:
        try:
            d, t, ctx, _ = _dev_ctx(task, seeds[0])
        except (IOError, OSError):
            continue
        ts = M._side(t, ctx, "target")
        img = M.markup_for(ts, t)
        with open(os.path.join(out_root, "markups",
                               "%s_target.png" % task), "wb") as f:
            f.write(M.encode_png(img))

    # ---- (b) MOKA low-level prompt variants (real VLM, dev scenes) --------
    variants = []
    # demo_image is always ON (the task's fairness floor: MOKA is shown the
    # same demonstration our pipeline is).  The sweep is over the two
    # legibility/prompt switches.
    for zoom_image, cot in ((False, False), (True, False), (False, True),
                            (True, True)):
        demo_image = True
        rows = []
        for task in tasks:
            subtask, dsel = None, None
            for seed in seeds:
                try:
                    d, t, ctx, pair = _dev_ctx(task, seed)
                except (IOError, OSError):
                    continue
                t0 = time.time()
                try:
                    r = M.run_moka_real(d, t, ctx, variant="dev",
                                        subtask=subtask,
                                        demo_selection=dsel,
                                        demo_image=demo_image,
                                        zoom_image=zoom_image, cot=cot,
                                        snap=rec["moka_mapping_chosen"]["snap"],
                                        n_proposals=rec["moka_mapping_chosen"][
                                            "n_proposals"])
                    subtask = r["subtask"]
                    dsel = r["demo_selection"]
                    rows.append({"task": task, "seed": seed, "ok": True,
                                 "mark_correct": r["mark_choice_correct"],
                                 "selected_mark_err_m":
                                     r["selected_mark_err_m"],
                                 "yaw_err_deg": r["yaw_err_deg"],
                                 "anchor_err_m": r["anchor_err_m"],
                                 "t": round(time.time() - t0, 2)})
                except Exception as e:
                    rows.append({"task": task, "seed": seed, "ok": False,
                                 "error": repr(e)})
                print("[freeze prompt zoom=%s cot=%s] %-11s %d ok=%s "
                      "mark_ok=%s err=%s" % (
                          zoom_image, cot, task, seed, rows[-1]["ok"],
                          rows[-1].get("mark_correct"),
                          ("%.0fmm" % (1e3 * rows[-1]["selected_mark_err_m"]))
                          if rows[-1]["ok"] else "-"), flush=True)
        ok = [r for r in rows if r["ok"]]
        variants.append({
            "demo_image": demo_image, "zoom_image": zoom_image, "cot": cot,
            "n": len(rows), "n_ok": len(ok),
            "mark_correct": int(sum(1 for r in ok if r["mark_correct"])),
            "median_mark_err_mm": (1e3 * float(np.median(
                [r["selected_mark_err_m"] for r in ok])) if ok else None),
            "median_abs_yaw_err_deg": (float(np.median(
                [abs(r["yaw_err_deg"]) for r in ok])) if ok else None),
            "rows": rows})
        print("VARIANT zoom=%s cot=%s: mark-correct %d/%d, median mark err "
              "%s mm, median |yaw err| %s deg"
              % (zoom_image, cot, variants[-1]["mark_correct"], len(ok),
                 variants[-1]["median_mark_err_mm"],
                 variants[-1]["median_abs_yaw_err_deg"]), flush=True)
    rec["moka_prompt_variants"] = variants
    chosen = min(variants, key=lambda v: (v["median_mark_err_mm"]
                                          if v["median_mark_err_mm"]
                                          is not None else 1e9))
    rec["moka_demo_image_chosen"] = bool(chosen["demo_image"])
    rec["moka_zoom_image_chosen"] = bool(chosen["zoom_image"])
    rec["moka_cot_chosen"] = bool(chosen["cot"])

    # ---- (c) our `pour` discrete prompt on dev scenes ----------------------
    solver, log = make_recording_solver()
    prompts = task_prompts()
    pour_rows = []
    for seed in seeds:
        try:
            d, t, ctx, pair = _dev_ctx("pour", seed)
        except (IOError, OSError):
            continue
        p_d = ctx.percept(d, "demo")
        p_t = ctx.percept(t, "target")
        orc = oracle.solve(p_d, p_t, ctx.T_gt,
                           ctx.keyframe_tcp("pre_grasp"), seed=KMEANS_SEED)
        q = query_choice_with("pour", bc.load_rgb(t, p_t["camera"]),
                              p_t["cands"], solver, log, prompts["pour"])
        oc = orc["target_choice"]
        pour_rows.append({
            "seed": seed, "oracle": {k: oc[k] for k in ("phi1", "phi2",
                                                        "phi3")},
            "vlm": q["choice"],
            "pair_set": (q["choice"] is not None
                         and {q["choice"]["phi1"], q["choice"]["phi2"]}
                         == {oc["phi1"], oc["phi2"]}),
            "exact": (q["choice"] is not None
                      and q["choice"]["phi1"] == oc["phi1"]
                      and q["choice"]["phi2"] == oc["phi2"])})
        print("[freeze pour-prompt] seed=%d oracle=(%s,%s) vlm=%s exact=%s"
              % (seed, oc["phi1"] + 1, oc["phi2"] + 1,
                 None if q["choice"] is None else
                 (q["choice"]["phi1"] + 1, q["choice"]["phi2"] + 1),
                 pour_rows[-1]["exact"]), flush=True)
    rec["pour_prompt"] = {"prompt": POUR_PROMPT, "rows": pour_rows,
                          "n_exact": sum(1 for r in pour_rows if r["exact"]),
                          "n_pair_set": sum(1 for r in pour_rows
                                            if r["pair_set"]),
                          "n": len(pour_rows)}
    rec["ours_task_prompts"] = prompts
    _write_json(os.path.join(out_root, "freeze.json"), rec)
    print("\nFROZEN: moka mapping %r, demo_image=%s zoom=%s cot=%s, pour "
          "prompt exact %d/%d"
          % (rec["moka_mapping_chosen"], rec["moka_demo_image_chosen"],
             rec["moka_zoom_image_chosen"], rec["moka_cot_chosen"],
             rec["pour_prompt"]["n_exact"], rec["pour_prompt"]["n"]))
    return rec


def query_choice_with(task, rgb, cands, solver, log, prompt):
    """campaign_d.query_choice with an explicit prompt string."""
    n0 = len(log)
    t0 = time.time()
    out = {"parse_failed": False, "error": None}
    try:
        ch = solver.solve(image=rgb, candidates_uv=cands.centers2d,
                          task_description=prompt)
        out["choice"] = {"phi1": ch.phi1, "phi2": ch.phi2,
                         "phi3": int(ch.phi3)}
    except Exception as e:
        out["parse_failed"] = True
        out["error"] = repr(e)
        out["choice"] = None
    out["latency_s"] = round(time.time() - t0, 2)
    out["n_api_calls"] = len(log) - n0
    out["raw_replies"] = [r["text"] for r in log[n0:]]
    return out


# ---------------------------------------------------------------------------
# phase: plan  (MOKA high-level + demo-side mark selection, once per task)
# ---------------------------------------------------------------------------

def plan_task(task, out_root=OUT_ROOT, require=True):
    path = os.path.join(out_root, task, "moka_plan.json")
    if os.path.exists(path):
        return _read_json(path)
    if not require:
        return None
    d, t, ctx, _ = pair_ctx(task, SEEDS_J[0])
    dside = mm.demo_side(d, ctx)
    dside_km = mm.demo_side(d, ctx, candidates="kmeans")
    rgb_d = bc.load_rgb(d, dside["percept"]["camera"])
    t0 = time.time()
    subtask, hinfo = mm.query_high_level(
        rgb_d, mm.TASK_INSTRUCTIONS[task])
    k = int(np.asarray(dside["marks_uv"]).shape[0])
    dsel, dinfo = mm.query_low_level(
        mm.markup_for(dside, d), subtask, k, demo_image=None,
        zoom_image=(mm.zoom_for(dside, d)
                    if MOKA_CFG["moka_real"]["zoom_image"] else None),
        cot=MOKA_CFG["moka_real"]["cot"])
    M = np.asarray(dside["marks_xyz"])
    ig_oracle = int(np.argmin(np.linalg.norm(M - dside["g_d"], axis=1)))
    dsel_km, dinfo_km = mm.query_low_level(
        mm.markup_for(dside_km, d), subtask,
        int(np.asarray(dside_km["marks_uv"]).shape[0]), demo_image=None,
        zoom_image=(mm.zoom_for(dside_km, d)
                    if MOKA_CFG["moka_real"]["zoom_image"] else None),
        cot=MOKA_CFG["moka_real"]["cot"])
    rec = {"task": task, "subtask": subtask,
           "demo_selection_ourcands": dsel_km,
           "high_level_attempts": hinfo["attempts"],
           "high_level_reply": hinfo["replies"][-1][-1500:],
           "demo_selection": dsel,
           "demo_query_attempts": dinfo["attempts"],
           "demo_reply": dinfo["replies"][-1][-1500:],
           "demo_marks_xyz": M.tolist(),
           "demo_grasp": np.asarray(dside["g_d"]).tolist(),
           "demo_oracle_grasp_mark": ig_oracle,
           "demo_mark_correct": bool(dsel["grasp_keypoint"] == ig_oracle),
           "demo_mark_err_m": float(np.linalg.norm(
               M[dsel["grasp_keypoint"]] - dside["g_d"])),
           "n_vlm_calls": (hinfo["attempts"] + dinfo["attempts"]
                           + dinfo_km["attempts"]),
           "time_s": round(time.time() - t0, 1)}
    _write_json(path, rec)
    print("[%s plan] object_grasped=%r motion=%r demo grasp mark P%d "
          "(oracle P%d, err %.0f mm)"
          % (task, subtask["object_grasped"], subtask["motion_direction"],
             dsel["grasp_keypoint"] + 1, ig_oracle + 1,
             1e3 * rec["demo_mark_err_m"]), flush=True)
    return rec


# ---------------------------------------------------------------------------
# phase: select  (one MOKA query + one ours query per (task, seed))
# ---------------------------------------------------------------------------

def select_task(task, seeds=SEEDS_J, out_root=OUT_ROOT,
                do_ourcands=True):
    from pipeline.campaign_d import make_recording_solver
    plan = plan_task(task, out_root)
    prompts = task_prompts()
    solver, log = make_recording_solver()

    # our cached demo-side discrete query (mirrors campaign_d)
    demo_path = os.path.join(out_root, task, "ours_demo_query.json")
    d0, t0c, ctx0, _ = pair_ctx(task, seeds[0])
    p_d0 = ctx0.percept(d0, "demo")
    if not os.path.exists(demo_path):
        d1o, d2o = oracle.demo_axial_choice(p_d0["cands"])
        q = query_choice_with(task, bc.load_rgb(d0, p_d0["camera"]),
                              p_d0["cands"], solver, log, prompts[task])
        _write_json(demo_path, {"task": task, "side": "demo",
                                "oracle_choice": {"phi1": d1o, "phi2": d2o,
                                                  "phi3": 0},
                                "task_prompt": prompts[task], "vlm": q})
    demo_q = _read_json(demo_path)

    n_new = 0
    for seed in seeds:
        sdir = os.path.join(out_root, task, str(seed))
        mpath = os.path.join(sdir, "moka_query.json")
        kpath = os.path.join(sdir, "moka_ourcands_query.json")
        opath = os.path.join(sdir, "ours_query.json")
        want = [mpath, opath] + ([kpath] if do_ourcands else [])
        if all(os.path.exists(x) for x in want):
            continue
        t_start = time.time()
        d, t, ctx, pair = pair_ctx(task, seed)
        inst = pair["target_instance"]
        demo_pose = pair["demo_object_poses"][inst]
        target_pose = pair["target_object_poses"][inst]

        if not os.path.exists(mpath):
            tside = mm._side(t, ctx, "target")
            dside = mm.demo_side(d, ctx)
            k = int(np.asarray(tside["marks_uv"]).shape[0])
            dimg = (mm.demo_reference_image(dside, d)
                    if MOKA_CFG["moka_real"]["demo_image"] else None)
            zimg = (mm.zoom_for(tside, t)
                    if MOKA_CFG["moka_real"]["zoom_image"] else None)
            rec = {"task": task, "seed": seed, "k_marks": k,
                   "camera": tside["percept"]["camera"],
                   "model": mm._model(None), "tier": "medium"}
            try:
                sel, info = mm.query_low_level(mm.markup_for(tside, t),
                                               plan["subtask"], k,
                                               demo_image=dimg,
                                               zoom_image=zimg,
                                               cot=MOKA_CFG["moka_real"]["cot"])
                rec["selection"] = sel
                rec["attempts"] = info["attempts"]
                rec["raw"] = info["raw"]
                rec["reply"] = info["replies"][-1][-1500:]
                rec["parse_failed"] = False
            except Exception as e:
                rec["selection"] = None
                rec["parse_failed"] = True
                rec["error"] = repr(e)
                rec["attempts"] = 2
            # scoring info (never seen by the method)
            g_gt = transform_points(ctx.T_gt,
                                    np.asarray(dside["g_d"])[None])[0]
            dists = np.linalg.norm(np.asarray(tside["marks_xyz"]) - g_gt,
                                   axis=1)
            rec["oracle_grasp_mark"] = int(np.argmin(dists))
            rec["best_possible_mark_err_m"] = float(dists.min())
            if rec["selection"] is not None:
                rec["mark_correct"] = bool(
                    rec["selection"]["grasp_keypoint"]
                    == rec["oracle_grasp_mark"])
                rec["selected_mark_err_m"] = float(
                    dists[rec["selection"]["grasp_keypoint"]])
            rec["time_s"] = round(time.time() - t_start, 1)
            _write_json(mpath, rec)
            n_new += 1

        if do_ourcands and not os.path.exists(kpath):
            tside = mm._side(t, ctx, "target", candidates="kmeans")
            dside = mm.demo_side(d, ctx, candidates="kmeans")
            k = int(np.asarray(tside["marks_uv"]).shape[0])
            dimg = (mm.demo_reference_image(dside, d)
                    if MOKA_CFG["moka_real"]["demo_image"] else None)
            rec = {"task": task, "seed": seed, "k_marks": k,
                   "candidates": "kmeans",
                   "camera": tside["percept"]["camera"], "tier": "medium"}
            try:
                sel, info = mm.query_low_level(
                    mm.markup_for(tside, t), plan["subtask"], k,
                    demo_image=dimg,
                    zoom_image=(mm.zoom_for(tside, t)
                                if MOKA_CFG["moka_real"]["zoom_image"]
                                else None),
                    cot=MOKA_CFG["moka_real"]["cot"])
                rec.update({"selection": sel, "attempts": info["attempts"],
                            "raw": info["raw"], "parse_failed": False,
                            "reply": info["replies"][-1][-1500:]})
            except Exception as e:
                rec.update({"selection": None, "parse_failed": True,
                            "error": repr(e), "attempts": 2})
            g_gt = transform_points(ctx.T_gt,
                                    np.asarray(dside["g_d"])[None])[0]
            dists = np.linalg.norm(np.asarray(tside["marks_xyz"]) - g_gt,
                                   axis=1)
            rec["oracle_grasp_mark"] = int(np.argmin(dists))
            rec["best_possible_mark_err_m"] = float(dists.min())
            if rec["selection"] is not None:
                rec["mark_correct"] = bool(
                    rec["selection"]["grasp_keypoint"]
                    == rec["oracle_grasp_mark"])
                rec["selected_mark_err_m"] = float(
                    dists[rec["selection"]["grasp_keypoint"]])
            _write_json(kpath, rec)
            n_new += 1

        if not os.path.exists(opath):
            p_t = ctx.percept(t, "target")
            orc = oracle.solve(p_d0, p_t, ctx.T_gt,
                               ctx.keyframe_tcp("pre_grasp"), seed=KMEANS_SEED)
            q = query_choice_with(task, bc.load_rgb(t, p_t["camera"]),
                                  p_t["cands"], solver, log, prompts[task])
            oc = {kk: orc["target_choice"][kk] for kk in ("phi1", "phi2",
                                                          "phi3")}
            rec = {"task": task, "seed": seed, "side": "target",
                   "camera": p_t["camera"], "oracle_choice": oc,
                   "demo_vlm_choice": demo_q["vlm"]["choice"],
                   "demo_oracle_choice": demo_q["oracle_choice"],
                   "vlm": q, "T_gt": ctx.T_gt.tolist(),
                   "demo_pos": demo_pose["pos"],
                   "target_pos": target_pose["pos"]}
            if q["choice"] is not None:
                v = q["choice"]
                rec["agree"] = {
                    "phi1": v["phi1"] == oc["phi1"],
                    "phi2": v["phi2"] == oc["phi2"],
                    "pair_set": {v["phi1"], v["phi2"]} == {oc["phi1"],
                                                           oc["phi2"]},
                    "joint_raw": (v["phi1"] == oc["phi1"]
                                  and v["phi2"] == oc["phi2"]
                                  and v["phi3"] == oc["phi3"])}
            _write_json(opath, rec)
            n_new += 1
        print("[%s %d select] moka_mark_ok=%s ours_pair_ok=%s (%.1fs)"
              % (task, seed,
                 _read_json(mpath).get("mark_correct"),
                 (_read_json(opath).get("agree") or {}).get("pair_set"),
                 time.time() - t_start), flush=True)
    return n_new


# ---------------------------------------------------------------------------
# phase: execute
# ---------------------------------------------------------------------------

def _fail_stage(p_d_ok, p_t_ok, rot, trans, grasped):
    if not (p_d_ok and p_t_ok):
        return "perception"
    if trans is not None and (trans > 0.05 or rot > 20.0):
        return "mapping"
    if grasped is False:
        return "grasp"
    return "post_action"


def _tile_waypoints(waypoints, sel, tside, g_t):
    """MOKA's tile channel (adaptation A8): replace the demo template's
    free-space `approach` waypoint by the centre of the VLM-selected
    pre-contact tile at the declared height."""
    wps = [dict(w) for w in waypoints]
    tile = (sel or {}).get("pre_contact_tile")
    if not tile:
        return wps, None
    cd = tside["cd"]
    h, w = np.asarray(cd["depth"]).shape[:2]
    try:
        uv = mm.tile_centre_uv(tile, w, h)
    except ValueError:
        return wps, None
    K = np.asarray(cd["K"], dtype=np.float64)
    T_wc = np.asarray(cd["T_world_cam"], dtype=np.float64)
    z_ref = float(g_t[2])
    if str(sel.get("pre_contact_height", "same")).startswith("ab"):
        z_ref += mm.ABOVE_HEIGHT_M
    # ray through the tile centre, intersected with the plane z = z_ref
    d_cam = np.array([(uv[0] - K[0, 2]) / K[0, 0],
                      (uv[1] - K[1, 2]) / K[1, 1], 1.0])
    d_w = T_wc[:3, :3] @ d_cam
    o_w = T_wc[:3, 3]
    if abs(d_w[2]) < 1e-6:
        return wps, None
    lam = (z_ref - o_w[2]) / d_w[2]
    if lam <= 0:
        return wps, None
    p = o_w + lam * d_w
    for i, wp in enumerate(wps):
        if wp.get("label") == "approach":
            wps[i] = dict(wp, pos=[float(p[0]), float(p[1]), float(p[2])])
            return wps, p.tolist()
    return wps, None


def execute_task(task, seeds=SEEDS_J, arms=ALL_ARMS, out_root=OUT_ROOT):
    from pipeline import campaign as cg
    from pipeline import retarget_runner as rr
    from simtasks import envs, motion

    root = data_root(task)
    plan = plan_task(task, out_root, require=("moka_marks_oracle" != arms
                                              and len(arms) > 1))
    dqp = os.path.join(out_root, task, "ours_demo_query.json")
    demo_q = _read_json(dqp) if os.path.exists(dqp) else None
    demo = rr.load_demo(task, root)
    env = cg.make_tier_env(task, "medium")
    n_new = 0
    try:
        for seed in seeds:
            sdir = os.path.join(out_root, task, str(seed))
            todo = [a for a in arms
                    if not os.path.exists(os.path.join(
                        sdir, "rollout_%s.json" % a))]
            if not todo:
                continue
            d, t, ctx, pair = pair_ctx(task, seed, root=root)
            inst = pair["target_instance"]
            demo_pose = pair["demo_object_poses"][inst]
            target_pose = pair["target_object_poses"][inst]
            p_d = ctx.percept(d, "demo")
            p_t = ctx.percept(t, "target")
            mp = os.path.join(sdir, "moka_query.json")
            kp = os.path.join(sdir, "moka_ourcands_query.json")
            op = os.path.join(sdir, "ours_query.json")
            mq = _read_json(mp) if os.path.exists(mp) else None
            kq = _read_json(kp) if os.path.exists(kp) else None
            oq = _read_json(op) if os.path.exists(op) else None
            todo = [a for a in todo
                    if not (a == "ours_vlm" and oq is None)
                    and not (a == "moka_real_ourcands" and kq is None)
                    and not (a in ("moka_real", "moka_real_strict",
                                   "moka_real_wp") and mq is None)]
            if not todo:
                continue

            for arm in todo:
                t0 = time.time()
                res = {"task": task, "seed": seed, "method": arm,
                       "variant": arm, "tier": "medium", "success": False,
                       "failure_stage": None, "camera": p_t["camera"],
                       "T_gt": ctx.T_gt.tolist(), "data_root": root,
                       "perception": {
                           "demo": {"sanity_ok": p_d["sanity_ok"],
                                    "centroid_err_m": p_d["centroid_err_m"]},
                           "target": {"sanity_ok": p_t["sanity_ok"],
                                      "centroid_err_m":
                                          p_t["centroid_err_m"]}}}
                try:
                    if arm == "ours_vlm":
                        res.update(_ours_vlm_map(task, ctx, p_d, p_t, demo_q,
                                                 oq))
                    else:
                        res.update(_moka_map(
                            arm, task, d, t, ctx,
                            kq if arm == "moka_real_ourcands" else mq, plan))
                except Exception as e:
                    traceback.print_exc()
                    res["failure_stage"] = "method_error"
                    res["error"] = repr(e)

                if res.get("T_map") is None:
                    res["time_s"] = round(time.time() - t0, 1)
                    _write_json(os.path.join(sdir, "rollout_%s.json" % arm),
                                res)
                    n_new += 1
                    print("[%s %d %-17s] success=False stage=%s (no exec)"
                          % (task, seed, arm, res["failure_stage"]),
                          flush=True)
                    continue

                T_map = np.asarray(res["T_map"], dtype=np.float64)
                rot, tr = map_errors(T_map, ctx.T_gt, demo_pose["pos"],
                                     target_pose["pos"])
                res["rot_err_deg"] = rot
                res["trans_err_m"] = tr
                wps = res.pop("_waypoints", None)
                if wps is None:
                    if arm.startswith("moka"):
                        wps = rr.retarget_waypoint_dicts(
                            demo["waypoints"], T_map,
                            demo_grasp=np.asarray(res["demo_grasp"]),
                            target_grasp=np.asarray(res["target_grasp"]))
                    else:
                        wps = rr.retarget_waypoint_dicts(demo["waypoints"],
                                                         T_map)
                envs.reset_with_seed(env, pair["target_seed"])
                ex = motion.execute_waypoints(
                    env, wps, grasp_check=rr._grasp_check(task))
                success = bool(envs.success_checker(task)(env))
                res.update({"grasped": ex["grasped"],
                            "waypoints_converged": ex["converged"],
                            "n_exec_steps": ex["n_steps"],
                            "success": success})
                if task == "pour":
                    res["pour_orientation"] = _json_safe(
                        envs.pour_orientation_trace(env))
                if not success:
                    res["failure_stage"] = _fail_stage(
                        p_d["sanity_ok"], p_t["sanity_ok"], rot, tr,
                        ex["grasped"])
                res["time_s"] = round(time.time() - t0, 1)
                _write_json(os.path.join(sdir, "rollout_%s.json" % arm), res)
                n_new += 1
                print("[%s %d %-17s] success=%s stage=%-11s rot=%5.1f "
                      "trans=%5.1fmm (%.1fs)"
                      % (task, seed, arm, success, str(res["failure_stage"]),
                         rot, 1e3 * tr, res["time_s"]), flush=True)
    finally:
        env.close()
    return n_new


def _moka_map(arm, task, demo_cap, target_cap, ctx, mq, plan):
    """Build one MOKA arm's T_map from the CACHED mark selection."""
    cfg = MOKA_CFG[arm]
    cands = cfg.get("candidates", "contour_fps")
    if arm == "moka_marks_oracle":
        r = mm.run_moka_marks_oracle(demo_cap, target_cap, ctx, variant=arm,
                                     snap=cfg["snap"],
                                     n_proposals=cfg["n_proposals"],
                                     candidates=cands)
        return _moka_result(r, n_vlm=0)
    if mq.get("selection") is None:
        return {"T_map": None, "failure_stage": "vlm_selection",
                "error": mq.get("error"), "moka": {"parse_failed": True}}
    sel = dict(mq["selection"])
    r = mm.run_moka_real(
        demo_cap, target_cap, ctx, variant=arm, subtask=plan["subtask"],
        demo_selection=(plan["demo_selection_ourcands"]
                        if cands == "kmeans" else plan["demo_selection"]),
        selection=sel, snap=cfg["snap"], n_proposals=cfg["n_proposals"],
        candidates=cands)
    out = _moka_result(r, n_vlm=mq.get("attempts", 1))
    if arm == "moka_real_wp":
        tside = mm._side(target_cap, ctx, "target",
                         n_proposals=cfg["n_proposals"])
        from pipeline import retarget_runner as rr
        base = rr.retarget_waypoint_dicts(
            ctx.waypoints, np.asarray(r["T_map"]),
            demo_grasp=np.asarray(r["demo_grasp"]),
            target_grasp=np.asarray(r["target_grasp"]))
        wps, tile_pos = _tile_waypoints(base, sel, tside,
                                        np.asarray(r["target_grasp"]))
        out["_waypoints"] = wps
        out["moka"]["tile_waypoint"] = tile_pos
        out["moka"]["pre_contact_tile"] = sel.get("pre_contact_tile")
    return out


def _moka_result(r, n_vlm):
    keys = ("demo_grasp", "target_grasp", "selection", "demo_selection",
            "delta_yaw_deg", "axis_channel_used", "anchor_err_m",
            "selected_mark_err_m", "best_possible_mark_err_m",
            "mark_choice_correct", "oracle_mark", "yaw_err_deg",
            "gt_yaw_deg", "grasp_proposal_dist_target_m",
            "grasp_proposal_snapped", "n_proposals_target", "n_marks",
            "mask_pixels_target", "demo_axis_deg", "target_axis_deg")
    out = {"T_map": np.asarray(r["T_map"]).tolist(),
           "demo_grasp": r["demo_grasp"], "target_grasp": r["target_grasp"],
           "n_vlm_calls_rollout": n_vlm,
           "moka": {k: r.get(k) for k in keys}}
    return out


def _ours_vlm_map(task, ctx, p_d, p_t, demo_q, oq):
    """Our pipeline's T_map from the CACHED VLM discrete answers
    (registration ON, grasp-point correction OFF -- the correction consumes
    oracle phi4/phi5, so it is disabled exactly as in Campaign D)."""
    dchoice = demo_q["vlm"]["choice"]
    tchoice = oq["vlm"]["choice"]
    out = {"vlm": {"demo_choice": dchoice, "target_choice": tchoice,
                   "oracle_choice": oq["oracle_choice"],
                   "agree": oq.get("agree")},
           "registration": True, "correction": False}
    if dchoice is None or tchoice is None:
        out["T_map"] = None
        out["failure_stage"] = "vlm_discrete"
        return out
    try:
        demo_alk = alk_from_candidates(p_d["cands"], dchoice["phi1"],
                                       dchoice["phi2"], phi3=False)
        tgt_alk = alk_from_candidates(p_t["cands"], tchoice["phi1"],
                                      tchoice["phi2"],
                                      phi3=bool(tchoice["phi3"]))
    except ValueError as e:
        out["T_map"] = None
        out["failure_stage"] = "alk_degenerate"
        out["error"] = str(e)
        return out
    T0 = procrustes(demo_alk, tgt_alk)
    reg = bounded_registration(p_d["cands"].points3d, p_t["cands"].points3d,
                               T0)
    out["T_map"] = np.asarray(reg["T"]).tolist()
    out["T_init"] = np.asarray(T0).tolist()
    out["chamfer_before_m"] = float(reg["chamfer_init"])
    out["chamfer_after_m"] = float(reg["chamfer_final"])
    cond = bc.conditioning(np.asarray(demo_alk))
    out["conditioning"] = cond["sigma23"]
    out["n_vlm_calls_rollout"] = 1
    return out


# ---------------------------------------------------------------------------
# phase: report
# ---------------------------------------------------------------------------

def _wilson(k, n):
    if n == 0:
        return "-"
    lo, hi = st.wilson_ci(k, n)
    return "%d/%d = %.2f [%.2f, %.2f]" % (k, n, k / n, lo, hi)


def _ref_success(task, seeds, method):
    out = {}
    for seed in seeds:
        p = os.path.join(REF_ROOT_FOR[task], task, str(seed),
                         "rollout_%s.json" % method)
        if os.path.exists(p):
            out[seed] = int(bool(_read_json(p).get("success")))
    return out


def collect(tasks=TASKS_J, seeds=SEEDS_J, out_root=OUT_ROOT):
    data = {}
    for task in tasks:
        rows = []
        for seed in seeds:
            row = {"seed": seed}
            sdir = os.path.join(out_root, task, str(seed))
            for arm in ALL_ARMS:
                p = os.path.join(sdir, "rollout_%s.json" % arm)
                if os.path.exists(p):
                    row[arm] = _read_json(p)
            for nm, fn in (("moka_q", "moka_query.json"),
                           ("moka_oc_q", "moka_ourcands_query.json"),
                           ("ours_q", "ours_query.json")):
                p = os.path.join(sdir, fn)
                if os.path.exists(p):
                    row[nm] = _read_json(p)
            rows.append(row)
        pp = os.path.join(out_root, task, "moka_plan.json")
        data[task] = {"rows": rows,
                      "plan": _read_json(pp) if os.path.exists(pp) else None,
                      "ref_ours_full": _ref_success(task, seeds, "ours_full"),
                      "ref_moka_oracle": _ref_success(task, seeds,
                                                      "moka_oracle")}
    return data


def report(tasks=TASKS_J, seeds=SEEDS_J, out_root=OUT_ROOT):
    data = collect(tasks, seeds, out_root)
    ARMS_TBL = ["ours_vlm", "moka_real", "moka_real_strict", "moka_real_wp",
                "moka_real_ourcands", "moka_marks_oracle"]
    md = ["# Campaign J -- deployable mark-based baseline (real VLM): MOKA vs "
          "ours\n",
          "Medium tier, paired seeds %d..%d (N=%d/task), model %s.  "
          "`box_open` uses data_medium_a2 + Campaign-A2 references; `pour` "
          "uses the corrected orientation-aware criterion (Campaign A2).  "
          "Raw per-rollout records under results/campaign_j/.\n"
          % (seeds[0], seeds[-1], len(seeds), mm.VLM_DEFAULTS["model"])]
    tex = []
    summary = {"arms": ARMS_TBL, "per_task": {}, "pooled": {}}

    # ---- table 1: success ---------------------------------------------------
    md.append("## 1. End-to-end success\n")
    md.append("| task | " + " | ".join(ARMS_TBL)
              + " | ours_full (oracle ref) | moka_oracle (ref) |")
    md.append("|---" * (len(ARMS_TBL) + 3) + "|")
    pooled = {a: [0, 0] for a in ARMS_TBL}
    pooled_ref = {"ours_full": [0, 0], "moka_oracle": [0, 0]}
    succ_vec = {a: {} for a in ARMS_TBL}
    for task in tasks:
        d = data[task]
        cells = []
        for a in ARMS_TBL:
            s = {r["seed"]: int(bool(r[a]["success"]))
                 for r in d["rows"] if a in r}
            succ_vec[a][task] = s
            pooled[a][0] += sum(s.values())
            pooled[a][1] += len(s)
            cells.append(_wilson(sum(s.values()), len(s)))
        for nm in ("ours_full", "moka_oracle"):
            ref = d["ref_%s" % nm]
            pooled_ref[nm][0] += sum(ref.values())
            pooled_ref[nm][1] += len(ref)
            cells.append(_wilson(sum(ref.values()), len(ref)))
        md.append("| %s | %s |" % (task, " | ".join(cells)))
        summary["per_task"][task] = {
            a: [sum(succ_vec[a][task].values()), len(succ_vec[a][task])]
            for a in ARMS_TBL}
        summary["per_task"][task].update(
            {nm: [sum(d["ref_%s" % nm].values()), len(d["ref_%s" % nm])]
             for nm in ("ours_full", "moka_oracle")})
    md.append("| **pooled (all rows)** | %s | %s |" % (
        " | ".join("**%s**" % _wilson(*pooled[a]) for a in ARMS_TBL),
        " | ".join(_wilson(*pooled_ref[nm])
                   for nm in ("ours_full", "moka_oracle"))))
    summary["pooled"] = {a: pooled[a] for a in ARMS_TBL}
    summary["pooled"].update(pooled_ref)

    # pooled over the tasks whose real-VLM arms are COMPLETE -- the honest
    # headline, because an incomplete task contributes a tiny, unrepresentative
    # n to the "all rows" pooled cell
    complete = [t for t in tasks
                if len(succ_vec["ours_vlm"].get(t, {})) >= 55
                and len(succ_vec["moka_real"].get(t, {})) >= 55]
    summary["complete_tasks"] = complete
    if complete and len(complete) != len(tasks):
        cells = []
        for a in ARMS_TBL:
            k = sum(sum(succ_vec[a].get(t, {}).values()) for t in complete)
            n = sum(len(succ_vec[a].get(t, {})) for t in complete)
            cells.append("**%s**" % _wilson(k, n))
            summary.setdefault("pooled_complete", {})[a] = [k, n]
        for nm in ("ours_full", "moka_oracle"):
            k = sum(data[t]["ref_%s" % nm].get(s_, 0)
                    for t in complete for s_ in seeds)
            n = sum(len(data[t]["ref_%s" % nm]) for t in complete)
            cells.append(_wilson(k, n))
            summary.setdefault("pooled_complete", {})[nm] = [k, n]
        md.append("| **pooled (%d complete real-VLM tasks: %s)** | %s |"
                  % (len(complete), ", ".join(complete), " | ".join(cells)))
        md.append("")
        md.append("Rows whose real-VLM arms are INCOMPLETE (the local vLLM "
                  "serving environment was removed from the host mid-run, so "
                  "no further VLM queries could be issued): %s.  Their "
                  "`moka_marks_oracle` column needs no VLM and is complete.\n"
                  % ", ".join(t for t in tasks if t not in complete))
    else:
        md.append("")

    # ---- table 2: paired McNemar ours_vlm vs each MOKA arm ------------------
    md.append("## 2. Paired McNemar (exact) -- ours_vlm vs each MOKA arm, "
              "Holm-corrected across the %d tasks\n" % len(tasks))
    md.append("| comparison | task | b (ours win) | c (moka win) | p | "
              "p_Holm |")
    md.append("|---|---|---|---|---|---|")
    summary["mcnemar"] = {}
    for a in ARMS_TBL[1:]:
        pv, rowsx = [], []
        for task in tasks:
            sa = succ_vec["ours_vlm"][task]
            sb = succ_vec[a][task]
            sh = sorted(set(sa) & set(sb))
            if not sh:
                continue
            r = st.mcnemar_from_vectors([sa[s] for s in sh],
                                        [sb[s] for s in sh])
            rowsx.append((task, r.b, r.c, r.pvalue))
            pv.append(r.pvalue)
        holm = st.holm_bonferroni(pv) if pv else []
        for (task, b, c, p), ph in zip(rowsx, holm):
            md.append("| ours_vlm vs %s | %s | %d | %d | %.3g | %.3g |"
                      % (a, task, b, c, p, ph))
        # pooled over the PER-TASK PAIRED intersection (concatenating raw
        # per-arm vectors would misalign pairs whenever the two arms have
        # different seed sets)
        sa, sb = [], []
        for task in tasks:
            va = succ_vec["ours_vlm"].get(task, {})
            vb = succ_vec[a].get(task, {})
            for s_ in sorted(set(va) & set(vb)):
                sa.append(va[s_])
                sb.append(vb[s_])
        n = len(sa)
        if n:
            rp = st.mcnemar_from_vectors(sa, sb)
            md.append("| ours_vlm vs %s | **pooled (n=%d paired)** | %d | %d "
                      "| **%.3g** | -- |" % (a, n, rp.b, rp.c, rp.pvalue))
            summary["mcnemar"][a] = {
                "per_task": [{"task": t_, "b": b, "c": c, "p": p,
                              "p_holm": ph}
                             for (t_, b, c, p), ph in zip(rowsx, holm)],
                "pooled": {"b": rp.b, "c": rp.c, "p": rp.pvalue, "n": n}}
    md.append("")

    # ---- table 3: MOKA failure-stage breakdown ------------------------------
    md.append("## 3. MOKA failure-stage breakdown (`moka_real`)\n")
    md.append("Stages: **perception** (segmentation sanity check failed) / "
              "**selection** (no parseable selection, or the VLM chose a mark "
              "other than the one nearest the ground-truth-mapped demo grasp "
              "point) / **mapping** (mark correct, but the anchor is >%.0f cm "
              "off or the grasp->function axis is >%.0f deg off the "
              "ground-truth object yaw, folded into +-90 deg because a "
              "parallel-jaw grasp is pi-symmetric) / **grasp** (mapping ok, "
              "gripper did not close on the object) / **execution** "
              "(grasped, post-action criterion not met).  NOTE: the "
              "campaign-wide `failure_stage` field uses the object-level "
              ">5 cm / >20 deg rule, which mislabels benign 180 deg template "
              "rotations on pi-symmetric bar grasps; this table does not.\n"
              % (100 * ANCHOR_TOL_M, YAW_TOL_DEG))
    md.append("| task | n fail | perception | selection | mapping | grasp | "
              "execution |")
    md.append("|---|---|---|---|---|---|---|")
    stage_pool = {k: 0 for k in ("perception", "selection", "mapping",
                                 "grasp", "execution")}
    n_fail_pool = 0
    summary["moka_stages"] = {}
    for task in tasks:
        cnt = {k: 0 for k in stage_pool}
        nf = 0
        for r in data[task]["rows"]:
            m = r.get("moka_real")
            if m is None or m.get("success"):
                continue
            nf += 1
            cnt[_moka_stage(m)] += 1
        for k in cnt:
            stage_pool[k] += cnt[k]
        n_fail_pool += nf
        md.append("| %s | %d | %d | %d | %d | %d | %d |"
                  % (task, nf, cnt["perception"], cnt["selection"],
                     cnt["mapping"], cnt["grasp"], cnt["execution"]))
        summary["moka_stages"][task] = dict(cnt, n_fail=nf)
    md.append("| **pooled** | %d | %d | %d | %d | %d | %d |\n"
              % (n_fail_pool, stage_pool["perception"],
                 stage_pool["selection"], stage_pool["mapping"],
                 stage_pool["grasp"], stage_pool["execution"]))
    summary["moka_stages"]["pooled"] = dict(stage_pool, n_fail=n_fail_pool)

    md.append("### 3b. Same breakdown for `moka_real_ourcands` "
              "(MOKA's interface on OUR k-means marks)\n")
    md.append("| task | n fail | perception | selection | mapping | grasp | "
              "execution |")
    md.append("|---|---|---|---|---|---|---|")
    pool2 = {k: 0 for k in stage_pool}
    nf2t = 0
    summary["moka_stages_ourcands"] = {}
    for task in tasks:
        cnt = {k: 0 for k in stage_pool}
        nf = 0
        for r in data[task]["rows"]:
            m = r.get("moka_real_ourcands")
            if m is None or m.get("success"):
                continue
            nf += 1
            cnt[_moka_stage(m)] += 1
        for k in cnt:
            pool2[k] += cnt[k]
        nf2t += nf
        md.append("| %s | %d | %d | %d | %d | %d | %d |"
                  % (task, nf, cnt["perception"], cnt["selection"],
                     cnt["mapping"], cnt["grasp"], cnt["execution"]))
        summary["moka_stages_ourcands"][task] = dict(cnt, n_fail=nf)
    md.append("| **pooled** | %d | %d | %d | %d | %d | %d |\n"
              % (nf2t, pool2["perception"], pool2["selection"],
                 pool2["mapping"], pool2["grasp"], pool2["execution"]))
    summary["moka_stages_ourcands"]["pooled"] = dict(pool2, n_fail=nf2t)

    # ---- table 4: interface diagnostics ------------------------------------
    md.append("## 4. Mark-selection and mapping diagnostics\n")
    md.append("| task | MOKA mark-choice acc | mark quantisation floor "
              "(median best-mark err) | MOKA selected-mark err | MOKA |yaw "
              "err| | ours phi pair-set acc | ours rot err | ours trans err |")
    md.append("|---|---|---|---|---|---|---|---|")
    summary["diagnostics"] = {}
    for task in tasks:
        rows = data[task]["rows"]
        mk = [r["moka_q"] for r in rows if "moka_q" in r]
        ok = [q for q in mk if q.get("selection") is not None]
        acc = sum(1 for q in ok if q.get("mark_correct"))
        best = [q["best_possible_mark_err_m"] for q in mk
                if q.get("best_possible_mark_err_m") is not None]
        selerr = [q["selected_mark_err_m"] for q in ok
                  if q.get("selected_mark_err_m") is not None]
        yaw = [abs(r["moka_real"]["moka"]["yaw_err_deg"]) for r in rows
               if "moka_real" in r
               and (r["moka_real"].get("moka") or {}).get("yaw_err_deg")
               is not None]
        oq = [r["ours_q"] for r in rows if "ours_q" in r]
        opair = sum(1 for q in oq if (q.get("agree") or {}).get("pair_set"))
        orot = [r["ours_vlm"]["rot_err_deg"] for r in rows
                if "ours_vlm" in r and r["ours_vlm"].get("rot_err_deg")
                is not None]
        otr = [r["ours_vlm"]["trans_err_m"] for r in rows
               if "ours_vlm" in r and r["ours_vlm"].get("trans_err_m")
               is not None]
        mrot = [r["moka_real"]["rot_err_deg"] for r in rows
                if "moka_real" in r and r["moka_real"].get("rot_err_deg")
                is not None]
        mtr = [r["moka_real"]["trans_err_m"] for r in rows
               if "moka_real" in r and r["moka_real"].get("trans_err_m")
               is not None]
        f = lambda v, s=1.0: ("%.0f" % (s * float(np.median(v)))) if v else "-"
        md.append("| %s | %d/%d | %s mm | %s mm | %s deg | %d/%d | %s deg | "
                  "%s mm |" % (task, acc, len(ok), f(best, 1e3), f(selerr, 1e3),
                               f(yaw), opair, len(oq),
                               ("%.1f" % float(np.median(orot))) if orot
                               else "-", f(otr, 1e3)))
        summary["diagnostics"][task] = {
            "moka_mark_acc": [acc, len(ok)],
            "moka_best_mark_err_mm": (1e3 * float(np.median(best))
                                      if best else None),
            "moka_selected_mark_err_mm": (1e3 * float(np.median(selerr))
                                          if selerr else None),
            "moka_abs_yaw_err_deg": (float(np.median(yaw)) if yaw else None),
            "moka_rot_err_deg": (float(np.median(mrot)) if mrot else None),
            "moka_trans_err_mm": (1e3 * float(np.median(mtr))
                                  if mtr else None),
            "ours_pair_set_acc": [opair, len(oq)],
            "ours_rot_err_deg": (float(np.median(orot)) if orot else None),
            "ours_trans_err_mm": (1e3 * float(np.median(otr))
                                  if otr else None)}
    md.append("")

    # ---- table 5: query counts ---------------------------------------------
    md.append("## 5. VLM query counts\n")
    md.append("| method | per task (cached, amortised) | per rollout | "
              "total for 5 x 60 |")
    md.append("|---|---|---|---|")
    qc = _query_counts(data, tasks)
    for name, row in qc.items():
        md.append("| %s | %s | %s | %d |" % (name, row["per_task"],
                                             row["per_rollout"], row["total"]))
    summary["query_counts"] = qc
    md.append("")

    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(os.path.join(TABLES_DIR, "campaign_j.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    _write_tex(os.path.join(TABLES_DIR, "campaign_j.tex"), tasks, ARMS_TBL,
               summary)
    _write_json(os.path.join(out_root, "summary.json"), summary)
    print("\n".join(md))
    return summary


ANCHOR_TOL_M = 0.03      # mapping-stage anchor tolerance
YAW_TOL_DEG = 30.0       # mapping-stage yaw tolerance (pi-symmetric)


def _moka_stage(m):
    """Diagnostic failure stage for a MOKA rollout (report table 3).

    The campaign-wide `failure_stage` heuristic calls a rollout a MAPPING
    failure when the OBJECT-level transform is off by >5 cm or >20 deg.  That
    is the wrong instrument for MOKA: a parallel-jaw grasp on a bar is
    pi-symmetric, so a 180 deg `T_map` can still execute the demonstrated
    motion perfectly (observed repeatedly on `nut_loosen`).  This breakdown
    therefore attributes MOKA's failures to the stage that actually broke:

      perception  segmentation sanity check failed;
      selection   the VLM picked a mark other than the one nearest the
                  ground-truth-mapped demonstrated grasp point (or produced
                  no parseable selection);
      mapping     the mark was right but the resulting anchor is >3 cm from
                  the ground-truth-mapped grasp point, or the grasp->function
                  axis is >30 deg off the ground-truth object yaw (folded
                  into +-90 deg for the pi-symmetric jaw);
      grasp       mapping ok, gripper did not close on the object;
      execution   grasped, post-action criterion not met.
    """
    perc = m.get("perception") or {}
    if not (perc.get("demo", {}).get("sanity_ok", True)
            and perc.get("target", {}).get("sanity_ok", True)):
        return "perception"
    mo = m.get("moka") or {}
    if m.get("failure_stage") in ("vlm_selection", "method_error"):
        return "selection"
    if mo.get("mark_choice_correct") is False:
        return "selection"
    anch = mo.get("anchor_err_m")
    yaw = mo.get("yaw_err_deg")
    yaw_pi = None if yaw is None else abs((float(yaw) + 90.0) % 180.0 - 90.0)
    if (anch is not None and anch > ANCHOR_TOL_M) or \
       (yaw_pi is not None and yaw_pi > YAW_TOL_DEG):
        return "mapping"
    if m.get("grasped") is False:
        return "grasp"
    return "execution"


def _query_counts(data, tasks):
    out = {}
    n_roll = sum(len([r for r in data[t]["rows"] if "ours_vlm" in r])
                 for t in tasks)
    n_moka = sum(len([r for r in data[t]["rows"] if "moka_real" in r])
                 for t in tasks)
    ours_task = sum(1 for t in tasks)          # one cached demo query / task
    ours_calls = sum(
        (data[t]["rows"][0].get("ours_q", {}).get("vlm", {}) or {})
        .get("n_api_calls", 1) for t in tasks if data[t]["rows"])
    moka_plan = sum((data[t]["plan"] or {}).get("n_vlm_calls", 0)
                    for t in tasks)
    moka_roll = sum(r["moka_q"].get("attempts", 1)
                    for t in tasks for r in data[t]["rows"]
                    if "moka_q" in r)
    ours_roll = sum((r["ours_q"]["vlm"].get("n_api_calls", 1))
                    for t in tasks for r in data[t]["rows"]
                    if "ours_q" in r)
    out["ours_vlm"] = {
        "per_task": "1 demo mark-selection query (%d API calls total)"
                    % ours_calls,
        "per_rollout": "1 query (%.2f API calls incl. retries)"
                       % (ours_roll / max(1, n_roll)),
        "total": int(ours_calls + ours_roll)}
    out["moka_real"] = {
        "per_task": "1 high-level + 1 demo mark-selection query "
                    "(%d API calls total)" % moka_plan,
        "per_rollout": "1 query (%.2f API calls incl. retries)"
                       % (moka_roll / max(1, n_moka)),
        "total": int(moka_plan + moka_roll)}
    out["moka_real_strict / _wp"] = {
        "per_task": "shares moka_real's cached queries",
        "per_rollout": "0 (replays the same selection)", "total": 0}
    oc_roll = sum(r["moka_oc_q"].get("attempts", 1)
                  for t in tasks for r in data[t]["rows"]
                  if "moka_oc_q" in r)
    n_oc = sum(len([r for r in data[t]["rows"]
                    if "moka_real_ourcands" in r]) for t in tasks)
    out["moka_real_ourcands"] = {
        "per_task": "1 demo mark-selection query on our k-means marks",
        "per_rollout": "1 query (%.2f API calls incl. retries)"
                       % (oc_roll / max(1, n_oc)),
        "total": int(oc_roll)}
    out["moka_marks_oracle"] = {"per_task": "0", "per_rollout": "0",
                                "total": 0}
    return out


def _write_tex(path, tasks, arms, summary):
    lines = ["% Campaign J: real-VLM mark-based MOKA vs ours (medium tier, "
             "60 paired seeds/task)",
             "\\begin{tabular}{l" + "c" * (len(arms) + 2) + "}",
             "\\toprule",
             "task & " + " & ".join(a.replace("_", "\\_") for a in arms)
             + " & ours\\_full & moka\\_oracle \\\\",
             "\\midrule"]
    for task in tasks:
        s = summary["per_task"][task]
        cells = ["%d/%d" % tuple(s[a]) for a in arms]
        cells += ["%d/%d" % tuple(s[nm])
                  for nm in ("ours_full", "moka_oracle")]
        lines.append("%s & %s \\\\" % (task.replace("_", "\\_"),
                                       " & ".join(cells)))
    lines.append("\\midrule")
    p = summary["pooled"]
    cells = ["\\textbf{%d/%d}" % tuple(p[a]) for a in arms]
    cells += ["%d/%d" % tuple(p[nm]) for nm in ("ours_full", "moka_oracle")]
    lines.append("pooled & %s \\\\" % " & ".join(cells))
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True,
                   choices=["freeze", "plan", "select", "execute", "report"])
    p.add_argument("--tasks", nargs="*", default=list(TASKS_J))
    p.add_argument("--arms", nargs="*", default=list(ALL_ARMS))
    p.add_argument("--seed-start", type=int, default=SEEDS_J[0])
    p.add_argument("--seed-end", type=int, default=SEEDS_J[-1])
    # Campaign M (second-VLM generality) reuses this runner verbatim with a
    # different output root and a different served model (ALK_VLM_MODEL).
    p.add_argument("--out-root", default=OUT_ROOT)
    p.add_argument("--skip-ourcands", action="store_true",
                   help="do not issue the moka_real_ourcands diagnostic query")
    a = p.parse_args(argv)
    seeds = tuple(range(a.seed_start, a.seed_end + 1))
    root = a.out_root
    if a.phase == "freeze":
        freeze(tuple(a.tasks), out_root=root)
    elif a.phase == "plan":
        for t in a.tasks:
            plan_task(t, out_root=root)
    elif a.phase == "select":
        for t in a.tasks:
            select_task(t, seeds, out_root=root,
                        do_ourcands=not a.skip_ourcands)
    elif a.phase == "execute":
        for t in a.tasks:
            execute_task(t, seeds, tuple(a.arms), out_root=root)
    else:
        report(tuple(a.tasks), seeds, out_root=root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
