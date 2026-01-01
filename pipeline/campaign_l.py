"""Campaign L -- deployable relational-constraint baseline: ReKep (real VLM).

Motivation
----------
Campaign J closed the mark-based question family (MOKA, real VLM).  The
relational-constraint family (ReKep, Huang et al. 2024) was measured nowhere
credible: the old sim row was renamed `gt_corr_procrustes` because it was
weighted Procrustes with oracle correspondences, not ReKep, and the hardware
ReKep was an N=10 pilot.  Campaign L implements ReKep's actual interface --
keypoints proposed from vision (DINOv2 features + clustering, ReKep's own
algorithm), a VLM WRITING PYTHON CONSTRAINT FUNCTIONS over those keypoints,
and a scipy solve of the pose under the generated constraints -- adapted to
the benchmark's one-shot retargeting protocol, and runs it head-to-head
against `ours_vlm` (campaign J) driven by the SAME local Qwen3-VL server on
the same scenes, seeds, executor and corrected success criteria.

Arms
----
  rekep_real                ReKep keypoints (DINOv2) + real-VLM-authored
                            constraints + ReKep solver.
  rekep_real_ourcands       identical, but the keypoint proposal is OUR
                            k-means candidate pool (separates "the constraint
                            interface is weaker" from "the proposal is
                            weaker"; mirrors campaign J's ourcands arm).
  rekep_oracle_constraints  the same constraint STRUCTURE authored from
                            ground truth (GT-NN correspondences) on the same
                            DINOv2 keypoints + the same solver: the ceiling
                            of the constraint representation, isolating the
                            authoring channel (analogous to
                            moka_marks_oracle).
References read from campaigns J (ours_vlm, moka_real) and A/A2 (ours_full,
moka_oracle, gt_corr_procrustes[= old rollout_rekep]).

Grid: 5 tasks x 60 paired seeds {1000..1059}, medium tier, corrected
criteria (box_open on data_medium_a2, pour scored orientation-aware).

Phases (all resumable)
  propose  -- DINOv2 keypoint proposals (torch venv subprocess, CPU), cached.
  freeze   -- keypoint-config + prompt selection on NON-EVAL scenes (easy
              tier seeds 1..5 + demo scenes).  Writes freeze.json.
  query    -- one constraint-authoring VLM query per (task, seed) and
              proposal set.  No simulator needed.
  execute  -- sandbox-exec cached code, solve, roll out.  MUJOCO_GL=egl.
  report   -- results/tables/campaign_l.{md,tex} + summary.json.
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

from alkbench import transform_points, rotation_angle_deg   # noqa: E402
from baselines import common as bc                          # noqa: E402
from baselines import moka_marks as mm                      # noqa: E402
from baselines import rekep_real as rk                      # noqa: E402
from pipeline import oracle                                 # noqa: E402
from stats import tests as st                               # noqa: E402

OUT_ROOT = os.path.join(SIMBENCH, "results", "campaign_l")
TABLES_DIR = os.path.join(SIMBENCH, "results", "tables")
EASY_DATA_ROOT = os.path.join(SIMBENCH, "data")
KP_ROOT = os.path.join(OUT_ROOT, "kp")

TASKS_L = ("nut_loosen", "cap_twist", "rim_grasp", "pour", "box_open")
SEEDS_L = tuple(range(1000, 1060))
DEV_SEEDS = (1, 2, 3, 4, 5)
DEMO_PAIR_SEED = 0
K = 8
KMEANS_SEED = 0

DATA_ROOT_FOR = {t: os.path.join(SIMBENCH, "data_medium") for t in TASKS_L}
DATA_ROOT_FOR["box_open"] = os.path.join(SIMBENCH, "data_medium_a2")
REF_ROOT_FOR = {t: os.path.join(SIMBENCH, "results", "campaign_a")
                for t in TASKS_L}
REF_ROOT_FOR["pour"] = os.path.join(SIMBENCH, "results", "campaign_a2")
REF_ROOT_FOR["box_open"] = os.path.join(SIMBENCH, "results", "campaign_a2")
CAMPAIGN_J_ROOT = os.path.join(SIMBENCH, "results", "campaign_j")

ARMS_L = ("rekep_real", "rekep_real_ourcands", "rekep_oracle_constraints",
          "rekep_oracle_constraints_ourcands")

# ---------------------------------------------------------------------------
# FROZEN configuration (set by the freeze phase on non-eval data; the values
# below are overwritten at import time from freeze.json when it exists)
# ---------------------------------------------------------------------------

KP_SWEEP = (
    dict(num_candidates=5, min_dist=0.06, upscale=1),    # ReKep verbatim
    dict(num_candidates=5, min_dist=0.06, upscale=3),
    dict(num_candidates=8, min_dist=0.03, upscale=1),
    dict(num_candidates=8, min_dist=0.03, upscale=3),
    dict(num_candidates=12, min_dist=0.02, upscale=3),
)
KP_FROZEN = dict(KP_SWEEP[0])          # replaced by freeze.json
PROMPT_ZOOM_FROZEN = True              # replaced by freeze.json
# failure-decomposition constants (frozen; sanity-checked on dev data):
GT_EPS_PER_CONSTRAINT_M = 0.02   # mean allowed violation at T_gt
SOLVE_EPS = 1.0                  # hinge units (= 200 x 5 mm)


def _load_freeze():
    global KP_FROZEN, PROMPT_ZOOM_FROZEN
    p = os.path.join(OUT_ROOT, "freeze.json")
    if os.path.exists(p):
        with open(p) as f:
            rec = json.load(f)
        KP_FROZEN = dict(rec["kp_chosen"])
        PROMPT_ZOOM_FROZEN = bool(rec["prompt_zoom_chosen"])


# ---------------------------------------------------------------------------
# small utilities (mirroring campaign_j)
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


def root_tag(root):
    b = os.path.basename(root.rstrip("/"))
    return {"data": "easy", "data_medium": "med",
            "data_medium_a2": "meda2"}.get(b, b)


def data_root(task):
    return DATA_ROOT_FOR[task]


def pair_ctx(task, seed, root=None):
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
                           noise_seed=seed, random4_seed=seed)
    return demo_cap, tgt_cap, ctx, pair


def map_errors(T_est, T_gt, demo_pos, target_pos):
    T_est = np.asarray(T_est, dtype=np.float64)
    T_gt = np.asarray(T_gt, dtype=np.float64)
    rot = float(rotation_angle_deg(T_est[:3, :3] @ T_gt[:3, :3].T))
    mapped = transform_points(T_est, np.asarray(demo_pos, np.float64)[None])[0]
    return rot, float(np.linalg.norm(mapped
                                     - np.asarray(target_pos, np.float64)))


# ---------------------------------------------------------------------------
# keypoint proposal plumbing
# ---------------------------------------------------------------------------

def kp_dir(cfg):
    return os.path.join(KP_ROOT, rk.kp_cfg_tag(cfg))


def ensure_kp(task, seed, role, cfg, root=None, run_worker=False):
    """Return (pixels, xyz, rec) for one scene's DINOv2 keypoints, writing
    the request (and optionally running the worker) if missing."""
    root = root or data_root(task)
    tag = root_tag(root)
    stem = rk.kp_request_stem(task, tag,
                              DEMO_PAIR_SEED if role == "demo" else seed,
                              role)
    d = kp_dir(cfg)
    jp = os.path.join(d, stem + ".json")
    if not os.path.exists(jp):
        demo_cap, tgt_cap, ctx, _ = pair_ctx(task, seed, root=root)
        cap = demo_cap if role == "demo" else tgt_cap
        rk.write_kp_request(d, stem, cap, ctx)
        if run_worker:
            rk.run_kp_worker(d, cfg)
    return rk.load_kp(d, stem)


def propose(tasks=TASKS_L, dev=True, eval_=True):
    """Write all requests, then run the worker once per config dir."""
    _load_freeze()
    cfgs = list(KP_SWEEP) if dev else []
    if eval_:
        if KP_FROZEN not in [dict(c) for c in cfgs]:
            cfgs.append(dict(KP_FROZEN))
    for cfg in cfgs:
        d = kp_dir(cfg)
        for task in tasks:
            if dev:
                for seed in DEV_SEEDS:
                    try:
                        demo_cap, tgt_cap, ctx, _ = pair_ctx(
                            task, seed, root=EASY_DATA_ROOT)
                    except (IOError, OSError):
                        continue
                    rk.write_kp_request(d, rk.kp_request_stem(
                        task, "easy", DEMO_PAIR_SEED, "demo"), demo_cap, ctx)
                    rk.write_kp_request(d, rk.kp_request_stem(
                        task, "easy", seed, "target"), tgt_cap, ctx)
            if eval_ and dict(cfg) == dict(KP_FROZEN):
                tag = root_tag(data_root(task))
                for seed in SEEDS_L:
                    demo_cap, tgt_cap, ctx, _ = pair_ctx(task, seed)
                    rk.write_kp_request(d, rk.kp_request_stem(
                        task, tag, DEMO_PAIR_SEED, "demo"), demo_cap, ctx)
                    rk.write_kp_request(d, rk.kp_request_stem(
                        task, tag, seed, "target"), tgt_cap, ctx)
        print("[propose] running worker for %s" % rk.kp_cfg_tag(cfg),
              flush=True)
        out = rk.run_kp_worker(d, cfg)
        print(out.splitlines()[-1] if out.strip() else "(no output)",
              flush=True)


# ---------------------------------------------------------------------------
# shared per-pair assembly
# ---------------------------------------------------------------------------

def side_arrays(task, seed, ctx, demo_cap, tgt_cap, proposal, root=None,
                cfg=None):
    """Demo/target keypoints + demo grasp anchor for one pair.

    proposal: "dino" (ReKep's own, frozen config) or "kmeans" (our pool)."""
    p_d = ctx.percept(demo_cap, "demo")
    p_t = ctx.percept(tgt_cap, "target")
    if proposal == "kmeans":
        d_uv = np.asarray(p_d["cands"].centers2d, dtype=np.float64)
        d_xyz = np.asarray(p_d["cands"].candidates3d, dtype=np.float64)
        t_uv = np.asarray(p_t["cands"].centers2d, dtype=np.float64)
        t_xyz = np.asarray(p_t["cands"].candidates3d, dtype=np.float64)
    else:
        cfg = cfg or KP_FROZEN
        d_uv, d_xyz, _ = ensure_kp(task, seed, "demo", cfg, root=root)
        t_uv, t_xyz, _ = ensure_kp(task, seed, "target", cfg, root=root)
    g_d, proj_dist = oracle.demo_grasp_point(p_d["cands"],
                                             ctx.keyframe_tcp("pre_grasp"))
    g_uv = mm._world_to_pixel(g_d, p_d["K"], p_d["T_world_cam"])
    return {"p_d": p_d, "p_t": p_t, "d_uv": d_uv, "d_xyz": d_xyz,
            "t_uv": t_uv, "t_xyz": t_xyz, "g_d": np.asarray(g_d),
            "g_uv": g_uv, "proj_dist": proj_dist}


def build_images(sd, demo_cap, tgt_cap, p_d, p_t, zoom=True):
    rgb_d = bc.load_rgb(demo_cap, p_d["camera"])
    rgb_t = bc.load_rgb(tgt_cap, p_t["camera"])
    demo_img = rk.draw_rekep_markup(rgb_d, sd["d_uv"], grasp_uv=sd["g_uv"])
    tgt_img = rk.draw_rekep_markup(rgb_t, sd["t_uv"])
    dz = tz = None
    if zoom:
        dz = rk.draw_rekep_zoom(rgb_d, sd["d_uv"], grasp_uv=sd["g_uv"])
        tz = rk.draw_rekep_zoom(rgb_t, sd["t_uv"])
    return demo_img, tgt_img, dz, tz


def author_and_check(task, sd, demo_img, tgt_img, dz, tz, max_attempts=2):
    """Query the VLM for constraint code; retry once if the code fails to
    parse / exec / validate (the analogue of campaign J's parse retry)."""
    kd, kt = sd["d_xyz"].shape[0], sd["t_xyz"].shape[0]
    out = {"k_demo": int(kd), "k_target": int(kt), "attempts": 0,
           "replies": []}
    last = None
    for attempt in range(max_attempts):
        try:
            code, info = rk.query_constraints(
                mm.TASK_INSTRUCTIONS[task], demo_img, tgt_img, kd, kt,
                demo_zoom=dz, target_zoom=tz,
                temperature=0.0 if attempt == 0 else 0.3, max_retries=0)
            out["attempts"] += info["attempts"]
            out["replies"] += [r[-3000:] for r in info["replies"]]
        except Exception as e:
            out["attempts"] += 1
            last = "query: %r" % e
            continue
        try:
            fns, meta = rk.exec_safe(code)
            if not fns:
                raise ValueError("no constraint functions defined")
            rk.validate_constraints(fns, sd["g_d"], sd["d_xyz"], sd["t_xyz"])
            out.update({"code": code, "meta": meta,
                        "diag": rk.code_diagnostics(code),
                        "authoring_ok": True, "error": None})
            return out
        except Exception as e:
            last = "exec: %r" % e
    out.update({"code": out.get("code"), "authoring_ok": False,
                "error": last})
    return out


def solve_from_code(code, sd, seed=0):
    fns, meta = rk.exec_safe(code)
    if not fns:
        raise ValueError("no constraint functions defined")
    rk.validate_constraints(fns, sd["g_d"], sd["d_xyz"], sd["t_xyz"])
    T, info = rk.solve_constraints(fns, sd["g_d"], sd["d_xyz"], sd["t_xyz"],
                                   seed=seed)
    return T, info, fns, meta


def hinge_diagnostics(fns, T_sol, T_gt, sd):
    h_sol, per_sol = rk.hinge_cost(fns, T_sol, sd["g_d"], sd["d_xyz"],
                                   sd["t_xyz"])
    h_gt, per_gt = rk.hinge_cost(fns, np.asarray(T_gt), sd["g_d"],
                                 sd["d_xyz"], sd["t_xyz"])
    n = max(1, len(fns))
    return {"hinge_sol": float(h_sol), "hinge_gt": float(h_gt),
            "per_constraint_sol": per_sol, "per_constraint_gt": per_gt,
            "gt_mean_violation_m": float(h_gt / (rk.PENALTY * n)),
            "n_constraints": n}


# ---------------------------------------------------------------------------
# phase: freeze  (NON-EVAL scenes only)
# ---------------------------------------------------------------------------

def freeze(tasks=TASKS_L, seeds=DEV_SEEDS, out_root=OUT_ROOT):
    rec = {"dev_data_root": EASY_DATA_ROOT, "dev_seeds": list(seeds),
           "tasks": list(tasks)}
    os.makedirs(os.path.join(out_root, "markups"), exist_ok=True)

    # ---- (a) keypoint-config sweep under ORACLE constraints (no VLM) ------
    sweep = []
    for cfg in KP_SWEEP:
        rows = []
        for task in tasks:
            for seed in seeds:
                try:
                    d, t, ctx, pair = pair_ctx(task, seed,
                                               root=EASY_DATA_ROOT)
                except (IOError, OSError):
                    continue
                try:
                    sd = side_arrays(task, seed, ctx, d, t, "dino",
                                     root=EASY_DATA_ROOT, cfg=cfg)
                except Exception as e:
                    rows.append({"task": task, "seed": seed, "ok": False,
                                 "error": repr(e)})
                    continue
                fns, meta = rk.oracle_constraint_fns(
                    sd["d_xyz"], sd["t_xyz"], sd["g_d"], ctx.T_gt)
                t0 = time.time()
                T, info = rk.solve_constraints(fns, sd["g_d"], sd["d_xyz"],
                                               sd["t_xyz"])
                inst = pair["target_instance"]
                rot, tr = map_errors(T, ctx.T_gt,
                                     pair["demo_object_poses"][inst]["pos"],
                                     pair["target_object_poses"][inst]["pos"])
                rows.append({"task": task, "seed": seed, "ok": True,
                             "rot_deg": rot, "trans_m": tr,
                             "k_demo": int(sd["d_xyz"].shape[0]),
                             "k_target": int(sd["t_xyz"].shape[0]),
                             "solve_s": round(time.time() - t0, 2),
                             "cost_final": info["cost_final"]})
        ok = [r for r in rows if r["ok"]]
        entry = {"cfg": dict(cfg), "tag": rk.kp_cfg_tag(cfg),
                 "n": len(rows), "n_ok": len(ok),
                 "median_rot_deg": (float(np.median([r["rot_deg"]
                                                     for r in ok]))
                                    if ok else None),
                 "median_trans_mm": (1e3 * float(np.median(
                     [r["trans_m"] for r in ok])) if ok else None),
                 "median_k": (float(np.median([r["k_target"] for r in ok]))
                              if ok else None),
                 "median_solve_s": (float(np.median([r["solve_s"]
                                                     for r in ok]))
                                    if ok else None),
                 "rows": rows}
        sweep.append(entry)
        print("[freeze kp] %-14s rot %s deg trans %s mm k~%s (%d ok)"
              % (entry["tag"], entry["median_rot_deg"],
                 entry["median_trans_mm"], entry["median_k"], len(ok)),
              flush=True)
    rec["kp_sweep"] = sweep
    best = min([s for s in sweep if s["n_ok"]],
               key=lambda s: (s["median_trans_mm"], s["median_rot_deg"]))
    rec["kp_chosen"] = best["cfg"]
    rec["kp_chosen_is_verbatim"] = (dict(best["cfg"])
                                    == dict(KP_SWEEP[0]))

    # ---- (b) markup renders for the record --------------------------------
    for task in tasks:
        try:
            d, t, ctx, _ = pair_ctx(task, seeds[0], root=EASY_DATA_ROOT)
            sd = side_arrays(task, seeds[0], ctx, d, t, "dino",
                             root=EASY_DATA_ROOT, cfg=best["cfg"])
            di, ti, dz, tz = build_images(sd, d, t, sd["p_d"], sd["p_t"])
            for nm, img in (("demo", di), ("target", ti), ("demo_zoom", dz),
                            ("target_zoom", tz)):
                with open(os.path.join(out_root, "markups", "%s_%s.png"
                                       % (task, nm)), "wb") as f:
                    f.write(rk.encode_png(img))
        except Exception as e:
            print("[freeze markup] %s failed: %r" % (task, e), flush=True)

    # ---- (c) prompt variant sweep (real VLM, dev scenes) -------------------
    variants = []
    for zoom in (True, False):
        rows = []
        for task in tasks:
            for seed in seeds[:3]:
                try:
                    d, t, ctx, pair = pair_ctx(task, seed,
                                               root=EASY_DATA_ROOT)
                    sd = side_arrays(task, seed, ctx, d, t, "dino",
                                     root=EASY_DATA_ROOT, cfg=best["cfg"])
                except Exception as e:
                    rows.append({"task": task, "seed": seed, "ok": False,
                                 "error": repr(e)})
                    continue
                di, ti, dz, tz = build_images(sd, d, t, sd["p_d"],
                                              sd["p_t"], zoom=zoom)
                t0 = time.time()
                q = author_and_check(task, sd, di, ti, dz, tz)
                row = {"task": task, "seed": seed,
                       "ok": bool(q.get("authoring_ok")),
                       "attempts": q["attempts"],
                       "latency_s": round(time.time() - t0, 1)}
                if q.get("authoring_ok"):
                    try:
                        T, info, fns, meta = solve_from_code(q["code"], sd)
                        inst = pair["target_instance"]
                        rot, tr = map_errors(
                            T, ctx.T_gt,
                            pair["demo_object_poses"][inst]["pos"],
                            pair["target_object_poses"][inst]["pos"])
                        row.update({"rot_deg": rot, "trans_m": tr,
                                    "n_constraints": meta["n_constraints"],
                                    "uses_demo": q["diag"][
                                        "uses_demo_keypoints"]})
                        row.update(hinge_diagnostics(fns, T, ctx.T_gt, sd))
                    except Exception as e:
                        row["ok"] = False
                        row["error"] = "solve: %r" % e
                else:
                    row["error"] = q.get("error")
                rows.append(row)
                print("[freeze prompt zoom=%s] %-11s %d ok=%s rot=%s "
                      "trans=%s (%.0fs)"
                      % (zoom, task, seed, row["ok"],
                         ("%.0f" % row["rot_deg"]) if "rot_deg" in row
                         else "-",
                         ("%.0fmm" % (1e3 * row["trans_m"]))
                         if "trans_m" in row else "-", row["latency_s"]),
                      flush=True)
        ok = [r for r in rows if r["ok"]]
        variants.append({
            "zoom": zoom, "n": len(rows), "n_ok": len(ok),
            "median_rot_deg": (float(np.median([r["rot_deg"] for r in ok]))
                               if ok else None),
            "median_trans_mm": (1e3 * float(np.median([r["trans_m"]
                                                       for r in ok]))
                                if ok else None),
            "median_gt_violation_mm": (1e3 * float(np.median(
                [r["gt_mean_violation_m"] for r in ok])) if ok else None),
            "rows": rows})
        print("VARIANT zoom=%s: ok %d/%d, median rot %s deg, trans %s mm"
              % (zoom, len(ok), len(rows),
                 variants[-1]["median_rot_deg"],
                 variants[-1]["median_trans_mm"]), flush=True)
    rec["prompt_variants"] = variants
    chosen = max(variants, key=lambda v: (v["n_ok"],
                                          -(v["median_trans_mm"] or 1e9)))
    rec["prompt_zoom_chosen"] = bool(chosen["zoom"])
    rec["prompt_template"] = rk.PROMPT_TEMPLATE
    rec["task_instructions"] = dict(mm.TASK_INSTRUCTIONS)
    rec["failure_decomposition"] = {
        "gt_eps_per_constraint_m": GT_EPS_PER_CONSTRAINT_M,
        "solve_eps_hinge": SOLVE_EPS,
        "note": "authoring rejects GT iff mean violation at T_gt > "
                "gt_eps; solver failure iff hinge(sol) > hinge(T_gt) + "
                "solve_eps while GT is feasible."}
    rec["solver"] = {"penalty": rk.PENALTY, "maxfun": rk.SOLVER_MAXFUN,
                     "trans_bound_m": rk.TRANS_BOUND,
                     "optimizer": "dual_annealing + SLSQP polish, seed 0"}
    _write_json(os.path.join(out_root, "freeze.json"), rec)
    print("\nFROZEN: kp %r (verbatim=%s), zoom=%s"
          % (rec["kp_chosen"], rec["kp_chosen_is_verbatim"],
             rec["prompt_zoom_chosen"]))
    return rec


# ---------------------------------------------------------------------------
# phase: query  (one constraint-authoring query per (task, seed) and arm)
# ---------------------------------------------------------------------------

def query_task(task, seeds=SEEDS_L, out_root=OUT_ROOT,
               proposals=("dino", "kmeans")):
    _load_freeze()
    n_new = 0
    for seed in seeds:
        sdir = os.path.join(out_root, task, str(seed))
        paths = {"dino": os.path.join(sdir, "rekep_query.json"),
                 "kmeans": os.path.join(sdir, "rekep_ourcands_query.json")}
        todo = [p for p in proposals if not os.path.exists(paths[p])]
        if not todo:
            continue
        d, t, ctx, pair = pair_ctx(task, seed)
        for prop in todo:
            t0 = time.time()
            rec = {"task": task, "seed": seed, "proposal": prop,
                   "tier": "medium", "model": mm._model(None),
                   "zoom": PROMPT_ZOOM_FROZEN}
            try:
                sd = side_arrays(task, seed, ctx, d, t, prop)
                di, ti, dz, tz = build_images(sd, d, t, sd["p_d"],
                                              sd["p_t"],
                                              zoom=PROMPT_ZOOM_FROZEN)
                q = author_and_check(task, sd, di, ti, dz, tz)
                rec.update(q)
                # scoring info (never seen by the method)
                g_gt = transform_points(ctx.T_gt, sd["g_d"][None])[0]
                dists = np.linalg.norm(sd["t_xyz"] - g_gt, axis=1)
                rec["oracle_grasp_keypoint"] = int(np.argmin(dists))
                rec["best_possible_kp_err_m"] = float(dists.min())
                gk = (rec.get("meta") or {}).get("grasp_keypoint")
                if gk is not None and 0 <= gk < len(dists):
                    rec["grasp_kp_err_m"] = float(dists[gk])
                    rec["grasp_kp_correct"] = bool(
                        gk == rec["oracle_grasp_keypoint"])
            except Exception as e:
                traceback.print_exc()
                rec.update({"authoring_ok": False,
                            "error": "pipeline: %r" % e})
            rec["time_s"] = round(time.time() - t0, 1)
            _write_json(paths[prop], rec)
            n_new += 1
            print("[%s %d query %-6s] ok=%s n_con=%s uses_demo=%s (%.0fs)"
                  % (task, seed, prop, rec.get("authoring_ok"),
                     (rec.get("meta") or {}).get("n_constraints"),
                     (rec.get("diag") or {}).get("uses_demo_keypoints"),
                     rec["time_s"]), flush=True)
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


def _arm_map(arm, task, seed, ctx, d, t, queries):
    """Build one arm's T_map (+ diagnostics) from cached artefacts."""
    prop = "kmeans" if arm.endswith("_ourcands") else "dino"
    sd = side_arrays(task, seed, ctx, d, t, prop)
    out = {"k_demo": int(sd["d_xyz"].shape[0]),
           "k_target": int(sd["t_xyz"].shape[0]),
           "demo_grasp": sd["g_d"].tolist(), "proposal": prop}
    if arm.startswith("rekep_oracle_constraints"):
        fns, meta = rk.oracle_constraint_fns(sd["d_xyz"], sd["t_xyz"],
                                             sd["g_d"], ctx.T_gt)
        T, info = rk.solve_constraints(fns, sd["g_d"], sd["d_xyz"],
                                       sd["t_xyz"])
        out.update({"T_map": np.asarray(T).tolist(), "solver": info,
                    "oracle_meta": meta, "n_vlm_calls_rollout": 0})
        out.update(hinge_diagnostics(fns, T, ctx.T_gt, sd))
        return out
    q = queries[prop]
    out["n_vlm_calls_rollout"] = q.get("attempts", 1)
    out["authoring_ok"] = bool(q.get("authoring_ok"))
    out["grasp_keypoint"] = (q.get("meta") or {}).get("grasp_keypoint")
    out["code_diag"] = q.get("diag")
    if not q.get("authoring_ok"):
        out.update({"T_map": None, "failure_stage": "authoring_unrunnable",
                    "error": q.get("error")})
        return out
    T, info, fns, meta = solve_from_code(q["code"], sd)
    out.update({"T_map": np.asarray(T).tolist(), "solver": info,
                "meta": meta})
    out.update(hinge_diagnostics(fns, T, ctx.T_gt, sd))
    return out


def execute_task(task, seeds=SEEDS_L, arms=ARMS_L, out_root=OUT_ROOT):
    from pipeline import campaign as cg
    from pipeline import retarget_runner as rr
    from simtasks import envs, motion

    _load_freeze()
    root = data_root(task)
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
            d, t, ctx, pair = pair_ctx(task, seed)
            inst = pair["target_instance"]
            demo_pose = pair["demo_object_poses"][inst]
            target_pose = pair["target_object_poses"][inst]
            p_d = ctx.percept(d, "demo")
            p_t = ctx.percept(t, "target")
            queries = {}
            for prop, fn in (("dino", "rekep_query.json"),
                             ("kmeans", "rekep_ourcands_query.json")):
                p = os.path.join(sdir, fn)
                queries[prop] = _read_json(p) if os.path.exists(p) else None
            todo = [a for a in todo
                    if not (a == "rekep_real" and queries["dino"] is None)
                    and not (a == "rekep_real_ourcands"
                             and queries["kmeans"] is None)]
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
                    res.update(_arm_map(arm, task, seed, ctx, d, t, queries))
                except Exception as e:
                    traceback.print_exc()
                    res["failure_stage"] = "method_error"
                    res["error"] = repr(e)
                    res.setdefault("T_map", None)

                if res.get("T_map") is None:
                    res["time_s"] = round(time.time() - t0, 1)
                    _write_json(os.path.join(sdir, "rollout_%s.json" % arm),
                                res)
                    n_new += 1
                    print("[%s %d %-24s] success=False stage=%s (no exec)"
                          % (task, seed, arm, res["failure_stage"]),
                          flush=True)
                    continue

                T_map = np.asarray(res["T_map"], dtype=np.float64)
                rot, tr = map_errors(T_map, ctx.T_gt, demo_pose["pos"],
                                     target_pose["pos"])
                res["rot_err_deg"] = rot
                res["trans_err_m"] = tr
                g_gt = transform_points(ctx.T_gt,
                                        np.asarray(res["demo_grasp"])[None])[0]
                g_est = transform_points(T_map,
                                         np.asarray(res["demo_grasp"])[None])[0]
                res["anchor_err_m"] = float(np.linalg.norm(g_est - g_gt))
                wps = rr.retarget_waypoint_dicts(demo["waypoints"], T_map)
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
                print("[%s %d %-24s] success=%s stage=%-11s rot=%5.1f "
                      "trans=%5.1fmm (%.1fs)"
                      % (task, seed, arm, success, str(res["failure_stage"]),
                         rot, 1e3 * tr, res["time_s"]), flush=True)
    finally:
        env.close()
    return n_new


# ---------------------------------------------------------------------------
# phase: corr  (correspondence diagnostic over the cached code, no VLM/sim)
# ---------------------------------------------------------------------------

def corr_task(task, seeds=SEEDS_L, out_root=OUT_ROOT):
    """For every cached constraint program, measure whether its declared
    demo->target keypoint pairs are geometrically correct, and what the
    keypoint pool would have allowed at best."""
    _load_freeze()
    rows = []
    for seed in seeds:
        sdir = os.path.join(out_root, task, str(seed))
        for prop, fn in (("dino", "rekep_query.json"),
                         ("kmeans", "rekep_ourcands_query.json")):
            p = os.path.join(sdir, fn)
            if not os.path.exists(p):
                continue
            q = _read_json(p)
            row = {"task": task, "seed": seed, "proposal": prop,
                   "authoring_ok": bool(q.get("authoring_ok"))}
            if q.get("code"):
                d, t, ctx, _ = pair_ctx(task, seed)
                sd = side_arrays(task, seed, ctx, d, t, prop)
                row.update(rk.correspondence_accuracy(
                    q["code"], sd["d_xyz"], sd["t_xyz"], ctx.T_gt))
                gk = (q.get("meta") or {}).get("grasp_keypoint")
                row["grasp_kp_correct"] = q.get("grasp_kp_correct")
                row["grasp_kp_err_m"] = q.get("grasp_kp_err_m")
                row["best_possible_kp_err_m"] = q.get("best_possible_kp_err_m")
                row["k_demo"] = int(sd["d_xyz"].shape[0])
                row["k_target"] = int(sd["t_xyz"].shape[0])
            rows.append(row)
    _write_json(os.path.join(out_root, task, "corr.json"), {"rows": rows})
    ok = [r for r in rows if r.get("n_pairs")]
    npair = sum(r["n_pairs"] for r in ok)
    ncorr = sum(r["n_correct"] for r in ok)
    print("[%s corr] %d programs, %d declared pairs, %d correct (%.0f%%)"
          % (task, len(ok), npair, ncorr,
             100.0 * ncorr / max(1, npair)), flush=True)
    return rows


# ---------------------------------------------------------------------------
# phase: report
# ---------------------------------------------------------------------------

def _wilson(k, n):
    if n == 0:
        return "-"
    lo, hi = st.wilson_ci(k, n)
    return "%d/%d = %.2f [%.2f, %.2f]" % (k, n, k / n, lo, hi)


def _ref_success(ref_root, task, seeds, fname):
    out = {}
    for seed in seeds:
        p = os.path.join(ref_root, task, str(seed), fname)
        if os.path.exists(p):
            out[seed] = int(bool(_read_json(p).get("success")))
    return out


def rekep_stage(r):
    """Refined failure attribution for a rekep_real* rollout (see
    freeze.json failure_decomposition)."""
    perc = r.get("perception") or {}
    if not (perc.get("demo", {}).get("sanity_ok", True)
            and perc.get("target", {}).get("sanity_ok", True)):
        return "perception"
    if r.get("failure_stage") in ("authoring_unrunnable", "method_error"):
        return "authoring_unrunnable"
    if r.get("gt_mean_violation_m") is not None \
            and r["gt_mean_violation_m"] > GT_EPS_PER_CONSTRAINT_M:
        return "authoring_rejects_gt"
    if r.get("hinge_sol") is not None and r.get("hinge_gt") is not None \
            and r["hinge_sol"] > r["hinge_gt"] + SOLVE_EPS:
        return "solver"
    rot = r.get("rot_err_deg")
    rot_fold = None if rot is None else min(abs(rot), abs(180.0 - abs(rot)))
    if (r.get("anchor_err_m") is not None
            and r["anchor_err_m"] > 0.03) \
            or (rot_fold is not None and rot_fold > 30.0):
        return "authoring_underdetermined"
    if r.get("grasped") is False:
        return "grasp"
    return "execution"


STAGES = ("perception", "authoring_unrunnable", "authoring_rejects_gt",
          "solver", "authoring_underdetermined", "grasp", "execution")


def collect(tasks=TASKS_L, seeds=SEEDS_L, out_root=OUT_ROOT):
    data = {}
    for task in tasks:
        rows = []
        for seed in seeds:
            row = {"seed": seed}
            sdir = os.path.join(out_root, task, str(seed))
            for arm in ARMS_L:
                p = os.path.join(sdir, "rollout_%s.json" % arm)
                if os.path.exists(p):
                    row[arm] = _read_json(p)
            for nm, fn in (("rekep_q", "rekep_query.json"),
                           ("rekep_oc_q", "rekep_ourcands_query.json")):
                p = os.path.join(sdir, fn)
                if os.path.exists(p):
                    row[nm] = _read_json(p)
            rows.append(row)
        data[task] = {
            "rows": rows,
            "ref_ours_vlm": _ref_success(CAMPAIGN_J_ROOT, task, seeds,
                                         "rollout_ours_vlm.json"),
            "ref_moka_real": _ref_success(CAMPAIGN_J_ROOT, task, seeds,
                                          "rollout_moka_real.json"),
            "ref_ours_full": _ref_success(REF_ROOT_FOR[task], task, seeds,
                                          "rollout_ours_full.json"),
            "ref_moka_oracle": _ref_success(REF_ROOT_FOR[task], task, seeds,
                                            "rollout_moka_oracle.json"),
            "ref_gt_corr_procrustes": _ref_success(
                REF_ROOT_FOR[task], task, seeds, "rollout_rekep.json")}
    return data


REFS = ("ours_vlm", "moka_real", "ours_full", "moka_oracle",
        "gt_corr_procrustes")


def report(tasks=TASKS_L, seeds=SEEDS_L, out_root=OUT_ROOT):
    _load_freeze()
    data = collect(tasks, seeds, out_root)
    ARMS_TBL = list(ARMS_L)
    md = ["# Campaign L -- deployable relational-constraint baseline "
          "(real VLM): ReKep vs ours\n",
          "Medium tier, paired seeds %d..%d (N=%d/task), model %s.  "
          "`box_open` uses data_medium_a2 + Campaign-A2 references; `pour` "
          "is scored with the corrected orientation-aware criterion.  "
          "`ours_vlm` / `moka_real` columns replay campaign J's paired "
          "rollouts (same seeds, same VLM).  Raw records under "
          "results/campaign_l/.\n"
          % (seeds[0], seeds[-1], len(seeds), mm.VLM_DEFAULTS["model"])]
    summary = {"arms": ARMS_TBL, "per_task": {}, "pooled": {}}

    # ---- table 1: success --------------------------------------------------
    md.append("## 1. End-to-end success\n")
    md.append("| task | " + " | ".join(ARMS_TBL)
              + " | " + " | ".join("%s (ref)" % r for r in REFS) + " |")
    md.append("|---" * (len(ARMS_TBL) + len(REFS) + 1) + "|")
    pooled = {a: [0, 0] for a in ARMS_TBL}
    pooled_ref = {r: [0, 0] for r in REFS}
    succ_vec = {a: {} for a in ARMS_TBL}
    ref_vec = {r: {} for r in REFS}
    for task in tasks:
        dd = data[task]
        cells = []
        for a in ARMS_TBL:
            s = {r["seed"]: int(bool(r[a]["success"]))
                 for r in dd["rows"] if a in r}
            succ_vec[a][task] = s
            pooled[a][0] += sum(s.values())
            pooled[a][1] += len(s)
            cells.append(_wilson(sum(s.values()), len(s)))
        for nm in REFS:
            ref = dd["ref_%s" % nm]
            ref_vec[nm][task] = ref
            pooled_ref[nm][0] += sum(ref.values())
            pooled_ref[nm][1] += len(ref)
            cells.append(_wilson(sum(ref.values()), len(ref)))
        md.append("| %s | %s |" % (task, " | ".join(cells)))
        summary["per_task"][task] = {
            a: [sum(succ_vec[a][task].values()), len(succ_vec[a][task])]
            for a in ARMS_TBL}
        summary["per_task"][task].update(
            {nm: [sum(dd["ref_%s" % nm].values()), len(dd["ref_%s" % nm])]
             for nm in REFS})
    md.append("| **pooled** | %s | %s |" % (
        " | ".join("**%s**" % _wilson(*pooled[a]) for a in ARMS_TBL),
        " | ".join(_wilson(*pooled_ref[nm]) for nm in REFS)))
    summary["pooled"] = {a: pooled[a] for a in ARMS_TBL}
    summary["pooled"].update(pooled_ref)
    md.append("")

    # ---- table 2: paired McNemar -------------------------------------------
    md.append("## 2. Paired McNemar (exact), Holm-corrected across the %d "
              "tasks\n" % len(tasks))
    md.append("| comparison | task | b (first wins) | c (second wins) | p | "
              "p_Holm |")
    md.append("|---|---|---|---|---|---|")
    summary["mcnemar"] = {}

    def _mcnemar_block(name, va_by_task, vb_by_task):
        pv, rowsx = [], []
        for task in tasks:
            va = va_by_task.get(task, {})
            vb = vb_by_task.get(task, {})
            sh = sorted(set(va) & set(vb))
            if not sh:
                continue
            r = st.mcnemar_from_vectors([va[s] for s in sh],
                                        [vb[s] for s in sh])
            rowsx.append((task, r.b, r.c, r.pvalue))
            pv.append(r.pvalue)
        holm = st.holm_bonferroni(pv) if pv else []
        for (task, b, c, p), ph in zip(rowsx, holm):
            md.append("| %s | %s | %d | %d | %.3g | %.3g |"
                      % (name, task, b, c, p, ph))
        sa, sb = [], []
        for task in tasks:
            va = va_by_task.get(task, {})
            vb = vb_by_task.get(task, {})
            for s_ in sorted(set(va) & set(vb)):
                sa.append(va[s_])
                sb.append(vb[s_])
        if sa:
            rp = st.mcnemar_from_vectors(sa, sb)
            md.append("| %s | **pooled (n=%d paired)** | %d | %d | **%.3g** "
                      "| -- |" % (name, len(sa), rp.b, rp.c, rp.pvalue))
            summary["mcnemar"][name] = {
                "per_task": [{"task": t_, "b": b, "c": c, "p": p,
                              "p_holm": ph}
                             for (t_, b, c, p), ph in zip(rowsx, holm)],
                "pooled": {"b": rp.b, "c": rp.c, "p": rp.pvalue,
                           "n": len(sa)}}

    for a in ARMS_TBL:
        _mcnemar_block("ours_vlm vs %s" % a, ref_vec["ours_vlm"],
                       succ_vec[a])
    _mcnemar_block("moka_real vs rekep_real", ref_vec["moka_real"],
                   succ_vec["rekep_real"])
    _mcnemar_block("rekep_real vs rekep_oracle_constraints",
                   succ_vec["rekep_real"],
                   succ_vec["rekep_oracle_constraints"])
    md.append("")

    # ---- table 3: failure-stage breakdown -----------------------------------
    for arm in ("rekep_real", "rekep_real_ourcands"):
        md.append("## 3%s. Failure-stage breakdown (`%s`)\n"
                  % ("" if arm == "rekep_real" else "b", arm))
        md.append("Stages: **perception** (segmentation sanity) / "
                  "**authoring_unrunnable** (no code, banned tokens, "
                  "exec/validation error after retry) / "
                  "**authoring_rejects_gt** (the generated constraints "
                  "exclude the ground-truth pose: mean violation at T_gt > "
                  "%.0f mm/constraint) / **solver** (a GT-feasible "
                  "constraint set, but the optimiser returned a point with "
                  "hinge > hinge(T_gt) + %.1f) / "
                  "**authoring_underdetermined** (constraints satisfied at "
                  "the solution AND at GT, but the pose is wrong: anchor "
                  ">3 cm or pi-folded rotation >30 deg -- the constraints "
                  "do not pin the pose down) / **grasp** / **execution**.\n"
                  % (1e3 * GT_EPS_PER_CONSTRAINT_M, SOLVE_EPS))
        md.append("| task | n fail | " + " | ".join(STAGES) + " |")
        md.append("|---" * (len(STAGES) + 2) + "|")
        pool = {k: 0 for k in STAGES}
        nfp = 0
        summary["stages_%s" % arm] = {}
        for task in tasks:
            cnt = {k: 0 for k in STAGES}
            nf = 0
            for r in data[task]["rows"]:
                m = r.get(arm)
                if m is None or m.get("success"):
                    continue
                nf += 1
                cnt[rekep_stage(m)] += 1
            for k in cnt:
                pool[k] += cnt[k]
            nfp += nf
            md.append("| %s | %d | %s |" % (
                task, nf, " | ".join(str(cnt[k]) for k in STAGES)))
            summary["stages_%s" % arm][task] = dict(cnt, n_fail=nf)
        md.append("| **pooled** | %d | %s |\n" % (
            nfp, " | ".join(str(pool[k]) for k in STAGES)))
        summary["stages_%s" % arm]["pooled"] = dict(pool, n_fail=nfp)

    # ---- table 4: authoring / solver diagnostics ----------------------------
    md.append("## 4. Constraint-authoring and mapping diagnostics "
              "(`rekep_real`)\n")
    md.append("| task | authoring ok | uses demo kps | median n_con | "
              "grasp-kp correct | median k_t | median GT violation | "
              "rot err | trans err | oracle-arm rot | oracle-arm trans |")
    md.append("|---" * 11 + "|")
    summary["diagnostics"] = {}
    for task in tasks:
        rows = data[task]["rows"]
        qs = [r["rekep_q"] for r in rows if "rekep_q" in r]
        ok = [q for q in qs if q.get("authoring_ok")]
        used = [q for q in ok
                if (q.get("diag") or {}).get("uses_demo_keypoints")]
        ncon = [(q.get("meta") or {}).get("n_constraints") for q in ok]
        ncon = [n for n in ncon if n]
        gkc = [q for q in ok if q.get("grasp_kp_correct")]
        gk_tot = [q for q in ok if q.get("grasp_kp_correct") is not None]
        kt = [q.get("k_target") for q in qs if q.get("k_target")]
        rr_ = [r["rekep_real"] for r in rows if "rekep_real" in r]
        gtv = [m["gt_mean_violation_m"] for m in rr_
               if m.get("gt_mean_violation_m") is not None]
        rot = [m["rot_err_deg"] for m in rr_
               if m.get("rot_err_deg") is not None]
        tr = [m["trans_err_m"] for m in rr_
              if m.get("trans_err_m") is not None]
        oc = [r["rekep_oracle_constraints"] for r in rows
              if "rekep_oracle_constraints" in r]
        orot = [m["rot_err_deg"] for m in oc
                if m.get("rot_err_deg") is not None]
        otr = [m["trans_err_m"] for m in oc
               if m.get("trans_err_m") is not None]
        f = lambda v, s=1.0, fmt="%.0f": (fmt % (s * float(np.median(v)))) \
            if v else "-"
        md.append("| %s | %d/%d | %d/%d | %s | %d/%d | %s | %s mm | %s deg "
                  "| %s mm | %s deg | %s mm |"
                  % (task, len(ok), len(qs), len(used), len(ok),
                     f(ncon), len(gkc), len(gk_tot), f(kt),
                     f(gtv, 1e3), f(rot, 1.0, "%.1f"), f(tr, 1e3),
                     f(orot, 1.0, "%.1f"), f(otr, 1e3)))
        summary["diagnostics"][task] = {
            "authoring_ok": [len(ok), len(qs)],
            "uses_demo": [len(used), len(ok)],
            "median_n_constraints": float(np.median(ncon)) if ncon else None,
            "grasp_kp_correct": [len(gkc), len(gk_tot)],
            "median_k_target": float(np.median(kt)) if kt else None,
            "median_gt_violation_mm": (1e3 * float(np.median(gtv))
                                       if gtv else None),
            "rekep_rot_err_deg": float(np.median(rot)) if rot else None,
            "rekep_trans_err_mm": (1e3 * float(np.median(tr))
                                   if tr else None),
            "oracle_rot_err_deg": float(np.median(orot)) if orot else None,
            "oracle_trans_err_mm": (1e3 * float(np.median(otr))
                                    if otr else None)}
    md.append("")

    # ---- table 4b: correspondence diagnostic --------------------------------
    md.append("## 4b. What the VLM's constraints actually assert (the "
              "decisive diagnostic)\n")
    md.append("A generated program is only as good as the demo->target "
              "keypoint pairs it ties together.  A declared pair (i, j) is "
              "**correct** when the ground-truth-mapped demo keypoint i lands "
              "within 3 cm of target keypoint j AND is that target "
              "keypoint's nearest mapped demo keypoint.\n")
    md.append("| task | proposal | programs | declared pairs | correct pairs "
              "| pair accuracy | median pair err | median k_demo/k_target |")
    md.append("|---" * 8 + "|")
    summary["correspondence"] = {}
    for task in tasks:
        p = os.path.join(out_root, task, "corr.json")
        if not os.path.exists(p):
            continue
        crows = _read_json(p)["rows"]
        for prop in ("dino", "kmeans"):
            rs = [r for r in crows if r["proposal"] == prop
                  and r.get("n_pairs")]
            if not rs:
                continue
            npair = sum(r["n_pairs"] for r in rs)
            ncorr = sum(r["n_correct"] for r in rs)
            errs = [e for r in rs for e in r["pair_err_m"] if e is not None]
            kd = [r["k_demo"] for r in rs]
            kt = [r["k_target"] for r in rs]
            md.append("| %s | %s | %d | %d | %d | %.2f | %.0f mm | %.0f/%.0f |"
                      % (task, prop, len(rs), npair, ncorr,
                         ncorr / max(1, npair),
                         1e3 * float(np.median(errs)) if errs else -1,
                         float(np.median(kd)), float(np.median(kt))))
            summary["correspondence"].setdefault(task, {})[prop] = {
                "programs": len(rs), "declared_pairs": npair,
                "correct_pairs": ncorr,
                "pair_accuracy": ncorr / max(1, npair),
                "median_pair_err_mm": (1e3 * float(np.median(errs))
                                       if errs else None),
                "median_k_demo": float(np.median(kd)),
                "median_k_target": float(np.median(kt))}
    md.append("")

    # ---- table 5: query counts ----------------------------------------------
    md.append("## 5. VLM query counts\n")
    md.append("| method | per task (cached) | per rollout | total |")
    md.append("|---|---|---|---|")
    qc = {}
    for nm, key, arm in (("rekep_real", "rekep_q", "rekep_real"),
                         ("rekep_real_ourcands", "rekep_oc_q",
                          "rekep_real_ourcands")):
        calls = sum(r[key].get("attempts", 1)
                    for t in tasks for r in data[t]["rows"] if key in r)
        n = sum(1 for t in tasks for r in data[t]["rows"] if key in r)
        qc[nm] = {"per_task": "0 (no per-task cache; the demo images ride "
                              "in every query)",
                  "per_rollout": "1 query (%.2f API calls incl. retries)"
                                 % (calls / max(1, n)),
                  "total": int(calls)}
        md.append("| %s | %s | %s | %d |" % (nm, qc[nm]["per_task"],
                                             qc[nm]["per_rollout"],
                                             qc[nm]["total"]))
    qc["rekep_oracle_constraints"] = {"per_task": "0", "per_rollout": "0",
                                      "total": 0}
    md.append("| rekep_oracle_constraints | 0 | 0 | 0 |")
    summary["query_counts"] = qc
    md.append("")

    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(os.path.join(TABLES_DIR, "campaign_l.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    _write_tex(os.path.join(TABLES_DIR, "campaign_l.tex"), tasks, ARMS_TBL,
               summary)
    _write_json(os.path.join(out_root, "summary.json"), summary)
    print("\n".join(md))
    return summary


def _write_tex(path, tasks, arms, summary):
    cols = list(arms) + list(REFS)
    lines = ["% Campaign L: real-VLM relational-constraint ReKep vs ours "
             "(medium tier, 60 paired seeds/task)",
             "\\begin{tabular}{l" + "c" * len(cols) + "}",
             "\\toprule",
             "task & " + " & ".join(c.replace("_", "\\_") for c in cols)
             + " \\\\",
             "\\midrule"]
    for task in tasks:
        s = summary["per_task"][task]
        cells = ["%d/%d" % tuple(s[c]) if s.get(c) and s[c][1] else "-"
                 for c in cols]
        lines.append("%s & %s \\\\" % (task.replace("_", "\\_"),
                                       " & ".join(cells)))
    lines.append("\\midrule")
    p = summary["pooled"]
    cells = ["\\textbf{%d/%d}" % tuple(p[c]) if p.get(c) and p[c][1]
             else "-" for c in cols]
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
                   choices=["propose", "freeze", "query", "execute",
                            "corr", "report"])
    p.add_argument("--tasks", nargs="*", default=list(TASKS_L))
    p.add_argument("--arms", nargs="*", default=list(ARMS_L))
    p.add_argument("--proposals", nargs="*", default=["dino", "kmeans"])
    p.add_argument("--seed-start", type=int, default=SEEDS_L[0])
    p.add_argument("--seed-end", type=int, default=SEEDS_L[-1])
    p.add_argument("--no-dev", action="store_true")
    p.add_argument("--no-eval", action="store_true")
    # Campaign M (second-VLM generality) reuses this runner verbatim with a
    # different output root and a different served model (ALK_VLM_MODEL).
    # freeze.json is deliberately still read from results/campaign_l/ -- the
    # keypoint config and prompt stay frozen across models.
    p.add_argument("--out-root", default=OUT_ROOT)
    a = p.parse_args(argv)
    seeds = tuple(range(a.seed_start, a.seed_end + 1))
    root = a.out_root
    if a.phase == "propose":
        propose(tuple(a.tasks), dev=not a.no_dev, eval_=not a.no_eval)
    elif a.phase == "freeze":
        freeze(tuple(a.tasks), out_root=root)
    elif a.phase == "query":
        _load_freeze()
        for t in a.tasks:
            query_task(t, seeds, out_root=root,
                       proposals=tuple(a.proposals))
    elif a.phase == "execute":
        for t in a.tasks:
            execute_task(t, seeds, tuple(a.arms), out_root=root)
    elif a.phase == "corr":
        for t in a.tasks:
            corr_task(t, seeds, out_root=root)
    else:
        report(tuple(a.tasks), seeds, out_root=root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
