"""Campaign K -- two review-blocking completions on the Campaign J grid.

JOB A  `ours_vlm_adaptive`
    The paper's RECOMMENDED configuration -- selective conditioning-adaptive
    widening (alkbench.registration.adaptive_registration, +-90 deg about the
    ALK principal axis when the demo-ALK sigma-ratio trigger fires), NO
    slender depth-consistency prior, NO grasp-point correction -- deployed
    with the real VLM.  Campaign J's `ours_vlm` cached every discrete VLM
    answer per scene (results/campaign_j/<task>/<seed>/ours_query.json and
    <task>/ours_demo_query.json); the adaptive stage changes only the
    registration step downstream of those answers, so this campaign re-runs
    the EXECUTION with adaptive registration while REUSING the recorded
    answers -- zero new VLM queries.  Output:
    results/campaign_k/ours_vlm_adaptive/<task>/<seed>/rollout_ours_vlm_adaptive.json

    The "no depth prior" half of the recommendation is the default here by
    construction: the depth prior lives in candidate/ALK generation
    (oracle.solve(depth_consistent="auto") on the ORACLE path,
    alk_from_candidates(depth_consistent=...) otherwise) and this campaign
    calls alk_from_candidates exactly as Campaign J's `ours_vlm` did
    (depth_consistent False).  The geometry therefore matches Campaign G's
    frozen `sel90_nodp` setting: adaptive_registration with its default
    base bounds (15 deg / 20 mm), default trigger threshold, and
    axis_bound_deg = 90.

JOB B  `moka_real_ourcands` completed to 60 seeds x 5 tasks
    Campaign J ran this arm (MOKA's mark-selection interface + motion
    mapping on OUR k-means candidate marks) to 60 seeds only on box_open;
    the other four tasks stopped at a 6-seed pilot when the VLM host went
    away.  This campaign completes the missing 54 seeds x 4 tasks with NEW
    VLM queries under the SAME freeze (prompts, MOKA_CFG, plan caches --
    Campaign J's moka_plan.json / ours_demo_query.json are symlinked in
    verbatim, and pipeline.campaign_j.select_task / execute_task are called
    with out_root pointed here, so every code path is Campaign J's own).
    Campaign J's existing pilot queries/rollouts are symlinked in and
    reused, NOT re-run.  Output:
    results/campaign_k/moka_real_ourcands/<task>/<seed>/

Campaign J trees are READ-ONLY inputs; nothing under results/campaign_j/ is
modified.

Phases (all resumable)
    setup     -- build the moka_real_ourcands symlink tree
    select    -- JOB B's missing VLM queries (the only phase needing the VLM)
    execute   -- JOB B rollouts (arm moka_real_ourcands, no VLM)
    adaptive  -- JOB A rollouts (no VLM)
    report    -- results/tables/campaign_k.{md,tex} + summary.json
"""
import argparse
import os
import sys
import time
import traceback

import numpy as np

SIMBENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SIMBENCH not in sys.path:
    sys.path.insert(0, SIMBENCH)

from alkbench import (alk_from_candidates, procrustes,          # noqa: E402
                      transform_points)
from alkbench.registration import adaptive_registration         # noqa: E402
from pipeline import campaign_j as cj                           # noqa: E402
from stats import tests as st                                   # noqa: E402

OUT_ROOT = os.path.join(SIMBENCH, "results", "campaign_k")
OUT_B = os.path.join(OUT_ROOT, "moka_real_ourcands")
OUT_A = os.path.join(OUT_ROOT, "ours_vlm_adaptive")
TABLES_DIR = os.path.join(SIMBENCH, "results", "tables")
J_ROOT = cj.OUT_ROOT                     # results/campaign_j (read-only)
TASKS = cj.TASKS_J
SEEDS = cj.SEEDS_J

ARM_A = "ours_vlm_adaptive"
ARM_B = "moka_real_ourcands"


def _read(p):
    return cj._read_json(p)


def _link(src, dst):
    """Symlink src -> dst if src exists and dst doesn't."""
    if os.path.exists(src) and not os.path.lexists(dst):
        os.symlink(os.path.abspath(src), dst)


# ---------------------------------------------------------------------------
# phase: setup (JOB B symlink tree; campaign_j stays untouched)
# ---------------------------------------------------------------------------

def setup(tasks=TASKS, seeds=SEEDS):
    n = 0
    for task in tasks:
        tdir = os.path.join(OUT_B, task)
        os.makedirs(tdir, exist_ok=True)
        for fn in ("moka_plan.json", "ours_demo_query.json"):
            _link(os.path.join(J_ROOT, task, fn), os.path.join(tdir, fn))
        for seed in seeds:
            sdir = os.path.join(tdir, str(seed))
            os.makedirs(sdir, exist_ok=True)
            for fn in ("moka_query.json", "ours_query.json",
                       "moka_ourcands_query.json",
                       "rollout_%s.json" % ARM_B):
                _link(os.path.join(J_ROOT, task, str(seed), fn),
                      os.path.join(sdir, fn))
            n += 1
    print("setup: %d seed dirs linked under %s" % (n, OUT_B))


# ---------------------------------------------------------------------------
# phase: select / execute (JOB B -- delegate to campaign_j with out_root here)
# ---------------------------------------------------------------------------

def select(tasks, seeds):
    for task in tasks:
        cj.select_task(task, seeds, out_root=OUT_B)


def execute(tasks, seeds):
    for task in tasks:
        cj.execute_task(task, seeds, arms=(ARM_B,), out_root=OUT_B)


# ---------------------------------------------------------------------------
# phase: adaptive (JOB A -- recommended config on the cached VLM answers)
# ---------------------------------------------------------------------------

def _adaptive_map(p_d, p_t, demo_q, oq):
    """T_map from the CACHED VLM discrete answers, selective-widening
    adaptive registration (NO depth prior, NO grasp correction).  Mirrors
    campaign_j._ours_vlm_map except for the registration call."""
    dchoice = demo_q["vlm"]["choice"]
    tchoice = oq["vlm"]["choice"]
    out = {"vlm": {"demo_choice": dchoice, "target_choice": tchoice,
                   "oracle_choice": oq["oracle_choice"],
                   "agree": oq.get("agree")},
           "registration": True, "correction": False, "adaptive": True,
           "depth_prior": False}
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
    reg = adaptive_registration(p_d["cands"].points3d, p_t["cands"].points3d,
                                T0, demo_alk)
    ab = reg["adaptive"]
    out["T_map"] = np.asarray(reg["T"]).tolist()
    out["T_init"] = np.asarray(T0).tolist()
    out["chamfer_before_m"] = float(reg["chamfer_init"])
    out["chamfer_after_m"] = float(reg["chamfer_final"])
    out["adaptive_registration"] = {
        "adaptive_triggered": bool(ab["adaptive_triggered"]),
        "axis_rot_bound_deg": ab["axis_rot_bound_deg"],
        "coarse_axis_theta_deg": reg["coarse_axis_theta_deg"],
        "widened_axis_world": (None if reg["axis_world"] is None
                               else np.asarray(reg["axis_world"]).tolist()),
        "sigma_ratios": {
            "singular_values": ab["singular_values"],
            "sigma23": ab["sigma23"],
            "sigma_ratio": ab["sigma_ratio"],
            "sigma_ratio_threshold": ab["sigma_ratio_threshold"]}}
    out["n_vlm_calls_rollout"] = 0        # answers reused from Campaign J
    return out


def adaptive(tasks, seeds):
    from pipeline import campaign as cg
    from pipeline import retarget_runner as rr
    from simtasks import envs, motion

    for task in tasks:
        root = cj.data_root(task)
        dqp = os.path.join(J_ROOT, task, "ours_demo_query.json")
        demo_q = _read(dqp)
        demo = rr.load_demo(task, root)
        env = None
        try:
            for seed in seeds:
                sdir = os.path.join(OUT_A, task, str(seed))
                outp = os.path.join(sdir, "rollout_%s.json" % ARM_A)
                if os.path.exists(outp):
                    continue
                oqp = os.path.join(J_ROOT, task, str(seed),
                                   "ours_query.json")
                if not os.path.exists(oqp):
                    print("[%s %d] no cached ours_query.json, skipping"
                          % (task, seed), flush=True)
                    continue
                oq = _read(oqp)
                t0 = time.time()
                d, t, ctx, pair = cj.pair_ctx(task, seed, root=root)
                inst = pair["target_instance"]
                demo_pose = pair["demo_object_poses"][inst]
                target_pose = pair["target_object_poses"][inst]
                p_d = ctx.percept(d, "demo")
                p_t = ctx.percept(t, "target")
                res = {"task": task, "seed": seed, "method": ARM_A,
                       "variant": ARM_A, "tier": "medium", "success": False,
                       "failure_stage": None, "camera": p_t["camera"],
                       "T_gt": ctx.T_gt.tolist(), "data_root": root,
                       "query_source": oqp,
                       "perception": {
                           "demo": {"sanity_ok": p_d["sanity_ok"],
                                    "centroid_err_m": p_d["centroid_err_m"]},
                           "target": {"sanity_ok": p_t["sanity_ok"],
                                      "centroid_err_m":
                                          p_t["centroid_err_m"]}}}
                try:
                    res.update(_adaptive_map(p_d, p_t, demo_q, oq))
                except Exception as e:
                    traceback.print_exc()
                    res["failure_stage"] = "method_error"
                    res["error"] = repr(e)
                if res.get("T_map") is None:
                    res["time_s"] = round(time.time() - t0, 1)
                    cj._write_json(outp, res)
                    print("[%s %d %s] success=False stage=%s (no exec)"
                          % (task, seed, ARM_A, res["failure_stage"]),
                          flush=True)
                    continue
                T_map = np.asarray(res["T_map"], dtype=np.float64)
                rot, tr = cj.map_errors(T_map, ctx.T_gt, demo_pose["pos"],
                                        target_pose["pos"])
                res["rot_err_deg"] = rot
                res["trans_err_m"] = tr
                # no grasp correction, exactly as ours_vlm in Campaign J
                wps = rr.retarget_waypoint_dicts(demo["waypoints"], T_map)
                if env is None:
                    env = cg.make_tier_env(task, "medium")
                envs.reset_with_seed(env, pair["target_seed"])
                ex = motion.execute_waypoints(
                    env, wps, grasp_check=rr._grasp_check(task))
                success = bool(envs.success_checker(task)(env))
                res.update({"grasped": ex["grasped"],
                            "waypoints_converged": ex["converged"],
                            "n_exec_steps": ex["n_steps"],
                            "success": success})
                if task == "pour":
                    res["pour_orientation"] = cj._json_safe(
                        envs.pour_orientation_trace(env))
                if not success:
                    res["failure_stage"] = cj._fail_stage(
                        p_d["sanity_ok"], p_t["sanity_ok"], rot, tr,
                        ex["grasped"])
                res["time_s"] = round(time.time() - t0, 1)
                cj._write_json(outp, res)
                trig = (res.get("adaptive_registration") or {}).get(
                    "adaptive_triggered")
                print("[%s %d %s] success=%s stage=%-11s rot=%5.1f "
                      "trans=%5.1fmm trig=%s (%.1fs)"
                      % (task, seed, ARM_A, success,
                         str(res["failure_stage"]), rot, 1e3 * tr, trig,
                         res["time_s"]), flush=True)
        finally:
            if env is not None:
                env.close()


# ---------------------------------------------------------------------------
# phase: report
# ---------------------------------------------------------------------------

def _load_rollouts(root, task, arm, seeds=SEEDS):
    out = {}
    for seed in seeds:
        p = os.path.join(root, task, str(seed), "rollout_%s.json" % arm)
        if os.path.exists(p):
            out[seed] = _read(p)
    return out


def _wilson(k, n):
    if n == 0:
        return "-"
    lo, hi = st.wilson_ci(k, n)
    return "%d/%d = %.2f [%.2f, %.2f]" % (k, n, k / n, lo, hi)


def _mcnemar_rows(sa, sb, tasks):
    """Per-task + pooled McNemar of paired success dicts sa[task][seed]."""
    pv, rows = [], []
    for task in tasks:
        va, vb = sa.get(task, {}), sb.get(task, {})
        shared = sorted(set(va) & set(vb))
        if not shared:
            continue
        r = st.mcnemar_from_vectors([va[s] for s in shared],
                                    [vb[s] for s in shared])
        rows.append([task, r.b, r.c, r.pvalue, None, len(shared)])
        pv.append(r.pvalue)
    holm = st.holm_bonferroni(pv) if pv else []
    for row, ph in zip(rows, holm):
        row[4] = ph
    xa, xb = [], []
    for task in tasks:
        va, vb = sa.get(task, {}), sb.get(task, {})
        for s in sorted(set(va) & set(vb)):
            xa.append(va[s])
            xb.append(vb[s])
    pooled = (st.mcnemar_from_vectors(xa, xb) if xa else None)
    return rows, pooled, len(xa)


def report(tasks=TASKS, seeds=SEEDS):
    md = ["# Campaign K -- recommended configuration deployed "
          "(ours_vlm_adaptive) + moka_real_ourcands at full N\n",
          ("Medium tier, paired seeds %d..%d (N=%d/task), model "
           "Qwen/Qwen3-VL-32B-Instruct-FP8, same scenes/executor/criteria "
           "as Campaign J (box_open: data_medium_a2; pour: orientation-"
           "aware criterion).  `ours_vlm_adaptive` REUSES Campaign J's "
           "cached per-scene VLM answers (0 new queries); "
           "`moka_real_ourcands` completes Campaign J's 6-seed pilot with "
           "new queries under the identical freeze.  Raw records under "
           "results/campaign_k/.\n") % (seeds[0], seeds[-1], len(seeds))]
    summary = {"tasks": list(tasks)}

    # ---- load everything ----------------------------------------------------
    A = {t: _load_rollouts(OUT_A, t, ARM_A, seeds) for t in tasks}
    base = {t: _load_rollouts(J_ROOT, t, "ours_vlm", seeds) for t in tasks}
    moka = {t: _load_rollouts(J_ROOT, t, "moka_real", seeds) for t in tasks}
    B = {t: _load_rollouts(OUT_B, t, ARM_B, seeds) for t in tasks}
    oq = {t: {s: _read(os.path.join(J_ROOT, t, str(s), "ours_query.json"))
              for s in seeds
              if os.path.exists(os.path.join(J_ROOT, t, str(s),
                                             "ours_query.json"))}
          for t in tasks}
    kq = {t: {s: _read(os.path.join(OUT_B, t, str(s),
                                    "moka_ourcands_query.json"))
              for s in seeds
              if os.path.exists(os.path.join(OUT_B, t, str(s),
                                             "moka_ourcands_query.json"))}
          for t in tasks}
    mq = {t: {s: _read(os.path.join(J_ROOT, t, str(s), "moka_query.json"))
              for s in seeds
              if os.path.exists(os.path.join(J_ROOT, t, str(s),
                                             "moka_query.json"))}
          for t in tasks}

    def succ(d):
        return {t: {s: int(bool(d[t][s]["success"])) for s in d[t]}
                for t in tasks}

    sA, sBase, sMoka, sB = succ(A), succ(base), succ(moka), succ(B)

    # ---- JOB A table 1: success ---------------------------------------------
    md.append("## A1. End-to-end success -- recommended configuration vs "
              "Campaign J arms\n")
    md.append("| task | ours_vlm_adaptive (recommended cfg) | ours_vlm "
              "(fixed bounds, Campaign J) | moka_real (Campaign J) |")
    md.append("|---|---|---|---|")
    pool = {k: [0, 0] for k in ("A", "base", "moka")}
    summary["jobA_success"] = {}
    for t in tasks:
        cells = []
        for key, sv in (("A", sA), ("base", sBase), ("moka", sMoka)):
            k, n = sum(sv[t].values()), len(sv[t])
            pool[key][0] += k
            pool[key][1] += n
            cells.append(_wilson(k, n))
        md.append("| %s | %s |" % (t, " | ".join(cells)))
        summary["jobA_success"][t] = {
            "ours_vlm_adaptive": [sum(sA[t].values()), len(sA[t])],
            "ours_vlm": [sum(sBase[t].values()), len(sBase[t])],
            "moka_real": [sum(sMoka[t].values()), len(sMoka[t])]}
    md.append("| **pooled** | %s |"
              % " | ".join("**%s**" % _wilson(*pool[k])
                           for k in ("A", "base", "moka")))
    summary["jobA_success"]["pooled"] = {
        "ours_vlm_adaptive": pool["A"], "ours_vlm": pool["base"],
        "moka_real": pool["moka"]}
    md.append("")

    # ---- JOB A table 2: paired McNemar --------------------------------------
    md.append("## A2. Paired McNemar (exact), Holm across tasks\n")
    md.append("| comparison | task | b (adaptive win) | c (other win) | p | "
              "p_Holm | n |")
    md.append("|---|---|---|---|---|---|---|")
    summary["jobA_mcnemar"] = {}
    for name, other in (("ours_vlm", sBase), ("moka_real", sMoka)):
        rows, pooled, n = _mcnemar_rows(sA, other, tasks)
        for t_, b, c, p, ph, nn in rows:
            md.append("| adaptive vs %s | %s | %d | %d | %.3g | %.3g | %d |"
                      % (name, t_, b, c, p, ph, nn))
        md.append("| adaptive vs %s | **pooled** | %d | %d | **%.3g** | -- "
                  "| %d |" % (name, pooled.b, pooled.c, pooled.pvalue, n))
        summary["jobA_mcnemar"][name] = {
            "per_task": [{"task": t_, "b": b, "c": c, "p": p, "p_holm": ph,
                          "n": nn} for t_, b, c, p, ph, nn in rows],
            "pooled": {"b": pooled.b, "c": pooled.c, "p": pooled.pvalue,
                       "n": n}}
    md.append("")

    # ---- JOB A table 3: trigger behaviour under VLM answers ------------------
    md.append("## A3. Adaptive trigger under VLM-driven (possibly corrupted) "
              "initializations\n")
    md.append("The trigger reads the sigma-ratio of the DEMO ALK built from "
              "the VLM's demo answers; `widened` = trigger fired and the "
              "+-90 deg axis search ran.\n")
    md.append("| task | rollouts with T_map | widened | trigger rate | "
              "median sigma-ratio | median coarse axis rotation applied "
              "(widened only) |")
    md.append("|---|---|---|---|---|---|")
    summary["jobA_trigger"] = {}
    for t in tasks:
        recs = [r for r in A[t].values()
                if r.get("adaptive_registration") is not None]
        trig = [r for r in recs
                if r["adaptive_registration"]["adaptive_triggered"]]
        ratios = [r["adaptive_registration"]["sigma_ratios"]["sigma_ratio"]
                  for r in recs]
        thetas = [abs(r["adaptive_registration"]["coarse_axis_theta_deg"])
                  for r in trig]
        md.append("| %s | %d | %d | %.2f | %.3f | %s deg |"
                  % (t, len(recs), len(trig),
                     len(trig) / max(1, len(recs)),
                     float(np.median(ratios)) if ratios else float("nan"),
                     ("%.0f" % float(np.median(thetas))) if thetas else "-"))
        summary["jobA_trigger"][t] = {
            "n": len(recs), "n_triggered": len(trig),
            "median_sigma_ratio": (float(np.median(ratios)) if ratios
                                   else None),
            "median_abs_coarse_theta_deg": (float(np.median(thetas))
                                            if thetas else None)}
    md.append("")

    # ---- JOB A table 4: VLM-error rollouts, adaptive vs not -------------------
    md.append("## A4. Does the adaptive stage rescue or damage VLM-error "
              "rollouts?\n")
    md.append("Split by whether the VLM's target-scene discrete answers "
              "agree with the oracle (joint_raw = phi1, phi2, phi3 all "
              "exact; pair-set = the unordered {phi1, phi2} pair matches).  "
              "Same scenes, same cached answers -- the ONLY difference is "
              "the registration stage.\n")
    md.append("| subset | n | success adaptive | success fixed-bounds | "
              "median rot err adaptive | median rot err fixed | median "
              "trans err adaptive | median trans err fixed |")
    md.append("|---|---|---|---|---|---|---|---|")
    summary["jobA_vlm_error"] = {}

    def agree_of(t, s, key):
        a = (oq[t][s].get("agree") or {})
        return bool(a.get(key))

    subsets = [
        ("joint agree", lambda t, s: agree_of(t, s, "joint_raw")),
        ("joint DISagree", lambda t, s: not agree_of(t, s, "joint_raw")),
        ("pair-set agree", lambda t, s: agree_of(t, s, "pair_set")),
        ("pair-set DISagree", lambda t, s: not agree_of(t, s, "pair_set")),
    ]
    for name, pred in subsets:
        ka = na = kb = 0
        rotA, rotB, trA, trB = [], [], [], []
        for t in tasks:
            for s in sorted(set(sA[t]) & set(sBase[t]) & set(oq[t])):
                if not pred(t, s):
                    continue
                na += 1
                ka += sA[t][s]
                kb += sBase[t][s]
                for src, rv, tv in ((A, rotA, trA), (base, rotB, trB)):
                    r = src[t][s]
                    if r.get("rot_err_deg") is not None:
                        rv.append(r["rot_err_deg"])
                        tv.append(r["trans_err_m"])
        f = lambda v, s_=1.0, fmt="%.1f": ((fmt % (s_ * float(np.median(v))))
                                           if v else "-")
        md.append("| %s | %d | %s | %s | %s deg | %s deg | %s mm | %s mm |"
                  % (name, na, _wilson(ka, na), _wilson(kb, na),
                     f(rotA), f(rotB), f(trA, 1e3, "%.0f"),
                     f(trB, 1e3, "%.0f")))
        summary["jobA_vlm_error"][name] = {
            "n": na, "success_adaptive": [ka, na],
            "success_fixed": [kb, na],
            "median_rot_adaptive_deg": (float(np.median(rotA)) if rotA
                                        else None),
            "median_rot_fixed_deg": (float(np.median(rotB)) if rotB
                                     else None),
            "median_trans_adaptive_mm": (1e3 * float(np.median(trA))
                                         if trA else None),
            "median_trans_fixed_mm": (1e3 * float(np.median(trB))
                                      if trB else None)}
    # paired McNemar inside the disagree subset (pooled)
    for key, label in (("joint_raw", "joint DISagree"),
                       ("pair_set", "pair-set DISagree")):
        xa, xb = [], []
        for t in tasks:
            for s in sorted(set(sA[t]) & set(sBase[t]) & set(oq[t])):
                if not agree_of(t, s, key):
                    xa.append(sA[t][s])
                    xb.append(sBase[t][s])
        if xa:
            r = st.mcnemar_from_vectors(xa, xb)
            md.append("")
            md.append("Paired McNemar within %s (pooled, n=%d): adaptive "
                      "wins b=%d, fixed wins c=%d, p=%.3g."
                      % (label, len(xa), r.b, r.c, r.pvalue))
            summary["jobA_vlm_error"]["mcnemar_" + key] = {
                "n": len(xa), "b": r.b, "c": r.c, "p": r.pvalue}
    md.append("")

    # ---- JOB B table 1: success at full N ------------------------------------
    md.append("## B1. moka_real vs moka_real_ourcands at full N "
              "(candidate-generation ablation)\n")
    md.append("Same MOKA interface, prompts, motion mapping; the only "
              "difference is the candidate pool the marks are drawn from "
              "(contour-FPS vs our k-means interior points).\n")
    md.append("| task | moka_real (contour-FPS cands) | moka_real_ourcands "
              "(k-means cands) | ours_vlm (full pipeline) |")
    md.append("|---|---|---|---|")
    poolB = {k: [0, 0] for k in ("moka", "oc", "ours")}
    summary["jobB_success"] = {}
    for t in tasks:
        cells = []
        for key, sv in (("moka", sMoka), ("oc", sB), ("ours", sBase)):
            k, n = sum(sv[t].values()), len(sv[t])
            poolB[key][0] += k
            poolB[key][1] += n
            cells.append(_wilson(k, n))
        md.append("| %s | %s |" % (t, " | ".join(cells)))
        summary["jobB_success"][t] = {
            "moka_real": [sum(sMoka[t].values()), len(sMoka[t])],
            "moka_real_ourcands": [sum(sB[t].values()), len(sB[t])],
            "ours_vlm": [sum(sBase[t].values()), len(sBase[t])]}
    md.append("| **pooled** | %s |"
              % " | ".join("**%s**" % _wilson(*poolB[k])
                           for k in ("moka", "oc", "ours")))
    summary["jobB_success"]["pooled"] = {
        "moka_real": poolB["moka"], "moka_real_ourcands": poolB["oc"],
        "ours_vlm": poolB["ours"]}
    md.append("")

    # ---- JOB B table 2: McNemar ----------------------------------------------
    md.append("## B2. Paired McNemar, moka_real_ourcands vs moka_real "
              "(and vs ours_vlm)\n")
    md.append("| comparison | task | b (ourcands win) | c (other win) | p | "
              "p_Holm | n |")
    md.append("|---|---|---|---|---|---|---|")
    summary["jobB_mcnemar"] = {}
    for name, other in (("moka_real", sMoka), ("ours_vlm", sBase)):
        rows, pooled, n = _mcnemar_rows(sB, other, tasks)
        for t_, b, c, p, ph, nn in rows:
            md.append("| ourcands vs %s | %s | %d | %d | %.3g | %.3g | %d |"
                      % (name, t_, b, c, p, ph, nn))
        md.append("| ourcands vs %s | **pooled** | %d | %d | **%.3g** | -- "
                  "| %d |" % (name, pooled.b, pooled.c, pooled.pvalue, n))
        summary["jobB_mcnemar"][name] = {
            "per_task": [{"task": t_, "b": b, "c": c, "p": p, "p_holm": ph,
                          "n": nn} for t_, b, c, p, ph, nn in rows],
            "pooled": {"b": pooled.b, "c": pooled.c, "p": pooled.pvalue,
                       "n": n}}
    md.append("")

    # ---- JOB B table 3: selection diagnostics ---------------------------------
    md.append("## B3. Mark-selection diagnostics at full N\n")
    md.append("| task | contour-FPS: mark acc | quantisation floor | "
              "selected err | k-means: mark acc | quantisation floor | "
              "selected err |")
    md.append("|---|---|---|---|---|---|---|")
    summary["jobB_diag"] = {}
    for t in tasks:
        row = []
        rec = {}
        for tag, qs in (("fps", mq[t]), ("km", kq[t])):
            vals = list(qs.values())
            ok = [q for q in vals if q.get("selection") is not None]
            acc = sum(1 for q in ok if q.get("mark_correct"))
            best = [q["best_possible_mark_err_m"] for q in vals
                    if q.get("best_possible_mark_err_m") is not None]
            sel = [q["selected_mark_err_m"] for q in ok
                   if q.get("selected_mark_err_m") is not None]
            row += ["%d/%d" % (acc, len(ok)),
                    ("%.0f mm" % (1e3 * float(np.median(best)))) if best
                    else "-",
                    ("%.0f mm" % (1e3 * float(np.median(sel)))) if sel
                    else "-"]
            rec[tag] = {"mark_acc": [acc, len(ok)],
                        "median_best_mm": (1e3 * float(np.median(best))
                                           if best else None),
                        "median_sel_mm": (1e3 * float(np.median(sel))
                                          if sel else None)}
        md.append("| %s | %s |" % (t, " | ".join(row)))
        summary["jobB_diag"][t] = rec
    md.append("")

    # ---- JOB B table 4: failure stages (diagnostic taxonomy) -------------------
    md.append("## B4. Failure stages, moka_real_ourcands at full N "
              "(diagnostic taxonomy of campaign_j.report)\n")
    md.append("| task | n fail | perception | selection | mapping | grasp | "
              "execution |")
    md.append("|---|---|---|---|---|---|---|")
    poolS = {k: 0 for k in ("perception", "selection", "mapping", "grasp",
                            "execution")}
    nfp = 0
    summary["jobB_stages"] = {}
    for t in tasks:
        cnt = {k: 0 for k in poolS}
        nf = 0
        for r in B[t].values():
            if r.get("success"):
                continue
            nf += 1
            cnt[cj._moka_stage(r)] += 1
        for k in cnt:
            poolS[k] += cnt[k]
        nfp += nf
        md.append("| %s | %d | %d | %d | %d | %d | %d |"
                  % (t, nf, cnt["perception"], cnt["selection"],
                     cnt["mapping"], cnt["grasp"], cnt["execution"]))
        summary["jobB_stages"][t] = dict(cnt, n_fail=nf)
    md.append("| **pooled** | %d | %d | %d | %d | %d | %d |"
              % (nfp, poolS["perception"], poolS["selection"],
                 poolS["mapping"], poolS["grasp"], poolS["execution"]))
    summary["jobB_stages"]["pooled"] = dict(poolS, n_fail=nfp)
    md.append("")

    # ---- write ----------------------------------------------------------------
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(os.path.join(TABLES_DIR, "campaign_k.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    _write_tex(os.path.join(TABLES_DIR, "campaign_k.tex"), tasks, summary)
    cj._write_json(os.path.join(OUT_ROOT, "summary.json"), summary)
    print("\n".join(md))
    return summary


def _write_tex(path, tasks, summary):
    lines = ["% Campaign K: recommended config deployed (ours_vlm_adaptive)"
             " + moka_real_ourcands at full N (medium tier, 60 seeds/task)",
             "\\begin{tabular}{lccccc}",
             "\\toprule",
             "task & ours\\_vlm\\_adaptive & ours\\_vlm & moka\\_real & "
             "moka\\_real\\_ourcands & moka\\_real\\_ourcands$-$"
             "moka\\_real \\\\",
             "\\midrule"]
    for t in tasks:
        a = summary["jobA_success"][t]
        b = summary["jobB_success"][t]
        d = (b["moka_real_ourcands"][0] - b["moka_real"][0])
        lines.append("%s & %d/%d & %d/%d & %d/%d & %d/%d & %+d \\\\"
                     % (t.replace("_", "\\_"),
                        a["ours_vlm_adaptive"][0], a["ours_vlm_adaptive"][1],
                        a["ours_vlm"][0], a["ours_vlm"][1],
                        a["moka_real"][0], a["moka_real"][1],
                        b["moka_real_ourcands"][0],
                        b["moka_real_ourcands"][1], d))
    lines.append("\\midrule")
    a = summary["jobA_success"]["pooled"]
    b = summary["jobB_success"]["pooled"]
    d = b["moka_real_ourcands"][0] - b["moka_real"][0]
    lines.append("pooled & \\textbf{%d/%d} & %d/%d & %d/%d & "
                 "%d/%d & %+d \\\\"
                 % (a["ours_vlm_adaptive"][0], a["ours_vlm_adaptive"][1],
                    a["ours_vlm"][0], a["ours_vlm"][1],
                    a["moka_real"][0], a["moka_real"][1],
                    b["moka_real_ourcands"][0], b["moka_real_ourcands"][1],
                    d))
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True,
                   choices=["setup", "select", "execute", "adaptive",
                            "report"])
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--seed-start", type=int, default=SEEDS[0])
    p.add_argument("--seed-end", type=int, default=SEEDS[-1])
    a = p.parse_args(argv)
    seeds = tuple(range(a.seed_start, a.seed_end + 1))
    if a.phase == "setup":
        setup(tuple(a.tasks), seeds)
    elif a.phase == "select":
        select(tuple(a.tasks), seeds)
    elif a.phase == "execute":
        execute(tuple(a.tasks), seeds)
    elif a.phase == "adaptive":
        adaptive(tuple(a.tasks), seeds)
    else:
        report(tuple(a.tasks), seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
