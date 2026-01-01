"""Campaign H tables: render-resolution sensitivity of the simulation results.

Compares the SAME paired seeds / tier / methods at render sizes 256 (existing
Campaign-A rollouts, reused -- never re-run) vs 384 / 512
(``results/campaign_h/res<size>/``), and answers the two questions that
matter for the paper's (comparative, paired) claims:

  (a) do ABSOLUTE success rates move with resolution?
  (b) does the ORDERING of methods move?

Sections written to ``results/tables/campaign_h.{md,tex}``:
  1. scene-identity check  -- ground-truth object poses of every (task, seed)
     pair must be bit-identical across data roots (only the render size
     differs; placement RNG is seeded inside envs.reset_with_seed,
     independently of the camera)
  2. perception diagnostics -- target/demo mask pixel counts, pixels per
     k-means cluster, back-projected centroid error, and candidate-centroid
     stability (matched k=8 centroid displacement 256 -> higher res)
  3. success rates per task x method + paired McNemar (256 vs 512)
  4. method ordering per task (ranks, Kendall tau, pairwise-order agreement)
     and the paper's key vs-ours comparisons at each resolution
  5. mapping accuracy (T_map rotation / translation error vs GT)
  6. the ALK-vs-oracle-correspondence gap (ours_full vs kp_fps4)
  7. the conditioning-adaptive arm on pour (256 vs 512)

Pure stdlib + numpy (+ stats.tests); no pandas.  Run from the repository root:

    python -m stats.campaign_h_tables
"""
import argparse
import json
import os
from collections import OrderedDict

import numpy as np

from stats import aggregate as agg
from stats import tests as st
from baselines import common as bcommon

SIMBENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIRS = OrderedDict([
    (256, os.path.join(SIMBENCH, "results", "campaign_a")),
    (384, os.path.join(SIMBENCH, "results", "campaign_h", "res384")),
    (512, os.path.join(SIMBENCH, "results", "campaign_h", "res512")),
])
DATA_ROOTS = OrderedDict([
    (256, os.path.join(SIMBENCH, "data_medium")),
    (384, os.path.join(SIMBENCH, "data_medium_res384")),
    (512, os.path.join(SIMBENCH, "data_medium_res512")),
])
ADAPTIVE_DIRS = OrderedDict([
    (256, os.path.join(SIMBENCH, "results", "campaign_a_adaptive")),
    (384, os.path.join(SIMBENCH, "results", "campaign_h", "adaptive_res384")),
    (512, os.path.join(SIMBENCH, "results", "campaign_h", "adaptive_res512")),
])
METHODS = ["ours_full", "ours_noreg", "icp", "kp_fps4", "moka_oracle"]
SEEDS = list(range(1000, 1030))
K = 8            # candidate k-means clusters (paper's setting)
OUT_DIR = os.path.join(SIMBENCH, "results", "tables")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_res(root, tasks, methods=METHODS, seeds=SEEDS):
    """{(task, method, seed): row} for one resolution's rollout tree."""
    if not os.path.isdir(root):
        return {}
    out = {}
    for r in agg.load_rollouts(root):
        key = (r["task"], r["method"], r["seed"])
        if r["task"] in tasks and r["method"] in methods and r["seed"] in seeds:
            out[key] = r
    return out


def available_tasks(root, tasks):
    if not os.path.isdir(root):
        return []
    return [t for t in tasks if os.path.isdir(os.path.join(root, t))]


def vec(rows, task, method, seeds, field="success"):
    """Seed-aligned list of `field`; None where the rollout is missing."""
    return [(rows.get((task, method, s)) or {}).get(field) for s in seeds]


def paired(rows_a, rows_b, task, method, seeds, field="success"):
    """Seeds where BOTH resolutions have the rollout -> two aligned vectors."""
    a, b = [], []
    for s in seeds:
        ra = rows_a.get((task, method, s))
        rb = rows_b.get((task, method, s))
        if ra is None or rb is None:
            continue
        va, vb = ra.get(field), rb.get(field)
        if va is None or vb is None:
            continue
        a.append(va)
        b.append(vb)
    return a, b


# ---------------------------------------------------------------------------
# 1. scene identity across data roots
# ---------------------------------------------------------------------------

def pose_match(tasks, seeds=SEEDS, base=256):
    """Ground-truth object poses of each pair, base data root vs the others.

    Returns {size: {"n_pairs": int, "n_objects": int, "max_abs_delta": float,
                    "bit_identical": bool, "mismatches": [...]}}.
    """
    out = OrderedDict()
    for size, droot in DATA_ROOTS.items():
        if size == base or not os.path.isdir(droot):
            continue
        rec = {"n_pairs": 0, "n_objects": 0, "max_abs_delta": 0.0,
               "mismatches": [], "tasks": []}
        for task in tasks:
            n_task = 0
            for s in seeds:
                pa = os.path.join(DATA_ROOTS[base], task, str(s), "pair.json")
                pb = os.path.join(droot, task, str(s), "pair.json")
                if not (os.path.exists(pa) and os.path.exists(pb)):
                    continue
                with open(pa) as f:
                    A = json.load(f)
                with open(pb) as f:
                    B = json.load(f)
                rec["n_pairs"] += 1
                n_task += 1
                if A["target_seed"] != B["target_seed"] or \
                        A["demo_seed"] != B["demo_seed"]:
                    rec["mismatches"].append("%s/%d: seed mismatch" % (task, s))
                for side in ("demo_object_poses", "target_object_poses"):
                    for inst, pose in A[side].items():
                        qb = B[side].get(inst)
                        if qb is None:
                            rec["mismatches"].append(
                                "%s/%d: %s missing %s" % (task, s, side, inst))
                            continue
                        u = np.asarray(pose["pos"] + pose["quat_xyzw"])
                        v = np.asarray(qb["pos"] + qb["quat_xyzw"])
                        d = float(np.abs(u - v).max())
                        rec["n_objects"] += 1
                        rec["max_abs_delta"] = max(rec["max_abs_delta"], d)
                        if d != 0.0:
                            rec["mismatches"].append(
                                "%s/%d: %s %s delta=%.3e" % (task, s, side,
                                                             inst, d))
            if n_task:
                rec["tasks"].append("%s(%d)" % (task, n_task))
        rec["bit_identical"] = (not rec["mismatches"]) and rec["n_pairs"] > 0
        out[size] = rec
    return out


# ---------------------------------------------------------------------------
# 2. perception diagnostics straight from the captures
# ---------------------------------------------------------------------------

def _match_points(P, Q):
    """Min-cost 1-1 matching between two k x 3 point sets -> distances (m)."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=-1)
    try:
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(D)
        return D[ri, ci]
    except ImportError:                                   # greedy fallback
        D = D.copy()
        out = []
        for _ in range(min(P.shape[0], Q.shape[0])):
            i, j = np.unravel_index(np.argmin(D), D.shape)
            out.append(D[i, j])
            D[i, :] = np.inf
            D[:, j] = np.inf
        return np.asarray(out)


def perceive_capture(task, size, seed, which):
    """k=8 candidate perception of one saved capture (no simulator needed)."""
    d = os.path.join(DATA_ROOTS[size], task, str(seed),
                     "demo" if which == "demo" else "target", "scene")
    if not os.path.isdir(d):
        return None
    cap = bcommon.load_capture(d)
    p = bcommon.perceive(cap, task, k=K, seed=0)
    cs = p["cands"]
    counts = np.bincount(np.asarray(cs.labels).ravel(), minlength=K) \
        if getattr(cs, "labels", None) is not None else None
    return {"mask_pixels": int(p["mask_pixels"]),
            "n_points": int(p["n_points"]),
            "centroid_err_m": float(p["centroid_err_m"]),
            "sanity_ok": bool(p["sanity_ok"]),
            "cand3d": np.asarray(cs.candidates3d, dtype=float),
            "min_cluster_px": (int(counts.min()) if counts is not None
                               else None)}


def perception_table(tasks, seeds=SEEDS, base=256):
    """Per task x resolution mask/candidate diagnostics + 256->res stability."""
    out = OrderedDict()
    for task in tasks:
        per_res = OrderedDict()
        cache = {}
        for size in DATA_ROOTS:
            if not os.path.isdir(os.path.join(DATA_ROOTS[size], task)):
                continue
            recs = {"target": [], "demo": []}
            for s in seeds:
                for which in ("target", "demo"):
                    p = perceive_capture(task, size, s, which)
                    if p is None:
                        continue
                    cache[(size, s, which)] = p
                    recs[which].append(p)
            if not recs["target"]:
                continue

            def med(rs, key):
                v = [r[key] for r in rs if r[key] is not None]
                return float(np.median(v)) if v else None

            per_res[size] = {
                "n_scenes": len(recs["target"]),
                "mask_px_target_med": med(recs["target"], "mask_pixels"),
                "mask_px_demo_med": med(recs["demo"], "mask_pixels"),
                "px_per_cluster_target": (med(recs["target"], "mask_pixels") / K
                                          if recs["target"] else None),
                "min_cluster_px_target_med": med(recs["target"],
                                                 "min_cluster_px"),
                "centroid_err_mm_target_med":
                    1e3 * med(recs["target"], "centroid_err_m"),
                "centroid_err_mm_demo_med":
                    1e3 * med(recs["demo"], "centroid_err_m"),
                "sanity_fail": sum(1 for r in recs["target"]
                                   if not r["sanity_ok"])
                + sum(1 for r in recs["demo"] if not r["sanity_ok"]),
            }
        # candidate-centroid stability: matched displacement base -> size
        for size in list(per_res):
            if size == base:
                continue
            dists = []
            for s in seeds:
                a = cache.get((base, s, "target"))
                b = cache.get((size, s, "target"))
                if a is None or b is None:
                    continue
                dists.append(_match_points(a["cand3d"], b["cand3d"]))
            if dists:
                d = 1e3 * np.concatenate(dists)            # mm
                per_res[size]["cand_shift_mm_med"] = float(np.median(d))
                per_res[size]["cand_shift_mm_p90"] = float(
                    np.percentile(d, 90))
                per_res[size]["cand_shift_mm_max"] = float(d.max())
        out[task] = per_res
    return out


def alk_shift(rows_by_res, tasks, seeds=SEEDS, base=256):
    """Per-task displacement (mm) of the four ALK points between resolutions
    (ours_full rollouts; world frame, index-aligned by construction)."""
    out = OrderedDict()
    for task in tasks:
        per_res = OrderedDict()
        for size, rows in rows_by_res.items():
            if size == base:
                continue
            for which in ("demo_alk", "target_alk"):
                d = []
                for s in seeds:
                    ra = rows_by_res[base].get((task, "ours_full", s))
                    rb = rows.get((task, "ours_full", s))
                    if not ra or not rb:
                        continue
                    A = _oracle_field(ra, which)
                    B = _oracle_field(rb, which)
                    if A is None or B is None:
                        continue
                    d.append(np.linalg.norm(np.asarray(A) - np.asarray(B),
                                            axis=-1))
                if d:
                    v = 1e3 * np.concatenate(d)
                    per_res.setdefault(size, {})[which] = {
                        "med_mm": float(np.median(v)),
                        "p90_mm": float(np.percentile(v, 90)),
                        "n": int(v.size)}
        out[task] = per_res
    return out


def _oracle_field(row, key):
    """Read `oracle.<key>` out of the raw rollout JSON behind a tidy row."""
    path = row.get("source_file")
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        rec = json.load(f)
    orc = rec.get("oracle") or {}
    return orc.get(key)


# ---------------------------------------------------------------------------
# 3. success rates + paired McNemar across resolutions
# ---------------------------------------------------------------------------

def success_table(rows_by_res, tasks, methods=METHODS, seeds=SEEDS, base=256):
    out = OrderedDict()
    for task in list(tasks) + ["POOLED"]:
        tt = tasks if task == "POOLED" else [task]
        per_method = OrderedDict()
        for m in methods:
            entry = {}
            for size, rows in rows_by_res.items():
                v = [x for t in tt for x in vec(rows, t, m, seeds)
                     if x is not None]
                if not v:
                    continue
                k, n = int(sum(v)), len(v)
                lo, hi = st.wilson_ci(k, n)
                entry[size] = {"k": k, "n": n, "rate": k / float(n),
                               "ci": [lo, hi]}
            for size, rows in rows_by_res.items():
                if size == base or size not in entry:
                    continue
                a, b = [], []
                for t in tt:
                    x, y = paired(rows_by_res[base], rows, t, m, seeds)
                    a += x
                    b += y
                if not a:
                    continue
                r = st.mcnemar_from_vectors(b, a)   # b=higher res, a=256
                bs = st.paired_bootstrap_diff(b, a)
                entry[size].update({
                    "n_paired": len(a),
                    "delta": (sum(b) - sum(a)) / float(len(a)),
                    "b_hi_only": r.b, "c_lo_only": r.c, "p": r.pvalue,
                    "boot_ci": [bs.lo, bs.hi]})
            per_method[m] = entry
        out[task] = per_method
    return out


def holm_pooled(succ, methods=METHODS, size=512):
    """Holm over the 5 POOLED (across tasks) 256-vs-`size` tests, one per
    method -- the family the verdict paragraph rests on."""
    keys, pv = [], []
    for m in methods:
        e = succ["POOLED"][m].get(size, {})
        if "p" in e:
            keys.append(m)
            pv.append(e["p"])
    adj = st.holm_bonferroni(pv) if pv else []
    return OrderedDict(zip(keys, adj))


def holm_over_family(succ, tasks, methods=METHODS, size=512):
    """Holm over the family of {task x method} 256-vs-`size` McNemar tests."""
    keys, pv = [], []
    for task in tasks:
        for m in methods:
            e = succ[task][m].get(size, {})
            if "p" in e:
                keys.append((task, m))
                pv.append(e["p"])
    adj = st.holm_bonferroni(pv) if pv else []
    return OrderedDict(zip(keys, adj)), dict(zip(keys, pv))


# ---------------------------------------------------------------------------
# 4. ordering
# ---------------------------------------------------------------------------

def kendall_tau(x, y):
    """Kendall tau-b of two equal-length score vectors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    con = dis = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                tx += 1
                ty += 1
            elif dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx * dy > 0:
                con += 1
            else:
                dis += 1
    n0 = n * (n - 1) / 2.0
    den = np.sqrt((n0 - tx) * (n0 - ty))
    return float((con - dis) / den) if den > 0 else float("nan")


def ordering_table(succ, tasks, methods=METHODS, base=256):
    out = OrderedDict()
    for task in list(tasks) + ["POOLED"]:
        e = succ[task]
        sizes = [s for s in RES_DIRS if all(s in e[m] for m in methods)]
        row = {"rates": OrderedDict(), "ranks": OrderedDict(), "tau": {},
               "pair_agreement": {}}
        for s in sizes:
            rates = [e[m][s]["rate"] for m in methods]
            row["rates"][s] = OrderedDict(zip(methods, rates))
            order = np.argsort([-r for r in rates], kind="stable")
            ranks = np.empty(len(methods), dtype=int)
            for pos, idx in enumerate(order):
                ranks[idx] = pos + 1
            row["ranks"][s] = OrderedDict(zip(methods, ranks.tolist()))
        for s in sizes:
            if s == base:
                continue
            a = [e[m][base]["rate"] for m in methods]
            b = [e[m][s]["rate"] for m in methods]
            row["tau"][s] = kendall_tau(a, b)
            same = tot = 0
            flips = []
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    da, db = a[i] - a[j], b[i] - b[j]
                    tot += 1
                    # ties count as agreement unless the sign strictly reverses
                    if da * db < 0:
                        flips.append("%s vs %s (%+.2f -> %+.2f)"
                                     % (methods[i], methods[j], da, db))
                    else:
                        same += 1
            row["pair_agreement"][s] = {"same": same, "total": tot,
                                        "flips": flips}
        out[task] = row
    return out


def vs_ours_table(rows_by_res, tasks, ours="ours_full", methods=METHODS,
                  seeds=SEEDS):
    """The paper's comparative claims: ours vs each baseline, per resolution
    (paired McNemar within a resolution, pooled over the tasks present)."""
    out = OrderedDict()
    others = [m for m in methods if m != ours]
    for size, rows in rows_by_res.items():
        per = OrderedDict()
        pv = []
        for m in others:
            a, b = [], []
            for t in tasks:
                for s in seeds:
                    ra = rows.get((t, ours, s))
                    rb = rows.get((t, m, s))
                    if not ra or not rb:
                        continue
                    if ra.get("success") is None or rb.get("success") is None:
                        continue
                    a.append(ra["success"])
                    b.append(rb["success"])
            if not a:
                continue
            r = st.mcnemar_from_vectors(a, b)
            bs = st.paired_bootstrap_diff(a, b)
            per[m] = {"n": len(a), "ours": sum(a), "other": sum(b),
                      "diff": (sum(a) - sum(b)) / float(len(a)),
                      "b": r.b, "c": r.c, "p": r.pvalue,
                      "boot_ci": [bs.lo, bs.hi]}
            pv.append(r.pvalue)
        adj = st.holm_bonferroni(pv) if pv else []
        for m, p in zip([m for m in others if m in per], adj):
            per[m]["p_holm"] = p
        out[size] = per
    return out


def pairwise_order(rows_by_res, tasks, methods=METHODS, seeds=SEEDS,
                   base=256, alpha=0.05):
    """Every method PAIR at every resolution: paired difference + McNemar,
    pooled over the campaign's tasks.

    The validity question is not whether a rank order is byte-stable (ties
    and near-ties reshuffle freely at N=%d per task) but whether any pair
    whose difference the paper would CLAIM (significant at one resolution)
    reverses at another.  Returns (pairs, flips, sig_flips).
    """
    out = OrderedDict()
    for i, mi in enumerate(methods):
        for mj in methods[i + 1:]:
            per = OrderedDict()
            for size, rows in rows_by_res.items():
                a, b = [], []
                for t in tasks:
                    for s in seeds:
                        ra = rows.get((t, mi, s))
                        rb = rows.get((t, mj, s))
                        if not ra or not rb:
                            continue
                        if ra.get("success") is None or \
                                rb.get("success") is None:
                            continue
                        a.append(ra["success"])
                        b.append(rb["success"])
                if not a:
                    continue
                r = st.mcnemar_from_vectors(a, b)
                per[size] = {"n": len(a), "ki": sum(a), "kj": sum(b),
                             "diff": (sum(a) - sum(b)) / float(len(a)),
                             "p": r.pvalue, "sig": r.pvalue < alpha}
            out[(mi, mj)] = per
    flips, sig_flips = [], []
    for key, per in out.items():
        if base not in per:
            continue
        d0 = per[base]["diff"]
        for size, e in per.items():
            if size == base:
                continue
            # a REVERSAL means strictly opposite signs; a pair that is tied at
            # one resolution and separated at the other is not an order flip
            if np.sign(d0) * np.sign(e["diff"]) < 0:
                tag = "%s vs %s: %+.3f @256 -> %+.3f @%d (p %s -> %s)" % (
                    key[0], key[1], d0, e["diff"], size,
                    fmt_p(per[base]["p"]), fmt_p(e["p"]))
                flips.append(tag)
                if per[base]["sig"] or e["sig"]:
                    sig_flips.append(tag)
    return out, flips, sig_flips


def gap_table(rows_by_res, tasks, a_method="ours_full", b_method="kp_fps4",
              seeds=SEEDS):
    """ALK vs oracle-correspondence gap, per task and pooled, per resolution."""
    out = OrderedDict()
    for task in list(tasks) + ["POOLED"]:
        tt = tasks if task == "POOLED" else [task]
        per = OrderedDict()
        for size, rows in rows_by_res.items():
            a, b = [], []
            for t in tt:
                for s in seeds:
                    ra = rows.get((t, a_method, s))
                    rb = rows.get((t, b_method, s))
                    if not ra or not rb:
                        continue
                    if ra.get("success") is None or rb.get("success") is None:
                        continue
                    a.append(ra["success"])
                    b.append(rb["success"])
            if not a:
                continue
            r = st.mcnemar_from_vectors(a, b)
            per[size] = {"n": len(a), "a": sum(a), "b": sum(b),
                         "gap": (sum(a) - sum(b)) / float(len(a)),
                         "b_only": r.b, "c_only": r.c, "p": r.pvalue}
        out[task] = per
    return out


# ---------------------------------------------------------------------------
# 5. mapping accuracy
# ---------------------------------------------------------------------------

def error_table(rows_by_res, tasks, methods=METHODS, seeds=SEEDS):
    out = OrderedDict()
    for task in tasks:
        per_method = OrderedDict()
        for m in methods:
            entry = OrderedDict()
            for size, rows in rows_by_res.items():
                rot = [x for s in seeds
                       for x in [(rows.get((task, m, s)) or {}).get("rot_err")]
                       if x is not None]
                tr = [x for s in seeds
                      for x in [(rows.get((task, m, s)) or {}).get("trans_err")]
                      if x is not None]
                if not rot:
                    continue
                ini = [x for s in seeds
                       for x in [(rows.get((task, m, s)) or {}).get(
                           "rot_err_init")]
                       if x is not None]
                stages = {}
                for s in seeds:
                    row = rows.get((task, m, s))
                    if row and not row.get("success"):
                        stages[row.get("failure_stage")] = stages.get(
                            row.get("failure_stage"), 0) + 1
                entry[size] = {
                    "n": len(rot),
                    "rot_med": float(np.median(rot)),
                    "rot_mean": float(np.mean(rot)),
                    "rot_init_med": float(np.median(ini)) if ini else None,
                    "trans_mm_med": 1e3 * float(np.median(tr)) if tr else None,
                    "trans_mm_mean": 1e3 * float(np.mean(tr)) if tr else None,
                    "failure_stages": stages,
                }
            per_method[m] = entry
        out[task] = per_method
    return out


# ---------------------------------------------------------------------------
# 7. adaptive arm
# ---------------------------------------------------------------------------

def adaptive_table(task="pour", seeds=SEEDS, base=256):
    """conditioning-adaptive ours_full at each resolution + vs its own
    non-adaptive rollouts at the same resolution (paired)."""
    ad = OrderedDict()
    for size, root in ADAPTIVE_DIRS.items():
        rows = load_res(root, [task], methods=["ours_full"], seeds=seeds)
        if rows:
            ad[size] = rows
    base_rows = OrderedDict()
    for size, root in RES_DIRS.items():
        rows = load_res(root, [task], methods=["ours_full", "kp_fps4"],
                        seeds=seeds)
        if rows:
            base_rows[size] = rows
    out = OrderedDict()
    for size, rows in ad.items():
        v = [x for x in vec(rows, task, "ours_full", seeds) if x is not None]
        k, n = int(sum(v)), len(v)
        lo, hi = st.wilson_ci(k, n)
        e = {"k": k, "n": n, "rate": k / float(n), "ci": [lo, hi]}
        rot = [x for s in seeds
               for x in [(rows.get((task, "ours_full", s)) or {}).get("rot_err")]
               if x is not None]
        e["rot_med"] = float(np.median(rot)) if rot else None
        # vs non-adaptive at the SAME resolution
        if size in base_rows:
            a, b = paired(rows, base_rows[size], task, "ours_full", seeds)
            if a:
                r = st.mcnemar_from_vectors(a, b)
                e["vs_nonadaptive"] = {"n": len(a), "adaptive": sum(a),
                                       "baseline": sum(b), "b": r.b, "c": r.c,
                                       "p": r.pvalue}
            a2, b2 = [], []
            for s in seeds:
                ra = rows.get((task, "ours_full", s))
                rb = base_rows[size].get((task, "kp_fps4", s))
                if ra and rb and ra.get("success") is not None \
                        and rb.get("success") is not None:
                    a2.append(ra["success"])
                    b2.append(rb["success"])
            if a2:
                r2 = st.mcnemar_from_vectors(a2, b2)
                e["vs_kp_fps4"] = {"n": len(a2), "adaptive": sum(a2),
                                   "kp_fps4": sum(b2), "b": r2.b, "c": r2.c,
                                   "p": r2.pvalue}
        # adaptive trigger rate
        trig = 0
        tot = 0
        for s in seeds:
            row = rows.get((task, "ours_full", s))
            if not row:
                continue
            path = row.get("source_file")
            with open(path) as f:
                rec = json.load(f)
            arg = rec.get("adaptive_registration") or {}
            if "adaptive_triggered" in arg:
                tot += 1
                trig += bool(arg["adaptive_triggered"])
        e["trigger"] = "%d/%d" % (trig, tot)
        out[size] = e
    # paired 256 vs higher-res adaptive
    for size in list(out):
        if size == base or base not in ad:
            continue
        a, b = paired(ad[base], ad[size], task, "ours_full", seeds)
        if a:
            r = st.mcnemar_from_vectors(b, a)
            out[size]["vs_256_adaptive"] = {
                "n": len(a), "res256": sum(a), "res%d" % size: sum(b),
                "b": r.b, "c": r.c, "p": r.pvalue}
    return out


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def fmt_p(p):
    if p is None:
        return "-"
    if p < 1e-4:
        return "<1e-4"
    return "%.3g" % p


def md_report(res):
    sizes = res["sizes"]
    hi = res["hi_sizes"]
    tasks = res["tasks"]
    L = []
    A = L.append
    A("# Campaign H — render-resolution sensitivity (auto-generated)")
    A("")
    A("Sizes compared: %s (256 = existing Campaign-A rollouts, reused). "
      "Tier medium, paired seeds %d..%d, methods %s."
      % (", ".join(str(s) for s in sizes), SEEDS[0], SEEDS[-1],
         ", ".join(METHODS)))
    A("")

    # 1 scene identity
    A("## 1. Scene identity across render sizes (ground-truth poses)")
    A("")
    A("| data root | pairs | object poses compared | max abs delta | "
      "mismatches | bit-identical |")
    A("|---|---|---|---|---|---|")
    for size, rec in res["pose_match"].items():
        A("| data_medium_res%d vs data_medium | %d | %d | %.1e | %s | %s |"
          % (size, rec["n_pairs"], rec["n_objects"], rec["max_abs_delta"],
             len(rec["mismatches"]),
             "YES" if rec["bit_identical"] else "NO — " +
             "; ".join(rec["mismatches"][:3])))
    A("")

    # 2 perception
    A("## 2. Perception diagnostics (target-side unless noted)")
    A("")
    A("| task | res | mask px (med) | px/cluster | min cluster px | "
      "back-proj centroid err (mm) | demo mask px | cand. shift vs 256 "
      "med/p90 (mm) |")
    A("|---|---|---|---|---|---|---|---|")
    for task, per_res in res["perception"].items():
        for size, e in per_res.items():
            A("| %s | %d | %.0f | %.1f | %s | %.1f | %.0f | %s |"
              % (task, size, e["mask_px_target_med"],
                 e["px_per_cluster_target"],
                 ("%s" % e["min_cluster_px_target_med"]),
                 e["centroid_err_mm_target_med"], e["mask_px_demo_med"],
                 ("%.1f / %.1f" % (e["cand_shift_mm_med"],
                                   e["cand_shift_mm_p90"]))
                 if "cand_shift_mm_med" in e else "—"))
    A("")
    if res["alk_shift"]:
        A("ALK point displacement between resolutions (ours_full, mm):")
        A("")
        A("| task | res | demo ALK med/p90 | target ALK med/p90 |")
        A("|---|---|---|---|")
        for task, per_res in res["alk_shift"].items():
            for size, e in per_res.items():
                d = e.get("demo_alk")
                t = e.get("target_alk")
                A("| %s | %d | %s | %s |"
                  % (task, size,
                     "%.1f / %.1f" % (d["med_mm"], d["p90_mm"]) if d else "—",
                     "%.1f / %.1f" % (t["med_mm"], t["p90_mm"]) if t else "—"))
        A("")

    # 3 success
    A("## 3. Success rates and paired McNemar (256 vs higher resolution)")
    A("")
    head = "| task | method | " + " | ".join("k/n @%d" % s for s in sizes) \
        + " | " + " | ".join("delta %d-256 (McNemar b/c, p)" % s for s in hi) \
        + " |"
    A(head)
    A("|" + "---|" * (2 + len(sizes) + len(hi)))
    for task in list(tasks) + ["POOLED"]:
        for m in METHODS:
            e = res["success"][task][m]
            cells = []
            for s in sizes:
                cells.append("%d/%d = %.2f" % (e[s]["k"], e[s]["n"],
                                               e[s]["rate"])
                             if s in e else "—")
            for s in hi:
                if s in e and "p" in e[s]:
                    holm = res["holm"].get((task, m))
                    cells.append("%+.2f (%d/%d, p=%s%s)"
                                 % (e[s]["delta"], e[s]["b_hi_only"],
                                    e[s]["c_lo_only"], fmt_p(e[s]["p"]),
                                    "" if holm is None
                                    else ", holm=%s" % fmt_p(holm)))
                else:
                    cells.append("—")
            A("| %s | %s | %s |" % (task, m, " | ".join(cells)))
    A("")
    A("(b/c = seeds that succeed only at the higher resolution / only at 256; "
      "Holm over the %d {task x method} tests at 512.)" % res["n_holm"])
    A("")
    A("Pooled 256-vs-512 per method, Holm over that 5-test family: %s"
      % "; ".join("%s p_holm=%s" % (m, fmt_p(p))
                  for m, p in res["holm_pooled"].items()))
    A("")

    # 4 ordering
    A("## 4. Method ordering")
    A("")
    A("| task | res | " + " | ".join(METHODS) + " | rank order |")
    A("|" + "---|" * (3 + len(METHODS)))
    for task in list(tasks) + ["POOLED"]:
        row = res["ordering"][task]
        for s, rates in row["rates"].items():
            order = sorted(METHODS, key=lambda m: (-rates[m], m))
            A("| %s | %d | %s | %s |"
              % (task, s, " | ".join("%.2f" % rates[m] for m in METHODS),
                 " > ".join(order)))
    A("")
    A("| task | res | Kendall tau vs 256 | pairwise order agreement | flips |")
    A("|---|---|---|---|---|")
    for task in list(tasks) + ["POOLED"]:
        row = res["ordering"][task]
        for s, tau in row["tau"].items():
            pa = row["pair_agreement"][s]
            A("| %s | %d | %.3f | %d/%d | %s |"
              % (task, s, tau, pa["same"], pa["total"],
                 "; ".join(pa["flips"]) if pa["flips"] else "none"))
    A("")
    A("### 4b. The paper's comparative claims at each resolution "
      "(ours_full vs baseline, pooled over the tasks of this campaign)")
    A("")
    A("| res | baseline | ours k | baseline k | n | diff | McNemar b/c | p | "
      "p_holm |")
    A("|---|---|---|---|---|---|---|---|---|")
    for size, per in res["vs_ours"].items():
        for m, e in per.items():
            A("| %d | %s | %d | %d | %d | %+.3f | %d/%d | %s | %s |"
              % (size, m, e["ours"], e["other"], e["n"], e["diff"], e["b"],
                 e["c"], fmt_p(e["p"]), fmt_p(e.get("p_holm"))))
    A("")

    A("### 4c. Every method pair at every resolution (pooled over the "
      "campaign's tasks): does any CLAIMABLE ordering reverse?")
    A("")
    A("| pair | " + " | ".join("diff @%d (p)" % s for s in sizes)
      + " | sign stable |")
    A("|" + "---|" * (2 + len(sizes)))
    for (mi, mj), per in res["pairwise"].items():
        cells = []
        for s in sizes:
            cells.append("%+.3f (%s%s)" % (per[s]["diff"], fmt_p(per[s]["p"]),
                                           "*" if per[s]["sig"] else "")
                         if s in per else "—")
        signs = [np.sign(per[s]["diff"]) for s in per]
        rev = any(a * b < 0 for a in signs for b in signs)
        A("| %s vs %s | %s | %s |"
          % (mi, mj, " | ".join(cells), "NO (reverses)" if rev else "yes"))
    A("")
    A("Per task (all %d method pairs, sign of the paired difference):"
      % (len(METHODS) * (len(METHODS) - 1) // 2))
    A("")
    A("| task | pairs reversing sign | of those, significant (p<0.05) at some "
      "resolution |")
    A("|---|---|---|")
    for task, e in res["pairwise_per_task"].items():
        A("| %s | %d | %s |"
          % (task, len(e["flips"]),
             "%d — %s" % (len(e["sig_flips"]), "; ".join(e["sig_flips"]))
             if e["sig_flips"] else "0 (none)"))
    A("")
    A("(* = McNemar p < 0.05 at that resolution.  Sign flips: %s. "
      "Sign flips involving a pair that is significant at some resolution: "
      "%s.)"
      % ("; ".join(res["pairwise_flips"]) if res["pairwise_flips"] else "none",
         "; ".join(res["pairwise_sig_flips"]) if res["pairwise_sig_flips"]
         else "NONE"))
    A("")

    # 5 errors
    A("## 5. Mapping accuracy vs ground truth (T_map)")
    A("")
    A("| task | method | " + " | ".join(
        "rot med (deg) @%d" % s for s in sizes) + " | " + " | ".join(
        "trans med (mm) @%d" % s for s in sizes) + " |")
    A("|" + "---|" * (2 + 2 * len(sizes)))
    for task, per_method in res["errors"].items():
        for m, e in per_method.items():
            cells = ["%.1f" % e[s]["rot_med"] if s in e else "—"
                     for s in sizes]
            cells += ["%.1f" % e[s]["trans_mm_med"] if s in e else "—"
                      for s in sizes]
            A("| %s | %s | %s |" % (task, m, " | ".join(cells)))
    A("")
    A("Closed-form (Procrustes-only) init rotation error and failure stages, "
      "ours_full:")
    A("")
    A("| task | res | rot init med (deg) | rot final med (deg) | "
      "failure stages |")
    A("|---|---|---|---|---|")
    for task, per_method in res["errors"].items():
        e = per_method.get("ours_full", {})
        for s in sizes:
            if s not in e:
                continue
            st_ = e[s]["failure_stages"]
            A("| %s | %d | %s | %.1f | %s |"
              % (task, s,
                 "%.1f" % e[s]["rot_init_med"]
                 if e[s]["rot_init_med"] is not None else "—",
                 e[s]["rot_med"],
                 ", ".join("%s:%d" % (k, v) for k, v in sorted(
                     st_.items(), key=lambda kv: str(kv[0]))) or "none"))
    A("")

    # 6 gap
    A("## 6. ALK vs oracle-correspondence gap (ours_full − kp_fps4)")
    A("")
    A("| task | res | ours_full | kp_fps4 | n | gap | McNemar b/c | p |")
    A("|---|---|---|---|---|---|---|---|")
    for task, per in res["gap"].items():
        for size, e in per.items():
            A("| %s | %d | %d | %d | %d | %+.3f | %d/%d | %s |"
              % (task, size, e["a"], e["b"], e["n"], e["gap"], e["b_only"],
                 e["c_only"], fmt_p(e["p"])))
    A("")

    # 7 adaptive
    if res["adaptive"]:
        A("## 7. Conditioning-adaptive arm on pour")
        A("")
        A("| res | adaptive k/n | rot med (deg) | trigger | vs non-adaptive "
          "same res (b/c, p) | vs kp_fps4 same res (b/c, p) | vs 256 adaptive |")
        A("|---|---|---|---|---|---|---|")
        for size, e in res["adaptive"].items():
            vn = e.get("vs_nonadaptive")
            vk = e.get("vs_kp_fps4")
            v2 = e.get("vs_256_adaptive")
            A("| %d | %d/%d = %.2f | %s | %s | %s | %s | %s |"
              % (size, e["k"], e["n"], e["rate"],
                 "%.1f" % e["rot_med"] if e["rot_med"] is not None else "—",
                 e["trigger"],
                 "%d vs %d (%d/%d, p=%s)" % (vn["adaptive"], vn["baseline"],
                                             vn["b"], vn["c"], fmt_p(vn["p"]))
                 if vn else "—",
                 "%d vs %d (%d/%d, p=%s)" % (vk["adaptive"], vk["kp_fps4"],
                                             vk["b"], vk["c"], fmt_p(vk["p"]))
                 if vk else "—",
                 "(%d/%d, p=%s)" % (v2["b"], v2["c"], fmt_p(v2["p"]))
                 if v2 else "—"))
        A("")
    return "\n".join(L)


def tex_report(res):
    """LaTeX version of the two tables the paper needs (success + ordering)."""
    sizes = res["sizes"]
    hi = res["hi_sizes"]
    tasks = res["tasks"]
    esc = lambda s: str(s).replace("_", r"\_")
    L = []
    A = L.append
    A("% Campaign H -- render-resolution sensitivity (auto-generated by")
    A("% stats/campaign_h_tables.py).  256 = reused Campaign-A rollouts.")
    ncol = 2 + len(sizes) + len(hi)
    A(r"\begin{table}[t]")
    A(r"\centering\small")
    A(r"\caption{Render-resolution sensitivity: success at %s px on the same "
      r"paired seeds (medium tier, $N=%d$ per cell). $b/c$ = seeds succeeding "
      r"only at the higher resolution / only at 256; exact McNemar, Holm over "
      r"the %d task$\times$method tests at 512.}"
      % ("/".join(str(s) for s in sizes), len(SEEDS), res["n_holm"]))
    A(r"\label{tab:campaign-h-success}")
    A(r"\begin{tabular}{ll" + "c" * (ncol - 2) + "}")
    A(r"\toprule")
    A("task & method & " + " & ".join(r"$k/n$@%d" % s for s in sizes) + " & "
      + " & ".join(r"$\Delta_{%d-256}$ ($b/c$, $p$)" % s for s in hi)
      + r" \\")
    A(r"\midrule")
    for task in list(tasks) + ["POOLED"]:
        for m in METHODS:
            e = res["success"][task][m]
            cells = ["%d/%d" % (e[s]["k"], e[s]["n"]) if s in e else "--"
                     for s in sizes]
            for s in hi:
                if s in e and "p" in e[s]:
                    holm = res["holm"].get((task, m))
                    cells.append("$%+.2f$ (%d/%d, %s%s)"
                                 % (e[s]["delta"], e[s]["b_hi_only"],
                                    e[s]["c_lo_only"],
                                    fmt_p(e[s]["p"]).replace("<", r"$<$"),
                                    "" if holm is None else
                                    ", h=%s" % fmt_p(holm).replace(
                                        "<", r"$<$")))
                else:
                    cells.append("--")
            A("%s & %s & %s" % (esc(task), esc(m), " & ".join(cells)) + r" \\")
        if task != "POOLED":
            A(r"\midrule")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table}")
    A("")
    A(r"\begin{table}[t]")
    A(r"\centering\small")
    A(r"\caption{Method ordering is resolution-stable: success rates per "
      r"render size, Kendall $\tau$ and pairwise-order agreement against 256.}")
    A(r"\label{tab:campaign-h-ordering}")
    A(r"\begin{tabular}{ll" + "c" * (len(METHODS) + 2) + "}")
    A(r"\toprule")
    A("task & px & " + " & ".join(esc(m) for m in METHODS)
      + r" & $\tau$ vs 256 & order agree \\")
    A(r"\midrule")
    for task in list(tasks) + ["POOLED"]:
        row = res["ordering"][task]
        for s, rates in row["rates"].items():
            tau = row["tau"].get(s)
            pa = row["pair_agreement"].get(s)
            A("%s & %d & %s & %s & %s"
              % (esc(task), s,
                 " & ".join("%.2f" % rates[m] for m in METHODS),
                 "--" if tau is None else "%.2f" % tau,
                 "--" if pa is None else "%d/%d" % (pa["same"], pa["total"]))
              + r" \\")
        if task != "POOLED":
            A(r"\midrule")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table}")
    A("")
    A(r"\begin{table}[t]")
    A(r"\centering\small")
    A(r"\caption{What the higher render size actually buys: target-mask "
      r"pixels, pixels per $k$-means candidate cluster ($k=8$), "
      r"back-projected centroid error against ground truth, and the matched "
      r"displacement of the eight candidate centroids relative to 256 px "
      r"(medians over %d target scenes per cell).}" % len(SEEDS))
    A(r"\label{tab:campaign-h-perception}")
    A(r"\begin{tabular}{llcccc}")
    A(r"\toprule")
    A(r"task & px & mask px & px/cluster & centroid err (mm) & "
      r"cand.\ shift vs 256 (mm) \\")
    A(r"\midrule")
    for task, per_res in res["perception"].items():
        for size, e in per_res.items():
            A("%s & %d & %.0f & %.0f & %.1f & %s"
              % (esc(task), size, e["mask_px_target_med"],
                 e["px_per_cluster_target"], e["centroid_err_mm_target_med"],
                 ("%.1f" % e["cand_shift_mm_med"])
                 if "cand_shift_mm_med" in e else "--") + r" \\")
        A(r"\midrule")
    L[-1] = r"\bottomrule"
    A(r"\end{tabular}")
    A(r"\end{table}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def build(tasks=None, seeds=SEEDS):
    hi_roots = OrderedDict((s, r) for s, r in RES_DIRS.items()
                           if s != 256 and os.path.isdir(r))
    if tasks is None:
        tasks = []
        for s, r in hi_roots.items():
            for t in sorted(os.listdir(r)):
                if os.path.isdir(os.path.join(r, t)) and t not in tasks:
                    tasks.append(t)
    rows_by_res = OrderedDict()
    for size, root in RES_DIRS.items():
        rows = load_res(root, tasks, seeds=seeds)
        if rows:
            rows_by_res[size] = rows
    sizes = list(rows_by_res)
    hi_sizes = [s for s in sizes if s != 256]
    succ = success_table(rows_by_res, tasks, seeds=seeds)
    holm, _ = holm_over_family(succ, tasks, size=512 if 512 in sizes
                               else hi_sizes[-1])
    res = {
        "sizes": sizes, "hi_sizes": hi_sizes, "tasks": tasks,
        "seeds": [seeds[0], seeds[-1]], "methods": METHODS,
        "pose_match": pose_match(tasks, seeds=seeds),
        "perception": perception_table(tasks, seeds=seeds),
        "alk_shift": alk_shift(rows_by_res, tasks, seeds=seeds),
        "success": succ,
        "holm": holm,
        "n_holm": len(holm),
        "holm_pooled": holm_pooled(succ, size=512 if 512 in sizes
                                   else hi_sizes[-1]),
        "ordering": ordering_table(succ, tasks),
        "vs_ours": vs_ours_table(rows_by_res, tasks, seeds=seeds),
        "pairwise": None,   # filled below (needs its own unpacking)
        "gap": gap_table(rows_by_res, tasks, seeds=seeds),
        "errors": error_table(rows_by_res, tasks, seeds=seeds),
        "adaptive": adaptive_table(seeds=seeds),
        "rows_by_res": rows_by_res,
    }
    pw, flips, sig_flips = pairwise_order(rows_by_res, tasks, seeds=seeds)
    res["pairwise"] = pw
    res["pairwise_flips"] = flips
    res["pairwise_sig_flips"] = sig_flips
    per_task = OrderedDict()
    for t in tasks:
        _, f, sf = pairwise_order(rows_by_res, [t], seeds=seeds)
        per_task[t] = {"flips": f, "sig_flips": sf}
    res["pairwise_per_task"] = per_task
    return res


def _json_safe(x):
    if isinstance(x, dict):
        return {("%s|%s" % k if isinstance(k, tuple) else str(k)):
                _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json_safe(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--prefix", default="campaign_h")
    args = ap.parse_args(argv)
    res = build(tasks=args.tasks)
    os.makedirs(args.out_dir, exist_ok=True)
    rows = [r for rows in res["rows_by_res"].values() for r in rows.values()]
    csv_path = os.path.join(args.out_dir, "%s_rollouts.csv" % args.prefix)
    agg.write_csv(sorted(rows, key=lambda r: (r["source_file"] or "")),
                  csv_path)
    md = md_report(res)
    tex = tex_report(res)
    for name, text in (("%s.md" % args.prefix, md),
                       ("%s.tex" % args.prefix, tex)):
        p = os.path.join(args.out_dir, name)
        with open(p, "w") as f:
            f.write(text + "\n")
        print("wrote %s" % p)
    summary = os.path.join(SIMBENCH, "results", "campaign_h", "summary.json")
    os.makedirs(os.path.dirname(summary), exist_ok=True)
    dump = {k: v for k, v in res.items() if k != "rows_by_res"}
    with open(summary, "w") as f:
        json.dump(_json_safe(dump), f, indent=2)
    print("wrote %s\nwrote %s" % (csv_path, summary))
    print(md)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
