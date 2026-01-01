"""Campaign F tables: which ALK points enter the closed-form Procrustes sum.

The earlier three-point construction sums orthogonal Procrustes over THREE point
pairs and its pipeline figure's "Closed-Form Alignment" box labels C1, C3, C4
-- one axial endpoint plus both lateral centroids, dropping c2.  This
benchmark's implementation sums over all four ALK points.  Campaign F turns
that discrepancy into a paired ablation over the identical protocol as
Campaign A (5 tasks x 60 seeds {1000..1059}, medium tier, oracle discrete,
registration ON, correction ON):

    alk4                current default, {c1,c2,c3,c4}  (== campaign_a
                        ours_full: same code path, byte-identical numerics)
    alk3_c134           the paper's original, {c1,c3,c4}
    alk3_c123           {c1,c2,c3}                        (completeness)
    alk2_c12            {c1,c2}, rank-1, roll frozen      (completeness)
    alk4_adaptive       alk4 + conditioning-adaptive registration (paper Section 3.7)
    alk3_c134_adaptive  alk3_c134 + adaptive: does the widened axis search
                        rescue the 3-point construction too?

Reads results/campaign_f/<task>/<seed>/rollout_<method>.json and writes

  results/tables/campaign_f.md    -- conditioning table, per-task + pooled
                                    success with McNemar, error medians,
                                    failure stages, adaptive sub-table
  results/tables/campaign_f.tex   -- LaTeX versions of the two paper tables
  results/tables/campaign_f_rollouts.csv

Statistics (frozen protocol, baselines/docs/CODE_MAP.md + stats/tests.py):
  * Wilson CI next to every success rate.
  * McNemar exact, paired by seed, for alk4 vs each other subset per task and
    pooled over (task, seed); Holm over the per-task family of the primary
    alk4-vs-alk3_c134 test.
  * Paired bootstrap CI for the success-rate difference (effect size).
  * Wilcoxon-free error comparison: medians + paired median difference of the
    rotation/translation errors (the paired sign test's b/c counts come from
    the McNemar machinery on the "which method is closer to GT" indicator).

Conditioning (analytic, no rollouts needed): singular values of the CENTERED
demo configuration and both theory numbers -- sigma2+sigma3 (paper Section 3.7: the
closed-form rotation error scales like 1/(sigma2+sigma3)) and the scale-free
ratio (sigma2+sigma3)/sigma1 that the adaptive trigger compares to
tau_sigma = 0.40.  Read off the demo ALK stored in each task's rollouts (the
demo scene is fixed per task, so the demo ALK is one constant per task).

Usage:
  python -m stats.campaign_f_tables [--root results/campaign_f]
"""
import argparse
import csv
import json
import os

import numpy as np

from alkbench.alk import ALK_SUBSETS
from alkbench.registration import SIGMA_RATIO_THRESHOLD
from stats.tests import (wilson_ci, mcnemar_from_vectors, paired_bootstrap_diff,
                         holm_bonferroni)

SIMBENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(SIMBENCH, "results", "campaign_f")
DEFAULT_OUT_DIR = os.path.join(SIMBENCH, "results", "tables")
CAMPAIGN_A = os.path.join(SIMBENCH, "results", "campaign_a")

TASKS = ["nut_loosen", "cap_twist", "rim_grasp", "box_open", "pour"]
BASE_METHODS = ["alk4", "alk3_c134", "alk3_c123", "alk2_c12"]
ADAPTIVE_METHODS = ["alk4_adaptive", "alk3_c134_adaptive"]
METHODS = BASE_METHODS + ADAPTIVE_METHODS
REF = "alk4"                      # the current default: everything vs this
PRIMARY = "alk3_c134"             # the paper's original construction
SUBSET_LABEL = {
    "alk4": "{c1,c2,c3,c4}",
    "alk3_c134": "{c1,c3,c4}",
    "alk3_c123": "{c1,c2,c3}",
    "alk2_c12": "{c1,c2}",
    "alk4_adaptive": "{c1,c2,c3,c4} +adaptive",
    "alk3_c134_adaptive": "{c1,c3,c4} +adaptive",
}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _sigma_ratio(rec):
    """(sigma2+sigma3)/sigma1 of the configuration the rollout actually
    SOLVED (conditioning_detail is written for the solved subset)."""
    sv = (rec.get("conditioning_detail") or {}).get("singular_values")
    if not sv or not sv[0]:
        return None
    return float((sv[1] + sv[2]) / sv[0])


def load_rows(root):
    """One flat row per rollout_<method>.json under <root>/<task>/<seed>/."""
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if not (name.startswith("rollout_") and name.endswith(".json")):
                continue
            path = os.path.join(dirpath, name)
            with open(path) as f:
                rec = json.load(f)
            rows.append({
                "task": rec.get("task"),
                "seed": int(rec["seed"]),
                "method": rec.get("method") or rec.get("variant"),
                "tier": rec.get("tier"),
                "success": int(bool(rec.get("success"))),
                "grasped": rec.get("grasped"),
                "rot_err": rec.get("rot_err_deg"),
                "trans_err": rec.get("trans_err_m"),
                "rot_err_init": rec.get("rot_err_init_deg"),
                "trans_err_init": rec.get("trans_err_init_m"),
                "chamfer_after": rec.get("chamfer_after_m"),
                "failure_stage": rec.get("failure_stage"),
                "conditioning": rec.get("conditioning"),
                "sigma_ratio": _sigma_ratio(rec),
                "alk_subset": rec.get("alk_subset"),
                "adaptive": rec.get("adaptive"),
                "adaptive_triggered": (
                    ((rec.get("adaptive_registration") or {})
                     .get("sigma_ratios") or {}) and
                    (rec.get("adaptive_registration") or {})
                    .get("adaptive_triggered")),
                "demo_alk": (rec.get("oracle") or {}).get("demo_alk"),
                "source_file": path,
            })
    return rows


def index(rows):
    """dict[(task, seed, method)] -> row."""
    return {(r["task"], r["seed"], r["method"]): r for r in rows}


def methods_present(rows):
    seen = {r["method"] for r in rows}
    return [m for m in METHODS if m in seen]


def tasks_present(rows):
    seen = {r["task"] for r in rows}
    return [t for t in TASKS if t in seen]


def paired_seeds(ix, task, m_a, m_b):
    """Seeds with a rollout for BOTH methods (the paired unit set)."""
    sa = {s for (t, s, m) in ix if t == task and m == m_a}
    sb = {s for (t, s, m) in ix if t == task and m == m_b}
    return sorted(sa & sb)


# ---------------------------------------------------------------------------
# conditioning (analytic)
# ---------------------------------------------------------------------------

def config_conditioning(points):
    P = np.asarray(points, dtype=np.float64)
    X = P - P.mean(axis=0)
    s = np.linalg.svd(X, compute_uv=False)
    s = np.concatenate([s, np.zeros(3)])[:3]
    sigma23 = float(s[1] + s[2])
    return {"n": int(P.shape[0]),
            "sv": [float(v) for v in s],
            "sigma23": sigma23,
            "ratio": float(sigma23 / s[0]) if s[0] > 0 else 0.0}


def demo_alks(rows, tasks):
    """task -> (4,3) demo ALK from the non-adaptive rollouts (constant per
    task: the demo scene is fixed and the depth prior is off).  Raises if a
    task's rollouts disagree, which would invalidate the analytic table."""
    out = {}
    for t in tasks:
        alks = [np.asarray(r["demo_alk"]) for r in rows
                if r["task"] == t and r["demo_alk"] and not r["adaptive"]]
        if not alks:
            continue
        A = np.stack(alks)
        if not np.allclose(A, A[0], atol=1e-12):
            raise AssertionError("demo ALK varies across %s rollouts" % t)
        out[t] = A[0]
    return out


def noise_sensitivity(alk, sigma_m, n_trials=2000, seed=0):
    """Monte-Carlo test of the Prop-3 prediction on ONE demo configuration.

    Draws a random ground-truth rotation, builds the noiseless target
    configuration, perturbs BOTH configurations with iid isotropic Gaussian
    keypoint noise of std `sigma_m`, re-solves the closed form over each named
    subset, and returns the median / p90 rotation error (deg) per subset.
    Paper Section 3.7 predicts the error scales like eps / (sigma2 + sigma3) of the
    SOLVED configuration, so this isolates the conditioning consequence from
    everything downstream (registration, execution).  Deterministic given
    `seed`; the SAME noise draw and the SAME ground truth are shared by all
    subsets (paired comparison).
    """
    from scipy.spatial.transform import Rotation

    from alkbench import align_keypoints, rotation_angle_deg

    A = np.asarray(alk, dtype=np.float64)
    rng = np.random.RandomState(seed)
    R_true = Rotation.random(n_trials, random_state=rng).as_matrix()
    noise_d = rng.normal(scale=sigma_m, size=(n_trials, 4, 3))
    noise_t = rng.normal(scale=sigma_m, size=(n_trials, 4, 3))
    out = {}
    for name, idx in ALK_SUBSETS.items():
        rows_ = list(idx)
        errs = []
        for i in range(n_trials):
            P = A + noise_d[i]
            Q = A @ R_true[i].T + noise_t[i]
            T = align_keypoints(P[rows_], Q[rows_])
            errs.append(rotation_angle_deg(T[:3, :3] @ R_true[i].T))
        e = np.asarray(errs)
        out[name] = {"median": float(np.median(e)),
                     "p90": float(np.percentile(e, 90))}
    return out


def conditioning_table(rows, tasks):
    """task -> subset name -> conditioning dict, for all named subsets."""
    out = {}
    for t, alk in demo_alks(rows, tasks).items():
        out[t] = {name: config_conditioning(alk[list(idx)])
                  for name, idx in ALK_SUBSETS.items()}
    return out


# ---------------------------------------------------------------------------
# success / paired tests
# ---------------------------------------------------------------------------

def success_counts(rows, method, task=None):
    sub = [r for r in rows if r["method"] == method
           and (task is None or r["task"] == task)]
    return sum(r["success"] for r in sub), len(sub)


def paired_test(ix, tasks, m_a, m_b, seed=0):
    """McNemar + paired bootstrap for m_a vs m_b over the given tasks
    (pairing unit = (task, seed))."""
    a, b = [], []
    for t in tasks:
        for s in paired_seeds(ix, t, m_a, m_b):
            a.append(ix[(t, s, m_a)]["success"])
            b.append(ix[(t, s, m_b)]["success"])
    if not a:
        return None
    mc = mcnemar_from_vectors(a, b)
    bs = paired_bootstrap_diff(a, b, seed=seed)
    return {"n": len(a), "k_a": int(sum(a)), "k_b": int(sum(b)),
            "b": mc.b, "c": mc.c, "p": mc.pvalue,
            "diff": bs.diff, "lo": bs.lo, "hi": bs.hi}


def error_stats(rows, method, key, task=None):
    vals = [r[key] for r in rows if r["method"] == method
            and (task is None or r["task"] == task) and r[key] is not None]
    if not vals:
        return None
    v = np.asarray(vals, dtype=float)
    return {"n": v.size, "median": float(np.median(v)),
            "mean": float(v.mean()),
            "p90": float(np.percentile(v, 90))}


def paired_error_diff(ix, tasks, m_a, m_b, key):
    """Median of the per-pair difference err(m_a) - err(m_b), plus the sign
    test on 'm_a is closer to GT than m_b' (exact binomial via McNemar's
    discordant-count machinery: b = m_a better, c = m_b better)."""
    d = []
    for t in tasks:
        for s in paired_seeds(ix, t, m_a, m_b):
            va, vb = ix[(t, s, m_a)][key], ix[(t, s, m_b)][key]
            if va is None or vb is None:
                continue
            d.append(va - vb)
    if not d:
        return None
    d = np.asarray(d, dtype=float)
    better_a = (d < 0).astype(int)
    better_b = (d > 0).astype(int)
    mc = mcnemar_from_vectors(better_a, better_b)
    return {"n": int(d.size), "median_diff": float(np.median(d)),
            "n_a_better": int(better_a.sum()), "n_b_better": int(better_b.sum()),
            "p_sign": mc.pvalue}


def stage_counts(rows, method, task=None):
    out = {}
    for r in rows:
        if r["method"] != method or (task is not None and r["task"] != task):
            continue
        if r["success"]:
            continue
        out[r["failure_stage"] or "unknown"] = \
            out.get(r["failure_stage"] or "unknown", 0) + 1
    return out


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def _fmt_rate(k, n, ci=True):
    if not n:
        return "—"
    if ci:
        lo, hi = wilson_ci(k, n)
        return "%d/%d = %.2f [%.2f, %.2f]" % (k, n, k / n, lo, hi)
    return "%d/%d = %.2f" % (k, n, k / n)


def _pm(x, fmt="%+.3f"):
    """Signed number without a '-0.000' artifact."""
    return fmt % (x if x else 0.0)


def _fmt_p(p):
    if p is None:
        return "—"
    return "%.3g" % p


def _fmt_stages(d):
    if not d:
        return "—"
    return ", ".join("%s %d" % (k, v)
                     for k, v in sorted(d.items(), key=lambda kv: -kv[1]))


def markdown_report(rows, ix):
    tasks = tasks_present(rows)
    methods = methods_present(rows)
    base = [m for m in BASE_METHODS if m in methods]
    lines = [
        "# Campaign F tables — which ALK points enter the Procrustes sum",
        "",
        "Protocol identical to Campaign A: medium tier, seeds 1000..1059, "
        "oracle discrete answers, bounded registration ON, grasp correction "
        "ON. Rollouts under `results/campaign_f/<task>/<seed>/"
        "rollout_<method>.json`.",
        "",
        "| method | Procrustes sum | note |",
        "|---|---|---|",
        "| alk4 | {c1,c2,c3,c4} | current default (identical code path to "
        "campaign_a `ours_full`) |",
        "| alk3_c134 | {c1,c3,c4} | the earlier three-point construction ("
        "$\\sum_{j=1}^{3}$, figure box C1/C3/C4) |",
        "| alk3_c123 | {c1,c2,c3} | completeness: axis + one lateral |",
        "| alk2_c12 | {c1,c2} | completeness: rank-1, roll frozen |",
        "| *_adaptive | as above | + conditioning-adaptive registration "
        "(paper Section 3.7), trigger read on the SOLVED subset |",
        "",
    ]

    # -- 1. conditioning -----------------------------------------------------
    cond = conditioning_table(rows, tasks)
    lines += [
        "## 1. Conditioning of the demo configuration (analytic)",
        "",
        "Singular values of the CENTERED demo configuration (m), "
        "$\\sigma_2+\\sigma_3$ (paper Section 3.7: closed-form rotation error scales "
        "like $1/(\\sigma_2+\\sigma_3)$) and the scale-free ratio "
        "$(\\sigma_2+\\sigma_3)/\\sigma_1$ compared by the adaptive trigger "
        "to $\\tau_\\sigma = %.2f$. The demo scene is fixed per task, so each "
        "row is one exact constant, not an average." % SIGMA_RATIO_THRESHOLD,
        "",
        "| task | subset | sigma1 | sigma2 | sigma3 | sigma2+sigma3 | ratio | "
        "ratio < tau |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in tasks:
        if t not in cond:
            continue
        for name in ("alk4", "alk3_c134", "alk3_c123", "alk2_c12"):
            c = cond[t][name]
            lines.append(
                "| %s | %s %s | %.5f | %.5f | %.5f | **%.5f** | **%.3f** | %s |"
                % (t if name == "alk4" else "", name, SUBSET_LABEL[name],
                   c["sv"][0], c["sv"][1], c["sv"][2], c["sigma23"],
                   c["ratio"],
                   "yes" if c["ratio"] < SIGMA_RATIO_THRESHOLD else "no"))
    lines.append("")
    lines += [
        "Relative change of the two theory numbers when c2 is dropped "
        "(alk3_c134 vs alk4):", "",
        "| task | sigma23 alk4 | sigma23 alk3_c134 | change | ratio alk4 | "
        "ratio alk3_c134 | change |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in tasks:
        if t not in cond:
            continue
        c4, c3 = cond[t]["alk4"], cond[t]["alk3_c134"]
        lines.append("| %s | %.5f | %.5f | %+.1f%% | %.3f | %.3f | %+.1f%% |"
                     % (t, c4["sigma23"], c3["sigma23"],
                        100.0 * (c3["sigma23"] / c4["sigma23"] - 1.0),
                        c4["ratio"], c3["ratio"],
                        100.0 * (c3["ratio"] / c4["ratio"] - 1.0)))
    lines.append("")

    # -- 1b. Monte-Carlo consequence of the conditioning --------------------
    sigmas_mm = (2.0, 5.0)
    alks = demo_alks(rows, tasks)
    lines += [
        "Consequence for the closed form alone (Monte Carlo, %d trials per "
        "cell, shared noise draws and shared random ground-truth rotation "
        "across subsets, seed 0): median / p90 rotation error (deg) of the "
        "closed-form solve when every keypoint is perturbed by iid isotropic "
        "Gaussian noise. This is what the perturbation scaling of Section 3.7 predicts "
        "(error ~ eps/(sigma2+sigma3) of the SOLVED configuration), isolated "
        "from registration and execution." % 2000,
        "",
        "| task | keypoint noise | " + " | ".join(
            "%s %s" % (n, SUBSET_LABEL[n]) for n in BASE_METHODS) + " |",
        "|---|---|" + "---|" * len(BASE_METHODS),
    ]
    for t in tasks:
        if t not in alks:
            continue
        for s_mm in sigmas_mm:
            ns = noise_sensitivity(alks[t], 1e-3 * s_mm)
            lines.append("| %s | %g mm | " % (t if s_mm == sigmas_mm[0] else "",
                                              s_mm)
                         + " | ".join("%.1f / %.1f" % (ns[n]["median"],
                                                       ns[n]["p90"])
                                      for n in BASE_METHODS) + " |")
    lines.append("")

    # -- 2. success ----------------------------------------------------------
    lines += ["## 2. End-to-end success (paired, 60 seeds/task)", "",
              "| task | " + " | ".join(base) + " |",
              "|---|" + "---|" * len(base)]
    for t in tasks:
        cells = [_fmt_rate(*success_counts(rows, m, t)) for m in base]
        lines.append("| %s | " % t + " | ".join(cells) + " |")
    pooled = [_fmt_rate(*success_counts(rows, m)) for m in base]
    lines.append("| **pooled** | " + " | ".join("**%s**" % c for c in pooled)
                 + " |")
    lines.append("")

    # -- 3. McNemar ----------------------------------------------------------
    lines += ["## 3. Paired tests vs the 4-point default (`alk4`)", "",
              "McNemar exact on the shared seeds; b = alk4 wins, c = the "
              "other method wins; diff = rate(other) - rate(alk4) with a "
              "paired-bootstrap 95% CI. Holm correction runs over the "
              "per-task family of the PRIMARY comparison (alk4 vs "
              "alk3_c134).",
              "",
              "| comparison | scope | n | k(alk4) | k(other) | diff [95% CI] "
              "| b/c | p | p_holm |",
              "|---|---|---|---|---|---|---|---|---|"]
    primary = []
    for t in tasks:
        r = paired_test(ix, [t], REF, PRIMARY)
        if r:
            primary.append((t, r))
    adj = holm_bonferroni([r["p"] for _, r in primary]) if primary else []
    for (t, r), pa in zip(primary, adj):
        lines.append(
            "| alk4 vs %s | %s | %d | %d | %d | %s [%s, %s] | %d/%d "
            "| %s | %s |"
            % (PRIMARY, t, r["n"], r["k_a"], r["k_b"], _pm(-r["diff"]),
               _pm(-r["hi"]), _pm(-r["lo"]), r["b"], r["c"],
               _fmt_p(r["p"]), _fmt_p(pa)))
    for m in [mm for mm in base if mm != REF]:
        r = paired_test(ix, tasks, REF, m)
        if not r:
            continue
        lines.append(
            "| alk4 vs %s | **pooled** | %d | %d | %d | %s [%s, %s] "
            "| %d/%d | **%s** | — |"
            % (m, r["n"], r["k_a"], r["k_b"], _pm(-r["diff"]), _pm(-r["hi"]),
               _pm(-r["lo"]), r["b"], r["c"], _fmt_p(r["p"])))
    lines.append("")

    # -- 4a. closed-form-only errors -----------------------------------------
    lines += ["## 4a. Closed-form alignment error (T_init vs GT, BEFORE "
              "bounded registration)", "",
              "The subsets differ ONLY in T_init, so this is where the "
              "conditioning consequence is visible before the shared bounded "
              "registration partly absorbs it.", "",
              "| task | metric | " + " | ".join(base) + " |",
              "|---|---|" + "---|" * len(base)]
    for t in tasks + ["pooled"]:
        tt = None if t == "pooled" else t
        for key, lbl, scale, fmt in (
                ("rot_err_init", "rot err deg (med / p90)", 1.0,
                 "%.1f / %.1f"),
                ("trans_err_init", "trans err mm (med / p90)", 1e3,
                 "%.1f / %.1f")):
            cells = []
            for m in base:
                st = error_stats(rows, m, key, tt)
                cells.append(fmt % (scale * st["median"], scale * st["p90"])
                             if st else "—")
            head = t if key == "rot_err_init" else ""
            if t == "pooled":
                head = "**pooled**"
                cells = ["**%s**" % c for c in cells]
            lines.append("| %s | %s | " % (head, lbl) + " | ".join(cells)
                         + " |")
    lines.append("")
    lines += ["Paired per-seed closed-form error difference vs alk4 (sign "
              "test over pairs):", "",
              "| metric | method | n | median(other - alk4) | other better | "
              "alk4 better | p |",
              "|---|---|---|---|---|---|---|"]
    for key, lbl, scale, unit in (("rot_err_init", "rotation", 1.0, "deg"),
                                  ("trans_err_init", "translation", 1e3,
                                   "mm")):
        for m in [mm for mm in base if mm != REF]:
            d = paired_error_diff(ix, tasks, m, REF, key)
            if not d:
                continue
            lines.append("| %s | %s | %d | %s %s | %d | %d | %s |"
                         % (lbl, m, d["n"],
                            _pm(scale * d["median_diff"], "%+.2f"), unit,
                            d["n_a_better"], d["n_b_better"],
                            _fmt_p(d["p_sign"])))
    lines.append("")

    # -- 4. errors -----------------------------------------------------------
    lines += ["## 4b. Mapping error (final T_map vs GT, after registration)",
              "",
              "| task | metric | " + " | ".join(base) + " |",
              "|---|---|" + "---|" * len(base)]
    for t in tasks:
        for key, lbl, scale, fmt in (("rot_err", "rot err deg (med / p90)",
                                      1.0, "%.1f / %.1f"),
                                     ("trans_err", "trans err mm (med / p90)",
                                      1e3, "%.1f / %.1f")):
            cells = []
            for m in base:
                st = error_stats(rows, m, key, t)
                cells.append(fmt % (scale * st["median"], scale * st["p90"])
                             if st else "—")
            lines.append("| %s | %s | " % (t if key == "rot_err" else "", lbl)
                         + " | ".join(cells) + " |")
    for key, lbl, scale, fmt in (("rot_err", "rot err deg (med / p90)", 1.0,
                                  "%.1f / %.1f"),
                                 ("trans_err", "trans err mm (med / p90)",
                                  1e3, "%.1f / %.1f")):
        cells = []
        for m in base:
            st = error_stats(rows, m, key)
            cells.append(fmt % (scale * st["median"], scale * st["p90"])
                         if st else "—")
        lines.append("| **pooled** | %s | " % lbl
                     + " | ".join("**%s**" % c for c in cells) + " |")
    lines.append("")
    lines += ["Paired per-seed error difference vs alk4 (negative median = "
              "the other method is closer to GT; sign test over pairs):", "",
              "| metric | method | n | median(other - alk4) | other better | "
              "alk4 better | p |",
              "|---|---|---|---|---|---|---|"]
    for key, lbl, scale, unit in (("rot_err", "rotation", 1.0, "deg"),
                                  ("trans_err", "translation", 1e3, "mm")):
        for m in [mm for mm in base if mm != REF]:
            d = paired_error_diff(ix, tasks, m, REF, key)
            if not d:
                continue
            lines.append("| %s | %s | %d | %s %s | %d | %d | %s |"
                         % (lbl, m, d["n"],
                            _pm(scale * d["median_diff"], "%+.2f"), unit,
                            d["n_a_better"], d["n_b_better"],
                            _fmt_p(d["p_sign"])))
    lines.append("")

    # -- 5. failure stages ---------------------------------------------------
    lines += ["## 5. Failure stages", "",
              "| task | " + " | ".join(base) + " |",
              "|---|" + "---|" * len(base)]
    for t in tasks:
        lines.append("| %s | " % t + " | ".join(
            _fmt_stages(stage_counts(rows, m, t)) for m in base) + " |")
    lines.append("| **pooled** | " + " | ".join(
        _fmt_stages(stage_counts(rows, m)) for m in base) + " |")
    lines.append("")

    # -- 6. adaptive ---------------------------------------------------------
    adapt = [m for m in ADAPTIVE_METHODS if m in methods]
    if adapt:
        atasks = [t for t in tasks
                  if any(r["task"] == t and r["method"] in adapt
                         for r in rows)]
        lines += ["## 6. Does conditioning-adaptive registration rescue the "
                  "3-point construction?", "",
                  "Run on the worst-conditioned tasks only. `triggered` = "
                  "how many rollouts had ratio < tau_sigma and therefore got "
                  "the widened axis search (for the subset methods the "
                  "trigger reads the SOLVED 3-point configuration). "
                  "`ratio` is the median (sigma2+sigma3)/sigma1 of the "
                  "configuration each method actually solves -- under "
                  "`adaptive` the auto slender depth prior "
                  "(depth_consistent='auto') alters the ALK on slender "
                  "objects, so it differs from the analytic table above.",
                  "",
                  "| task | method | success | ratio (med) | triggered | "
                  "rot err deg (med) | trans err mm (med) |",
                  "|---|---|---|---|---|---|---|"]
        for t in atasks:
            for m in [mm for mm in (BASE_METHODS[:2] + adapt) if mm in methods]:
                k, n = success_counts(rows, m, t)
                if not n:
                    continue
                trig = sum(1 for r in rows if r["task"] == t
                           and r["method"] == m and r["adaptive_triggered"])
                sr = error_stats(rows, m, "rot_err", t)
                st = error_stats(rows, m, "trans_err", t)
                ra = error_stats(rows, m, "sigma_ratio", t)
                lines.append("| %s | %s | %s | %s | %s | %s | %s |"
                             % (t if m == BASE_METHODS[0] else "", m,
                                _fmt_rate(k, n),
                                "%.3f" % ra["median"] if ra else "—",
                                "%d/%d" % (trig, n)
                                if m.endswith("adaptive") else "—",
                                "%.1f" % sr["median"] if sr else "—",
                                "%.1f" % (1e3 * st["median"]) if st else "—"))
        lines.append("")
        lines += ["Paired McNemar within the adaptive-tested tasks:", "",
                  "| comparison | scope | n | k_a | k_b | b/c | p |",
                  "|---|---|---|---|---|---|---|"]
        pairs = [("alk3_c134", "alk3_c134_adaptive"),
                 ("alk4", "alk4_adaptive"),
                 ("alk4_adaptive", "alk3_c134_adaptive")]
        for m_a, m_b in pairs:
            if m_a not in methods or m_b not in methods:
                continue
            for scope in [[t] for t in atasks] + [atasks]:
                r = paired_test(ix, scope, m_a, m_b)
                if not r:
                    continue
                lines.append("| %s vs %s | %s | %d | %d | %d | %d/%d | %s |"
                             % (m_a, m_b,
                                scope[0] if len(scope) == 1 else "**pooled**",
                                r["n"], r["k_a"], r["k_b"], r["b"], r["c"],
                                _fmt_p(r["p"])))
        lines.append("")

    # -- 7. cross-check vs campaign A ----------------------------------------
    xc = crosscheck_campaign_a(ix)
    if xc:
        lines += ["## 7. Cross-check: `alk4` reproduces campaign_a "
                  "`ours_full`", "",
                  "`alk4` runs the untouched default code path, so its "
                  "rollouts must match the stored Campaign-A ones "
                  "seed-by-seed.", "",
                  "| task | n compared | success mismatches | max |rot| diff "
                  "(deg) | max |trans| diff (m) |",
                  "|---|---|---|---|---|"]
        for t, d in sorted(xc.items()):
            lines.append("| %s | %d | %d | %.2e | %.2e |"
                         % (t, d["n"], d["mismatch"], d["max_rot"],
                            d["max_trans"]))
        lines.append("")
    return "\n".join(lines)


def crosscheck_campaign_a(ix, root_a=CAMPAIGN_A):
    """Compare alk4 rollouts against campaign_a's ours_full (same code path)."""
    out = {}
    for (t, s, m), r in sorted(ix.items()):
        if m != REF:
            continue
        pa = os.path.join(root_a, t, str(s), "rollout_ours_full.json")
        if not os.path.exists(pa):
            continue
        with open(pa) as f:
            a = json.load(f)
        d = out.setdefault(t, {"n": 0, "mismatch": 0, "max_rot": 0.0,
                               "max_trans": 0.0})
        d["n"] += 1
        d["mismatch"] += int(bool(a.get("success")) != bool(r["success"]))
        if a.get("rot_err_deg") is not None and r["rot_err"] is not None:
            d["max_rot"] = max(d["max_rot"],
                               abs(a["rot_err_deg"] - r["rot_err"]))
            d["max_trans"] = max(d["max_trans"],
                                 abs(a["trans_err_m"] - r["trans_err"]))
    return out


# ---------------------------------------------------------------------------
# LaTeX (the two paper tables)
# ---------------------------------------------------------------------------

def latex_report(rows, ix):
    tasks = tasks_present(rows)
    methods = methods_present(rows)
    base = [m for m in BASE_METHODS if m in methods]
    esc = lambda s: s.replace("_", r"\_")
    cond = conditioning_table(rows, tasks)

    out = ["% Campaign F: ALK point-subset ablation (auto-generated by "
           "stats/campaign_f_tables.py)",
           r"\begin{table}", r"\centering",
           r"\caption{Conditioning of the demo keypoint configuration: the "
           r"original paper's three-point sum $\{c_1,c_3,c_4\}$ vs the "
           r"four-point sum. $\sigma_2+\sigma_3$ governs the closed-form "
           r"rotation-error bound (Prop.~3); the ratio "
           r"$(\sigma_2+\sigma_3)/\sigma_1$ is what the adaptive trigger "
           r"compares to $\tau_\sigma=0.40$. Dropping $c_2$ removes the "
           r"axial extent, makes the configuration exactly coplanar "
           r"($\sigma_3=0$) and lowers $\sigma_2+\sigma_3$ on every task, "
           r"while INFLATING the scale-free ratio.}",
           r"\label{tab:alk-subset-conditioning}",
           r"\begin{tabular}{lcccc}", r"\toprule",
           r"& \multicolumn{2}{c}{$\sigma_2+\sigma_3$ (m)} & "
           r"\multicolumn{2}{c}{$(\sigma_2+\sigma_3)/\sigma_1$} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r"task & 4-pt & 3-pt & 4-pt & 3-pt \\", r"\midrule"]
    for t in tasks:
        if t not in cond:
            continue
        c4, c3 = cond[t]["alk4"], cond[t]["alk3_c134"]
        out.append("%s & %.4f & %.4f & %.3f & %.3f \\\\"
                   % (esc(t), c4["sigma23"], c3["sigma23"], c4["ratio"],
                      c3["ratio"]))
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    out += [r"\begin{table}", r"\centering",
            r"\caption{End-to-end success of the closed-form alignment point "
            r"subsets (medium tier, 60 paired seeds per task, oracle "
            r"discrete answers, bounded registration and grasp correction "
            r"on). $p$: exact McNemar vs the four-point default, paired by "
            r"seed.}",
            r"\label{tab:alk-subset-success}",
            r"\begin{tabular}{l" + "c" * len(base) + "c}", r"\toprule",
            "task & " + " & ".join(esc(m) for m in base)
            + r" & $p$ (4-pt vs 3-pt) \\", r"\midrule"]
    for t in tasks:
        cells = []
        for m in base:
            k, n = success_counts(rows, m, t)
            cells.append("%.2f" % (k / n) if n else "--")
        r = paired_test(ix, [t], REF, PRIMARY)
        out.append(esc(t) + " & " + " & ".join(cells)
                   + " & %s \\\\" % (_fmt_p(r["p"]) if r else "--"))
    out.append(r"\midrule")
    cells = []
    for m in base:
        k, n = success_counts(rows, m)
        cells.append("\\textbf{%.2f}" % (k / n) if n else "--")
    r = paired_test(ix, tasks, REF, PRIMARY)
    out.append("pooled & " + " & ".join(cells)
               + " & %s \\\\" % (_fmt_p(r["p"]) if r else "--"))
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


CSV_KEYS = ["task", "seed", "method", "tier", "alk_subset", "adaptive",
            "adaptive_triggered", "success", "grasped", "rot_err", "trans_err",
            "rot_err_init", "trans_err_init", "chamfer_after", "conditioning",
            "failure_stage", "source_file"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--prefix", default="campaign_f")
    args = ap.parse_args(argv)

    rows = load_rows(args.root)
    if not rows:
        raise SystemExit("no rollouts under %s" % args.root)
    ix = index(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "%s_rollouts.csv" % args.prefix), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_KEYS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    md = markdown_report(rows, ix)
    with open(os.path.join(args.out_dir, "%s.md" % args.prefix), "w") as f:
        f.write(md + "\n")
    with open(os.path.join(args.out_dir, "%s.tex" % args.prefix), "w") as f:
        f.write(latex_report(rows, ix) + "\n")
    print("wrote %s.{md,tex,_rollouts.csv} to %s (%d rollouts, %d methods)"
          % (args.prefix, args.out_dir, len(rows), len(methods_present(rows))))
    print(md)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
