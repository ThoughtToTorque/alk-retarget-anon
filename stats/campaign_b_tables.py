"""Campaign B tables: sensor-noise sensitivity (success vs noise level).

Reads results/campaign_b/<task>/<cell>/<seed>/rollout_<method>.json
(cell = s<depth_sigma_mm>_e<mask_erosion_px>) and writes

  results/tables/campaign_b.md    -- all tables (success grid pooled +
                                     per-task, McNemar full-vs-noreg per
                                     cell, error medians, trend tests)
  results/tables/campaign_b.tex   -- LaTeX versions of the two paper tables
  results/tables/campaign_b_rollouts.csv

Statistics (frozen protocol, baselines/docs/CODE_MAP.md + stats/tests.py):
  * Wilson CI next to every pooled success rate.
  * McNemar exact (paired by (task, seed)) for ours_full vs ours_noreg at
    every cell -- the registration-contribution test.
  * Trend: per paired unit i = (task, seed), d_i(level) = full_i - noreg_i
    in {-1, 0, 1}; OLS slope of mean(d) vs the noise level along one axis
    (depth sigma with erosion fixed, and vice versa), cluster bootstrap over
    units (resample units jointly across levels -> slope CI + one-sided p
    for H1: the registration advantage GROWS with noise).

Usage:
  python -m stats.campaign_b_tables [--root results/campaign_b]
"""
import argparse
import csv
import json
import os
import re

import numpy as np

from stats.tests import wilson_ci, mcnemar_from_vectors, holm_bonferroni

SIMBENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(SIMBENCH, "results", "campaign_b")
DEFAULT_OUT_DIR = os.path.join(SIMBENCH, "results", "tables")

METHODS = ["ours_full", "ours_noreg", "icp", "kp_fps4"]
DEPTH_SIGMAS_MM = (0.0, 1.5, 3.0, 6.0)
MASK_EROSIONS_PX = (0, 2, 4)
CELL_RE = re.compile(r"^s([0-9.]+)_e([0-9]+)$")


def cell_name(s, e):
    return "s%g_e%d" % (s, e)


ALL_CELLS = [(s, e) for e in MASK_EROSIONS_PX for s in DEPTH_SIGMAS_MM]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_rows(root):
    """One flat row per rollout; cell parsed from the path (works for the
    symlinked zero cell whose records carry no cell field)."""
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if not (name.startswith("rollout_") and name.endswith(".json")):
                continue
            path = os.path.join(dirpath, name)
            seed_dir, cell_dir = os.path.split(dirpath)
            task_dir, cell = os.path.split(seed_dir)
            m = CELL_RE.match(cell)
            if not m:
                continue
            with open(path) as f:
                rec = json.load(f)
            rows.append({
                "task": rec.get("task") or os.path.basename(task_dir),
                "seed": int(rec["seed"]),
                "method": rec.get("method") or rec.get("variant"),
                "cell": cell,
                "sigma_mm": float(m.group(1)),
                "erode_px": int(m.group(2)),
                "success": int(bool(rec.get("success"))),
                "rot_err": rec.get("rot_err_deg"),
                "trans_err": rec.get("trans_err_m"),
                "chamfer_after": rec.get("chamfer_after_m"),
                "failure_stage": rec.get("failure_stage"),
                "noise_seed": rec.get("noise_seed"),
                "source_file": path,
            })
    return rows


def index(rows):
    """dict[(task, seed, cell, method)] -> row (unique)."""
    ix = {}
    for r in rows:
        key = (r["task"], r["seed"], r["cell"], r["method"])
        ix[key] = r
    return ix


def sel(rows, **conds):
    out = rows
    for k, v in conds.items():
        out = [r for r in out if r.get(k) == v]
    return out


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def cells_present(rows):
    seen = {(r["sigma_mm"], r["erode_px"]) for r in rows}
    return [c for c in ALL_CELLS if c in seen]


def success_grid(rows, task=None):
    """dict[method][cell] -> (k, n)."""
    sub = sel(rows, task=task) if task else rows
    out = {}
    for m in METHODS:
        out[m] = {}
        for (s, e) in cells_present(rows):
            rr = sel(sub, method=m, cell=cell_name(s, e))
            if rr:
                out[m][cell_name(s, e)] = (sum(r["success"] for r in rr),
                                           len(rr))
    return out


def paired_vectors(ix, cell, m_a, m_b, tasks, seeds):
    """Aligned success vectors for two methods on one cell; units with both
    rollouts present only.  Returns (units, vec_a, vec_b)."""
    units, va, vb = [], [], []
    for t in tasks:
        for s in seeds:
            ra = ix.get((t, s, cell, m_a))
            rb = ix.get((t, s, cell, m_b))
            if ra is not None and rb is not None:
                units.append((t, s))
                va.append(ra["success"])
                vb.append(rb["success"])
    return units, va, vb


def mcnemar_by_cell(rows, ix, m_a="ours_full", m_b="ours_noreg"):
    tasks = sorted({r["task"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    out = []
    for (s, e) in cells_present(rows):
        cell = cell_name(s, e)
        units, va, vb = paired_vectors(ix, cell, m_a, m_b, tasks, seeds)
        if not units:
            continue
        res = mcnemar_from_vectors(va, vb)
        out.append({
            "cell": cell, "sigma_mm": s, "erode_px": e, "n": len(units),
            "k_a": int(np.sum(va)), "k_b": int(np.sum(vb)),
            "diff": (np.mean(va) - np.mean(vb)),
            "b": res.b, "c": res.c, "p": res.pvalue,
        })
    return out


def trend_test(rows, ix, axis, m_a="ours_full", m_b="ours_noreg",
               n_boot=10000, boot_seed=0):
    """Cluster-bootstrap trend of the paired advantage d = a - b along one
    noise axis (other axis fixed at 0).  axis in {"sigma", "erosion"}."""
    if axis == "sigma":
        levels = [(s, 0) for s in DEPTH_SIGMAS_MM]
        xs = np.array(DEPTH_SIGMAS_MM)
        unit_label = "per mm depth sigma"
    else:
        levels = [(0.0, e) for e in MASK_EROSIONS_PX]
        xs = np.array(MASK_EROSIONS_PX, dtype=float)
        unit_label = "per px erosion"
    levels = [lv for lv in levels if lv in cells_present(rows)]
    if len(levels) < 2:
        return None
    xs = xs[:len(levels)]
    tasks = sorted({r["task"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    # units complete at ALL levels of the axis
    units = None
    per_level = {}
    for lv in levels:
        u, va, vb = paired_vectors(ix, cell_name(*lv), m_a, m_b, tasks, seeds)
        per_level[lv] = dict(zip(u, np.asarray(va) - np.asarray(vb)))
        units = set(u) if units is None else units & set(u)
    units = sorted(units)
    if len(units) < 5:
        return None
    D = np.array([[per_level[lv][u] for lv in levels] for u in units],
                 dtype=float)                      # (n_units, n_levels)

    def slope(d_mat):
        y = d_mat.mean(axis=0)
        X = np.vstack([xs, np.ones_like(xs)]).T
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return coef[0]

    obs = slope(D)
    rng = np.random.default_rng(boot_seed)
    n = len(units)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        bs[i] = slope(D[rng.integers(0, n, size=n)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p_grow = float(np.mean(bs <= 0.0))   # one-sided: H1 slope > 0
    return {"axis": axis, "unit": unit_label, "levels": [list(l) for l in levels],
            "n_units": n, "mean_d_by_level": D.mean(axis=0).tolist(),
            "slope": float(obs), "ci": [float(lo), float(hi)],
            "p_one_sided_grow": p_grow}


def diff_in_diff(rows, ix, cell_hi, cell_lo="s0_e0",
                 m_a="ours_full", m_b="ours_noreg", n_boot=10000,
                 boot_seed=0):
    """Paired difference-in-differences: does the (a-b) advantage at the
    high-noise cell exceed the zero-noise cell?  Cluster bootstrap CI +
    one-sided p."""
    tasks = sorted({r["task"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    u1, va1, vb1 = paired_vectors(ix, cell_lo, m_a, m_b, tasks, seeds)
    u2, va2, vb2 = paired_vectors(ix, cell_hi, m_a, m_b, tasks, seeds)
    d1 = dict(zip(u1, np.asarray(va1) - np.asarray(vb1)))
    d2 = dict(zip(u2, np.asarray(va2) - np.asarray(vb2)))
    units = sorted(set(d1) & set(d2))
    if len(units) < 5:
        return None
    dd = np.array([d2[u] - d1[u] for u in units], dtype=float)
    rng = np.random.default_rng(boot_seed)
    n = len(units)
    bs = dd[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"cell_lo": cell_lo, "cell_hi": cell_hi, "n_units": n,
            "dd": float(dd.mean()), "ci": [float(lo), float(hi)],
            "p_one_sided_grow": float(np.mean(bs <= 0.0))}


def marginal_success(rows, ix, axis):
    """Success pooled along one noise axis (the other axis averaged over):
    dict[method][level] -> (k, n).  axis in {"sigma", "erosion"}."""
    tasks = sorted({r["task"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    levels = DEPTH_SIGMAS_MM if axis == "sigma" else MASK_EROSIONS_PX
    other = MASK_EROSIONS_PX if axis == "sigma" else DEPTH_SIGMAS_MM
    out = {}
    for m in METHODS:
        out[m] = {}
        for lv in levels:
            cells = [cell_name(lv, o) if axis == "sigma" else cell_name(o, lv)
                     for o in other]
            v = [ix[(t, s, c, m)]["success"] for t in tasks for s in seeds
                 for c in cells if (t, s, c, m) in ix]
            if v:
                out[m][lv] = (int(np.sum(v)), len(v))
    return out


def noise_pooled_registration(rows, ix, m_a="ours_full", m_b="ours_noreg",
                              n_boot=20000, boot_seed=0):
    """Registration advantage pooled over the 11 NOISE cells, with a cluster
    bootstrap over paired units (task, seed) -- cells within a unit share the
    scene, so units, not rollouts, are the resampling level.  Also reports
    the zero-noise cell and their difference (DiD)."""
    tasks = sorted({r["task"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    zero = cell_name(0, 0)
    noise = [cell_name(*c) for c in cells_present(rows)
             if cell_name(*c) != zero]
    units = [(t, s) for t in tasks for s in seeds
             if all((t, s, c, m) in ix for c in noise + [zero]
                    for m in (m_a, m_b))]
    if len(units) < 5 or not noise:
        return None
    A = np.array([[ix[(t, s, c, m_a)]["success"] for c in noise]
                  for t, s in units], dtype=float)
    B = np.array([[ix[(t, s, c, m_b)]["success"] for c in noise]
                  for t, s in units], dtype=float)
    d = (A - B).mean(axis=1)
    z = np.array([ix[(t, s, zero, m_a)]["success"]
                  - ix[(t, s, zero, m_b)]["success"] for t, s in units],
                 dtype=float)
    rng = np.random.default_rng(boot_seed)
    n = len(units)
    idx = rng.integers(0, n, size=(n_boot, n))
    bs = d[idx].mean(axis=1)
    bs_dd = (d - z)[idx].mean(axis=1)
    p_two = 2.0 * min(float((bs <= 0).mean()), float((bs >= 0).mean()))
    return {
        "n_units": n, "n_noise_cells": len(noise),
        "rate_a": float(A.mean()), "rate_b": float(B.mean()),
        "d": float(d.mean()),
        "ci": [float(x) for x in np.percentile(bs, [2.5, 97.5])],
        "p_two_sided": min(1.0, p_two),
        "d_zero": float(z.mean()),
        "dd": float((d - z).mean()),
        "dd_ci": [float(x) for x in np.percentile(bs_dd, [2.5, 97.5])],
        "dd_p_one_sided_grow": float((bs_dd <= 0).mean()),
    }


def paired_error_improvement(rows, ix, key="rot_err", m_a="ours_full",
                             m_b="ours_noreg"):
    """Per-erosion-level paired improvement b - a of an error metric
    (positive = a is better), plus P(a better)."""
    tasks = sorted({r["task"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    out = []
    for e in MASK_EROSIONS_PX:
        diffs, wins = [], 0
        for t in tasks:
            for s in seeds:
                for sg in DEPTH_SIGMAS_MM:
                    c = cell_name(sg, e)
                    ra, rb = ix.get((t, s, c, m_a)), ix.get((t, s, c, m_b))
                    if not ra or not rb:
                        continue
                    a, b = ra.get(key), rb.get(key)
                    if a is None or b is None:
                        continue
                    diffs.append(b - a)
                    wins += int(a < b)
        if diffs:
            d = np.asarray(diffs)
            out.append({"erode_px": e, "n": d.size,
                        "median": float(np.median(d)),
                        "mean": float(d.mean()),
                        "p_a_better": wins / float(d.size)})
    return out


def median_by_cell(rows, key, method):
    out = {}
    for (s, e) in cells_present(rows):
        vals = [r[key] for r in sel(rows, method=method, cell=cell_name(s, e))
                if r.get(key) is not None]
        if vals:
            out[cell_name(s, e)] = float(np.median(vals))
    return out


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def _fmt_rate(k, n, ci=True):
    if ci:
        lo, hi = wilson_ci(k, n)
        return "%d/%d = %.2f [%.2f, %.2f]" % (k, n, k / n, lo, hi)
    return "%d/%d (%.2f)" % (k, n, k / n)


def markdown_report(rows, ix):
    cells = cells_present(rows)
    names = [cell_name(*c) for c in cells]
    lines = ["# Campaign B tables — sensor-noise sensitivity",
             "",
             "Cells: depth sigma (mm) x mask erosion (px); rollouts under "
             "`results/campaign_b/<task>/<cell>/<seed>/`.",
             ""]

    # -- pooled success grid -------------------------------------------------
    lines += ["## Success vs noise cell (pooled over tasks)", ""]
    grid = success_grid(rows)
    lines.append("| method | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for m in METHODS:
        row = ["%s" % m]
        for c in names:
            kn = grid[m].get(c)
            row.append("%d/%d=%.2f" % (kn[0], kn[1], kn[0] / kn[1])
                       if kn else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # -- per-task grids -------------------------------------------------------
    for task in sorted({r["task"] for r in rows}):
        lines += ["### %s" % task, ""]
        g = success_grid(rows, task=task)
        lines.append("| method | " + " | ".join(names) + " |")
        lines.append("|---|" + "---|" * len(names))
        for m in METHODS:
            row = [m]
            for c in names:
                kn = g[m].get(c)
                row.append("%d/%d" % kn if kn else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # -- marginal axes ---------------------------------------------------------
    lines += ["## Marginal noise axes (the other axis averaged over)", "",
              "### Success vs depth sigma (mm)", ""]
    ms = marginal_success(rows, ix, "sigma")
    lv_s = sorted({lv for m in METHODS for lv in ms[m]})
    lines.append("| method | " + " | ".join("%g mm" % lv for lv in lv_s) + " |")
    lines.append("|---|" + "---|" * len(lv_s))
    for m in METHODS:
        lines.append("| %s | " % m + " | ".join(
            _fmt_rate(*ms[m][lv], ci=False) if lv in ms[m] else "—"
            for lv in lv_s) + " |")
    lines += ["", "### Success vs mask erosion (px)", ""]
    me = marginal_success(rows, ix, "erosion")
    lv_e = sorted({lv for m in METHODS for lv in me[m]})
    lines.append("| method | " + " | ".join("%d px" % lv for lv in lv_e) + " |")
    lines.append("|---|" + "---|" * len(lv_e))
    for m in METHODS:
        lines.append("| %s | " % m + " | ".join(
            _fmt_rate(*me[m][lv], ci=False) if lv in me[m] else "—"
            for lv in lv_e) + " |")
    lines.append("")

    # -- McNemar full vs noreg -----------------------------------------------
    lines += ["## Registration contribution: McNemar ours_full vs "
              "ours_noreg per cell", "",
              "Holm correction over the family of 12 cells.", ""]
    lines.append("| cell | n | full | noreg | diff | discordant b/c | "
                 "p (exact) | p_holm |")
    lines.append("|---|---|---|---|---|---|---|---|")
    mc = mcnemar_by_cell(rows, ix)
    adj = holm_bonferroni([r["p"] for r in mc]) if mc else []
    for r, pa in zip(mc, adj):
        lines.append("| %s | %d | %d | %d | %+.3f | %d/%d | %.4g | %.4g |"
                     % (r["cell"], r["n"], r["k_a"], r["k_b"], r["diff"],
                        r["b"], r["c"], r["p"], pa))
    lines.append("")
    np_ = noise_pooled_registration(rows, ix)
    if np_:
        lines += [
            "**Noise-pooled test** (the %d noise cells pooled; cluster "
            "bootstrap over %d paired (task, seed) units, since cells within "
            "a unit share the scene): ours_full %.3f vs ours_noreg %.3f, "
            "d = %+.4f, 95%% CI [%+.4f, %+.4f], p = %.4g (two-sided). "
            "Zero-noise cell d = %+.4f; difference-in-differences "
            "%+.4f, 95%% CI [%+.4f, %+.4f], one-sided p(grow) = %.4f."
            % (np_["n_noise_cells"], np_["n_units"], np_["rate_a"],
               np_["rate_b"], np_["d"], np_["ci"][0], np_["ci"][1],
               np_["p_two_sided"], np_["d_zero"], np_["dd"],
               np_["dd_ci"][0], np_["dd_ci"][1],
               np_["dd_p_one_sided_grow"]), ""]

    # -- error medians ---------------------------------------------------------
    lines += ["## Median rotation error (deg) per cell", ""]
    lines.append("| method | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for m in METHODS:
        med = median_by_cell(rows, "rot_err", m)
        lines.append("| %s | " % m + " | ".join(
            "%.1f" % med[c] if c in med else "—" for c in names) + " |")
    lines += ["", "## Median chamfer-after (mm) per cell", ""]
    lines.append("| method | " + " | ".join(names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for m in METHODS:
        med = median_by_cell(rows, "chamfer_after", m)
        lines.append("| %s | " % m + " | ".join(
            "%.1f" % (1e3 * med[c]) if c in med else "—" for c in names)
            + " |")
    lines.append("")

    # -- paired error improvement ---------------------------------------------
    lines += ["## Paired error improvement from registration "
              "(noreg − full, positive = registration helps)", ""]
    for key, lbl, scale, unit in (("rot_err", "rotation", 1.0, "deg"),
                                  ("chamfer_after", "chamfer", 1e3, "mm")):
        pei = paired_error_improvement(rows, ix, key=key)
        if not pei:
            continue
        lines.append("| erosion | n | median %s improvement (%s) | mean | "
                     "P(full better) |" % (lbl, unit))
        lines.append("|---|---|---|---|---|")
        for r in pei:
            lines.append("| %d px | %d | %+.2f | %+.2f | %.3f |"
                         % (r["erode_px"], r["n"], scale * r["median"],
                            scale * r["mean"], r["p_a_better"]))
        lines.append("")

    # -- trend tests -----------------------------------------------------------
    lines += ["## Trend of the paired advantage d = full − noreg", ""]
    for axis in ("sigma", "erosion"):
        tr = trend_test(rows, ix, axis)
        if tr is None:
            continue
        lines.append(
            "- **%s axis** (%s; n=%d paired units): mean d by level %s; "
            "slope %+0.4f %s, 95%% CI [%+.4f, %+.4f], one-sided p(grow) "
            "= %.4f" % (axis, "/".join(cell_name(*l) for l in
                                       [tuple(x) for x in tr["levels"]]),
                        tr["n_units"],
                        "[" + ", ".join("%+.3f" % v
                                        for v in tr["mean_d_by_level"]) + "]",
                        tr["slope"], tr["unit"], tr["ci"][0], tr["ci"][1],
                        tr["p_one_sided_grow"]))
    dd = diff_in_diff(rows, ix, cell_hi=cell_name(6, 4))
    if dd:
        lines.append(
            "- **difference-in-differences** (%s vs %s, n=%d): "
            "Δd = %+.3f, 95%% CI [%+.3f, %+.3f], one-sided p(grow) = %.4f"
            % (dd["cell_hi"], dd["cell_lo"], dd["n_units"], dd["dd"],
               dd["ci"][0], dd["ci"][1], dd["p_one_sided_grow"]))
    lines.append("")
    return "\n".join(lines)


def latex_report(rows, ix):
    cells = cells_present(rows)
    names = [cell_name(*c) for c in cells]
    esc = lambda s: s.replace("_", r"\_")
    out = ["% Campaign B: sensor-noise sensitivity (auto-generated)",
           "% wide grids use table* + resizebox for the double-column class",
           r"\begin{table*}", r"\centering",
           r"\caption{Success rate vs sensor noise "
           r"(depth $\sigma$ mm / mask erosion px), pooled over 5 tasks "
           r"$\times$ 30 paired seeds ($n=150$ per cell).}",
           r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{l" + "c" * len(names) + "}", r"\toprule",
           "method & " + " & ".join(esc(n) for n in names) + r" \\",
           r"\midrule"]
    grid = success_grid(rows)
    for m in METHODS:
        cellsr = []
        for c in names:
            kn = grid[m].get(c)
            cellsr.append("%.2f" % (kn[0] / kn[1]) if kn else "--")
        out.append(esc(m) + " & " + " & ".join(cellsr) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]

    # marginal axes (compact paper table)
    ms = marginal_success(rows, ix, "sigma")
    me = marginal_success(rows, ix, "erosion")
    lv_s = sorted({lv for m in METHODS for lv in ms[m]})
    lv_e = sorted({lv for m in METHODS for lv in me[m]})
    out += [r"\begin{table}", r"\centering",
            r"\caption{Marginal sensor-noise axes (each entry pooled over the "
            r"other axis, 5 tasks $\times$ 30 paired seeds).}",
            r"\begin{tabular}{l" + "c" * (len(lv_s) + len(lv_e)) + "}",
            r"\toprule",
            r"& \multicolumn{%d}{c}{depth $\sigma$ (mm)} & "
            r"\multicolumn{%d}{c}{mask erosion (px)} \\" % (len(lv_s),
                                                            len(lv_e)),
            "method & " + " & ".join(["%g" % lv for lv in lv_s]
                                     + ["%d" % lv for lv in lv_e]) + r" \\",
            r"\midrule"]
    for m in METHODS:
        vals = ["%.2f" % (ms[m][lv][0] / ms[m][lv][1]) if lv in ms[m] else "--"
                for lv in lv_s]
        vals += ["%.2f" % (me[m][lv][0] / me[m][lv][1]) if lv in me[m] else "--"
                 for lv in lv_e]
        out.append(esc(m) + " & " + " & ".join(vals) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    mc = mcnemar_by_cell(rows, ix)
    adj = holm_bonferroni([r["p"] for r in mc]) if mc else []
    out += [r"\begin{table}", r"\centering",
            r"\caption{Registration contribution under sensor noise: exact "
            r"McNemar ours\_full vs ours\_noreg per noise cell "
            r"(paired over task$\times$seed, Holm over the 12 cells).}",
            r"\begin{tabular}{lccccccc}", r"\toprule",
            r"cell & $n$ & full & noreg & $\Delta$ & $b/c$ & $p$ & "
            r"$p_{\mathrm{holm}}$ \\",
            r"\midrule"]
    for r, pa in zip(mc, adj):
        out.append("%s & %d & %d & %d & $%+.3f$ & %d/%d & %.3g & %.3g \\\\"
                   % (esc(r["cell"]), r["n"], r["k_a"], r["k_b"], r["diff"],
                      r["b"], r["c"], r["p"], pa))
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # median rotation error per cell (the flat-vs-degrading curve)
    out += [r"\begin{table*}", r"\centering",
            r"\caption{Median rotation error (deg) per noise cell: bounded "
            r"registration keeps ours\_full flat while the Procrustes-only "
            r"ablation degrades.}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{l" + "c" * len(names) + "}", r"\toprule",
            "method & " + " & ".join(esc(n) for n in names) + r" \\",
            r"\midrule"]
    for m in METHODS:
        med = median_by_cell(rows, "rot_err", m)
        out.append(esc(m) + " & " + " & ".join(
            "%.1f" % med[c] if c in med else "--" for c in names) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(out)


CSV_KEYS = ["task", "seed", "method", "cell", "sigma_mm", "erode_px",
            "success", "rot_err", "trans_err", "chamfer_after",
            "failure_stage", "noise_seed", "source_file"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--prefix", default="campaign_b")
    args = ap.parse_args(argv)

    rows = load_rows(args.root)
    if not rows:
        raise SystemExit("no rollouts under %s" % args.root)
    ix = index(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir,
                           "%s_rollouts.csv" % args.prefix), "w",
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
    n_cells = len(cells_present(rows))
    print("wrote %s.{md,tex,_rollouts.csv} to %s (%d rollouts, %d cells)"
          % (args.prefix, args.out_dir, len(rows), n_cells))
    print(md)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
