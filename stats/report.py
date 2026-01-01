"""Table generation for the paper (no figures here; tables only).

Produces, from the tidy rows of stats.aggregate:

  (a) main success-rate table   -- methods x tasks, k/n + Wilson CI,
      pairwise-vs-ours exact McNemar p (paired via shared seeds; Fisher
      exact fallback when seeds do not align), Holm-corrected within each
      task's family of vs-ours comparisons.  Markdown + LaTeX (booktabs).
  (b) error-decomposition table -- rotation / translation error
      distributions and failure-stage attribution counts.
  (c) decoupling analysis       -- P(success | discrete correct),
      P(success | discrete wrong), and the linearity fit of success rate
      vs injected discrete-error rate (OLS + R^2).  Under the decoupling
      claim, success(e) = (1-e)*P(s|correct) + e*P(s|wrong) is LINEAR in
      the injected error rate e, so a high R^2 with slope
      -(P(s|correct) - P(s|wrong)) is direct evidence for the claim.

Pure numpy + stats.tests; python 3.8 compatible; no scipy required.
"""

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from stats.tests import (wilson_ci, mcnemar_from_vectors, fisher_exact,
                             holm_bonferroni)
    from stats import aggregate as agg
except ImportError:  # pragma: no cover - script-style import
    from tests import (wilson_ci, mcnemar_from_vectors, fisher_exact,  # type: ignore
                       holm_bonferroni)
    import aggregate as agg  # type: ignore

__all__ = [
    "make_success_table",
    "success_table_markdown",
    "success_table_latex",
    "make_error_table",
    "error_table_markdown",
    "error_table_latex",
    "linear_fit",
    "decoupling_analysis",
    "decoupling_markdown",
]

POOLED = "ALL"


# ---------------------------------------------------------------------------
# (a) main success-rate table
# ---------------------------------------------------------------------------

def _success_by_seed(rows: List[Dict[str, Any]]) -> Dict[Any, int]:
    """(task, seed) -> success, for pairing.  Rows must be single-method."""
    out = {}
    for r in rows:
        if r.get("success") is None:
            continue
        out[(r.get("task"), r.get("seed"))] = int(r["success"])
    return out


def _compare_vs_ours(rows_m: List[Dict[str, Any]],
                     rows_ours: List[Dict[str, Any]]) -> Dict[str, Any]:
    """One method-vs-ours comparison.  Paired McNemar when the seed sets
    align (the protocol's shared-seed design); Fisher exact otherwise."""
    m = _success_by_seed(rows_m)
    o = _success_by_seed(rows_ours)
    shared = sorted(set(m) & set(o))
    if shared and len(shared) == len(m) == len(o):
        res = mcnemar_from_vectors([o[k] for k in shared],
                                   [m[k] for k in shared])
        return {"test": "mcnemar", "pvalue": res.pvalue,
                "detail": "b=%d,c=%d" % (res.b, res.c)}
    # unpaired fallback
    k_o, n_o = sum(o.values()), len(o)
    k_m, n_m = sum(m.values()), len(m)
    res = fisher_exact([[k_o, n_o - k_o], [k_m, n_m - k_m]])
    return {"test": "fisher", "pvalue": res.pvalue,
            "detail": "OR=%.3g" % res.odds_ratio}


def make_success_table(rows: List[Dict[str, Any]], ours: str = "full",
                       alpha: float = 0.05) -> Dict[str, Any]:
    """Build the main success-rate table structure.

    For every task (plus a POOLED row over all tasks) and every method:
    k, n, rate, Wilson (1-alpha) CI; for every method != ``ours``:
    p-value vs ours (McNemar if paired else Fisher) and the Holm-adjusted
    p within that task's family of vs-ours comparisons.

    Returns {'ours', 'alpha', 'tasks', 'methods',
             'cells': {(task, method): {...}}}.
    """
    rows = [r for r in rows if r.get("success") is not None]
    tasks = agg.unique(rows, "task")
    methods = agg.unique(rows, "method")
    if ours in methods:  # ours first, for table layout
        methods = [ours] + [m for m in methods if m != ours]
    cells: Dict[Any, Dict[str, Any]] = {}
    for task in tasks + [POOLED]:
        trows = rows if task == POOLED else agg.select(rows, task=task)
        ours_rows = [r for r in trows if r.get("method") == ours]
        pvals, pkeys = [], []
        for method in methods:
            mrows = [r for r in trows if r.get("method") == method]
            if not mrows:
                continue
            succ = agg.column(mrows, "success")
            k, n = int(sum(succ)), len(succ)
            lo, hi = wilson_ci(k, n, alpha)
            cell = {"k": k, "n": n, "rate": k / float(n),
                    "ci": (lo, hi), "pvalue": None, "p_holm": None,
                    "test": None, "detail": None}
            if method != ours and ours_rows:
                cmp_ = _compare_vs_ours(mrows, ours_rows)
                cell.update(cmp_)
                pvals.append(cmp_["pvalue"])
                pkeys.append((task, method))
            cells[(task, method)] = cell
        if pvals:  # Holm within this task's family of vs-ours comparisons
            for key, p_adj in zip(pkeys, holm_bonferroni(pvals)):
                cells[key]["p_holm"] = p_adj
    return {"ours": ours, "alpha": alpha, "tasks": tasks + [POOLED],
            "methods": methods, "cells": cells}


def _fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "--"
    if p < 1e-4:
        return "<1e-4"
    return "%.4f" % p


def _fmt_cell(cell: Dict[str, Any]) -> str:
    return "%d/%d = %.2f [%.2f, %.2f]" % (
        cell["k"], cell["n"], cell["rate"], cell["ci"][0], cell["ci"][1])


def success_table_markdown(table: Dict[str, Any]) -> str:
    """Long-format markdown rendering of :func:`make_success_table`."""
    conf = int(round(100 * (1 - table["alpha"])))
    lines = [
        "| Task | Method | k/n = rate [Wilson %d%% CI] | test | p vs %s | p (Holm) |"
        % (conf, table["ours"]),
        "|---|---|---|---|---:|---:|",
    ]
    for task in table["tasks"]:
        for method in table["methods"]:
            cell = table["cells"].get((task, method))
            if cell is None:
                continue
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                task, method, _fmt_cell(cell),
                cell["test"] or "--", _fmt_p(cell["pvalue"]),
                _fmt_p(cell["p_holm"])))
    return "\n".join(lines)


def _tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def success_table_latex(table: Dict[str, Any]) -> str:
    """booktabs LaTeX rendering of :func:`make_success_table`."""
    conf = int(round(100 * (1 - table["alpha"])))
    out = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Task & Method & $k/n$ [Wilson %d\%% CI] & test & $p$ vs %s & $p$ (Holm) \\"
        % (conf, _tex_escape(table["ours"])),
        r"\midrule",
    ]
    for task in table["tasks"]:
        first = True
        for method in table["methods"]:
            cell = table["cells"].get((task, method))
            if cell is None:
                continue
            out.append(r"%s & %s & $%d/%d = %.2f$ [%.2f, %.2f] & %s & %s & %s \\" % (
                _tex_escape(task) if first else "",
                _tex_escape(method), cell["k"], cell["n"], cell["rate"],
                cell["ci"][0], cell["ci"][1],
                cell["test"] or "--", _fmt_p(cell["pvalue"]),
                _fmt_p(cell["p_holm"])))
            first = False
        if task != table["tasks"][-1]:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# (b) error-decomposition table
# ---------------------------------------------------------------------------

def _dist(vals: Sequence[float]) -> Optional[Dict[str, float]]:
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if v.size == 0:
        return None
    return {"mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "median": float(np.median(v)), "n": int(v.size)}


def make_error_table(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Error decomposition per (task, method): rotation error [deg],
    translation error [mm], chamfer-after [mm] distributions plus
    failure-stage attribution counts.  Continuous quality is reported
    decoupled from the success rate, per the frozen protocol."""
    tasks = agg.unique(rows, "task")
    methods = agg.unique(rows, "method")
    cells = {}
    for task in tasks:
        for method in methods:
            sub = agg.select(rows, task=task, method=method)
            if not sub:
                continue
            stages: Dict[str, int] = {}
            for r in sub:
                st = r.get("failure_stage")
                if st:
                    stages[st] = stages.get(st, 0) + 1
            cells[(task, method)] = {
                "rot_err_deg": _dist(agg.column(sub, "rot_err")),
                "trans_err_mm": _dist([1e3 * v for v in agg.column(sub, "trans_err")]),
                "chamfer_after_mm": _dist([1e3 * v for v in agg.column(sub, "chamfer_after")]),
                "failure_stages": stages,
                "n": len(sub),
            }
    return {"tasks": tasks, "methods": methods, "cells": cells}


def _fmt_dist(d: Optional[Dict[str, float]]) -> str:
    if d is None:
        return "--"
    return "%.2f ± %.2f (med %.2f)" % (d["mean"], d["std"], d["median"])


def _fmt_stages(stages: Dict[str, int]) -> str:
    if not stages:
        return "none"
    return ", ".join("%s: %d" % (k, stages[k]) for k in sorted(stages))


def error_table_markdown(table: Dict[str, Any]) -> str:
    lines = [
        "| Task | Method | rot err [deg] | trans err [mm] | chamfer after [mm] | failures (stage: count) |",
        "|---|---|---|---|---|---|",
    ]
    for task in table["tasks"]:
        for method in table["methods"]:
            cell = table["cells"].get((task, method))
            if cell is None:
                continue
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                task, method, _fmt_dist(cell["rot_err_deg"]),
                _fmt_dist(cell["trans_err_mm"]),
                _fmt_dist(cell["chamfer_after_mm"]),
                _fmt_stages(cell["failure_stages"])))
    return "\n".join(lines)


def error_table_latex(table: Dict[str, Any]) -> str:
    out = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Task & Method & rot.\ err [deg] & trans.\ err [mm] & chamfer [mm] & failure stages \\",
        r"\midrule",
    ]
    for task in table["tasks"]:
        first = True
        for method in table["methods"]:
            cell = table["cells"].get((task, method))
            if cell is None:
                continue
            out.append(r"%s & %s & %s & %s & %s & %s \\" % (
                _tex_escape(task) if first else "",
                _tex_escape(method),
                _tex_escape(_fmt_dist(cell["rot_err_deg"])),
                _tex_escape(_fmt_dist(cell["trans_err_mm"])),
                _tex_escape(_fmt_dist(cell["chamfer_after_mm"])),
                _tex_escape(_fmt_stages(cell["failure_stages"]))))
            first = False
        if task != table["tasks"][-1]:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# (c) decoupling analysis
# ---------------------------------------------------------------------------

def linear_fit(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Ordinary least-squares line y = intercept + slope*x, with R^2.

    Used for the success-vs-injected-error-rate linearity check; at least
    two distinct x values required."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 2 or np.unique(x).size < 2:
        raise ValueError("need >= 2 points with distinct x")
    slope, intercept = np.polyfit(x, y, 1)
    yhat = intercept + slope * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2,
            "n": int(x.size)}


def decoupling_analysis(rows: List[Dict[str, Any]],
                        alpha: float = 0.05) -> Dict[str, Any]:
    """Decoupling evidence from rollouts carrying ``discrete_correct`` flags.

    Estimates P(success | discrete correct) and P(success | discrete wrong)
    with Wilson CIs, and — when rollouts also carry ``injected_error_rate``
    — the OLS fit of per-error-rate success rate vs injected rate e.  Under
    the conditional-decoupling proposition success(e) is linear in e with
    slope -(P(s|correct) - P(s|wrong)), so slope + R^2 quantify the claim.

    Rollouts without flags (e.g. pure-oracle runs) are ignored; the result
    reports how many labeled rollouts were available.
    """
    labeled = [r for r in rows
               if r.get("discrete_correct") is not None
               and r.get("success") is not None]
    out: Dict[str, Any] = {"n_labeled": len(labeled),
                           "given_correct": None, "given_wrong": None,
                           "fit": None, "points": None}
    for flag, name in ((1, "given_correct"), (0, "given_wrong")):
        sub = [r for r in labeled if r["discrete_correct"] == flag]
        if sub:
            succ = agg.column(sub, "success")
            k, n = int(sum(succ)), len(succ)
            out[name] = {"k": k, "n": n, "rate": k / float(n),
                         "ci": wilson_ci(k, n, alpha)}
    # linearity vs injected error rate
    with_rate = [r for r in labeled if r.get("injected_error_rate") is not None]
    rates = sorted({r["injected_error_rate"] for r in with_rate})
    if len(rates) >= 2:
        pts = []
        for e in rates:
            sub = [r for r in with_rate if r["injected_error_rate"] == e]
            succ = agg.column(sub, "success")
            pts.append({"error_rate": float(e), "k": int(sum(succ)),
                        "n": len(succ), "rate": sum(succ) / float(len(succ))})
        out["points"] = pts
        out["fit"] = linear_fit([p["error_rate"] for p in pts],
                                [p["rate"] for p in pts])
    return out


def decoupling_markdown(res: Dict[str, Any]) -> str:
    lines = ["Labeled rollouts (with discrete_correct flag): %d" % res["n_labeled"], ""]
    if res["n_labeled"] == 0:
        lines.append("*(no rollouts carry discrete-correctness flags; run the "
                     "VLM / error-injection phase to populate this analysis)*")
        return "\n".join(lines)
    for name, label in (("given_correct", "P(success | discrete correct)"),
                        ("given_wrong", "P(success | discrete wrong)")):
        c = res[name]
        if c is None:
            lines.append("- %s: no data" % label)
        else:
            lines.append("- %s = %d/%d = %.3f  [%.3f, %.3f]" % (
                label, c["k"], c["n"], c["rate"], c["ci"][0], c["ci"][1]))
    if res["fit"] is not None:
        f = res["fit"]
        lines += [
            "",
            "Linearity of success vs injected discrete-error rate e:",
            "",
            "| e | k/n | success rate |",
            "|---:|---:|---:|",
        ]
        for p in res["points"]:
            lines.append("| %.2f | %d/%d | %.3f |" % (
                p["error_rate"], p["k"], p["n"], p["rate"]))
        lines += [
            "",
            "OLS: success(e) = %.3f + (%.3f)*e,  R^2 = %.4f  (%d levels)" % (
                f["intercept"], f["slope"], f["r2"], f["n"]),
            "Decoupling prediction: slope = -(P(s|correct) - P(s|wrong)); "
            "R^2 near 1 supports the conditional-decoupling proposition.",
        ]
    else:
        lines.append("\n*(no injected_error_rate values; linearity fit skipped)*")
    return "\n".join(lines)
