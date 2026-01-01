"""Phase-4 Campaign C runner: VLM error injection (conditional-decoupling
test; paper Section 4.1).

Design (frozen in docs/REPRODUCE.md):
  ours_full (default flags: registration ON, correction ON, adaptive OFF,
  oracle discrete) x 5 tasks x injection rate rho in {0, 0.1, 0.2, 0.5, 1.0}
  x 40 seeds {1000..1039}, medium tier, on the SAME scene pairs as
  Campaign A (data_medium/).  The rho=0 cell REUSES Campaign A's
  ours_full rollouts for those seeds (identical configuration; noted in
  the findings) -- only rho > 0 is run here.

Injection semantics: pipeline.inject (joint corruption of the target-side
discrete choice with probability rho per rollout, seeded per
(task, seed, rho); demo side stays oracle).  The injector goes through
retarget_runner.run_pair's optional ``inject=`` hook; nothing else in the
shared pipeline is modified.

Per-rollout JSON: results/campaign_c/<task>/<seed>/rollout_ours_full_<tag>.json
with top-level ``injected`` / ``injected_error_rate`` / ``discrete_correct``
(stats/aggregate.py-compatible) plus the full ``injection`` record.
Resumable: existing rollout files are skipped.

Usage (one process may run several tasks sequentially; <=3 concurrent
processes while Campaign B shares the GPU):

  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m pipeline.campaign_c run \
      --tasks nut_loosen cap_twist --seed-start 1000 --seed-end 1039
  python -m pipeline.campaign_c report
"""
import argparse
import json
import os
import time
import traceback

from simtasks import envs
from pipeline import campaign, inject
from pipeline import retarget_runner as rr

SIMBENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_ROOT = os.path.join(SIMBENCH, "results", "campaign_c")
CAMPAIGN_A_ROOT = os.path.join(SIMBENCH, "results", "campaign_a")
TABLES_DIR = os.path.join(SIMBENCH, "results", "tables")

TASKS = ["nut_loosen", "cap_twist", "rim_grasp", "box_open", "pour"]
TIER = "medium"
SEED_START, SEED_END = 1000, 1039
RHOS = list(inject.RHOS)  # (0.1, 0.2, 0.5, 1.0); rho=0 <- campaign_a


def rho_tag(rho):
    return ("rho%.2f" % rho).replace(".", "p")


def result_path(out_root, task, seed, rho):
    return os.path.join(out_root, task, str(seed),
                        "rollout_ours_full_%s.json" % rho_tag(rho))


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def run_task(task, seeds, rhos, data_root, out_root):
    campaign.ensure_demo_assets(task, data_root)
    env = campaign.make_tier_env(task, TIER)
    n_done = 0
    try:
        for seed in seeds:
            for rho in rhos:
                path = result_path(out_root, task, seed, rho)
                if os.path.exists(path):
                    continue
                t0 = time.time()
                try:
                    injector = inject.make_injector(task, seed, rho)
                    r = rr.run_pair(task, seed, env=env, data_root=data_root,
                                    save=False, inject=injector)
                except Exception as e:
                    traceback.print_exc()
                    r = {"task": task, "seed": seed, "success": False,
                         "failure_stage": "exception", "error": repr(e),
                         "injection": {"rho": float(rho), "injected": None},
                         "time_s": round(time.time() - t0, 1)}
                r["method"] = "ours_full"
                r["variant"] = "ours_full_%s" % rho_tag(rho)
                r["tier"] = TIER
                inj = r.get("injection") or {}
                r["rho"] = float(rho)
                r["injected"] = inj.get("injected")
                r["injected_error_rate"] = float(rho)
                if inj.get("injected") is not None:
                    r["discrete_correct"] = not bool(inj["injected"])
                campaign._write_json(path, r)
                n_done += 1
                print("[%s %d rho=%.2f] injected=%s success=%s stage=%-14s "
                      "rot=%s trans=%s (%.1fs)"
                      % (task, seed, rho, r.get("injected"), r.get("success"),
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
# report
# ---------------------------------------------------------------------------

def load_rows(out_root=DEFAULT_OUT_ROOT, campaign_a_root=CAMPAIGN_A_ROOT,
              seeds=None):
    """Campaign C tidy rows: rho>0 rollouts from out_root + the rho=0 cell
    reused from Campaign A ours_full (tagged injected_error_rate=0,
    discrete_correct=1: oracle answers, no injection)."""
    from stats import aggregate as agg
    seeds = seeds if seeds is not None else range(SEED_START, SEED_END + 1)
    rows = [r for r in agg.load_rollouts(out_root)
            if r.get("method") == "ours_full"]
    for task in TASKS:
        for seed in seeds:
            p = os.path.join(campaign_a_root, task, str(seed),
                             "rollout_ours_full.json")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                rec = json.load(f)
            row = agg.row_from_record(rec, source_file=p)
            row["injected_error_rate"] = 0.0
            row["discrete_correct"] = 1
            row["variant"] = "ours_full_rho0p00"
            rows.append(row)
    return rows


def _wilson(k, n, alpha=0.05):
    from stats.tests import wilson_ci
    return {"k": int(k), "n": int(n),
            "rate": k / float(n) if n else None,
            "ci": wilson_ci(int(k), int(n), alpha) if n else None}


def _quantiles(vals, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    import numpy as np
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    if v.size == 0:
        return None
    out = {("p%d" % int(100 * q)): float(np.quantile(v, q)) for q in qs}
    out["mean"] = float(v.mean())
    out["n"] = int(v.size)
    return out


def _best_threshold(clean, injected):
    """Best single-threshold separation of the two error samples: threshold
    t maximizing balanced accuracy of (x > t -> injected)."""
    import numpy as np
    c = np.asarray([x for x in clean if x is not None], dtype=float)
    i = np.asarray([x for x in injected if x is not None], dtype=float)
    if c.size == 0 or i.size == 0:
        return None
    cand = np.unique(np.concatenate([c, i]))
    best = None
    for t in cand:
        acc = 0.5 * ((c <= t).mean() + (i > t).mean())
        if best is None or acc > best["balanced_acc"]:
            best = {"threshold": float(t), "balanced_acc": float(acc)}
    best["clean_le_thr"] = float((c <= best["threshold"]).mean())
    best["injected_gt_thr"] = float((i > best["threshold"]).mean())
    return best


def analyze(rows):
    """Per-task + pooled: success-vs-rho points, OLS linearity, P(s|clean)
    vs P(s|injected), slope check, and the error-magnitude regime split."""
    from stats import aggregate as agg
    from stats.report import decoupling_analysis

    out = {"per_task": {}, "pooled": None}
    for scope, sub in ([(t, agg.select(rows, task=t)) for t in TASKS]
                       + [("ALL", rows)]):
        if not sub:
            continue
        dec = decoupling_analysis(sub)
        entry = {"decoupling": dec}
        gc, gw = dec.get("given_correct"), dec.get("given_wrong")
        if dec.get("fit") and gc and gw:
            entry["slope_predicted"] = -(gc["rate"] - gw["rate"])
            entry["slope_fit"] = dec["fit"]["slope"]
        # error-regime signature (mapping errors; degenerate-ALK rollouts
        # carry no rot/trans error -> counted separately)
        clean = [r for r in sub if r.get("discrete_correct") == 1]
        inj = [r for r in sub if r.get("discrete_correct") == 0]
        entry["regimes"] = {
            "clean": {"rot_deg": _quantiles(agg.column(clean, "rot_err")),
                      "trans_mm": _quantiles(
                          [1e3 * v for v in agg.column(clean, "trans_err")]),
                      "map_bad": _map_bad(clean)},
            "injected": {"rot_deg": _quantiles(agg.column(inj, "rot_err")),
                         "trans_mm": _quantiles(
                             [1e3 * v for v in agg.column(inj, "trans_err")]),
                         "map_bad": _map_bad(inj),
                         "n_alk_degenerate": sum(
                             1 for r in inj
                             if r.get("failure_stage") == "alk_degenerate")},
            "sep_rot": _best_threshold(agg.column(clean, "rot_err"),
                                       agg.column(inj, "rot_err")),
            "sep_trans": _best_threshold(agg.column(clean, "trans_err"),
                                         agg.column(inj, "trans_err")),
        }
        if scope == "ALL":
            out["pooled"] = entry
        else:
            out["per_task"][scope] = entry
    return out


def _map_bad(sub, rot_thr=20.0, trans_thr=0.05):
    """Fraction of rollouts past the pipeline's own mapping-failure
    heuristic (rot > 20 deg or trans > 5 cm); degenerate-ALK rollouts
    (no T_map at all) count as bad."""
    n = k = 0
    for r in sub:
        rot, tr = r.get("rot_err"), r.get("trans_err")
        if rot is None and tr is None:
            if r.get("failure_stage") == "alk_degenerate":
                n += 1
                k += 1
            continue
        n += 1
        if (rot is not None and rot > rot_thr) or \
           (tr is not None and tr > trans_thr):
            k += 1
    return _wilson(k, n) if n else None


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def _fmt_ci(c):
    if not c or c.get("rate") is None:
        return "--"
    return "%d/%d = %.2f [%.2f, %.2f]" % (
        c["k"], c["n"], c["rate"], c["ci"][0], c["ci"][1])


def tables(res):
    """(markdown, latex) for results/tables/campaign_c.{md,tex}."""
    scopes = TASKS + ["ALL"]
    rhos = [0.0] + list(RHOS)

    md = ["# Campaign C — VLM error injection (success vs rho)",
          "",
          "rho=0 cell reused from Campaign A ours_full (same seeds "
          "%d..%d, identical configuration)." % (SEED_START, SEED_END),
          "",
          "| task | " + " | ".join("rho=%.1f" % r for r in rhos)
          + " | slope (fit) | slope (pred) | R^2 "
            "| P(s given clean) | P(s given injected) |",
          "|---|" + "---|" * (len(rhos) + 5)]
    tex = [r"\begin{tabular}{l" + "c" * (len(rhos) + 5) + "}",
           r"\toprule",
           "task & " + " & ".join(r"$\rho{=}%.1f$" % r for r in rhos)
           + r" & slope & pred.\ & $R^2$ & $P(s\mid\text{clean})$"
             r" & $P(s\mid\text{inj})$ \\",
           r"\midrule"]
    for scope in scopes:
        e = res["pooled"] if scope == "ALL" else res["per_task"].get(scope)
        if e is None:
            continue
        dec = e["decoupling"]
        pts = {p["error_rate"]: p for p in (dec.get("points") or [])}
        cells = []
        for r in rhos:
            p = pts.get(r)
            cells.append("%d/%d=%.2f" % (p["k"], p["n"], p["rate"])
                         if p else "--")
        fit = dec.get("fit") or {}
        gc, gw = dec.get("given_correct"), dec.get("given_wrong")
        md.append("| %s | %s | %.3f | %.3f | %.4f | %s | %s |" % (
            scope, " | ".join(cells),
            fit.get("slope", float("nan")),
            e.get("slope_predicted", float("nan")),
            fit.get("r2", float("nan")),
            _fmt_ci(gc), _fmt_ci(gw)))
        tex.append(r"%s & %s & $%.3f$ & $%.3f$ & %.4f & %s & %s \\" % (
            scope.replace("_", r"\_"), " & ".join("$%s$" % c for c in cells),
            fit.get("slope", float("nan")),
            e.get("slope_predicted", float("nan")),
            fit.get("r2", float("nan")),
            _fmt_ci(gc), _fmt_ci(gw)))
    tex += [r"\bottomrule", r"\end{tabular}"]

    md += ["", "## Error-magnitude regimes (clean vs injected, pooled)", ""]
    reg = res["pooled"]["regimes"]
    md += ["| group | rot err deg (med [p10, p90]) | trans err mm "
           "(med [p10, p90]) | frac past mapping heuristic |",
           "|---|---|---|---|"]
    for g in ("clean", "injected"):
        r_, t_ = reg[g]["rot_deg"], reg[g]["trans_mm"]
        md.append("| %s | %.1f [%.1f, %.1f] | %.1f [%.1f, %.1f] | %s |" % (
            g, r_["p50"], r_["p10"], r_["p90"],
            t_["p50"], t_["p10"], t_["p90"], _fmt_ci(reg[g]["map_bad"])))
    for name, key, unit in (("rotation", "sep_rot", "deg"),
                            ("translation", "sep_trans", "m")):
        s = reg[key]
        if s:
            md.append("")
            md.append("Best single-threshold %s separation: %.3g %s, "
                      "balanced acc %.3f (clean below: %.3f, injected "
                      "above: %.3f)." % (name, s["threshold"], unit,
                                         s["balanced_acc"], s["clean_le_thr"],
                                         s["injected_gt_thr"]))
    return "\n".join(md) + "\n", "\n".join(tex) + "\n"


def report(out_root=DEFAULT_OUT_ROOT):
    from stats import aggregate as agg
    rows = load_rows(out_root)
    res = analyze(rows)
    os.makedirs(TABLES_DIR, exist_ok=True)
    md, tex = tables(res)
    with open(os.path.join(TABLES_DIR, "campaign_c.md"), "w") as f:
        f.write(md)
    with open(os.path.join(TABLES_DIR, "campaign_c.tex"), "w") as f:
        f.write(tex)
    agg.write_csv(rows, os.path.join(TABLES_DIR, "campaign_c_rollouts.csv"))
    campaign._write_json(os.path.join(out_root, "summary.json"),
                         {"n_rows": len(rows), "analysis": res})
    print(md)
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["run", "report"])
    p.add_argument("--tasks", nargs="*", default=TASKS,
                   choices=sorted(envs.TASKS.keys()))
    p.add_argument("--rhos", nargs="*", type=float, default=RHOS)
    p.add_argument("--seed-start", type=int, default=SEED_START)
    p.add_argument("--seed-end", type=int, default=SEED_END, help="inclusive")
    p.add_argument("--data-root", default=campaign.TIER_DATA_ROOT[TIER])
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    args = p.parse_args(argv)

    if args.phase == "report":
        report(args.out_root)
        return 0
    seeds = list(range(args.seed_start, args.seed_end + 1))
    t0 = time.time()
    n = 0
    for task in args.tasks:
        n += run_task(task, seeds, args.rhos, args.data_root, args.out_root)
    print("done: %s, %d new rollouts in %.1f min"
          % (args.tasks, n, (time.time() - t0) / 60.0), flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
