"""CLI driver: per-rollout JSON tree -> paper tables (md + LaTeX).

    python -m stats.make_tables results/campaign_a --prefix campaign_a \
        [--ours ours_full] [--out-dir results/tables]

Writes <out-dir>/<prefix>_success.{md,tex}, <prefix>_errors.{md,tex}, plus
<prefix>_rollouts.csv (the tidy table) and, when discrete-correctness labels
exist, <prefix>_decoupling.md.  Pure stats.aggregate + stats.report.
"""
import argparse
import os

from stats import aggregate as agg
from stats import report

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", help="directory tree of rollout_*.json files")
    ap.add_argument("--ours", default="ours_full")
    ap.add_argument("--prefix", default="campaign_a")
    ap.add_argument("--out-dir", default=os.path.join(RESULTS_DIR, "tables"))
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)

    rows = agg.load_rollouts(args.root)
    if not rows:
        raise SystemExit("no rollout_*.json under %s" % args.root)
    os.makedirs(args.out_dir, exist_ok=True)

    def write(name, text):
        path = os.path.join(args.out_dir, name)
        with open(path, "w") as f:
            f.write(text + "\n")
        print("wrote %s" % path)

    agg.write_csv(rows, os.path.join(args.out_dir,
                                     "%s_rollouts.csv" % args.prefix))
    st = report.make_success_table(rows, ours=args.ours, alpha=args.alpha)
    write("%s_success.md" % args.prefix, report.success_table_markdown(st))
    write("%s_success.tex" % args.prefix, report.success_table_latex(st))
    et = report.make_error_table(rows)
    write("%s_errors.md" % args.prefix, report.error_table_markdown(et))
    write("%s_errors.tex" % args.prefix, report.error_table_latex(et))
    dec = report.decoupling_analysis(rows, alpha=args.alpha)
    if dec["n_labeled"]:
        write("%s_decoupling.md" % args.prefix, report.decoupling_markdown(dec))
    print("%d rollouts, %d tasks, %d methods"
          % (len(rows), len(agg.unique(rows, "task")),
             len(agg.unique(rows, "method"))))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
