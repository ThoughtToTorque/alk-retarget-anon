#!/usr/bin/env python
"""Mini benchmark: the paired multi-method comparison, at demo scale.

    python demo/mini_benchmark.py                  # all 5 tasks x 4 paired seeds
    python demo/mini_benchmark.py --tasks nut_loosen --seeds 20

Runs the SAME driver the paper's campaigns use (pipeline/campaign.py) on a
handful of paired scene-pair seeds, then aggregates with the same statistics
code (stats/aggregate.py, stats/tests.py) and prints per-task and pooled
success rates with Wilson 95% CIs and the exact paired McNemar p-value of
every method against the full method.

THIS IS A SCALED-DOWN ILLUSTRATION.  The paper reports 60 paired seeds per
task (seeds 1000..1059) over 5 tasks and 11 methods; at 4 seeds and 3
methods the confidence intervals are far too wide to support any claim.  Do
not quote these numbers -- see docs/REPRODUCE.md for the real commands.

The default runs all 5 tasks so nothing is cherry-picked; it takes roughly
8 minutes from a fresh checkout (60 rollouts, plus recording the 5 scripted
demonstrations once) and is resumable, so an interrupted run continues where
it stopped.

No GPU, no VLM, no network: the discrete variables are answered by the
simulator oracle, exactly as in the paper's large-N campaigns.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_TASKS = ["nut_loosen", "rim_grasp", "pour", "box_open",
                 "cap_twist"]
DEFAULT_METHODS = ["ours_full", "ours_noreg", "icp"]
DEFAULT_SEEDS = 4
SEED_START = 1000

PAPER_N = 60

METHOD_BLURB = {
    "ours_full": "full method (ALK Procrustes + bounded registration + "
                 "grasp correction)",
    "ours_noreg": "ablation: closed-form ALK Procrustes only, no bounded "
                  "registration",
    "ours_nocorr": "ablation: no grasp-point translation correction",
    "icp": "baseline B1: point-to-point ICP on the same object clouds "
           "(identity init)",
    "icp_centroid": "baseline B1: ICP with centroid-aligned init",
    "moka_oracle": "baseline B2: MOKA-style pixel marking (oracle mark)",
    "rekep": "baseline B3: ReKep-style keypoint-constraint optimisation",
}


GL_HINT = ("MuJoCo needs an EGL/osmesa GL stack; see the headless-rendering "
           "notes in the README, or run `python demo/quickstart.py --help`.")

_GL_MARKERS = ("egl", "opengl", "gl context", "glfw", "osmesa", "mujoco_gl",
               "framebuffer", "libgl", "display")


def _die(msg, hint=""):
    sys.stderr.write("\nmini_benchmark: %s\n%s\n" % (msg, hint))
    raise SystemExit(2)


def _looks_like_gl_error(exc):
    text = ("%s %s" % (type(exc).__name__, exc)).lower()
    return any(m in text for m in _GL_MARKERS)


def _import_stack():
    try:
        import mujoco  # noqa: F401
        import robosuite  # noqa: F401
    except ImportError as e:
        _die("robosuite / mujoco are not installed (%s)" % e,
             "From the repository root:  pip install -e .")
    except Exception as e:  # noqa: BLE001 -- mujoco picks its GL backend at
        # import time, so a bad/missing GL stack surfaces here
        if _looks_like_gl_error(e):
            _die("MuJoCo could not set up rendering (%s: %s)"
                 % (type(e).__name__, e), GL_HINT)
        raise
    try:
        from pipeline import campaign  # noqa: F401
        from stats import aggregate, tests  # noqa: F401
    except ImportError as e:
        _die("could not import the repository packages (%s)" % e,
             "Run this from a checkout of the repository.")


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def _vectors(rows, task, method, seeds):
    """0/1 success vector for (task, method), aligned to `seeds`."""
    from stats import aggregate as agg
    by_seed = {}
    for r in agg.select(rows, task=task, method=method):
        by_seed[int(r["seed"])] = 1 if r["success"] else 0
    return [by_seed.get(s) for s in seeds]


def _report(rows, tasks, methods, seeds, elapsed):
    from stats import tests as st

    print("")
    print("=" * 78)
    print("mini benchmark -- %d task(s) x %d paired seeds x %d methods"
          % (len(tasks), len(seeds), len(methods)))
    print("=" * 78)
    print("")
    for m in methods:
        print("  %-13s %s" % (m, METHOD_BLURB.get(m, "")))
    print("")

    vec = {}
    missing = []
    for t in tasks:
        for m in methods:
            v = _vectors(rows, t, m, seeds)
            if any(x is None for x in v):
                missing.append("%s/%s" % (t, m))
                v = [x for x in v if x is not None]
            vec[(t, m)] = v

    ref = methods[0]
    header = ("%-12s %-13s %5s %7s %-18s %10s"
              % ("task", "method", "k/n", "rate", "Wilson 95% CI",
                 "McNemar p"))
    print(header)
    print("-" * len(header))
    for t in tasks:
        for m in methods:
            v = vec[(t, m)]
            k, n = sum(v), len(v)
            lo, hi = st.wilson_ci(k, n) if n else (float("nan"),) * 2
            if m == ref or not n:
                pcol = "  (ref)" if m == ref else "     -"
            else:
                a, b = vec[(t, ref)], v
                if len(a) == len(b):
                    r = st.mcnemar_from_vectors(a, b)
                    pcol = "%10.3f" % r.pvalue
                else:
                    pcol = "     -"
            print("%-12s %-13s %2d/%-2d %6.2f  [%.2f, %.2f]      %s"
                  % (t, m, k, n, (float(k) / n) if n else float("nan"),
                     lo, hi, pcol))
        print("-" * len(header))

    # pooled over tasks (pairs are (task, seed), so pooling keeps the pairing)
    print("")
    print("pooled over %s" % ", ".join(tasks))
    print(header)
    print("-" * len(header))
    pooled = {}
    for m in methods:
        pooled[m] = [x for t in tasks for x in vec[(t, m)]]
    for m in methods:
        v = pooled[m]
        k, n = sum(v), len(v)
        lo, hi = st.wilson_ci(k, n) if n else (float("nan"),) * 2
        if m == ref:
            pcol = "  (ref)"
        else:
            a = pooled[ref]
            if len(a) == len(v):
                r = st.mcnemar_from_vectors(a, v)
                pcol = "%10.3f  (b=%d, c=%d discordant)" % (r.pvalue, r.b,
                                                            r.c)
            else:
                pcol = "     -"
        print("%-12s %-13s %2d/%-2d %6.2f  [%.2f, %.2f]      %s"
              % ("(all)", m, k, n, (float(k) / n) if n else float("nan"),
                 lo, hi, pcol))
    print("-" * len(header))
    if missing:
        print("")
        print("WARNING: missing rollout records for %s (crashed rollouts are "
              "recorded as failures; a missing FILE means the run was "
              "interrupted -- rerun, it resumes)" % ", ".join(missing))
    if "pour" in tasks:
        print("")
        print("NOTE on `pour`: it is the ill-conditioned task -- the ALK of "
              "the elongated")
        print("  block is nearly rank-1, so the closed-form rotation about "
              "the long axis is")
        print("  weakly determined.  Its rate at these DEFAULT settings is "
              "expected to be")
        print("  the lowest of the five; the conditioning-adaptive "
              "registration (add")
        print("  --methods ours_full and run pipeline.campaign --adaptive) is "
              "what addresses")
        print("  it.  See docs/REPRODUCE.md, campaigns A-adaptive and G.")
    print("")
    print("SCALE WARNING")
    print("  The paper's numbers come from %d paired seeds per task "
          "(seeds %d..%d)" % (PAPER_N, SEED_START, SEED_START + PAPER_N - 1))
    print("  over 5 tasks and 11 methods.  This run used %d seeds per task, "
          "so the" % len(seeds))
    print("  intervals above are wide and the p-values under-powered: it is "
          "an")
    print("  ILLUSTRATION of the protocol, not a reproduction of the "
          "results.")
    print("  See docs/REPRODUCE.md for the commands that produce the paper's "
          "tables.")
    print("")
    print("wall clock: %.1f min" % (elapsed / 60.0))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                   help="number of paired seeds per task, counted from %d "
                        "(default: %d)" % (SEED_START, DEFAULT_SEEDS))
    p.add_argument("--seed-start", type=int, default=SEED_START)
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                   help="first entry is the reference of the paired tests "
                        "(default: %s)" % " ".join(DEFAULT_METHODS))
    p.add_argument("--tier", default="medium",
                   help="randomization tier (default: medium, the paper's)")
    p.add_argument("--data-root", default=None,
                   help="scene-pair cache (default: <repo>/data_mini)")
    p.add_argument("--out-root", default=None,
                   help="rollout records (default: <repo>/demo_out/"
                        "mini_benchmark)")
    p.add_argument("--report-only", action="store_true",
                   help="skip execution, just aggregate existing records")
    args = p.parse_args(argv)

    _import_stack()
    from simtasks import envs
    from pipeline import campaign
    from stats import aggregate as agg

    for t in args.tasks:
        if t not in envs.TASKS:
            _die("unknown task %r (have %s)"
                 % (t, ", ".join(sorted(envs.TASKS))))
    for m in args.methods:
        if m not in campaign.KNOWN_METHODS:
            _die("unknown method %r (have %s)"
                 % (m, ", ".join(campaign.KNOWN_METHODS)))

    data_root = args.data_root or os.path.join(REPO_ROOT, "data_mini")
    out_root = args.out_root or os.path.join(REPO_ROOT, "demo_out",
                                             "mini_benchmark")
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    print("mini_benchmark: tasks=%s seeds=%d..%d methods=%s tier=%s"
          % (",".join(args.tasks), seeds[0], seeds[-1],
             ",".join(args.methods), args.tier))
    print("  scene pairs generated from their seeds into %s" % data_root)
    print("  rollout records into %s" % out_root)
    print("  %d rollouts total; existing records are reused (resumable)"
          % (len(args.tasks) * len(seeds) * len(args.methods)))
    t0 = time.time()
    if not args.report_only:
        for task in args.tasks:
            print("")
            print("--- %s ---" % task)
            try:
                campaign.run_task(task, seeds, args.methods, args.tier,
                                  data_root, out_root)
            except Exception as e:  # noqa: BLE001
                if _looks_like_gl_error(e):
                    _die("headless rendering failed (%s: %s)"
                         % (type(e).__name__, e), GL_HINT)
                raise
    rows = agg.load_rollouts(out_root)
    _report(rows, args.tasks, args.methods, seeds, time.time() - t0)
    print("records: %s" % out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
