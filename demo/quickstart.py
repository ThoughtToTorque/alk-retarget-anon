#!/usr/bin/env python
"""Quickstart: one full one-shot retargeting rollout, end to end, in ~1 minute.

    python demo/quickstart.py

No GPU, no VLM, no network, no API key and no downloaded dataset are needed.
The demo/target scene pair is *generated* from its seed (scene pairs are a
deterministic function of the seed, see simtasks/scene_pairs.py), the
scripted expert records the single demonstration, and the discrete
variables Phi are answered by the simulator oracle
(pipeline/oracle.py) -- the same substitution the paper's large-N campaigns
use, so a VLM server is only needed for the VLM-error study
(docs/VLM_SETUP.md).

What it prints
--------------
* the ALK quadruple (C1..C4) on the demo and on the target object,
* the closed-form Procrustes pose T0 and the bounded-registration pose
  T_map, each with its rotation / translation error against ground truth,
* Chamfer distance before and after bounded registration,
* whether the retargeted execution satisfied the task's success checker,
* the path of a PNG showing the demo cloud mapped into the target frame
  before and after registration.
"""
import argparse
import os
import sys
import time

# MuJoCo must be told to use EGL *before* it is imported, so that this file
# can be run as a plain `python demo/quickstart.py` with no env prefix.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_TASK = "nut_loosen"
DEFAULT_SEED = 1000

INSTALL_HINT = """
This demo needs the simulator dependencies.  From the repository root:

    pip install -e .

which installs numpy, scipy, robosuite==1.4.1, mujoco==3.2.3, imageio and
matplotlib.  See the README for the headless-rendering requirements.
"""

EGL_HINT = """
Headless rendering failed.  MuJoCo needs an EGL-capable GL stack:

  * on Linux with an NVIDIA GPU:  the driver already ships libEGL; make sure
    `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` is exported (this script sets both
    by default) and that `libEGL.so.1` is on the loader path;
  * on a machine with no GPU: install osmesa
    (`apt-get install libosmesa6-dev`) and run with
    `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`;
  * on a desktop with a display: `MUJOCO_GL=glfw`.
"""


def _die(msg, hint):
    sys.stderr.write("\nquickstart: %s\n%s\n" % (msg, hint))
    raise SystemExit(2)


_GL_MARKERS = ("egl", "opengl", "gl context", "glfw", "osmesa", "mujoco_gl",
               "framebuffer", "libgl", "display")


def _looks_like_gl_error(exc):
    text = ("%s %s" % (type(exc).__name__, exc)).lower()
    return any(m in text for m in _GL_MARKERS)


def _import_stack():
    """Import the simulator stack, turning ImportErrors into instructions."""
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as e:
        _die("missing core dependency (%s)" % e, INSTALL_HINT)
    try:
        import mujoco  # noqa: F401
        import robosuite  # noqa: F401
    except ImportError as e:
        _die("robosuite / mujoco are not installed (%s)" % e, INSTALL_HINT)
    except Exception as e:  # noqa: BLE001 -- mujoco resolves its GL backend
        # at import time, so a bad/missing GL stack surfaces here, not later
        if _looks_like_gl_error(e):
            _die("MuJoCo could not set up rendering (%s: %s)"
                 % (type(e).__name__, e), EGL_HINT)
        raise
    try:
        from pipeline import retarget_runner, perception  # noqa: F401
        from simtasks import capture, envs  # noqa: F401
    except ImportError as e:
        _die("could not import the repository packages (%s); run this from "
             "a checkout of the repository" % e, INSTALL_HINT)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _fmt_pose(T):
    import numpy as np
    T = np.asarray(T, dtype=float)
    lines = []
    for i in range(3):
        lines.append("    [% .5f % .5f % .5f | % .4f ]"
                     % (T[i, 0], T[i, 1], T[i, 2], T[i, 3]))
    return "\n".join(lines)


def _print_report(task, seed, res, elapsed):
    orc = res.get("oracle") or {}
    print("")
    print("=" * 72)
    print("one-shot retargeting -- task=%s  pair seed=%d" % (task, seed))
    print("=" * 72)
    print("")
    print("perception (ground-truth instance segmentation stands in for")
    print("GroundingDINO+SAM; k-means candidates, k=8, camera=%s)"
          % res.get("camera"))
    for role in ("demo", "target"):
        p = res["perception"][role]
        print("  %-6s mask=%6d px  cloud=%6d pts  centroid err=%5.1f mm  "
              "sanity=%s" % (role, p["mask_pixels"], p["n_points"],
                             1e3 * p["centroid_err_m"], p["sanity_ok"]))
    print("")
    print("discrete variables Phi (oracle answers; a VLM would answer these)")
    print("  demo   %s" % orc.get("demo_choice"))
    print("  target %s" % orc.get("target_choice"))
    print("")
    print("ALK quadruple (world frame, metres)")
    print("        %-24s %-24s" % ("demo C_i", "target C_i"))
    for i, (a, b) in enumerate(zip(orc["demo_alk"], orc["target_alk"]), 1):
        print("  C%d    [% .4f % .4f % .4f]  [% .4f % .4f % .4f]"
              % (i, a[0], a[1], a[2], b[0], b[1], b[2]))
    print("")
    print("closed-form pose T0 = Procrustes(ALK_demo, ALK_target)")
    print(_fmt_pose(res["T_init"]))
    print("  rotation error    %7.2f deg" % res["rot_err_init_deg"])
    print("  translation error %7.2f mm" % (1e3 * res["trans_err_init_m"]))
    print("")
    print("registered pose T_map = bounded Chamfer refinement of T0")
    print(_fmt_pose(res["T_map"]))
    print("  rotation error    %7.2f deg" % res["rot_err_deg"])
    print("  translation error %7.2f mm" % (1e3 * res["trans_err_m"]))
    print("")
    print("Chamfer distance (demo cloud mapped into the target frame)")
    print("  before registration %7.2f mm" % (1e3 * res["chamfer_before_m"]))
    print("  after  registration %7.2f mm" % (1e3 * res["chamfer_after_m"]))
    if res.get("grasp_correction_norm_m") is not None:
        print("")
        print("grasp-point translation correction  %.2f mm"
              % (1e3 * res["grasp_correction_norm_m"]))
    print("")
    print("execution in the target scene (retargeted waypoints, %d control "
          "steps)" % (res.get("n_exec_steps") or 0))
    print("  grasped              %s" % res.get("grasped"))
    print("  waypoints converged  %s" % (res.get("waypoints_converged"),))
    print("  TASK SUCCESS         %s%s"
          % (res["success"],
             "" if res["success"] else
             "   (failure stage: %s)" % res.get("failure_stage")))
    print("")
    print("wall clock: %.1f s" % elapsed)


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def _write_figure(task, seed, res, data_root, out_png):
    """Demo cloud mapped into the target frame, before vs after registration.

    Re-runs the (pure numpy) perception stage on the two saved scene
    captures -- deterministic, so the clouds are exactly the ones the
    rollout used -- and applies the T0 / T_map recorded in the rollout.
    """
    import numpy as np
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("figure skipped: matplotlib is not installed "
              "(pip install matplotlib)")
        return None

    from alkbench import transform_points
    from pipeline import perception, retarget_runner as rr
    from simtasks import capture

    demo = rr.load_demo(task, data_root)
    tgt_cap = capture.load_capture(
        os.path.join(data_root, task, str(seed), "target", "scene"))
    cam = res["camera"]
    p_demo = perception.perceive(demo["scene"], task, k=8, seed=0, camera=cam)
    p_tgt = perception.perceive(tgt_cap, task, k=8, seed=0, camera=cam)
    src = p_demo["cands"].points3d
    tgt = p_tgt["cands"].points3d

    def sub(P, n=3000):
        if P.shape[0] <= n:
            return P
        idx = np.random.RandomState(0).choice(P.shape[0], n, replace=False)
        return P[idx]

    tgt_s = sub(tgt)
    alk_d = np.asarray(res["oracle"]["demo_alk"], dtype=float)
    alk_t = np.asarray(res["oracle"]["target_alk"], dtype=float)

    panels = [("closed-form Procrustes T0", np.asarray(res["T_init"]),
               res["chamfer_before_m"], res["rot_err_init_deg"],
               res["trans_err_init_m"]),
              ("after bounded registration T_map", np.asarray(res["T_map"]),
               res["chamfer_after_m"], res["rot_err_deg"],
               res["trans_err_m"])]
    views = [("x", "y", 0, 1), ("x", "z", 0, 2)]

    mapped_all = [transform_points(np.asarray(T), sub(src))
                  for _, T, _, _, _ in panels]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for col, (title, T, ch, rot, trans) in enumerate(panels):
        mapped = mapped_all[col]
        mapped_alk = transform_points(np.asarray(T), alk_d)
        for row, (ax_l, ay_l, ai, aj) in enumerate(views):
            ax = axes[row][col]
            ax.scatter(tgt_s[:, ai], tgt_s[:, aj], s=1.5, c="0.65")
            ax.scatter(mapped[:, ai], mapped[:, aj], s=1.5, c="tab:blue",
                       alpha=0.6)
            ax.scatter(alk_t[:, ai], alk_t[:, aj], s=90, marker="o",
                       facecolors="none", edgecolors="k", linewidths=1.6)
            ax.scatter(mapped_alk[:, ai], mapped_alk[:, aj], s=70,
                       marker="x", c="tab:red", linewidths=2.0)
            ax.set_xlabel("%s [m]" % ax_l)
            ax.set_ylabel("%s [m]" % ay_l)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, lw=0.3, alpha=0.5)
            if row == 0:
                ax.set_title("%s\nChamfer %.2f mm | rot %.2f deg | "
                             "trans %.2f mm"
                             % (title, 1e3 * ch, rot, 1e3 * trans),
                             fontsize=10)
    # identical limits per row, so the two columns are visually comparable
    for row, (_, _, ai, aj) in enumerate(views):
        P = np.vstack([tgt_s] + mapped_all)
        pad = 0.012
        lo, hi = P[:, ai].min() - pad, P[:, ai].max() + pad
        lo2, hi2 = P[:, aj].min() - pad, P[:, aj].max() + pad
        half = 0.5 * max(hi - lo, hi2 - lo2)
        cx, cy = 0.5 * (lo + hi), 0.5 * (lo2 + hi2)
        for col in (0, 1):
            axes[row][col].set_xlim(cx - half, cx + half)
            axes[row][col].set_ylim(cy - half, cy + half)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], ls="", marker="o", ms=5, mfc="0.65", mec="0.65",
               label="target object cloud"),
        Line2D([], [], ls="", marker="o", ms=5, mfc="tab:blue",
               mec="tab:blue", label="demo cloud mapped by the pose"),
        Line2D([], [], ls="", marker="o", ms=9, mfc="none", mec="k",
               mew=1.6, label="target ALK $C_1..C_4$"),
        Line2D([], [], ls="", marker="x", ms=9, mec="tab:red", mew=2.0,
               label="mapped demo ALK $C_1..C_4$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               frameon=False)
    fig.suptitle("%s, pair seed %d: demo object cloud mapped into the target "
                 "frame" % (task, seed), fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    d = os.path.dirname(os.path.abspath(out_png))
    if d and not os.path.isdir(d):
        os.makedirs(d)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default=DEFAULT_TASK,
                   help="task key (default: %s)" % DEFAULT_TASK)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="scene-pair seed (default: %d)" % DEFAULT_SEED)
    p.add_argument("--data-root", default=None,
                   help="where to cache the generated scene pair "
                        "(default: <repo>/data)")
    p.add_argument("--out-png", default=None,
                   help="figure path (default: "
                        "<repo>/demo_out/quickstart_<task>_<seed>.png)")
    p.add_argument("--no-figure", action="store_true")
    args = p.parse_args(argv)

    _import_stack()
    from simtasks import envs, scene_pairs
    from pipeline import retarget_runner as rr

    if args.task not in envs.TASKS:
        raise SystemExit("unknown task %r (have %s)"
                         % (args.task, ", ".join(sorted(envs.TASKS))))
    data_root = args.data_root or scene_pairs.DATA_ROOT
    out_png = args.out_png or os.path.join(
        REPO_ROOT, "demo_out",
        "quickstart_%s_%d.png" % (args.task, args.seed))

    print("quickstart: task=%s seed=%d" % (args.task, args.seed))
    print("  scene pairs are generated from the seed; cache dir: %s"
          % data_root)
    print("  discrete answers: simulator oracle (no VLM / GPU / network)")
    print("  MUJOCO_GL=%s" % os.environ.get("MUJOCO_GL"))
    print("  ... generating the scene pair, recording the scripted demo and "
          "running one rollout")
    t0 = time.time()
    try:
        res = rr.run_pair(args.task, args.seed, registration=True,
                          correction=True, data_root=data_root, save=True)
    except Exception as e:  # noqa: BLE001 -- turn GL failures into advice
        if _looks_like_gl_error(e):
            _die("headless rendering failed (%s: %s)"
                 % (type(e).__name__, e), EGL_HINT)
        raise
    elapsed = time.time() - t0
    _print_report(args.task, args.seed, res, elapsed)

    print("rollout record: %s"
          % os.path.join(data_root, args.task, str(args.seed),
                         "rollout_%s.json" % res["variant"]))
    if not args.no_figure:
        path = _write_figure(args.task, args.seed, res, data_root, out_png)
        if path:
            print("figure:         %s" % path)
    print("total wall clock: %.1f s" % (time.time() - t0))
    return 0 if res["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
