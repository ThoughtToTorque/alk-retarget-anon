#!/usr/bin/env python
"""Record one retargeting rollout as an mp4 (offscreen agentview render).

    python demo/record_rollout.py --task nut_loosen --seed 1000 --out out.mp4

Same rollout as demo/quickstart.py (oracle discrete answers, registration
and grasp correction on), plus a frame captured from the offscreen renderer
every --every control steps of the target-scene execution, written as an
mp4 via imageio/ffmpeg.  Rendering is entirely optional machinery: without
this script the pipeline never renders during rollouts.
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="nut_loosen")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", default=None,
                   help="output mp4 path (default: "
                        "<repo>/demo_out/rollout_<task>_<seed>.mp4)")
    p.add_argument("--camera", default="agentview")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--every", type=int, default=2,
                   help="capture one frame every N control steps (default 2)")
    p.add_argument("--fps", type=int, default=25,
                   help="playback fps of the mp4 (default 25)")
    p.add_argument("--data-root", default=None)
    args = p.parse_args(argv)

    import imageio.v2 as imageio
    from simtasks import envs, scene_pairs
    from pipeline import retarget_runner as rr

    if args.task not in envs.TASKS:
        raise SystemExit("unknown task %r (have %s)"
                         % (args.task, ", ".join(sorted(envs.TASKS))))
    out = args.out or os.path.join(
        REPO_ROOT, "demo_out",
        "rollout_%s_%d.mp4" % (args.task, args.seed))
    d = os.path.dirname(os.path.abspath(out))
    if d and not os.path.isdir(d):
        os.makedirs(d)

    frames = []

    def grab(env, step_index):
        if step_index % args.every:
            return
        rgb = env.sim.render(camera_name=args.camera, width=args.width,
                             height=args.height, depth=False)
        frames.append(rgb[::-1])  # MuJoCo renders bottom-row-first

    print("record_rollout: task=%s seed=%d camera=%s %dx%d every=%d fps=%d"
          % (args.task, args.seed, args.camera, args.width, args.height,
             args.every, args.fps))
    res = rr.run_pair(args.task, args.seed, registration=True,
                      correction=True, data_root=args.data_root,
                      save=False, step_callback=grab)
    print("  success=%s  control steps=%s  frames=%d"
          % (res["success"], res.get("n_exec_steps"), len(frames)))
    imageio.mimwrite(out, frames, fps=args.fps, quality=8,
                     macro_block_size=1)
    print("  wrote %s (%.1f s at %d fps)"
          % (out, len(frames) / float(args.fps), args.fps))
    return 0 if res["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
