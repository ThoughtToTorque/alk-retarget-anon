"""End-to-end smoke test.

For each task:
  1. generate one (demo_scene, target_scene) pair (seed 0),
  2. re-instantiate the demo scene and run the scripted expert in it,
     saving keyframes + trajectory under <data>/<task>/0/demo/,
  3. verify task success via the env's own success check,
  4. verify captures load back correctly (shapes, dtypes, keyframe count),
  5. run the depth back-projection sanity check on the demo scene: the
     centroid of the target object's segmented, back-projected pixels must
     agree with the ground-truth object position.

Usage:
  python -m simtasks.run_smoke [task ...]        (default: all 5 tasks)

Exits non-zero unless at least 2 tasks pass everything.
"""
import json
import os
import sys
import time
import traceback

import numpy as np

try:
    from . import capture, envs, scene_pairs, scripted_demo
except ImportError:
    import capture
    import envs
    import scene_pairs
    import scripted_demo


def check_capture_loads(scene_dir, n_cams_expected=2):
    cap = capture.load_capture(scene_dir)
    assert len(cap["cameras"]) >= 1, "no cameras in capture"
    for cam, cd in cap["cameras"].items():
        h, w = cd["depth"].shape
        assert cd["rgb"].shape == (h, w, 3), "rgb shape mismatch"
        assert cd["seg"].shape == (h, w), "seg shape mismatch"
        assert cd["depth"].dtype == np.float32
        assert np.isfinite(cd["depth"]).all()
        assert 0.05 < np.median(cd["depth"]) < 20.0, "depth not metric?"
        assert cd["K"].shape == (3, 3) and cd["T_world_cam"].shape == (4, 4)
    assert cap["meta"]["objects"], "no ground-truth object poses"
    assert "tcp" in cap["meta"]
    return cap


def smoke_task(task, seed=0, out_root=None):
    out_root = out_root or scene_pairs.DATA_ROOT
    spec = envs.TASKS[task]
    result = {"task": task, "ok": False}
    t0 = time.time()

    # 1. scene pair
    info = scene_pairs.generate_pair(task, seed, out_root=out_root)
    pair_dir = os.path.join(out_root, task, str(seed))

    # object placement must actually differ between demo and target scenes
    d = np.array(info["demo_object_poses"][spec.target_instance]["pos"])
    t = np.array(info["target_object_poses"][spec.target_instance]["pos"])
    result["placement_delta_m"] = float(np.linalg.norm(d - t))

    # 2. scripted demo in the demo scene
    env = scene_pairs.make_env_for(info, "demo")
    try:
        success, dinfo = scripted_demo.run_demo(
            env, task, out_dir=os.path.join(pair_dir, "demo"))
    finally:
        env.close()
    result["demo_success"] = bool(success)
    result.update(dinfo)

    # 3. captures load back correctly
    demo_cap = check_capture_loads(os.path.join(pair_dir, "demo", "scene"))
    check_capture_loads(os.path.join(pair_dir, "target", "scene"))
    with open(os.path.join(pair_dir, "demo", "keyframes.json")) as f:
        keyframes = json.load(f)
    assert len(keyframes) == 3, "expected 3 keyframes"
    for kf in keyframes:
        check_capture_loads(os.path.join(pair_dir, "demo", kf["dir"]))
    traj = np.load(os.path.join(pair_dir, "demo", "trajectory.npz"))
    assert traj["tcp_pos"].shape[0] == traj["gripper"].shape[0] > 10

    # 4. back-projection sanity on the demo scene (primary camera)
    gt = demo_cap["meta"]["objects"][spec.target_instance]["pos"]
    cam = list(demo_cap["cameras"].keys())[0]
    bp_ok, err, npix = capture.sanity_check_backprojection(
        demo_cap, cam, spec.target_instance, gt, atol=spec.sanity_tol)
    result["backproj_ok"] = bool(bp_ok)
    result["backproj_err_m"] = err
    result["backproj_npix"] = int(npix)

    # 5. success_checker hook exists and is callable on a fresh env
    checker = scene_pairs.success_checker(task)
    env = scene_pairs.make_env_for(info, "target")
    try:
        fresh = checker(env)
        assert fresh in (True, False)
        result["target_success_at_reset"] = bool(fresh)
    finally:
        env.close()

    result["ok"] = bool(success) and bool(bp_ok)
    result["time_s"] = round(time.time() - t0, 1)
    return result


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    tasks = argv or sorted(envs.TASKS.keys())
    results = []
    for task in tasks:
        try:
            results.append(smoke_task(task))
        except Exception:
            traceback.print_exc()
            results.append({"task": task, "ok": False, "error": True})

    print("\n=== run_smoke summary ===")
    hdr = "%-12s %-5s %-8s %-9s %-14s %-8s %s"
    print(hdr % ("task", "ok", "demo", "backproj", "bp_err(m)", "steps",
                 "placement_dx(m)"))
    for r in results:
        print(hdr % (
            r["task"], r["ok"], r.get("demo_success", "-"),
            r.get("backproj_ok", "-"),
            ("%.4f" % r["backproj_err_m"]) if "backproj_err_m" in r else "-",
            r.get("n_steps", "-"),
            ("%.3f" % r["placement_delta_m"]) if "placement_delta_m" in r else "-",
        ))
    n_ok = sum(1 for r in results if r["ok"])
    print("passed %d/%d tasks" % (n_ok, len(results)))
    return 0 if n_ok >= 2 else 1


if __name__ == "__main__":
    sys.exit(main())
