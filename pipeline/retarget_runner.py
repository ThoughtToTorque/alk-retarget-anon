"""Full one-shot retargeting closed loop for a (task, seed) scene pair.

    demo capture  -> demo candidates -> demo ALK   (oracle discrete answers)
    target capture-> target candidates -> target ALK
    Procrustes on the two ALKs -> T0
    bounded Chamfer registration on the two WORLD-frame object clouds -> T_map
    T_map applied to the demo's commanded TCP waypoints as 4x4 poses
    (orientation rotated too), + grasp-point translation correction
    -> execute in the TARGET scene (reset_with_seed) through the shared
    motion primitives (simtasks.motion) -> env success check.

Flags: --no-registration (closed-form Procrustes only),
       --no-correction   (skip grasp-point translation correction),
       --adaptive        (DEFAULT OFF: conditioning-adaptive bounded
                          registration, paper Section 3.7 -- widened rotation search
                          about the ALK principal axis when the demo ALK is
                          nearly rank-1, + auto slender depth-consistency
                          prior; records adaptive_triggered / widened_axis /
                          sigma ratios in the rollout JSON),
       --alk-subset      (DEFAULT None = all four ALK points: which ALK
                          points enter the closed-form Procrustes sum,
                          alkbench.alk.ALK_SUBSETS -- Campaign F ablation of
                          the paper's {c1,c3,c4} construction).

Per-rollout JSON is written to  data/<task>/<seed>/rollout_<variant>.json
(chamfer before/after, T_map-vs-T_gt rotation/translation errors, oracle
discrete choices + diagnostics, success bool, failure stage guess).
"""
import argparse
import json
import os
import shutil
import time

import numpy as np

from alkbench import (procrustes, bounded_registration, chamfer_distance,
                      adaptive_registration, retarget, transform_points,
                      rotation_angle_deg, align_keypoints, ALK_SUBSETS,
                      alk_subset_indices, select_alk_subset)
from alkbench.registration import _subsample

from simtasks import capture, envs, motion, scene_pairs, scripted_demo
from pipeline import oracle, perception

# the demo recording (keyframes + waypoints) is shared by all seeds of a
# task (DEMO_SEEDS is fixed); its canonical location is the seed-0 pair dir
DEMO_PAIR_SEED = 0


# ---------------------------------------------------------------------------
# data plumbing
# ---------------------------------------------------------------------------

def ensure_pair(task, seed, env=None, data_root=None):
    """Load or generate the (demo, target) scene pair for (task, seed).

    With `env` given, scene captures reuse the live env (one reset instead of
    a full env construction); the demo scene capture is copied from the
    canonical seed-0 pair when available (it is identical by construction).
    """
    data_root = data_root or scene_pairs.DATA_ROOT
    pair_dir = os.path.join(data_root, task, str(seed))
    pair_json = os.path.join(pair_dir, "pair.json")
    if os.path.exists(pair_json):
        with open(pair_json) as f:
            return json.load(f)
    if env is None:
        return scene_pairs.generate_pair(task, seed, out_root=data_root)

    spec = envs.TASKS[task]
    demo_seed = scene_pairs.DEMO_SEEDS[task]
    tgt_seed = scene_pairs.target_seed_for(task, seed)
    demo_scene_dir = os.path.join(pair_dir, "demo", "scene")
    src_demo = os.path.join(data_root, task, str(DEMO_PAIR_SEED),
                            "demo", "scene")
    if os.path.isdir(src_demo) and seed != DEMO_PAIR_SEED:
        if not os.path.isdir(demo_scene_dir):
            shutil.copytree(src_demo, demo_scene_dir)
        with open(os.path.join(demo_scene_dir, "meta.json")) as f:
            demo_meta = json.load(f)
    else:
        envs.reset_with_seed(env, demo_seed)
        demo_meta = capture.capture_scene(env, demo_scene_dir,
                                          extra_state=spec.extra_state(env))
    envs.reset_with_seed(env, tgt_seed)
    target_meta = capture.capture_scene(
        env, os.path.join(pair_dir, "target", "scene"),
        extra_state=spec.extra_state(env))

    info = {
        "task": task,
        "env_name": spec.env_name,
        "seed": seed,
        "demo_seed": demo_seed,
        "target_seed": tgt_seed,
        "target_instance": spec.target_instance,
        "demo_object_poses": demo_meta["objects"],
        "target_object_poses": target_meta["objects"],
    }
    with open(pair_json, "w") as f:
        json.dump(info, f, indent=2)
    return info


def demo_dir_for(task, data_root=None):
    data_root = data_root or scene_pairs.DATA_ROOT
    return os.path.join(data_root, task, str(DEMO_PAIR_SEED), "demo")


def ensure_demo(task, env=None, data_root=None):
    """Make sure the shared demo recording for `task` exists (keyframes +
    waypoints from the scripted expert in the fixed demo scene)."""
    demo_dir = demo_dir_for(task, data_root)
    if (os.path.exists(os.path.join(demo_dir, "waypoints.json"))
            and os.path.exists(os.path.join(demo_dir, "keyframes.json"))):
        return demo_dir
    ensure_pair(task, DEMO_PAIR_SEED, env=env, data_root=data_root)
    own = env is None
    if own:
        env = envs.make_env(task)
    try:
        envs.reset_with_seed(env, scene_pairs.DEMO_SEEDS[task])
        success, _ = scripted_demo.run_demo(env, task, out_dir=demo_dir)
        if not success:
            raise RuntimeError("scripted demo failed for task %r" % task)
    finally:
        if own:
            env.close()
    return demo_dir


def load_demo(task, data_root=None):
    """Load the shared demo recording: scene capture, keyframes, waypoints."""
    demo_dir = demo_dir_for(task, data_root)
    with open(os.path.join(demo_dir, "keyframes.json")) as f:
        keyframes = json.load(f)
    with open(os.path.join(demo_dir, "waypoints.json")) as f:
        waypoints = json.load(f)
    scene = capture.load_capture(os.path.join(demo_dir, "scene"))
    return {"dir": demo_dir, "scene": scene, "keyframes": keyframes,
            "waypoints": waypoints}


# ---------------------------------------------------------------------------
# execution helpers
# ---------------------------------------------------------------------------

def object_geoms(env, task):
    """Contact geoms of the task's target object (for the grasp check)."""
    spec = envs.TASKS[task]
    if task in ("nut_loosen", "cap_twist"):
        nut = [n for n in env.nuts if n.name == spec.target_instance][0]
        return nut.contact_geoms
    if task == "rim_grasp":
        return env.objects[env.object_id].contact_geoms
    if task == "pour":
        return env.cube.contact_geoms
    if task == "box_open":
        return ["Door_handle"]
    raise KeyError(task)


def _grasp_check(task):
    def check(env):
        return bool(env._check_grasp(gripper=env.robots[0].gripper,
                                     object_geoms=object_geoms(env, task)))
    return check


def retarget_waypoint_dicts(waypoints, T_map, demo_grasp=None,
                            target_grasp=None):
    """Apply T_map (4x4, world frame) to a demo waypoint list; orientation is
    rotated too (T_map @ T_wp as 4x4).  Grasp-point translation correction is
    applied when both grasp points are given. Returns a new waypoint list."""
    W = np.stack([motion.pose_to_matrix(wp["pos"], wp["quat_xyzw"])
                  for wp in waypoints])
    W2 = retarget(T_map, W, demo_grasp=demo_grasp, target_grasp=target_grasp)
    out = []
    for wp, M in zip(waypoints, W2):
        pos, quat = motion.matrix_to_pose(M)
        nwp = dict(wp)
        nwp["pos"] = pos.tolist()
        nwp["quat_xyzw"] = quat.tolist()
        out.append(nwp)
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def map_errors(T_est, T_gt, demo_obj_pos, target_obj_pos):
    """Rotation error (deg) and translation error (m) of an estimated object
    transform vs ground truth, translation measured at the object center."""
    T_est = np.asarray(T_est, dtype=np.float64)
    T_gt = np.asarray(T_gt, dtype=np.float64)
    rot = float(rotation_angle_deg(T_est[:3, :3] @ T_gt[:3, :3].T))
    mapped = transform_points(T_est, np.asarray(demo_obj_pos)[None])[0]
    trans = float(np.linalg.norm(mapped - np.asarray(target_obj_pos)))
    return rot, trans


def _config_conditioning(points):
    """Singular values of a CENTERED keypoint configuration plus the two
    theory numbers: sigma2+sigma3 (paper Section 3.7: closed-form rotation error scales
    like 1/(sigma2+sigma3)) and the scale-free ratio (sigma2+sigma3)/sigma1
    that the adaptive-registration trigger compares to tau_sigma."""
    P = np.asarray(points, dtype=np.float64)
    X = P - P.mean(axis=0)
    s = np.linalg.svd(X, compute_uv=False)
    s = np.concatenate([s, np.zeros(3)])[:3]
    sigma23 = float(s[1] + s[2])
    return {"n_points": int(P.shape[0]),
            "singular_values": [float(v) for v in s],
            "sigma23": sigma23,
            "sigma_ratio": float(sigma23 / s[0]) if s[0] > 0 else 0.0}


def _json_safe(x):
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
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


def variant_name(registration, correction, adaptive=False, alk_subset=None):
    name = "full" if registration else "noreg"
    if not correction:
        name += "_nocorr"
    if adaptive:
        name += "_adaptive"
    if alk_subset:
        name += "_" + alk_subset
    return name


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------

def run_pair(task, seed, registration=True, correction=True, env=None,
             data_root=None, save=True, k=8, kmeans_seed=0, adaptive=False,
             inject=None, alk_subset=None, step_callback=None):
    """Run the full retargeting closed loop on one (task, seed) pair.

    `env` (optional): a live env for this task, reused across calls (it is
    reset to the pair's target seed).  Returns the result dict.

    `inject` (optional, DEFAULT None -- existing behavior unchanged):
    callable ``inject(orc, p_demo, p_tgt) -> (orc_or_None, record)`` applied
    to the oracle answers AFTER they are recorded (Campaign C error
    injection, pipeline.inject).  The record is stored under
    ``result["injection"]``; a None orc means the corrupted discrete choice
    produced a degenerate ALK and the rollout is recorded as a failure.

    `adaptive` (DEFAULT False -- never changes existing behavior): enables
    (1) conditioning-adaptive bounded registration
    (alkbench.registration.adaptive_registration: widened rotation search
    about the ALK principal axis when the demo ALK conditioning ratio is
    below threshold, paper Section 3.7) and (2) the auto slender depth-consistency
    prior (oracle.solve depth_consistent="auto").

    `step_callback` (DEFAULT None -- existing behavior unchanged): callable
    ``step_callback(env, step_index)`` invoked after every control step of
    the TARGET-scene execution only (demo/record_rollout.py uses it to
    render video frames offscreen).

    `alk_subset` (DEFAULT None -- existing behavior unchanged): name of the
    ALK point subset entering the closed-form Procrustes sum
    (alkbench.alk.ALK_SUBSETS).  None means all four points, i.e. the
    untouched default code path.  "alk3_c134" reproduces the hardware system's
    earlier three-point construction (sum over C1, C3, C4 -- c2 dropped from the
    alignment sum but still used for the halfplane split and, under
    `adaptive`, for the widened axis).  Everything downstream (bounded
    registration on the dense clouds, grasp correction, execution) is
    unchanged; only T0 differs.  Extra JSON keys (alk_subset,
    alk_subset_indices, alk_conditioning*) are written ONLY when a subset is
    requested, so default rollout JSONs stay byte-identical.
    """
    data_root = data_root or scene_pairs.DATA_ROOT
    t0 = time.time()
    variant = variant_name(registration, correction, adaptive, alk_subset)
    result = {"task": task, "seed": seed, "variant": variant,
              "registration": bool(registration),
              "correction": bool(correction),
              "adaptive": bool(adaptive),
              "success": False, "failure_stage": None}
    if alk_subset is not None:
        result["alk_subset"] = alk_subset
        result["alk_subset_indices"] = list(alk_subset_indices(alk_subset))

    pair = ensure_pair(task, seed, env=env, data_root=data_root)
    ensure_demo(task, env=env, data_root=data_root)
    demo = load_demo(task, data_root)
    pair_dir = os.path.join(data_root, task, str(seed))
    target_scene = capture.load_capture(os.path.join(pair_dir, "target",
                                                     "scene"))

    # ---- perception (same camera for both scenes) --------------------------
    cam = perception.pick_camera(demo["scene"], task)
    p_demo = perception.perceive(demo["scene"], task, k=k, seed=kmeans_seed,
                                 camera=cam)
    p_tgt = perception.perceive(target_scene, task, k=k, seed=kmeans_seed,
                                camera=cam)
    result["camera"] = cam
    result["perception"] = {
        "demo": {kk: p_demo[kk] for kk in
                 ("mask_pixels", "n_points", "centroid_err_m", "sanity_ok")},
        "target": {kk: p_tgt[kk] for kk in
                   ("mask_pixels", "n_points", "centroid_err_m", "sanity_ok")},
    }

    # ---- ground truth relative transform -----------------------------------
    inst = pair["target_instance"]
    demo_pose = pair["demo_object_poses"][inst]
    target_pose = pair["target_object_poses"][inst]
    T_gt = oracle.gt_relative_transform(demo_pose, target_pose)
    result["T_gt"] = T_gt.tolist()

    # ---- oracle discrete answers -------------------------------------------
    pre_grasp = [kf for kf in demo["keyframes"] if kf["name"] == "pre_grasp"][0]
    try:
        orc = oracle.solve(p_demo, p_tgt, T_gt,
                           np.asarray(pre_grasp["tcp_pos"]), seed=kmeans_seed,
                           depth_consistent="auto" if adaptive else False)
    except ValueError as e:  # degenerate halfplane split
        result["failure_stage"] = "alk_degenerate"
        result["error"] = str(e)
        result["time_s"] = round(time.time() - t0, 1)
        if save:
            _save_result(pair_dir, variant, result)
        return result
    result["oracle"] = _json_safe({
        "demo_choice": orc["demo_choice"],
        "target_choice": orc["target_choice"],
        "demo_alk": orc["demo_alk"],
        "target_alk": orc["target_alk"],
        "demo_grasp": orc["demo_grasp"],
        "target_grasp_centroid": orc["target_grasp_centroid"],
        # candidate pixel positions, so a real VLM can later be scored
        # against these oracle answers on the same marked-up image
        "demo_candidates_uv": p_demo["cands"].centers2d,
        "target_candidates_uv": p_tgt["cands"].centers2d,
        "diagnostics": orc["diagnostics"],
        "depth_consistent": orc["depth_consistent"],
        "slender": orc["slender"],
    })

    # ---- optional error injection (Campaign C; no-op when inject is None) --
    if inject is not None:
        orc_inj, inj_rec = inject(orc, p_demo, p_tgt)
        result["injection"] = _json_safe(inj_rec)
        if orc_inj is None:  # corrupted choice -> degenerate ALK -> failure
            result["failure_stage"] = "alk_degenerate"
            result["error"] = inj_rec.get("error")
            result["time_s"] = round(time.time() - t0, 1)
            if save:
                _save_result(pair_dir, variant, result)
            return result
        orc = orc_inj

    # ---- alignment: Procrustes (+ bounded registration) --------------------
    if alk_subset is None:
        T0 = procrustes(orc["demo_alk"], orc["target_alk"])
        cond_points = None
    else:
        P_sub = select_alk_subset(orc["demo_alk"], alk_subset)
        Q_sub = select_alk_subset(orc["target_alk"], alk_subset)
        T0 = align_keypoints(P_sub, Q_sub)
        cond_points = P_sub
        result["alk_conditioning"] = _json_safe(
            {"solved": _config_conditioning(P_sub),
             "alk4": _config_conditioning(orc["demo_alk"])})
    src = p_demo["cands"].points3d
    tgt = p_tgt["cands"].points3d
    if registration:
        if adaptive:
            reg = adaptive_registration(src, tgt, T0, orc["demo_alk"],
                                        cond_points=cond_points)
            ab = reg["adaptive"]
            result["adaptive_registration"] = _json_safe({
                "adaptive_triggered": ab["adaptive_triggered"],
                "widened_axis_demo": ab["axis_demo"],
                "widened_axis_world": reg["axis_world"],
                "axis_rot_bound_deg": ab["axis_rot_bound_deg"],
                "coarse_axis_theta_deg": reg["coarse_axis_theta_deg"],
                "sigma_ratios": {
                    "singular_values": ab["singular_values"],
                    "sigma23": ab["sigma23"],
                    "sigma_ratio": ab["sigma_ratio"],
                    "sigma_ratio_threshold": ab["sigma_ratio_threshold"],
                },
            })
        else:
            reg = bounded_registration(src, tgt, T0)
        T_map = reg["T"]
        chamfer_before = float(reg["chamfer_init"])
        chamfer_after = float(reg["chamfer_final"])
    else:
        T_map = T0
        src_s = _subsample(src, 2000, 0)
        tgt_s = _subsample(tgt, 2000, 0)
        chamfer_before = float(chamfer_distance(
            transform_points(T0, src_s), tgt_s))
        chamfer_after = chamfer_before

    rot0, trans0 = map_errors(T0, T_gt, demo_pose["pos"], target_pose["pos"])
    rot_err, trans_err = map_errors(T_map, T_gt, demo_pose["pos"],
                                    target_pose["pos"])
    result.update({
        "T_init": np.asarray(T0).tolist(),
        "T_map": np.asarray(T_map).tolist(),
        "chamfer_before_m": chamfer_before,
        "chamfer_after_m": chamfer_after,
        "rot_err_init_deg": rot0, "trans_err_init_m": trans0,
        "rot_err_deg": rot_err, "trans_err_m": trans_err,
    })

    # ---- retarget demo waypoints -------------------------------------------
    # Continuous target grasp point: projection of the T_map-mapped demo
    # grasp point onto the oracle-chosen fine grasp region (phi4/phi5 are the
    # discrete answers; the projection uses only the pipeline's own T_map, so
    # no ground truth leaks).  Using the region CENTROID instead would inject
    # the region quantization (several cm on large instances like the door)
    # straight into the translation correction.
    target_grasp = None
    if correction:
        region = np.asarray(orc["target_grasp_region_points"])
        mapped_gd = transform_points(T_map, orc["demo_grasp"][None])[0]
        target_grasp = region[np.argmin(
            np.linalg.norm(region - mapped_gd, axis=1))]
        result["target_grasp_used"] = target_grasp.tolist()
        dt = target_grasp - mapped_gd
        result["grasp_correction_m"] = dt.tolist()
        result["grasp_correction_norm_m"] = float(np.linalg.norm(dt))
    mapped_wps = retarget_waypoint_dicts(
        demo["waypoints"], T_map,
        demo_grasp=orc["demo_grasp"] if correction else None,
        target_grasp=target_grasp)

    # ---- execute in the target scene ----------------------------------------
    own_env = env is None
    if own_env:
        env = envs.make_env(task)
    try:
        envs.reset_with_seed(env, pair["target_seed"])
        exec_res = motion.execute_waypoints(env, mapped_wps,
                                            grasp_check=_grasp_check(task),
                                            step_callback=step_callback)
        success = bool(envs.success_checker(task)(env))
    finally:
        if own_env:
            env.close()

    result.update({
        "grasped": exec_res["grasped"],
        "waypoints_converged": exec_res["converged"],
        "n_exec_steps": exec_res["n_steps"],
        "success": success,
    })
    if task == "pour":  # orientation-aware criterion diagnostics (pour only)
        result["pour_orientation"] = _json_safe(
            envs.pour_orientation_trace(env))

    # ---- failure stage guess (one level deep) --------------------------------
    if not success:
        map_bad = trans_err > 0.05 or rot_err > 20.0
        if not (p_demo["sanity_ok"] and p_tgt["sanity_ok"]):
            stage = "perception"
        elif map_bad:
            stage = "mapping"
        elif exec_res["grasped"] is False:
            stage = "grasp"
        else:
            stage = "post_action"
        result["failure_stage"] = stage

    result["time_s"] = round(time.time() - t0, 1)
    if save:
        _save_result(pair_dir, variant, result)
    return result


def _save_result(pair_dir, variant, result):
    path = os.path.join(pair_dir, "rollout_%s.json" % variant)
    with open(path, "w") as f:
        json.dump(_json_safe(result), f, indent=2)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=sorted(envs.TASKS.keys()))
    p.add_argument("seed", type=int)
    p.add_argument("--no-registration", action="store_true",
                   help="closed-form Procrustes only (skip bounded Chamfer "
                        "registration)")
    p.add_argument("--no-correction", action="store_true",
                   help="skip the grasp-point translation correction")
    p.add_argument("--adaptive", action="store_true",
                   help="conditioning-adaptive bounded registration + auto "
                        "slender depth prior (default OFF; existing "
                        "behavior is unchanged without this flag)")
    p.add_argument("--alk-subset", default=None,
                   choices=sorted(ALK_SUBSETS),
                   help="ALK points entering the closed-form Procrustes sum "
                        "(default: all four, unchanged behavior); "
                        "alk3_c134 = the original paper's {c1,c3,c4}")
    p.add_argument("--data-root", default=None)
    args = p.parse_args(argv)
    res = run_pair(args.task, args.seed,
                   registration=not args.no_registration,
                   correction=not args.no_correction,
                   adaptive=args.adaptive,
                   alk_subset=args.alk_subset,
                   data_root=args.data_root)
    print(json.dumps(_json_safe(res), indent=2))
    return 0 if res["success"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
