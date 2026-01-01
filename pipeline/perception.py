"""Perception stage: saved capture -> world-frame object cloud + k=8
candidate keypoints.

Uses the ground-truth instance segmentation of the capture (the sim stand-in
for GroundingDINO+SAM), backprojects masked valid-depth pixels to the WORLD
frame via alkbench.candidates.backproject with the capture's T_world_cam
(OpenCV camera axes, verified conventions -- see simtasks/TASKS.md), and runs
the paper's candidate pipeline (2D k-means, k=8, 3D cluster centroids).

Every perceive() call re-runs the same sanity check as simtasks.run_smoke:
the cloud centroid must agree with the ground-truth object position within
the task's tolerance (surface-centroid vs body-center bias allowed).
"""
import numpy as np

from alkbench import compute_candidates

try:
    from simtasks import envs
except ImportError:  # pragma: no cover
    import envs

# Per-task camera priority: the first available camera in the capture wins.
# box_open: the Door env's agentview faces the robot's back; sideview is the
# informative view (TASKS.md).
DEFAULT_PRIORITY = ("agentview", "sideview", "frontview")
CAMERA_PRIORITY = {
    "box_open": ("sideview", "frontview", "agentview"),
}


def pick_camera(cap, task):
    """Choose the perception camera for a loaded capture."""
    priority = CAMERA_PRIORITY.get(task, DEFAULT_PRIORITY)
    for cam in priority:
        if cam in cap["cameras"]:
            return cam
    return sorted(cap["cameras"].keys())[0]


def instance_id_for(meta, instance_name):
    """Look up the segmentation id of an instance name in a capture meta."""
    for k, v in meta["instance_id_to_name"].items():
        if v == instance_name:
            return int(k)
    raise KeyError("instance %r not in capture id map" % instance_name)


def perceive(cap, task, k=8, seed=0, camera=None):
    """Run the perception stage on a loaded capture (simtasks.capture format).

    Parameters
    ----------
    cap : dict from simtasks.capture.load_capture
    task : task key (selects target instance + camera priority + sanity tol)
    k : number of candidate keypoints
    seed : k-means seed
    camera : optional explicit camera name

    Returns
    -------
    dict with keys:
      camera, instance_id, mask_pixels, cands (CandidateSet, WORLD frame),
      gt_pos (3,), centroid_err_m, sanity_ok, K (3,3), T_world_cam (4,4)
    """
    spec = envs.TASKS[task]
    cam = camera or pick_camera(cap, task)
    cd = cap["cameras"][cam]
    inst_id = instance_id_for(cap["meta"], spec.target_instance)
    mask = cd["seg"] == inst_id
    K = np.asarray(cd["K"], dtype=np.float64)
    T_wc = np.asarray(cd["T_world_cam"], dtype=np.float64)
    cands = compute_candidates(mask, cd["depth"],
                               K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                               extrinsic=T_wc, k=k, seed=seed)
    gt_pos = np.asarray(cap["meta"]["objects"][spec.target_instance]["pos"],
                        dtype=np.float64)
    err = float(np.linalg.norm(cands.points3d.mean(axis=0) - gt_pos))
    return {
        "camera": cam,
        "instance_id": inst_id,
        "mask_pixels": int(mask.sum()),
        "n_points": int(cands.points3d.shape[0]),
        "cands": cands,
        "gt_pos": gt_pos,
        "centroid_err_m": err,
        "sanity_ok": bool(err <= spec.sanity_tol),
        "K": K,
        "T_world_cam": T_wc,
    }
