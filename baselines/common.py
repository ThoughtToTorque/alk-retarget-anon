"""Shared plumbing for the Phase-3 baseline methods (B1-B4).

Everything here is importable WITHOUT robosuite/mujoco: captures are read
straight from disk (npy/json; RGB via imageio only when actually needed), and
perception re-uses the pure-numpy alkbench candidate pipeline.  The tiny
task table below duplicates the fields of ``simtasks.envs.TASKS`` that
perception needs (target instance name, sanity tolerance, camera priority) so
baselines never import the simulator -- keep it in sync with simtasks/envs.py.

All baseline methods share one signature (see runner_hooks):

    method(demo_capture, target_capture, ctx) -> result dict

where the captures are ``load_capture`` dicts and ``ctx`` is a
:class:`MethodContext` carrying the per-pair information (task, oracle T_gt,
demo keyframes/waypoints, seeds, noise settings).  Every result dict contains
at least ``T_map`` (4x4 world-frame demo->target object transform) so it plugs
into the execution path of ``pipeline/retarget_runner.py``
(``retarget_waypoint_dicts`` + ``simtasks.motion.execute_waypoints``).
"""
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

from alkbench import compute_candidates

# ---------------------------------------------------------------------------
# task table (kept in sync with simtasks.envs.TASKS -- perception fields only)
# ---------------------------------------------------------------------------

TASK_INFO = {
    "nut_loosen": {"target_instance": "RoundNut", "sanity_tol": 0.06},
    "rim_grasp": {"target_instance": "Can", "sanity_tol": 0.06},
    "pour": {"target_instance": "cube", "sanity_tol": 0.06},
    "box_open": {"target_instance": "Door", "sanity_tol": 0.30},
    "cap_twist": {"target_instance": "SquareNut", "sanity_tol": 0.06},
}

# same camera priorities as pipeline.perception
DEFAULT_PRIORITY = ("agentview", "sideview", "frontview")
CAMERA_PRIORITY = {
    "box_open": ("sideview", "frontview", "agentview"),
}


def pick_camera(cap, task):
    priority = CAMERA_PRIORITY.get(task, DEFAULT_PRIORITY)
    for cam in priority:
        if cam in cap["cameras"]:
            return cam
    return sorted(cap["cameras"].keys())[0]


# ---------------------------------------------------------------------------
# capture / pair loading (no robosuite import, unlike simtasks.capture)
# ---------------------------------------------------------------------------

def load_capture(scene_dir, load_rgb=False):
    """Load a simtasks capture directory.  RGB is optional (only the VLM
    variants need it) so the loader works without imageio installed."""
    with open(os.path.join(scene_dir, "meta.json")) as f:
        meta = json.load(f)
    out = {"meta": meta, "cameras": {}, "dir": scene_dir}
    for cam, cm in meta["cameras"].items():
        entry = {
            "depth": np.load(os.path.join(scene_dir, "%s_depth.npy" % cam)),
            "seg": np.load(os.path.join(scene_dir, "%s_seg.npy" % cam)),
            "K": np.asarray(cm["K"], dtype=np.float64),
            "T_world_cam": np.asarray(cm["T_world_cam"], dtype=np.float64),
            "rgb": None,
        }
        if load_rgb:
            entry["rgb"] = _imread(os.path.join(scene_dir,
                                                "%s_rgb.png" % cam))
        out["cameras"][cam] = entry
    return out


def _imread(path):
    try:
        import imageio.v2 as imageio  # imageio >= 2.16
    except ImportError:  # pragma: no cover
        import imageio
    return np.asarray(imageio.imread(path))[..., :3]


def load_rgb(capture, camera):
    """Lazily load (and cache) the RGB image of one camera of a capture."""
    cd = capture["cameras"][camera]
    if cd.get("rgb") is None:
        cd["rgb"] = _imread(os.path.join(capture["dir"],
                                         "%s_rgb.png" % camera))
    return cd["rgb"]


def load_pair(pair_dir, load_rgb_images=False):
    """Load one (task, seed) scene-pair directory (simtasks.scene_pairs
    layout): pair.json + demo scene/keyframes/waypoints + target scene."""
    with open(os.path.join(pair_dir, "pair.json")) as f:
        pair = json.load(f)
    demo_dir = os.path.join(pair_dir, "demo")
    with open(os.path.join(demo_dir, "keyframes.json")) as f:
        keyframes = json.load(f)
    with open(os.path.join(demo_dir, "waypoints.json")) as f:
        waypoints = json.load(f)
    return {
        "pair": pair,
        "demo_capture": load_capture(os.path.join(demo_dir, "scene"),
                                     load_rgb=load_rgb_images),
        "target_capture": load_capture(os.path.join(pair_dir, "target",
                                                    "scene"),
                                       load_rgb=load_rgb_images),
        "keyframes": keyframes,
        "waypoints": waypoints,
    }


# ---------------------------------------------------------------------------
# perception (mirrors pipeline.perception.perceive, minus the robosuite dep)
# ---------------------------------------------------------------------------

def instance_id_for(meta, instance_name):
    for k, v in meta["instance_id_to_name"].items():
        if v == instance_name:
            return int(k)
    raise KeyError("instance %r not in capture id map" % instance_name)


def perceive(cap, task, k=8, seed=0, camera=None, sensor_noise=None,
             noise_seed=0):
    """Capture -> world-frame object cloud + k candidate keypoints.

    ``sensor_noise`` (optional): a difficulty-tier dict
    {"depth_sigma_m": float, "mask_erosion_px": int} applied to the raw
    depth/mask before backprojection (hard tier; see baselines.difficulty).
    """
    info = TASK_INFO[task]
    cam = camera or pick_camera(cap, task)
    cd = cap["cameras"][cam]
    inst_id = instance_id_for(cap["meta"], info["target_instance"])
    mask = cd["seg"] == inst_id
    depth = cd["depth"]
    if sensor_noise:
        from baselines import difficulty
        depth, mask = difficulty.apply_sensor_noise(
            depth, mask, sensor_noise, np.random.RandomState(noise_seed))
    K = np.asarray(cd["K"], dtype=np.float64)
    T_wc = np.asarray(cd["T_world_cam"], dtype=np.float64)
    cands = compute_candidates(mask, depth,
                               K[0, 0], K[1, 1], K[0, 2], K[1, 2],
                               extrinsic=T_wc, k=k, seed=seed)
    gt_pos = np.asarray(
        cap["meta"]["objects"][info["target_instance"]]["pos"],
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
        "sanity_ok": bool(err <= info["sanity_tol"]),
        "K": K,
        "T_world_cam": T_wc,
    }


# ---------------------------------------------------------------------------
# method context
# ---------------------------------------------------------------------------

class MethodContext(object):
    """Per-pair context handed to every baseline method.

    Parameters
    ----------
    task : task key
    T_gt : (4,4) ground-truth relative object transform demo->target.  Used
        ONLY for the oracle discrete answers (keypoint correspondence, marked
        pixels) -- exactly the information the paper's VLM supplies; all
        continuous geometry stays estimated.  ``None`` for pure-VLM variants.
    keyframes / waypoints : demo recording (keyframes.json / waypoints.json)
    k, kmeans_seed : candidate pipeline parameters (defaults match pipeline)
    camera : explicit camera override (default: per-task priority)
    sigma_px : MOKA pixel-noise sigma (px)
    noise_seed : RNG seed for pixel/sensor noise
    registration : bounded Chamfer refinement on/off for the B4 stack
    sensor_noise : difficulty-tier sensor noise dict or None
    rekep_n_keypoints, random4_seed : method-specific knobs
    vlm_* : OpenAI-compatible endpoint config (defaults from environment,
        same conventions as alkbench.discrete.VLMSolver)
    """

    def __init__(self, task, T_gt=None, keyframes=None, waypoints=None,
                 k=8, kmeans_seed=0, camera=None,
                 sigma_px=0.0, noise_seed=0,
                 registration=True, sensor_noise=None,
                 rekep_n_keypoints=4, random4_seed=0,
                 vlm_model=None, vlm_base_url=None, vlm_api_key=None):
        self.task = task
        self.T_gt = None if T_gt is None else np.asarray(T_gt,
                                                         dtype=np.float64)
        self.keyframes = keyframes
        self.waypoints = waypoints
        self.k = int(k)
        self.kmeans_seed = int(kmeans_seed)
        self.camera = camera
        self.sigma_px = float(sigma_px)
        self.noise_seed = int(noise_seed)
        self.registration = bool(registration)
        self.sensor_noise = sensor_noise
        self.rekep_n_keypoints = int(rekep_n_keypoints)
        self.random4_seed = int(random4_seed)
        self.vlm_model = vlm_model
        self.vlm_base_url = vlm_base_url
        self.vlm_api_key = vlm_api_key
        self._percepts = {}

    # -- cached perception -------------------------------------------------
    def percept(self, capture, role):
        """Perceive a capture once per (role, camera) and cache the result.
        role is "demo" or "target" (cache key within this context)."""
        cam = self.camera or pick_camera(capture, self.task)
        key = (role, cam)
        if key not in self._percepts:
            self._percepts[key] = perceive(
                capture, self.task, k=self.k, seed=self.kmeans_seed,
                camera=cam, sensor_noise=self.sensor_noise,
                noise_seed=self.noise_seed)
        return self._percepts[key]

    def keyframe_tcp(self, name):
        """TCP position (3,) of a named demo keyframe (e.g. 'pre_grasp')."""
        if not self.keyframes:
            raise ValueError("MethodContext has no demo keyframes")
        for kf in self.keyframes:
            if kf["name"] == name:
                return np.asarray(kf["tcp_pos"], dtype=np.float64)
        raise KeyError("no keyframe named %r" % name)

    def require_T_gt(self, what):
        if self.T_gt is None:
            raise ValueError("%s needs ctx.T_gt (oracle discrete answers); "
                             "got None" % what)
        return self.T_gt


def context_from_pair(pair_dir, **overrides):
    """Convenience: build (demo_capture, target_capture, ctx) from a scene
    pair directory (data/<task>/<seed> or baselines/testdata/<task>/<seed>).
    T_gt is computed from the saved GT poses like the pipeline does."""
    from pipeline import oracle  # numpy/scipy only, no robosuite
    loaded = load_pair(pair_dir)
    pair = loaded["pair"]
    inst = pair["target_instance"]
    T_gt = oracle.gt_relative_transform(pair["demo_object_poses"][inst],
                                        pair["target_object_poses"][inst])
    kwargs = dict(task=pair["task"], T_gt=T_gt,
                  keyframes=loaded["keyframes"],
                  waypoints=loaded["waypoints"])
    kwargs.update(overrides)
    ctx = MethodContext(**kwargs)
    return loaded["demo_capture"], loaded["target_capture"], ctx, pair


# ---------------------------------------------------------------------------
# SE(3) / config-matrix helpers
# ---------------------------------------------------------------------------

def se3_from_rotvec(rotvec, t):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec,
                                                dtype=np.float64)).as_matrix()
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def is_valid_se3(T, tol=1e-6):
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    R = T[:3, :3]
    return (np.allclose(R @ R.T, np.eye(3), atol=1e-5)
            and abs(np.linalg.det(R) - 1.0) < 1e-5
            and np.allclose(T[3], [0, 0, 0, 1], atol=tol))


def shortest_arc_rotation(a, b):
    """Minimal rotation matrix mapping unit-ish vector a onto b (zero twist
    about the axis).  Antiparallel case: 180 deg about an arbitrary
    perpendicular axis."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = np.cross(a, b)
    d = float(np.dot(a, b))
    if np.linalg.norm(c) < 1e-12:
        if d > 0:
            return np.eye(3)
        # antiparallel: any perpendicular axis
        p = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            p = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, p)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * np.pi).as_matrix()
    axis = c / np.linalg.norm(c)
    angle = np.arctan2(np.linalg.norm(c), d)
    return Rotation.from_rotvec(axis * angle).as_matrix()


def conditioning(points):
    """Singular values of the CENTERED keypoint configuration and the
    theory-linked conditioning number sigma2 + sigma3 (Proposition 3: the
    Procrustes rotation-error bound scales with 1/(sigma2+sigma3); a rank-1
    configuration, e.g. 2 collinear points, has sigma2+sigma3 = 0 and leaves
    rotation about the axis unobservable)."""
    P = np.asarray(points, dtype=np.float64)
    X = P - P.mean(axis=0)
    s = np.linalg.svd(X, compute_uv=False)
    s = np.concatenate([s, np.zeros(3)])[:3]
    return {"singular_values": [float(v) for v in s],
            "sigma23": float(s[1] + s[2])}


def gt_nn_match(demo_points, target_candidates, T_gt):
    """Oracle correspondence: greedily match each T_gt-mapped demo point to
    the nearest UNUSED target candidate (injective; pairs assigned in order
    of ascending distance).  Returns (indices into target_candidates, dists).
    """
    from alkbench import transform_points
    P = np.asarray(demo_points, dtype=np.float64)
    C = np.asarray(target_candidates, dtype=np.float64)
    if P.shape[0] > C.shape[0]:
        raise ValueError("more demo points than target candidates")
    M = transform_points(T_gt, P)
    D = np.linalg.norm(M[:, None, :] - C[None, :, :], axis=2)
    idx = np.full(P.shape[0], -1, dtype=int)
    used = np.zeros(C.shape[0], dtype=bool)
    order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
    n_done = 0
    for i, j in order:
        if idx[i] >= 0 or used[j]:
            continue
        idx[i] = j
        used[j] = True
        n_done += 1
        if n_done == P.shape[0]:
            break
    dists = D[np.arange(P.shape[0]), idx]
    return idx, dists


def farthest_point_indices(points, n, start="max_norm"):
    """Deterministic farthest-point sampling over a small point set.
    start="max_norm": first index = point farthest from the centroid."""
    P = np.asarray(points, dtype=np.float64)
    if n > P.shape[0]:
        raise ValueError("FPS: n > number of points")
    d0 = np.linalg.norm(P - P.mean(axis=0), axis=1)
    idx = [int(np.argmax(d0))]
    dmin = np.linalg.norm(P - P[idx[0]], axis=1)
    while len(idx) < n:
        j = int(np.argmax(dmin))
        idx.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(P - P[j], axis=1))
    return idx


def base_result(method, T_map, **aux):
    """Uniform result schema: {"method", "T_map" (4x4 list), **aux}."""
    T = np.asarray(T_map, dtype=np.float64)
    out = {"method": method, "T_map": T.tolist()}
    out.update(aux)
    return out
