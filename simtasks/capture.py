"""Scene capture in the paper's demo format.

Per scene directory:
  <cam>_rgb.png    uint8 HxWx3, OpenCV convention (row 0 = top of image)
  <cam>_depth.npy  float32 HxW, METRIC depth in meters (z-distance along the
                   camera optical axis, i.e. directly usable for pinhole
                   back-projection)
  <cam>_seg.npy    int32 HxW instance segmentation; 0 = background, ids match
                   meta["instance_id_to_name"] (same id convention robosuite
                   uses for camera_segmentations="instance" obs)
  meta.json        camera intrinsics/extrinsics, instance id map, ground-truth
                   object poses, robot TCP pose, gripper state, extra state

Conventions (verified by the back-projection sanity check in this module):
  * All image arrays are stored top-row-first (OpenCV).  Raw
    env.sim.render() output is bottom-row-first (OpenGL) and is flipped here.
  * Intrinsics K are from robosuite camera_utils (pinhole, principal point at
    image center).
  * T_world_cam is the 4x4 camera pose in the world frame with OpenCV camera
    axes (x right, y down, z forward), from
    camera_utils.get_camera_extrinsic_matrix.
    world_point = T_world_cam @ [x_cam, y_cam, z_cam, 1].
  * Quaternions are stored xyzw (robosuite convention).
"""
import json
import os

import imageio
import numpy as np
import robosuite.utils.camera_utils as cu
import robosuite.utils.transform_utils as T

TCP_SITE = "gripper0_grip_site"


# ---------------------------------------------------------------------------
# instance segmentation id mapping
# ---------------------------------------------------------------------------

def get_instance_id_map(env):
    """id -> instance name; identical numbering to robosuite's built-in
    camera_segmentations='instance' observable (index in
    env.model.instances_to_ids insertion order, +1; 0 is background)."""
    names = list(env.model.instances_to_ids.keys())
    id_map = {0: "BACKGROUND"}
    for i, n in enumerate(names):
        id_map[i + 1] = n
    return id_map


def _geom_id_lut(env):
    """LUT geom_id -> instance id (0 for unmapped geoms)."""
    names = list(env.model.instances_to_ids.keys())
    name2id = {n: i + 1 for i, n in enumerate(names)}
    lut = np.zeros(env.sim.model.ngeom, dtype=np.int32)
    for gid, inst in env.model.geom_ids_to_instances.items():
        lut[gid] = name2id[inst]
    return lut


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render_camera(env, camera, height, width):
    """Render one camera; returns (rgb uint8 HxWx3, metric depth float32 HxW,
    instance seg int32 HxW), all in OpenCV (top-row-first) convention."""
    rgb, depth = env.sim.render(camera_name=camera, width=width, height=height,
                                depth=True)
    rgb = np.ascontiguousarray(rgb[::-1])
    depth = np.ascontiguousarray(depth[::-1]).astype(np.float32)
    depth_m = cu.get_real_depth_map(env.sim, depth).astype(np.float32)

    seg = env.sim.render(camera_name=camera, width=width, height=height,
                         segmentation=True)[::-1]
    geom_ids = seg[..., 1].astype(np.int64)
    lut = _geom_id_lut(env)
    inst = np.zeros(geom_ids.shape, dtype=np.int32)
    valid = (geom_ids >= 0) & (geom_ids < lut.shape[0])
    inst[valid] = lut[geom_ids[valid]]
    return rgb, depth_m, inst


def camera_params(env, camera, height, width):
    K = cu.get_camera_intrinsic_matrix(env.sim, camera, height, width)
    T_world_cam = cu.get_camera_extrinsic_matrix(env.sim, camera)
    return K, T_world_cam


# ---------------------------------------------------------------------------
# ground truth state
# ---------------------------------------------------------------------------

def get_tcp_pose(env):
    """TCP (grip site) pose in world frame -> (pos(3), quat_xyzw(4))."""
    sid = env.sim.model.site_name2id(TCP_SITE)
    pos = np.array(env.sim.data.site_xpos[sid])
    mat = np.array(env.sim.data.site_xmat[sid]).reshape(3, 3)
    return pos, T.mat2quat(mat)


def get_object_poses(env):
    """Ground-truth 6-DoF poses of every scene object.

    An 'object' is any segmentation instance that owns a '<name>_main' body
    (this excludes the robot / gripper / mount instances).  Returns
    {name: {"pos": [...], "quat_xyzw": [...], "body": body_name}}."""
    body_names = set(env.sim.model.body_names)
    poses = {}
    for name in env.model.instances_to_ids.keys():
        body = name + "_main"
        if body not in body_names:
            continue
        pos = env.sim.data.get_body_xpos(body)
        quat = T.convert_quat(env.sim.data.get_body_xquat(body), to="xyzw")
        poses[name] = {
            "pos": [float(x) for x in pos],
            "quat_xyzw": [float(x) for x in quat],
            "body": body,
        }
    return poses


# ---------------------------------------------------------------------------
# capture / load
# ---------------------------------------------------------------------------

def capture_scene(env, out_dir, cameras=None, height=None, width=None,
                  label=None, extra_state=None):
    """Save the current sim state in the paper's demo format.  Returns the
    meta dict."""
    cameras = list(cameras if cameras is not None
                   else getattr(env, "_simtasks_cameras", ["agentview"]))
    size = getattr(env, "_simtasks_cam_size", 256)
    height = height or size
    width = width or size
    os.makedirs(out_dir, exist_ok=True)

    cam_meta = {}
    for cam in cameras:
        rgb, depth, seg = render_camera(env, cam, height, width)
        K, T_wc = camera_params(env, cam, height, width)
        imageio.imwrite(os.path.join(out_dir, "%s_rgb.png" % cam), rgb)
        np.save(os.path.join(out_dir, "%s_depth.npy" % cam), depth)
        np.save(os.path.join(out_dir, "%s_seg.npy" % cam), seg)
        cam_meta[cam] = {
            "height": height,
            "width": width,
            "K": K.tolist(),
            "T_world_cam": T_wc.tolist(),
        }

    tcp_pos, tcp_quat = get_tcp_pose(env)
    gripper_qpos = [float(x) for x in
                    env.sim.data.qpos[env.robots[0]._ref_gripper_joint_pos_indexes]]
    meta = {
        "label": label,
        "task": getattr(env, "_simtasks_task", None),
        "cameras": cam_meta,
        "instance_id_to_name": {str(k): v
                                for k, v in get_instance_id_map(env).items()},
        "objects": get_object_poses(env),
        "tcp": {"pos": [float(x) for x in tcp_pos],
                "quat_xyzw": [float(x) for x in tcp_quat]},
        "gripper_qpos": gripper_qpos,
        "extra_state": extra_state or {},
        "conventions": {
            "image_origin": "top-left (OpenCV); flipped from mujoco's OpenGL render",
            "depth": "metric meters, z-distance along camera optical axis",
            "quat": "xyzw",
            "T_world_cam": "camera pose in world, OpenCV camera axes",
        },
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_capture(scene_dir):
    """Load a capture directory back into memory."""
    with open(os.path.join(scene_dir, "meta.json")) as f:
        meta = json.load(f)
    out = {"meta": meta, "cameras": {}}
    for cam, cm in meta["cameras"].items():
        out["cameras"][cam] = {
            "rgb": imageio.imread(os.path.join(scene_dir, "%s_rgb.png" % cam)),
            "depth": np.load(os.path.join(scene_dir, "%s_depth.npy" % cam)),
            "seg": np.load(os.path.join(scene_dir, "%s_seg.npy" % cam)),
            "K": np.array(cm["K"]),
            "T_world_cam": np.array(cm["T_world_cam"]),
        }
    return out


# ---------------------------------------------------------------------------
# back-projection + sanity check
# ---------------------------------------------------------------------------

def backproject_depth(depth, K):
    """Metric depth HxW -> camera-frame points HxWx3 (OpenCV axes)."""
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (us - cx) / fx * z
    y = (vs - cy) / fy * z
    return np.stack([x, y, z], axis=-1)


def instance_points_world(cam_data, instance_id):
    """World-frame 3D points of one segmented instance from a loaded capture
    camera entry."""
    mask = cam_data["seg"] == instance_id
    if not mask.any():
        return np.zeros((0, 3))
    pts_cam = backproject_depth(cam_data["depth"], cam_data["K"])[mask]
    T_wc = cam_data["T_world_cam"]
    return pts_cam @ T_wc[:3, :3].T + T_wc[:3, 3]


def sanity_check_backprojection(capture, camera, instance_name, gt_pos,
                                atol=0.06, min_pixels=20):
    """Back-project the target instance's segmented pixels and compare the
    centroid against the ground-truth object position.

    Returns (ok, err_meters, n_pixels)."""
    meta = capture["meta"]
    inst_id = None
    for k, v in meta["instance_id_to_name"].items():
        if v == instance_name:
            inst_id = int(k)
            break
    if inst_id is None:
        return False, float("inf"), 0
    pts = instance_points_world(capture["cameras"][camera], inst_id)
    if pts.shape[0] < min_pixels:
        return False, float("inf"), pts.shape[0]
    centroid = pts.mean(axis=0)
    err = float(np.linalg.norm(centroid - np.asarray(gt_pos)))
    return err <= atol, err, pts.shape[0]
