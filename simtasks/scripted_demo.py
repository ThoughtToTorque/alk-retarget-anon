"""Scripted expert demonstrations.

For each task, produces ONE successful demonstration consisting of
  * 3 keyframe captures (approach / pre_grasp / post_action) in the paper's
    demo format, each with a synchronized TCP pose, and
  * the full executed waypoint trajectory (per-control-step TCP poses +
    the commanded waypoint list).

Motions are scripted from ground-truth sim state and executed through the
OSC_POSE controller in absolute mode (control_delta=False): every env.step
receives [x, y, z, axis-angle(3), gripper] world-frame targets.  Targets are
interpolated from the current TCP pose in small increments so the OSC
controller tracks smoothly.
"""
import json
import os

import numpy as np
import robosuite.utils.transform_utils as T

try:
    from . import capture, envs, motion
except ImportError:  # allow running as a plain script
    import capture
    import envs
    import motion

# motion primitives moved to motion.py (shared with the retargeting runner);
# re-exported here for backward compatibility.
R_DOWN = motion.R_DOWN
GRIP_OPEN = motion.GRIP_OPEN
GRIP_CLOSE = motion.GRIP_CLOSE
BAR_GRASP_PHASE = motion.BAR_GRASP_PHASE
PAD_OFFSET = motion.PAD_OFFSET
rot_z = motion.rot_z
yaw_down_R = motion.yaw_down_R
Executor = motion.Executor
_quat_angle = motion.quat_angle


class KeyframeRecorder:
    """Captures the paper's 3 demo keyframes with synchronized TCP poses."""

    def __init__(self, env, spec, out_dir=None):
        self.env = env
        self.spec = spec
        self.out_dir = out_dir
        self.keyframes = []

    def capture(self, name, executor=None):
        entry = {"name": name, "index": len(self.keyframes)}
        pos, quat = capture.get_tcp_pose(self.env)
        entry["tcp_pos"] = pos.tolist()
        entry["tcp_quat_xyzw"] = quat.tolist()
        if executor is not None:
            entry["traj_index"] = len(executor.traj) - 1
        if self.out_dir is not None:
            kf_dir = os.path.join(self.out_dir, "keyframes",
                                  "%d_%s" % (len(self.keyframes), name))
            capture.capture_scene(self.env, kf_dir, label=name,
                                  extra_state=self.spec.extra_state(self.env))
            entry["dir"] = os.path.relpath(kf_dir, self.out_dir)
        self.keyframes.append(entry)

    def save(self):
        if self.out_dir is not None:
            with open(os.path.join(self.out_dir, "keyframes.json"), "w") as f:
                json.dump(self.keyframes, f, indent=2)


# ---------------------------------------------------------------------------
# grasp helpers
# ---------------------------------------------------------------------------

def _is_grasping(env, obj_geoms):
    return bool(env._check_grasp(gripper=env.robots[0].gripper,
                                 object_geoms=obj_geoms))


def _grasp_at(ex, kf, grasp_pos, yaw, obj_geoms, hover=0.10, descend_offset=0.0,
              capture_keyframes=True):
    """Approach from above, capture approach + pre_grasp keyframes, close.
    Returns True if the object is detected in the gripper."""
    env = ex.env
    R = yaw_down_R(yaw)
    grasp_pos = np.asarray(grasp_pos, dtype=float)
    ex.step_to(grasp_pos + [0, 0, hover], R, grip=GRIP_OPEN, label="approach")
    if capture_keyframes:
        kf.capture("approach", ex)
    ex.step_to(grasp_pos + [0, 0, descend_offset], R, grip=GRIP_OPEN,
               pos_tol=0.004, label="pre_grasp")
    if capture_keyframes:
        kf.capture("pre_grasp", ex)
    ex.set_gripper(GRIP_CLOSE)
    return _is_grasping(env, obj_geoms)


# ---------------------------------------------------------------------------
# per-task demonstrations
# ---------------------------------------------------------------------------

def _grasp_nut(env, ex, kf, nut_name):
    """Grasp a nut by its handle bar (with the orthogonal-yaw retry).
    Returns (grasped, grasp_yaw)."""
    sim = env.sim
    nut = [n for n in env.nuts if n.name == nut_name][0]
    handle = np.array(sim.data.get_site_xpos(nut_name + "_handle_site"))
    center = np.array(sim.data.get_site_xpos(nut_name + "_center_site"))
    bar = handle - center
    bar_yaw = float(np.arctan2(bar[1], bar[0]))
    grasp_yaw = bar_yaw + BAR_GRASP_PHASE

    grasped = _grasp_at(ex, kf, handle, grasp_yaw, nut.contact_geoms,
                        hover=0.10, descend_offset=-PAD_OFFSET)
    if not grasped:  # retry with the orthogonal yaw, without re-capturing
        ex.set_gripper(GRIP_OPEN)
        ex.step_to(handle + [0, 0, 0.10], yaw_down_R(grasp_yaw), label="retry_up")
        handle = np.array(sim.data.get_site_xpos(nut_name + "_handle_site"))
        grasp_yaw += np.pi / 2.0
        grasped = _grasp_at(ex, kf, handle, grasp_yaw, nut.contact_geoms,
                            hover=0.08, descend_offset=-PAD_OFFSET,
                            capture_keyframes=False)
    return grasped, grasp_yaw


def demo_nut_loosen(env, spec, ex, kf):
    """Bolt-loosening analog (object-centric): grasp the round nut by its
    handle bar and lift it clear off the table (>= 5 cm; the success checker
    compares against the settled initial height).  No world-fixed peg goal."""
    grasped, grasp_yaw = _grasp_nut(env, ex, kf, "RoundNut")

    # lift straight up well past the 5 cm success threshold and hold
    cur_pos, _ = capture.get_tcp_pose(env)
    R_carry = yaw_down_R(grasp_yaw)
    lift = np.array([cur_pos[0], cur_pos[1], cur_pos[2] + 0.15])
    ex.step_to(lift, R_carry, label="lift_off")
    for _ in range(10):  # hold, let any slip show up before the check
        ex._step(lift, R_carry, ex.grip)
    kf.capture("post_action", ex)
    return spec.check_success(env)


def demo_cap_twist(env, spec, ex, kf):
    """Cap-twisting analog (object-centric): grasp the square nut by its
    handle, lift it barely off the table, twist it ~45 deg about world z
    AROUND THE NUT CENTER (the TCP follows the arc so the center stays put),
    set it back down and release.  Success: >= 30 deg yaw change with the
    center within 3 cm of its initial xy (twist, not drag)."""
    sim = env.sim
    grasped, grasp_yaw = _grasp_nut(env, ex, kf, "SquareNut")
    center = np.array(sim.data.get_site_xpos("SquareNut_center_site"))

    # small lift so the twist does not fight table friction; the nut stays
    # within ~1.5 cm of the table (contact-rich, near-tabletop motion)
    tcp_pos, _ = capture.get_tcp_pose(env)
    R_g = yaw_down_R(grasp_yaw)
    lift = np.array([tcp_pos[0], tcp_pos[1], tcp_pos[2] + 0.015])
    ex.step_to(lift, R_g, label="lift_small")

    # twist about world z around the nut CENTER: rotate the TCP position on
    # its arc and the gripper orientation by the same angle
    twist = np.deg2rad(45.0)
    r = lift[:2] - center[:2]
    c, s = np.cos(twist), np.sin(twist)
    xy = center[:2] + np.array([c * r[0] - s * r[1], s * r[0] + c * r[1]])
    R_tw = rot_z(twist) @ R_g
    ex.step_to([xy[0], xy[1], lift[2]], R_tw, ori_tol=0.06, label="twist")

    # set down and release (the yaw change must persist without the gripper)
    ex.step_to([xy[0], xy[1], lift[2] - 0.012], R_tw, label="set_down")
    ex.set_gripper(GRIP_OPEN)
    tcp_pos, _ = capture.get_tcp_pose(env)
    ex.step_to(tcp_pos + [0, 0, 0.10], R_tw, label="retreat")
    kf.capture("post_action", ex)
    return spec.check_success(env)


def demo_rim_grasp(env, spec, ex, kf):
    """Rim-grasp analog (object-centric): grasp the can near its upper rim
    and lift it (>= 5 cm with both fingerpads in contact).  No world-fixed
    bin goal."""
    sim = env.sim
    can = env.objects[env.object_id]
    can_pos = np.array(sim.data.get_body_xpos("Can_main"))
    grasp = can_pos + [0.0, 0.0, 0.03]  # upper part of the cylinder (rim-ish)
    grasped = _grasp_at(ex, kf, grasp, 0.0, can.contact_geoms,
                        hover=0.12, descend_offset=0.0)

    cur_pos, _ = capture.get_tcp_pose(env)
    R = yaw_down_R(0.0)
    lift = np.array([cur_pos[0], cur_pos[1], cur_pos[2] + 0.12])
    ex.step_to(lift, R, label="lift_off")
    for _ in range(10):  # hold, let any slip show up before the check
        ex._step(lift, R, ex.grip)
    kf.capture("post_action", ex)
    return spec.check_success(env)


def demo_pour(env, spec, ex, kf):
    """Grasp the elongated block across its long axis, lift it, and execute a
    large wrist rotation (pouring analog; rotation-dominant post-action).

    The vessel is PourLift's elongated block (8.0 x 2.2 x 4.4 cm, long axis =
    body x): unlike the old near-symmetric cube it has an identifiable long
    axis (paper Sec. V scope), and the grasp is a bar grasp across that axis
    (nearest 180-deg gripper-equivalent yaw)."""
    sim = env.sim
    cube_pos = np.array(sim.data.get_body_xpos("cube_main"))
    cube_quat = T.convert_quat(sim.data.get_body_xquat("cube_main"), to="xyzw")
    long_yaw = T.mat2euler(T.quat2mat(cube_quat))[2]  # long-axis (body x) yaw
    # bar grasp across the long axis; fingers are 180-deg symmetric, so
    # reduce to the nearest equivalent yaw in [-pi/2, pi/2)
    yaw = (long_yaw + BAR_GRASP_PHASE + np.pi / 2.0) % np.pi - np.pi / 2.0
    grasped = _grasp_at(ex, kf, cube_pos, yaw, env.cube.contact_geoms,
                        hover=0.10, descend_offset=-0.005)

    R_g = yaw_down_R(yaw)
    lift = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.20])
    ex.step_to(lift, R_g, label="lift")
    # pouring rotation: tilt ~75 deg about the world x axis, hold, tilt back
    tilt = np.deg2rad(75.0)
    c, s = np.cos(tilt), np.sin(tilt)
    R_tilt = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) @ R_g
    ex.step_to(lift, R_tilt, ori_tol=0.06, label="pour_tilt")
    for _ in range(10):  # hold the pour
        ex._step(lift, R_tilt, ex.grip)
    kf.capture("post_action", ex)
    ex.step_to(lift, R_g, ori_tol=0.08, label="pour_back")
    return spec.check_success(env)


def _horizontal_R(n):
    """Gripper orientation for a horizontal approach along -n (n = outward
    door normal): approach axis (local z) points at the door, finger closing
    axis (local x) is vertical."""
    x = np.array([0.0, 0.0, 1.0])
    z = -np.asarray(n, dtype=float)
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def demo_box_open(env, spec, ex, kf):
    """Grasp the door handle bar with a horizontal approach and swing the
    door open along an arc around the hinge (a vertical descent onto the
    handle stalls against the arm's workspace limits in this env)."""
    sim = env.sim
    hinge_jid = sim.model.joint_name2id(env.door.joints[0])
    hinge = np.array(sim.data.xanchor[hinge_jid])

    handle_geom = sim.model.geom_name2id("Door_handle")
    bar_center = np.array(sim.data.geom_xpos[handle_geom])
    # outward door normal (side the handle sticks out of), robust to the
    # sampled door yaw
    door_mat = np.array(sim.data.get_body_xmat("Door_door"))
    n = door_mat[:, 1].copy()
    n[2] = 0.0
    n /= np.linalg.norm(n)
    door_pos = np.array(sim.data.get_body_xpos("Door_door"))
    if np.dot(n, bar_center - door_pos) < 0:
        n = -n
    R_g = _horizontal_R(n)

    # seat the bar DEEP between the fingerpads (PAD_OFFSET puts the pad
    # centers at the bar; the extra 6 mm buries the bar toward the palm so
    # the hold survives the lateral pull of the swing — at the pad-center
    # depth the hold was marginal and ~half of the sampled placements
    # slipped, see phase 2b notes)
    grasp_depth = PAD_OFFSET + 0.006
    ex.step_to(bar_center + n * 0.12, R_g, grip=GRIP_OPEN, ori_tol=0.06,
               label="approach")
    kf.capture("approach", ex)
    ex.step_to(bar_center - n * grasp_depth, R_g, grip=GRIP_OPEN,
               pos_tol=0.005, label="pre_grasp")
    kf.capture("pre_grasp", ex)
    ex.set_gripper(GRIP_CLOSE)

    # swing: command the TCP along the circle around the hinge axis; if the
    # handle slips out of the fingers, re-grasp at its current pose and
    # continue from the current hinge angle
    swing_total = np.deg2rad(55.0)
    n_arc = 16
    tcp_pos, _ = capture.get_tcp_pose(env)
    r_vec = tcp_pos[:2] - hinge[:2]
    a0 = 0.0
    regrasps = 0
    i = 1
    while i <= n_arc:
        a = a0 + (swing_total - a0) * i / n_arc
        c, s = np.cos(a - a0), np.sin(a - a0)
        xy = hinge[:2] + np.array([c * r_vec[0] - s * r_vec[1],
                                   s * r_vec[0] + c * r_vec[1]])
        R_i = rot_z(a) @ R_g
        ex.step_to([xy[0], xy[1], tcp_pos[2]], R_i, pos_tol=0.02,
                   ori_tol=0.2, max_steps=50, label="swing_%d" % i)
        if sim.data.qpos[env.hinge_qpos_addr] > 0.35:  # success margin
            break
        if not _is_grasping(env, ["Door_handle"]) and regrasps < 3:
            # slipped: re-approach the handle where it is now
            regrasps += 1
            a0 = float(sim.data.qpos[env.hinge_qpos_addr])
            bar_center = np.array(sim.data.geom_xpos[handle_geom])
            n_now = rot_z(a0) @ n
            R_g = _horizontal_R(n_now)
            ex.set_gripper(GRIP_OPEN, steps=8)
            ex.step_to(bar_center + n_now * 0.10, R_g, grip=GRIP_OPEN,
                       ori_tol=0.1, label="regrasp_standoff")
            ex.step_to(bar_center - n_now * grasp_depth, R_g, grip=GRIP_OPEN,
                       pos_tol=0.006, label="regrasp_insert")
            ex.set_gripper(GRIP_CLOSE)
            R_g = rot_z(-a0) @ R_g  # so that rot_z(a) @ R_g is current at a0
            tcp_pos, _ = capture.get_tcp_pose(env)
            r_vec = tcp_pos[:2] - hinge[:2]
            i = 1
            continue
        i += 1
    kf.capture("post_action", ex)
    return spec.check_success(env)


DEMOS = {
    "nut_loosen": demo_nut_loosen,
    "rim_grasp": demo_rim_grasp,
    "pour": demo_pour,
    "box_open": demo_box_open,
    "cap_twist": demo_cap_twist,
}


def run_demo(env, task, out_dir=None):
    """Run the scripted expert for `task` in the CURRENT env episode.

    Saves keyframes + trajectory under out_dir (if given).  Returns
    (success, info_dict)."""
    spec = envs.TASKS[task]
    ex = Executor(env)
    kf = KeyframeRecorder(env, spec, out_dir)
    success = bool(DEMOS[task](env, spec, ex, kf))
    kf.save()
    info = {"n_steps": len(ex.traj), "n_waypoints": len(ex.waypoints),
            "n_keyframes": len(kf.keyframes)}
    if out_dir is not None:
        ex.save(out_dir)
        with open(os.path.join(out_dir, "demo_meta.json"), "w") as f:
            json.dump({"task": task, "success": success, **info}, f, indent=2)
    return success, info
