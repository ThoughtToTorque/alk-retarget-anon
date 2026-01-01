"""Shared motion primitives: absolute-OSC waypoint execution.

Extracted from scripted_demo.py so both the scripted expert (demo side) and
the retargeting runner (target side) drive the arm through the exact same
machinery: absolute OSC_POSE targets (control_delta=False), interpolated in
<= max_step m / <= max_rot rad sub-goals per control step.

Also provides `execute_waypoints`, which replays a recorded/retargeted
waypoint list (the schema written by Executor.save -> waypoints.json) in a
live env, with per-label tolerances mirroring the scripted expert and the
demo's gripper open/close schedule.
"""
import numpy as np
import robosuite.utils.transform_utils as T

try:
    from . import capture
except ImportError:  # allow running as a plain script
    import capture

# canonical top-down grasp orientation (gripper z pointing down);
# same matrix robosuite's OSC uses as its default orientation
R_DOWN = np.array([[0.0, 1.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0]])

GRIP_OPEN = -1.0
GRIP_CLOSE = 1.0

# To grasp a bar whose axis has world yaw `theta`, command
#   R = Rz(theta + BAR_GRASP_PHASE) @ R_DOWN
# Empirically (verified on the nut handles) the Robotiq-85 finger closing
# axis is already perpendicular to the bar at phase 0.
BAR_GRASP_PHASE = 0.0

# The Robotiq-85 fingerpad centers sit ~1.3 cm ABOVE the gripper0_grip_site
# TCP frame; for thin bars (nut handles, door handle) the TCP must therefore
# be commanded slightly BELOW the bar center so the pads straddle the bar.
PAD_OFFSET = 0.013


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_down_R(theta):
    """Top-down grasp orientation with world-frame yaw theta applied."""
    return rot_z(theta) @ R_DOWN


def quat_angle(q1, q2):
    dot = min(1.0, abs(float(np.dot(q1, q2))))
    return 2.0 * np.arccos(dot)


_quat_angle = quat_angle  # backward-compat alias


def pose_to_matrix(pos, quat_xyzw):
    """(pos(3), quat_xyzw(4)) -> 4x4 homogeneous pose."""
    M = np.eye(4)
    M[:3, :3] = T.quat2mat(np.asarray(quat_xyzw, dtype=float))
    M[:3, 3] = np.asarray(pos, dtype=float)
    return M


def matrix_to_pose(M):
    """4x4 -> (pos(3), quat_xyzw(4))."""
    M = np.asarray(M, dtype=float)
    return M[:3, 3].copy(), T.mat2quat(M[:3, :3])


class Executor:
    """Steps the env toward absolute TCP pose targets with interpolated
    sub-goals; records the executed trajectory."""

    def __init__(self, env, max_step=0.02, max_rot=0.2, step_callback=None):
        self.env = env
        self.max_step = max_step      # max position increment per control step (m)
        self.max_rot = max_rot        # max orientation increment per control step (rad)
        self.grip = GRIP_OPEN
        self.traj = []                # per-control-step record
        self.waypoints = []           # commanded waypoint list
        # optional hook, called as step_callback(env, step_index) after every
        # recorded control step (used by demo/record_rollout.py to render
        # video frames offscreen; None = default behavior, no overhead)
        self.step_callback = step_callback
        self.record()

    # -- bookkeeping --------------------------------------------------------
    def record(self, action=None):
        pos, quat = capture.get_tcp_pose(self.env)
        self.traj.append({
            "tcp_pos": pos.tolist(),
            "tcp_quat_xyzw": quat.tolist(),
            "gripper": float(self.grip),
            "action": None if action is None else [float(a) for a in action],
        })
        if self.step_callback is not None:
            self.step_callback(self.env, len(self.traj) - 1)

    def _tcp(self):
        return capture.get_tcp_pose(self.env)

    def _step(self, pos, R, grip):
        aa = T.quat2axisangle(T.mat2quat(R))
        action = np.concatenate([pos, aa, [grip]])
        self.env.step(action)
        self.record(action)

    # -- motion primitives ---------------------------------------------------
    def step_to(self, pos, R=None, grip=None, pos_tol=0.007, ori_tol=0.08,
                max_steps=200, settle=5, label=None):
        """Move the TCP to an absolute world pose. Returns True if converged."""
        pos = np.asarray(pos, dtype=float)
        if grip is not None:
            self.grip = float(grip)
        if R is None:
            _, cur_quat = self._tcp()
            R = T.quat2mat(cur_quat)
        q_target = T.mat2quat(R)
        self.waypoints.append({
            "type": "move", "label": label, "pos": pos.tolist(),
            "quat_xyzw": q_target.tolist(), "gripper": float(self.grip),
            "traj_index": len(self.traj) - 1,
        })

        converged = False
        for _ in range(max_steps):
            cur_pos, cur_quat = self._tcp()
            d = pos - cur_pos
            dist = np.linalg.norm(d)
            ang = quat_angle(cur_quat, q_target)
            if dist < pos_tol and ang < ori_tol:
                converged = True
                break
            # interpolated sub-goal
            sub_pos = pos if dist <= self.max_step else cur_pos + d / dist * self.max_step
            frac = 1.0 if ang <= self.max_rot else self.max_rot / ang
            sub_q = T.quat_slerp(cur_quat, q_target, frac)
            self._step(sub_pos, T.quat2mat(sub_q), self.grip)
        for _ in range(settle):
            self._step(pos, R, self.grip)
        return converged

    def set_gripper(self, grip, steps=15):
        """Open/close the gripper in place."""
        self.grip = float(grip)
        cur_pos, cur_quat = self._tcp()
        R = T.quat2mat(cur_quat)
        self.waypoints.append({
            "type": "gripper", "label": "close" if grip > 0 else "open",
            "pos": cur_pos.tolist(), "quat_xyzw": cur_quat.tolist(),
            "gripper": float(grip), "traj_index": len(self.traj) - 1,
        })
        for _ in range(steps):
            self._step(cur_pos, R, self.grip)

    def save(self, out_dir):
        import json
        import os
        os.makedirs(out_dir, exist_ok=True)
        n = len(self.traj)
        np.savez(
            os.path.join(out_dir, "trajectory.npz"),
            tcp_pos=np.array([t["tcp_pos"] for t in self.traj]),
            tcp_quat_xyzw=np.array([t["tcp_quat_xyzw"] for t in self.traj]),
            gripper=np.array([t["gripper"] for t in self.traj]),
        )
        with open(os.path.join(out_dir, "waypoints.json"), "w") as f:
            json.dump(self.waypoints, f, indent=2)
        return n


# ---------------------------------------------------------------------------
# waypoint-list replay (target-side execution of retargeted demo waypoints)
# ---------------------------------------------------------------------------

# per-label step_to parameters, mirroring the tolerances the scripted expert
# used when it recorded the demo (scripted_demo.py); matched by exact label,
# then by prefix.
LABEL_PARAMS = {
    "pre_grasp": {"pos_tol": 0.004},
    "over_peg": {"pos_tol": 0.004},
    "insert": {"pos_tol": 0.004, "max_steps": 120},
    "lower": {"max_steps": 120},
    "align_yaw": {"ori_tol": 0.04},
    "twist": {"ori_tol": 0.06},
    "pour_tilt": {"ori_tol": 0.06},
    "pour_back": {"ori_tol": 0.08},
}
LABEL_PREFIX_PARAMS = {
    "swing": {"pos_tol": 0.02, "ori_tol": 0.2, "max_steps": 50},
    "regrasp": {"pos_tol": 0.006, "ori_tol": 0.1},
}


def _params_for(label):
    if label in LABEL_PARAMS:
        return dict(LABEL_PARAMS[label])
    if label:
        for prefix, p in LABEL_PREFIX_PARAMS.items():
            if str(label).startswith(prefix):
                return dict(p)
    return {}


def execute_waypoints(env, waypoints, grasp_check=None, executor=None,
                      step_callback=None):
    """Replay a waypoint list (schema of Executor.save / waypoints.json) in
    `env`.  Positions/orientations are taken as-is (absolute world targets);
    the gripper schedule is the list's own open/close events.

    Parameters
    ----------
    env : live robosuite env (absolute OSC_POSE)
    waypoints : list of dicts with keys type ('move'|'gripper'), label,
        pos(3), quat_xyzw(4), gripper (float)
    grasp_check : optional callable env -> bool, evaluated after every
        gripper-close event
    executor : optional pre-built Executor (else one is created)

    Returns
    -------
    dict with keys "executor", "n_steps", "grasped" (result of the last
    grasp_check after a close event, or None), "converged" (per-move list).
    """
    ex = (executor if executor is not None
          else Executor(env, step_callback=step_callback))
    grasped = None
    converged = []
    for wp in waypoints:
        if wp["type"] == "gripper":
            ex.set_gripper(float(wp["gripper"]))
            if wp["gripper"] > 0 and grasp_check is not None:
                grasped = bool(grasp_check(env))
        else:
            R = T.quat2mat(np.asarray(wp["quat_xyzw"], dtype=float))
            kw = _params_for(wp.get("label"))
            ok = ex.step_to(np.asarray(wp["pos"], dtype=float), R,
                            grip=float(wp["gripper"]), label=wp.get("label"),
                            **kw)
            converged.append(bool(ok))
    return {"executor": ex, "n_steps": len(ex.traj), "grasped": grasped,
            "converged": converged}
