"""Task environment factory.

Maps the 5 real-robot tasks of the paper (one-shot keypoint-anchored
retargeting) to robosuite 1.4.1 built-in environments.  See TASKS.md for the
mapping rationale and conventions.

All environments use:
  robot        UR5e (closest built-in analog of the paper's UR3)
  controller   OSC_POSE with control_delta=False (absolute world-frame
               end-effector pose targets -> easy to script and to retarget)
  control_freq 20 Hz
  cameras      agentview (primary) + sideview (secondary); rendering is done
               on demand from env.sim (use_camera_obs=False for speed).
"""
import numpy as np
import robosuite as suite
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_controller_config

DEFAULT_CAMERAS = ("agentview", "sideview")
DEFAULT_CAM_SIZE = 256


class TaskSpec:
    """Static description of one benchmark task."""

    def __init__(self, name, env_name, env_kwargs, target_instance, target_body,
                 obj_pos_key, description, sanity_tol=0.06, extra_state=None,
                 make_placement_initializer=None, success_fn=None):
        # optional factory for a WIDENED placement sampler (passed to
        # suite.make; persists across hard resets)
        self.make_placement_initializer = make_placement_initializer
        self.name = name
        self.env_name = env_name
        self.env_kwargs = dict(env_kwargs)
        # instance name as it appears in the segmentation instance map
        self.target_instance = target_instance
        # mujoco body giving the ground-truth 6-DoF object pose
        self.target_body = target_body
        # obs key holding the object position (informational)
        self.obj_pos_key = obj_pos_key
        self.description = description
        # tolerance (m) for the depth back-projection centroid sanity check
        self.sanity_tol = sanity_tol
        # optional callable env -> dict of extra ground-truth state
        self._extra_state = extra_state
        # OBJECT-CENTRIC success override (phase 2b): callable env -> bool
        # evaluated against GT sim state + the settled initial object pose
        # recorded by reset_with_seed.  When None, the env's own native
        # _check_success is used.  The paper's tasks all act on the object
        # itself (no world-fixed goals), so tasks whose robosuite-native
        # success is "object on world-fixed peg/bin" override it here.
        self.success_fn = success_fn

    def extra_state(self, env):
        return self._extra_state(env) if self._extra_state is not None else {}

    def check_success(self, env):
        if self.success_fn is not None:
            return bool(self.success_fn(env))
        return bool(env._check_success())


# ---------------------------------------------------------------------------
# object-centric success checkers (phase 2b)
#
# The paper's 5 real tasks have NO world-fixed goals — every goal is attached
# to the manipulated object (grasp cup rim, pour, open carton, twist cap,
# loosen bolt).  robosuite's native success for the NutAssembly / PickPlace
# analogs is "object placed on a WORLD-FIXED peg / bin", which object-relative
# one-shot retargeting misses by construction (phase 2 finding).  These
# checkers redefine success purely from the object's own state relative to
# its settled initial pose (recorded by reset_with_seed after settling).
# ---------------------------------------------------------------------------

def _initial_object_pose(env):
    st = getattr(env, "_simtasks_init_obj_pose", None)
    if st is None:
        raise RuntimeError(
            "no settled initial object pose recorded on this env; reset it "
            "with envs.reset_with_seed (which settles the scene and records "
            "the pose) before checking success")
    return st


def _body_pose(env, body):
    pos = np.array(env.sim.data.get_body_xpos(body))
    quat = T.convert_quat(np.array(env.sim.data.get_body_xquat(body)),
                          to="xyzw")
    return pos, quat


def _wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _grasping(env, obj_geoms):
    return bool(env._check_grasp(gripper=env.robots[0].gripper,
                                 object_geoms=obj_geoms))


def _nut_loosen_success(env):
    """Nut 'loosened/removed': lifted >= 5 cm above its settled initial
    resting height (held or set free without falling back)."""
    init = _initial_object_pose(env)
    pos, _ = _body_pose(env, "RoundNut_main")
    return bool(pos[2] - init["pos"][2] >= 0.05)


def _cap_twist_success(env):
    """Nut twisted IN PLACE: rotated >= 30 deg about world z relative to its
    settled initial yaw while its center stayed within 3 cm of the initial
    xy (a twist, not a drag)."""
    init = _initial_object_pose(env)
    pos, quat = _body_pose(env, "SquareNut_main")
    yaw = float(T.mat2euler(T.quat2mat(quat))[2])
    dyaw = abs(_wrap_angle(yaw - init["yaw"]))
    dxy = float(np.linalg.norm(pos[:2] - np.asarray(init["pos"])[:2]))
    return bool(dyaw >= np.deg2rad(30.0) and dxy <= 0.03)


def _rim_grasp_success(env):
    """Can lifted >= 5 cm in a stable grasp (both fingerpads in contact),
    any xy — the goal moves with the object, as in the paper."""
    init = _initial_object_pose(env)
    pos, _ = _body_pose(env, "Can_main")
    lifted = pos[2] - init["pos"][2] >= 0.05
    can = env.objects[env.object_id]
    return bool(lifted and _grasping(env, can.contact_geoms))


# ---------------------------------------------------------------------------
# pour: ORIENTATION-AWARE success (2026-08-16 criterion revision)
#
# The superseded criterion for `pour` was the env-native Lift check ("block
# above the table by 4 cm") evaluated at the END of the rollout, i.e. plain
# grasp-and-hold-through-the-tilt.  That criterion is BLIND to the very
# quantity the task is supposed to probe: the paper classifies pour as
# rotation-dominant, yet an execution whose object transform is off by the
# vessel's 180-deg body flip (Rz_body(pi), a symmetry of the elongated
# block) grasps the block identically, tilts it by the same 75 deg -- in the
# OPPOSITE sense -- and was scored a success.  Campaign E measured the dense
# family (icp / icp_centroid / kp_dense_init) taking exactly that flip on
# ~half the seeds (raw rotation error 136-173 deg vs GT) while no keypoint
# method ever took it.  A rotation-dominant task whose success functional
# cannot distinguish a 180-deg orientation error is measuring the wrong
# thing, so the criterion is made ORIENTATION-AWARE here.
#
# Construction (all quantities read off the FIXED demonstration; nothing
# else about the task, object, sampler or demo changed):
#   u_body   the vessel's body-frame axis that points away from gravity at
#            the demo's PRE-GRASP keyframe (the "up" face of the vessel);
#            the block rests flat, so u_body = +z_body.
#   ref      where the demonstration carried that axis, expressed in the
#            vessel's OWN settled initial frame:
#                ref = R_pre^T @ (R_post @ u_body),
#            with R_pre / R_post the demo object orientations at the
#            pre_grasp and post_action (pour apex) keyframes.  Because both
#            sides are body-frame quantities, `ref` transfers to any target
#            scene: under a PERFECT object transform the executed rollout
#            reproduces it exactly.
# Success then requires (a) the block still lifted at the end of the rollout
# (the unchanged env-native condition) and (b) that at some instant while
# lifted, the demonstrated up-axis reached within POUR_ORI_TOL_DEG of `ref`.
# Since the demonstration carries the up-axis 81 deg down from vertical,
# (b) forces both the right tilt MAGNITUDE and the right tilt SENSE.
#
# Margins (measured, see TASKS.md): a correct pour lands at 0 deg (exact
# under a perfect transform); the 180-deg-flipped execution sends the
# up-axis to the mirrored direction, 162 deg from `ref` at the apex and
# never closer than 81 deg at ANY instant of the rollout.  The tolerance is
# POUR_ORI_TOL_DEG = 45 deg: a correct pour passes with 45 deg of slack in
# the estimated object rotation, and a flipped one fails by a 36 deg margin
# at its most favourable instant.
# ---------------------------------------------------------------------------

# body-frame axis of the vessel pointing away from gravity at the demo's
# pre_grasp keyframe (the block rests flat on the table -> +z_body)
POUR_UP_AXIS_BODY = (0.0, 0.0, 1.0)

# where the DEMONSTRATION carries that axis, in the vessel's own settled
# initial frame (see pour_reference_from_demo(); computed from
# data/pour/0/demo/keyframes/{1_pre_grasp,2_post_action}/meta.json and
# pinned here so the checker needs no disk access.  81.0 deg from
# POUR_UP_AXIS_BODY: the demo's achieved pour tilt.)
POUR_TILT_REF_BODY = (0.44842472, -0.88001729, 0.15647630)

# angular tolerance of the orientation-aware pour check (deg)
POUR_ORI_TOL_DEG = 45.0


def pour_reference_from_demo(demo_dir):
    """Recompute (u_body, ref_body) from a recorded pour demonstration.

    `demo_dir` is a demo recording directory (…/pour/<seed>/demo) holding
    `keyframes/1_pre_grasp/meta.json` and `keyframes/2_post_action/meta.json`.
    Returns the two unit vectors pinned above as POUR_UP_AXIS_BODY /
    POUR_TILT_REF_BODY; `tests/test_pour_criterion.py` asserts they agree
    with the canonical demo.
    """
    import json
    import os

    def _quat(name):
        with open(os.path.join(demo_dir, "keyframes", name,
                               "meta.json")) as f:
            meta = json.load(f)
        obj = meta["objects"]["cube"]
        return T.quat2mat(np.asarray(obj["quat_xyzw"], dtype=np.float64))

    R_pre = _quat("1_pre_grasp")
    R_post = _quat("2_post_action")
    up = R_pre.T @ np.array([0.0, 0.0, 1.0])
    up /= np.linalg.norm(up)
    ref = R_pre.T @ (R_post @ up)
    return up, ref / np.linalg.norm(ref)


def _install_pour_tracker(env):
    """Record the vessel's pose after every control step (pour env only).

    The orientation-aware pour criterion is evaluated at the pour APEX,
    which is in the MIDDLE of the rollout (the demonstration tilts and then
    tilts back), so the end-of-rollout sim state alone cannot see it.  The
    wrapper only appends ground-truth poses to a python list — it draws no
    random numbers and touches no sim state — so nothing about the physics,
    the actions, or any other task changes.  `reset_with_seed` arms the
    recorder AFTER settling, so only the rollout proper is recorded.
    """
    if getattr(env, "_simtasks_pour_tracked", False):
        return env
    inner_step = env.step

    def step(action):
        out = inner_step(action)
        track = getattr(env, "_simtasks_pour_track", None)
        if track is not None:
            track.append(_body_pose(env, "cube_main"))
        return out

    env.step = step
    env._simtasks_pour_tracked = True
    return env


def _pour_lift_height(env):
    """Height above which the vessel counts as lifted (the env-native Lift
    margin: table top + 4 cm)."""
    return float(env.model.mujoco_arena.table_offset[2]) + 0.04


def _pour_success(env):
    """Pour succeeded: the vessel was lifted AND tilted the DEMONSTRATED way.

    Two conditions, both required:

    (a) at the end of the rollout the block is still lifted (unchanged
        env-native Lift condition: > 4 cm above the table top), and
    (b) at some instant while lifted, the block's demonstrated "up" body
        axis (POUR_UP_AXIS_BODY — the face pointing away from gravity at the
        demo's pre-grasp keyframe) reached within **POUR_ORI_TOL_DEG = 45
        deg** of POUR_TILT_REF_BODY, the direction the demonstration carried
        it to, expressed in the vessel's own settled initial frame.

    (b) is what makes this criterion orientation-aware.  The demonstration
    carries the up-axis 81 deg down from vertical, so passing within 45 deg
    of that direction requires the tilt to have the right magnitude AND the
    right SENSE.  An execution whose object transform is off by the block's
    180-deg body flip tilts the vessel the opposite way: its up-axis is
    162 deg from the reference at the apex and never closer than 81 deg at
    any instant, so it fails by a 36 deg margin.  A correct pour is exact
    (0 deg) under a perfect transform and keeps 45 deg of slack.

    Requires the per-step pose record installed by `make_env` and armed by
    `reset_with_seed` (see `_install_pour_tracker`).
    """
    init = _initial_object_pose(env)
    if not bool(env._check_success()):     # (a) still lifted at the end
        return False
    track = getattr(env, "_simtasks_pour_track", None)
    if not track:
        return False
    R_init = T.quat2mat(np.asarray(init["quat_xyzw"], dtype=np.float64))
    u_body = R_init.T @ np.array([0.0, 0.0, 1.0])
    ref = np.asarray(POUR_TILT_REF_BODY, dtype=np.float64)
    z_lift = _pour_lift_height(env)
    cos_tol = float(np.cos(np.deg2rad(POUR_ORI_TOL_DEG)))
    for pos, quat in track:                # (b) demonstrated tilt reached
        if pos[2] <= z_lift:
            continue
        axis = R_init.T @ (T.quat2mat(quat) @ u_body)
        if float(np.dot(axis, ref)) >= cos_tol:
            return True
    return False


def pour_orientation_trace(env):
    """Diagnostics for one pour rollout: the angle (deg) between the
    demonstrated up-axis and POUR_TILT_REF_BODY over the rollout, plus the
    achieved tilt from vertical.  Returns None when nothing was recorded."""
    track = getattr(env, "_simtasks_pour_track", None)
    if not track:
        return None
    init = getattr(env, "_simtasks_init_obj_pose", None)
    if init is None:
        return None
    R_init = T.quat2mat(np.asarray(init["quat_xyzw"], dtype=np.float64))
    u_body = R_init.T @ np.array([0.0, 0.0, 1.0])
    ref = np.asarray(POUR_TILT_REF_BODY, dtype=np.float64)
    z_lift = _pour_lift_height(env)
    best = None       # smallest angle-to-reference while lifted
    apex = None       # largest tilt from vertical while lifted
    for pos, quat in track:
        if pos[2] <= z_lift:
            continue
        R = T.quat2mat(quat)
        axis = R_init.T @ (R @ u_body)
        ang = float(np.degrees(np.arccos(
            float(np.clip(np.dot(axis, ref), -1.0, 1.0)))))
        tilt = float(np.degrees(np.arccos(
            float(np.clip((R @ u_body)[2], -1.0, 1.0)))))
        if best is None or ang < best[0]:
            best = (ang, tilt)
        if apex is None or tilt > apex[1]:
            apex = (ang, tilt)
    if best is None:
        return {"n_steps": len(track), "lifted_steps": 0,
                "min_ref_angle_deg": None, "apex_tilt_deg": None,
                "apex_ref_angle_deg": None}
    return {"n_steps": len(track),
            "lifted_steps": sum(1 for p, _ in track if p[2] > z_lift),
            "min_ref_angle_deg": best[0],
            "apex_tilt_deg": apex[1],
            "apex_ref_angle_deg": apex[0],
            "tolerance_deg": POUR_ORI_TOL_DEG}


# ---------------------------------------------------------------------------
# pour vessel env (campaign A revision): Lift with the cube swapped for an
# ELONGATED block.  Campaign A showed the near-cubic Lift cube is ~90-deg yaw
# symmetric, which is explicitly OUT OF SCOPE for the method (paper Sec. V:
# anchor-based keypoints need an identifiable object frame); dense baselines
# "won" the cube-pour column via symmetry-equivalent poses.  The pour vessel
# must therefore have a distinguishable long axis (cup/bottle analog).
# robosuite 1.4.1 ships no free-standing cup/bottle asset outside the
# PickPlace bins arena, so the smallest faithful change is a custom elongated
# BoxObject in the Lift arena: 8.0 x 2.2 x 4.4 cm (long axis = body x).  The
# 2.2 cm width and 4.4 cm height match the original cube's graspable cross
# section, so the expert's grasp geometry is unchanged.  All names are kept
# ("cube", body "cube_main", obs cube_pos/...) so every downstream consumer
# (perception instance map, baselines, success check) works unmodified.
# ---------------------------------------------------------------------------

from robosuite.environments.manipulation.lift import Lift as _Lift


class PourLift(_Lift):
    """Lift env with the cube replaced by an elongated block (pour vessel).

    Auto-registered with robosuite via the EnvMeta metaclass, so
    ``suite.make("PourLift", ...)`` works like any built-in env.
    """

    # half-sizes (m): 8.0 x 2.2 x 4.4 cm full dims, long axis = body x
    CUBE_HALF_SIZE = (0.040, 0.011, 0.022)

    def _load_model(self):
        # Replicates Lift._load_model with the object swapped: call the
        # GRANDPARENT (single-arm) model setup, then rebuild arena + object +
        # task exactly as Lift does but with the elongated BoxObject.
        super(_Lift, self)._load_model()

        from robosuite.models.arenas import TableArena
        from robosuite.models.objects import BoxObject
        from robosuite.models.tasks import ManipulationTask
        from robosuite.utils.mjcf_utils import CustomMaterial
        from robosuite.utils.placement_samplers import UniformRandomSampler

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4",
                      "shininess": "0.1"}
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        half = list(self.CUBE_HALF_SIZE)
        self.cube = BoxObject(
            name="cube",  # keep all downstream names unchanged
            size_min=half,
            size_max=half,
            rgba=[1, 0, 0, 1],
            material=redwood,
        )

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.cube)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube,
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )


def _pour_sampler():
    """Widened block placement for the pour task: Lift's default range is a
    tiny +-3 cm box; +-12 cm (+ uniform yaw) gives target scenes meaningful
    translation variation while staying well inside the UR5e workspace."""
    from robosuite.utils.placement_samplers import UniformRandomSampler
    return UniformRandomSampler(
        name="ObjectSampler",
        x_range=[-0.12, 0.12],
        y_range=[-0.12, 0.12],
        rotation=None,  # uniform yaw
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=np.array((0, 0, 0.8)),  # Lift table_offset
        z_offset=0.01,
    )


# box_open door placement bounds (phase 2b): the Door env's default sampler
# (x [0.07, 0.09], y [-0.01, 0.01], yaw [-pi/2-0.25, -pi/2] ~ [-104.3, -90]
# deg, about reference (-0.2, -0.35, 0.8)) straddles a UR5e-INFEASIBLE yaw
# pocket: for door yaw in ~[-102, -93] deg the OSC controller stalls 45-90 mm
# short of the handle pre-grasp pose (workspace/IK limit), the fingers close
# beside the bar, and the scripted expert was only ~50% reliable.  A 132-cell
# feasibility grid (tracking error + grasp proxy, then full-demo checks)
# found yaw [-130, -125] deg feasible over the FULL widened x/y box below
# (24/24 cells, 2-8 mm tracking error), so the sampler uses that region —
# with MORE translational variation than the env default (4 cm x 3 cm vs
# 2 cm x 2 cm).  See TASKS.md "box_open door placement bounds".
DOOR_SAMPLER_BOUNDS = {
    "x_range": [0.06, 0.10],
    "y_range": [-0.02, 0.01],
    "rotation": (np.deg2rad(-130.0), np.deg2rad(-125.0)),
}


def _door_sampler():
    """Narrowed door placement for box_open (see DOOR_SAMPLER_BOUNDS)."""
    from robosuite.utils.placement_samplers import UniformRandomSampler
    return UniformRandomSampler(
        name="DoorSampler",
        x_range=list(DOOR_SAMPLER_BOUNDS["x_range"]),
        y_range=list(DOOR_SAMPLER_BOUNDS["y_range"]),
        rotation=tuple(DOOR_SAMPLER_BOUNDS["rotation"]),
        rotation_axis="z",
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=np.array((-0.2, -0.35, 0.8)),  # Door table_offset
    )


def _door_extra(env):
    return {
        "hinge_qpos": float(env.sim.data.qpos[env.hinge_qpos_addr]),
        "handle_pos": [float(x) for x in
                       env.sim.data.get_site_xpos(env.door.important_sites["handle"])],
    }


TASKS = {
    # bolt loosening / threading analog: highest-precision insertion of a
    # round nut over a round peg
    "nut_loosen": TaskSpec(
        name="nut_loosen",
        env_name="NutAssemblyRound",
        env_kwargs={},
        target_instance="RoundNut",
        target_body="RoundNut_main",
        obj_pos_key="RoundNut_pos",
        description="Grasp the round nut by its handle and lift it clear off the table "
                    "(analog of bolt loosening/removal; high precision; object-centric "
                    "success: nut >= 5 cm above its settled initial height).",
        sanity_tol=0.06,
        success_fn=_nut_loosen_success,
    ),
    # rim/edge grasp of a cylindrical object, translation-dominant transport
    "rim_grasp": TaskSpec(
        name="rim_grasp",
        env_name="PickPlaceCan",
        env_kwargs={},
        target_instance="Can",
        target_body="Can_main",
        obj_pos_key="Can_pos",
        description="Grasp the can cylinder near its upper rim and lift it "
                    "(translation-dominant; object-centric success: can lifted >= 5 cm "
                    "with both fingerpads in contact).",
        sanity_tol=0.06,
        success_fn=_rim_grasp_success,
    ),
    # pouring analog: lift an object and rotate the wrist by a large angle.
    # NOTE: robosuite 1.4.1 has no cup/bottle built-in ENV; PourLift (above)
    # swaps Lift's near-symmetric cube for an ELONGATED block as the grasped
    # 'vessel' (identifiable long axis, paper Sec. V scope) and the demo adds
    # rotation waypoints (rotation-dominant post-action).  See TASKS.md.
    "pour": TaskSpec(
        name="pour",
        env_name="PourLift",
        env_kwargs={},
        target_instance="cube",
        target_body="cube_main",
        obj_pos_key="cube_pos",
        description="Grasp the elongated block across its long axis, lift it, and execute "
                    "a large wrist rotation (pouring analog; rotation-dominant; "
                    "orientation-aware success: lifted AND tilted the demonstrated way, "
                    "within %.0f deg of the demonstrated up-axis direction)."
                    % POUR_ORI_TOL_DEG,
        sanity_tol=0.06,
        make_placement_initializer=_pour_sampler,
        success_fn=_pour_success,
    ),
    # box-lid opening analog: rotation-translation coupled articulated motion
    "box_open": TaskSpec(
        name="box_open",
        env_name="Door",
        env_kwargs={"use_latch": False},
        target_instance="Door",
        target_body="Door_main",
        obj_pos_key="handle_pos",
        description="Grasp the door handle and swing the door open (rotation-translation "
                    "coupling analog of box-lid opening).",
        sanity_tol=0.30,  # instance mask covers the whole door assembly
        extra_state=_door_extra,
        make_placement_initializer=_door_sampler,
    ),
    # cap twisting analog: high-precision placement requiring yaw alignment
    "cap_twist": TaskSpec(
        name="cap_twist",
        env_name="NutAssemblySquare",
        env_kwargs={},
        target_instance="SquareNut",
        target_body="SquareNut_main",
        obj_pos_key="SquareNut_pos",
        description="Grasp the square nut and twist it in place about world z "
                    "(high-precision rotation analog of cap twisting; object-centric "
                    "success: >= 30 deg yaw change with center within 3 cm of start).",
        sanity_tol=0.06,
        success_fn=_cap_twist_success,
    ),
}


def make_controller_config(control_delta=False):
    cfg = load_controller_config(default_controller="OSC_POSE")
    cfg["control_delta"] = control_delta
    return cfg


def make_env(task, camera_names=DEFAULT_CAMERAS, camera_size=DEFAULT_CAM_SIZE,
             horizon=5000, **overrides):
    """Create the robosuite env for a benchmark task.

    Camera observations are disabled for speed; captures render on demand
    from env.sim (see capture.py).
    """
    spec = TASKS[task]
    kwargs = dict(
        robots="UR5e",
        controller_configs=make_controller_config(control_delta=False),
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        camera_names=list(camera_names),
        camera_heights=camera_size,
        camera_widths=camera_size,
        control_freq=20,
        ignore_done=True,
        horizon=horizon,
        hard_reset=True,
        reward_shaping=False,
    )
    kwargs.update(spec.env_kwargs)
    if spec.make_placement_initializer is not None:
        kwargs["placement_initializer"] = spec.make_placement_initializer()
    kwargs.update(overrides)
    env = suite.make(spec.env_name, **kwargs)
    # keep only cameras that exist in this arena (e.g. the bins arena of
    # PickPlace has no 'sideview'); fall back to 'frontview'
    available = set(env.sim.model.camera_names)
    cams = [c for c in camera_names if c in available]
    for fallback in ("sideview", "frontview"):
        if len(cams) >= 2:
            break
        if fallback in available and fallback not in cams:
            cams.append(fallback)
    env._simtasks_cameras = cams
    env._simtasks_cam_size = camera_size
    env._simtasks_task = task
    if task == "pour":  # per-step pose record for the orientation-aware check
        _install_pour_tracker(env)
    return env


def hide_sites(env):
    """Shrink and hide all mujoco sites so they pollute neither RGB renders
    nor segmentation masks (same trick robosuite applies internally when
    camera segmentation obs are enabled)."""
    env.sim.model.site_size[:] = 1e-8
    env.sim.model.site_rgba[:, 3] = 0.0
    env.sim.forward()


def settle(env, steps=20):
    """Let free objects drop onto the table after reset.

    robosuite placement samplers spawn objects with a drop height (e.g. the
    nuts start ~6 cm above the table), so ground-truth poses read right after
    reset() are stale.  Holds the current TCP pose with the absolute OSC
    controller for a few control steps until objects come to rest."""
    import robosuite.utils.transform_utils as T
    sid = env.sim.model.site_name2id("gripper0_grip_site")
    pos = np.array(env.sim.data.site_xpos[sid])
    mat = np.array(env.sim.data.site_xmat[sid]).reshape(3, 3)
    aa = T.quat2axisangle(T.mat2quat(mat))
    action = np.concatenate([pos, aa, [-1.0]])
    obs = None
    for _ in range(steps):
        obs, _, _, _ = env.step(action)
    return obs


def reset_with_seed(env, seed, settle_steps=20):
    """Deterministic reset: robosuite placement samplers draw from the global
    numpy RNG, so seeding it right before reset() makes object placement (and
    robot init noise) reproducible.  The scene is then settled so ground-truth
    object poses are physical (see settle())."""
    np.random.seed(seed)
    env._simtasks_pour_track = None   # disarm while resetting/settling
    obs = env.reset()
    hide_sites(env)
    if settle_steps:
        obs = settle(env, settle_steps)
    _record_initial_object_pose(env)
    if getattr(env, "_simtasks_pour_tracked", False):
        env._simtasks_pour_track = []  # arm: record the rollout proper
    return obs


def _record_initial_object_pose(env):
    """Store the SETTLED initial pose of the task's target object on the env
    (used by the object-centric success checkers)."""
    task = getattr(env, "_simtasks_task", None)
    if task is None:
        return
    spec = TASKS[task]
    pos, quat = _body_pose(env, spec.target_body)
    env._simtasks_init_obj_pose = {
        "pos": pos.tolist(),
        "quat_xyzw": quat.tolist(),
        "yaw": float(T.mat2euler(T.quat2mat(quat))[2]),
    }


def success_checker(task):
    """Return a callable env -> bool for later evaluation of the task."""
    spec = TASKS[task]
    return spec.check_success
