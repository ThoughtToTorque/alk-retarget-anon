# simtasks — simulation task suite for the one-shot retargeting benchmark

Reproduces, with robosuite 1.4.1 built-ins, the experimental pipeline of the
paper (single RGB-D demo with 3 keyframes -> target scene with re-randomized
object pose -> transfer).  This package covers the SIMULATION DATA side only:
environments, scene capture, scripted expert demos, scene-pair generation,
ground truth.  The retargeting algorithm lives elsewhere (alkbench).

Environment: python >= 3.8, robosuite 1.4.1, mujoco 3.2.3 (see the top-level
README).  Headless rendering requires `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`.
`h5py` must be installed (robosuite.utils.camera_utils imports it at module
level); it comes in as a robosuite dependency.

Run everything:

    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m simtasks.run_smoke

>  **Note on paths.**  Directories named `results/`, `data_medium/` and
>  `data_medium_a2/` below refer to the *archived record deposit* of the
>  paper's campaigns, not to this repository — no scene data or rollout
>  records are shipped here.  Scene pairs are regenerated from their seeds
>  on demand (`simtasks/scene_pairs.py`).  See `docs/REPRODUCE.md`.

## Task -> environment mapping

| task key     | paper task (real robot)          | robosuite env      | target object | why this analog |
|--------------|----------------------------------|--------------------|---------------|-----------------|
| `nut_loosen` | bolt loosening / removal         | NutAssemblyRound   | RoundNut      | high-precision bar grasp (4 mm clearance on the handle) + lift-off: grasp the nut by its handle bar and lift it clear of the table ("remove the loosened bolt") |
| `rim_grasp`  | rim grasp of a cylinder          | PickPlaceCan       | Can           | translation-dominant: grasp the can cylinder near its top ("rim") and lift it in a stable two-fingerpad grasp |
| `pour`       | pouring                          | PourLift (custom Lift subclass, + rotation waypoints) | cube (elongated block 8.0 x 2.2 x 4.4 cm) | rotation-dominant: robosuite 1.4.1 has **no cup/bottle built-in env**, so `envs.PourLift` swaps the Lift cube for an ELONGATED block as the grasped vessel (identifiable long axis — the near-cubic original was ~90-deg yaw symmetric, out of the method's scope, paper Sec. V; see campaign A). The scripted post-action is a 75 deg wrist rotation (pour tilt), then back. Success = object lifted AND tilted in the DEMONSTRATED sense (orientation-aware, see below) |
| `box_open`   | box-lid opening                  | Door (use_latch=False) | Door      | rotation-translation coupled articulated motion: grasp handle bar, swing the door around its hinge (success: hinge > 0.3 rad). ToolHang exists in 1.4.1 but is a two-stage insertion, not an articulated-opening analog, so Door was chosen |
| `cap_twist`  | cap twisting                     | NutAssemblySquare  | SquareNut     | high-precision in-place rotation: grasp the square nut by its handle and twist it ~45 deg counterclockwise (loosening direction) about world z around its own center, keeping it at table level (contact-rich) |

## Success semantics (redefined in phase 2b — object-centric)

Phase 2 found that the robosuite-NATIVE success checks for `nut_loosen`,
`cap_twist` and `rim_grasp` are "object placed on a WORLD-FIXED peg/bin".
The paper has NO world-fixed goals — all 5 real tasks act on the object
itself — so object-relative retargeting missed those goals by construction.
`envs.py` therefore overrides success with OBJECT-CENTRIC checkers computed
from GT sim state relative to the object's SETTLED initial pose (recorded by
`reset_with_seed` after settling; stored on the env as
`_simtasks_init_obj_pose`).  This also matches the paper's single-object
scope (relational two-object goals are explicitly out of scope there).

| task | success (evaluated at the end of the rollout) |
|---|---|
| `nut_loosen` | round nut >= 5 cm above its settled initial height (held, or set free without falling back) |
| `cap_twist`  | square nut yaw changed >= 30 deg about world z relative to its settled initial yaw AND its center stayed within 3 cm of the initial xy (twist, not drag) |
| `rim_grasp`  | can >= 5 cm above its settled initial height AND both fingerpads in contact (stable grasp), any xy |
| `pour`       | **orientation-aware** (`envs._pour_success`, criterion revised 2026-08-16): the block is still lifted at the end of the rollout AND, at some instant while lifted, its demonstrated "up" body axis came within **45 deg** of the direction the demonstration carried it to (see "pour success criterion" below). Supersedes the grasp-and-tilt check (lifted-only), which could not tell a correct pour from a 180-deg-flipped one |
| `box_open`   | env-native (Door): hinge > 0.3 rad (goal moves with the door body — already object-relative) |

### pour success criterion (orientation-aware; revised 2026-08-16)

**Why it changed.** The superseded `pour` criterion was the env-native Lift
check ("block > 4 cm above the table") evaluated at the END of the rollout —
grasp, lift, survive the 75 deg tilt, done.  The paper classifies `pour` as
**rotation-dominant**, but that criterion is blind to orientation: the
elongated block is symmetric under its own 180-deg body flip `Rz_body(pi)`,
so an execution whose estimated object transform is off by exactly that flip
grasps the block identically (a parallel-jaw grasp is 180-deg symmetric),
tilts it by the same 75 deg **in the opposite sense**, and scored a success.
Campaign E measured the dense-geometry family (`icp`, `icp_centroid`,
`kp_dense_init`) taking that flip on roughly half the seeds — raw rotation
error 136-173 deg from ground truth — while no keypoint method ever took it.
A rotation-dominant task whose success functional cannot distinguish a
180-deg orientation error is measuring the wrong thing, so the criterion was
made orientation-aware.  Nothing else changed: same env, same object, same
sampler, same demonstration, same waypoints.

**Construction** (`envs._pour_success`; constants `POUR_UP_AXIS_BODY`,
`POUR_TILT_REF_BODY`, `POUR_ORI_TOL_DEG`).  Both reference quantities are
read off the FIXED demonstration recording (`envs.pour_reference_from_demo`,
from `keyframes/1_pre_grasp` and `keyframes/2_post_action`):

- `u_body` — the vessel's body-frame axis pointing **away from gravity at
  the pre-grasp keyframe**, i.e. which face of the vessel is "up" before the
  grasp.  The block rests flat, so `u_body = +z_body`.
- `ref` — where the demonstration **carried** that axis, expressed in the
  vessel's own settled initial frame:
  `ref = R_pre^T (R_post u_body)`, with `R_pre` / `R_post` the demo object
  orientations at the pre-grasp and post-action (pour apex) keyframes.
  Measured: `ref = (0.4484, -0.8800, 0.1565)`, i.e. **81.0 deg** down from
  `u_body` — the demo's achieved pour tilt (75 deg commanded).

Because both are body-frame quantities the reference **transfers to any
target scene**: under a perfect object transform the retargeted rollout
reproduces `ref` exactly.  Success then requires

1. the block is still lifted at the end of the rollout (the unchanged
   env-native condition, > 4 cm above the table top), **and**
2. at some instant while lifted, the demonstrated up-axis came within
   **`POUR_ORI_TOL_DEG = 45 deg`** of `ref` (angle measured in the block's
   own settled initial frame).

Condition 2 is what makes the criterion orientation-aware: since `ref` sits
81 deg from vertical, passing within 45 deg of it forces both the right tilt
**magnitude** and the right tilt **sense**.

**Why 45 deg** (a decision, not a knob — both margins measured):

| execution | angle to `ref` | verdict |
|---|---|---|
| perfect transform (`T_gt`), analytic | 0 deg | pass |
| perfect transform, measured over seeds 1000-1009 | 1.7-4.4 deg | pass, 40+ deg of slack |
| scripted expert in the demo scene | 0.03 deg | pass |
| **180-deg-flipped transform, apex** | **162 deg** | fail |
| **180-deg-flipped transform, best instant of the whole rollout** | **~81 deg** | fail by **36 deg** |

A flipped execution can never do better than 81 deg (its most favourable
instant is the untilted start of the lift), so any tolerance below 75 deg
rejects it; 45 deg leaves a correct pour ~47 deg of slack in the estimated
object yaw and still rejects the flip by 36 deg.  The tolerance is stated in
`envs.POUR_ORI_TOL_DEG` and in the checker's docstring; pinned by
`tests/test_pour_criterion.py`.

**Mechanics.** The pour apex is in the MIDDLE of the rollout (the demo tilts
and then tilts back), so the end-of-rollout sim state cannot see it.
`envs.make_env("pour")` therefore installs `_install_pour_tracker`, a thin
`env.step` wrapper that appends the block's ground-truth pose to a list;
`reset_with_seed` arms it after settling.  The wrapper draws no random
numbers and writes no sim state, and is installed for `pour` only, so every
other task's rollouts are bit-identical to before.
`envs.pour_orientation_trace(env)` returns the per-rollout diagnostics
(`min_ref_angle_deg`, `apex_tilt_deg`, `apex_ref_angle_deg`).

**Consequence for the scripted expert.** The expert is a from-scratch script
that tilts about the **world** x axis, not a retargeting of the demo, so in
a target scene whose object yaw differs it pours in a different body-relative
direction and does NOT reproduce the demonstration (measured 1/5 on seeds
1000-1004 under the new criterion, though it lifts and tilts every time).
The expert is therefore no longer the right feasibility reference for
`pour`; the ground-truth-transform rollout (retargeted demo with `T_gt`) is,
and it passes 10/10 on seeds 1000-1009.  The demonstration itself is
unchanged and still passes its own criterion in the demo scene.

Robot: **UR5e** (closest built-in to the paper's UR3) with Robotiq-85
gripper (default), `OSC_POSE` controller with **`control_delta=False`**
(actions are absolute world-frame TCP pose: `[x y z, axis-angle(3), grip]`),
`control_freq=20`.  Note: robosuite applies **no clipping/scaling** to
absolute OSC actions, so world coordinates (e.g. z=1.05) pass through as-is.

Cameras: `agentview` (primary) + `sideview` (secondary; the PickPlace bins
arena has no sideview, so `frontview` is used there — `make_env` selects
automatically).  Note the Door env's `agentview` faces the robot's back;
`sideview`/`frontview` are the informative views for `box_open`.

## Files

- `envs.py` — `TASKS` spec table, `make_env(task)`, `reset_with_seed(env, seed)`,
  `success_checker(task)`.
- `capture.py` — `capture_scene`, `load_capture`, back-projection helpers,
  `sanity_check_backprojection`.
- `motion.py` — shared motion primitives (extracted from scripted_demo so the
  retargeting runner in `pipeline/` reuses the identical machinery):
  `Executor` (absolute-OSC interpolated waypoint tracking), constants
  (`R_DOWN`, `PAD_OFFSET`, `GRIP_OPEN/CLOSE`, `BAR_GRASP_PHASE`),
  `rot_z`/`yaw_down_R`, `pose_to_matrix`/`matrix_to_pose`, and
  `execute_waypoints(env, waypoints)` — replays a (possibly retargeted)
  waypoints.json-schema list with per-label tolerances mirroring the
  scripted expert and the list's own gripper open/close schedule.
- `scripted_demo.py` — `run_demo(env, task, out_dir)`: scripted expert,
  3 keyframes + full trajectory (re-exports the motion primitives for
  backward compatibility).
- `scene_pairs.py` — `generate_pair(task, seed)`, `load_pair`, `make_env_for`.
- `run_smoke.py` — end-to-end test (see above).

## Capture format (per scene directory)

```
<cam>_rgb.png     uint8 HxWx3
<cam>_depth.npy   float32 HxW, METRIC meters
<cam>_seg.npy     int32 HxW instance ids (0 = background)
meta.json         cameras {K 3x3, T_world_cam 4x4, h, w},
                  instance_id_to_name, objects {pos, quat_xyzw, body},
                  tcp {pos, quat_xyzw}, gripper_qpos, extra_state
```

### Depth / image conventions (verified empirically)

- `env.sim.render()` returns images **bottom-row-first** (OpenGL, which is
  also robosuite 1.4.1's default obs convention, `macros.IMAGE_CONVENTION =
  "opengl"`).  `capture.py` **flips everything vertically** so all saved
  arrays are **top-row-first (OpenCV)**.
- Depth is converted from mujoco's normalized buffer to metric meters with
  `robosuite.utils.camera_utils.get_real_depth_map` (z-distance along the
  optical axis, directly usable in the pinhole model).
- `K` from `get_camera_intrinsic_matrix` (principal point at image center);
  `T_world_cam` from `get_camera_extrinsic_matrix` — the camera pose in the
  world with **OpenCV camera axes** (x right, y down, z forward):
  `p_world = T_world_cam @ [x_cam, y_cam, z_cam, 1]`, with
  `x_cam = (u - cx) / fx * depth`, `y_cam = (v - cy) / fy * depth` for pixel
  `(u=col, v=row)` in the saved (OpenCV) arrays.
- **Automated sanity check** (`capture.sanity_check_backprojection`, run by
  `run_smoke`): back-project the target object's segmented pixels and compare
  the centroid to the ground-truth body position. Measured errors: 0.6–3 cm
  for the tabletop objects (surface-centroid vs body-center bias), ~5 cm for
  the whole-door instance (tolerance 30 cm since the instance covers the full
  door assembly).
- Quaternions everywhere are **xyzw** (robosuite convention); mujoco's wxyz
  body quaternions are converted on capture.

### Instance segmentation ids

The seg maps use the SAME id convention as robosuite's built-in
`camera_segmentations="instance"` observable: id = index of the instance in
`env.model.instances_to_ids` insertion order + 1, id 0 = background.
`meta.json:instance_id_to_name` stores the mapping explicitly.  Captures are
rendered from `env.sim.render(segmentation=True)` (geom ids) and remapped via
`env.model.geom_ids_to_instances` — vectorized, identical result, much faster
than the per-pixel python remap robosuite does per step.  All mujoco sites
are shrunk/hidden after reset (`envs.hide_sites`) so they pollute neither RGB
nor segmentation (robosuite does the same internally when segmentation obs
are on).

## Scripted expert (scripted_demo.py)

Absolute OSC pose targets, interpolated in <=2 cm / <=0.2 rad sub-goals per
control step.  Keyframes follow the paper: **approach / pre_grasp /
post_action**, each captured in full demo format with synchronized TCP pose;
the full executed trajectory (per-step TCP pose + gripper) is saved to
`trajectory.npz`, the commanded waypoints to `waypoints.json`.

Grasp geometry facts discovered empirically (encoded as constants):

- The Robotiq-85 **fingerpad centers sit ~1.3 cm above (behind) the
  `gripper0_grip_site` TCP frame**; for thin bars (nut handles, door handle)
  the TCP is commanded `PAD_OFFSET = 1.3 cm` past the bar center along the
  approach axis, else the fingers close above/short of the bar.
- The finger **closing axis is the gripper's local x**; with the canonical
  top-down orientation `R_DOWN = [[0,1,0],[1,0,0],[0,0,-1]]`, commanding
  `Rz(bar_yaw) @ R_DOWN` closes the fingers perpendicular to a bar with
  world yaw `bar_yaw` (`BAR_GRASP_PHASE = 0`).
- **`env.reset()` does NOT settle objects**: placement samplers spawn objects
  with a drop height (nuts start ~6 cm above the table), so ground truth read
  at reset is stale.  `envs.reset_with_seed` therefore holds the arm for 20
  control steps (`envs.settle`) before anything reads poses or captures.
- Door: a vertical descent onto the handle stalls against the UR5e's
  workspace limits; the expert approaches **horizontally** along the door
  normal, closes vertically on the bar, then tracks an arc around the hinge
  (re-grasping up to twice if the handle slips).

## Scene pairs (scene_pairs.py)

`generate_pair(task, seed)` writes under `<data-root>/<task>/<seed>/` (default `data/`, see `ALK_DATA_ROOT`):
a fixed, feasible **demo scene** (per-task `DEMO_SEEDS`, matching the paper's
single recorded demo) and a **target scene** re-randomized by the env's own
placement sampler (position + yaw) under the deterministic seed
`target_seed_for(task, seed)`.  Scenes are reproducible:
`make_env_for(pair_info, "demo"|"target")` re-instantiates them live via the
recorded seeds (robosuite samplers draw from the global numpy RNG).
`success_checker(task)` returns the env->bool hook for evaluating transferred
trajectories later.

Sampler ranges: yaw is uniformly randomized for the tabletop objects
(measured demo-vs-target yaw deltas of ~50 deg are typical); position ranges
are the envs' defaults except **pour**, where Lift's tiny +-3 cm range
is **widened to +-12 cm** (`envs._pour_sampler`, passed as
`placement_initializer`, persists across hard resets), and **box_open**,
whose door placement is re-bounded to the UR5e-feasible region (below).

### box_open door placement bounds (phase 2b)

The Door env's DEFAULT sampler (x [0.07, 0.09], y [-0.01, 0.01], yaw
[-104.3, -90] deg about reference (-0.2, -0.35, 0.8)) straddles a
UR5e-infeasible pocket: for door yaw in ~[-102, -93] deg the OSC controller
stalls **45-90 mm short of the handle pre-grasp pose** (workspace/IK limit,
measured TCP tracking error), the fingers close beside the bar, and the
scripted expert managed only ~5/10 placements.  A 132-cell feasibility grid
(approach + pre_grasp + close proxy: tracking error < 35 mm AND grasp holds)
over x [0.06, 0.10], y [-0.02, 0.01], yaw [-130, -90] deg found:

| door yaw (deg) | feasible cells |
|---|---|
| -90            | 14/15 (default x/y box) |
| -95 .. -100    | ~0/27 (the infeasible pocket) |
| -104 .. -106   | 27/27 |
| -110 .. -120   | 18/36 (marginal, 10-20 mm tracking error) |
| -125 .. -130   | 24/24 (2-8 mm tracking error) |

`envs.DOOR_SAMPLER_BOUNDS` therefore uses **x [0.06, 0.10], y [-0.02, 0.01],
yaw [-130, -125] deg** — the deep-feasible plateau, with MORE translational
variation than the env default (4 x 3 cm vs 2 x 2 cm) at the cost of a
narrower (5 deg) yaw range.  Full-demo verification over seeds 0-9: see
"What works" below.  The phase 2b expert also seats the handle bar 6 mm
deeper between the fingerpads (`grasp_depth = PAD_OFFSET + 0.006`), swings in
16 arc segments (was 12) and allows 3 re-grasps (was 2); at pad-center depth
the hold was marginal and about half of the default-box placements slipped.

### box_open MEDIUM-tier door placement bounds (narrowed 2026-08-16)

The section above is the **easy**-tier (phase 2b) window.  The **medium**
tier lives in `baselines/difficulty.py:MEDIUM_SAMPLER["box_open"]` and was
narrowed on 2026-08-16 for the same reason the pour criterion was fixed:
it was measuring the wrong thing.

The superseded medium window (x [0.05, 0.11], y [-0.04, 0.04], yaw
[-112.92, -81.41] deg) deliberately re-included the UR5e-infeasible yaw
pocket at [-102, -93] deg.  On it, the **scripted expert** — which is
scripted from ground truth and so carries zero retargeting error — scores
only **7/10** on the medium-tier target scenes of seeds 1000-1009 (failures
at door yaw -94.4 deg and -96.6 deg; the hinge never moved).  A tier on
which the oracle-scripted motion fails 3/10 for reachability reasons reads
out arm reach, not retargeting accuracy, which is what the benchmark exists
to measure — and it depressed every method (best 31/60 in Campaign A).

The window was re-bounded to the region where the expert is reliable, using
the same **>= 9/10 over 10 seeds** standard as the other tasks, measured by
running the full scripted demo directly in the medium-tier target scenes of
seeds 1000-1009.  Measured ladder (translation box fixed at x [0.05, 0.11],
y [-0.04, 0.04]):

| yaw window (deg) | scripted expert |
|---|---|
| [-112.92, -81.41] | **7/10** (superseded medium window) |
| [-113, -103] | 10/10 |
| [-110, -103] | 10/10 |
| [-107, -103] | 10/10 |
| [-120, -103] / [-125, -103] / [-130, -103] | 10/10 |
| [-130, -120] / [-130, -115] | 10/10 |
| **[-112.92, -103.0]** | **10/10** — ADOPTED |

Adopted: **x [0.05, 0.11], y [-0.04, 0.04], yaw [-112.92, -103.0] deg**
(`rotation=(-np.pi/2 - 0.40, np.deg2rad(-103.0))`), scripted-expert
reliability **10/10** (hinge 0.35-0.40 rad, zero re-grasps).  It is a
**strict subset** of the superseded window — the fix only removes
placements — and it still keeps more translational variation than the easy
tier (6 x 8 cm vs 4 x 3 cm) with a 9.9 deg yaw span vs the easy tier's
5 deg.  Wider all-feasible windows exist below -113 deg (down to -130 deg),
but they fall outside the superseded window and would move the tier in more
than one direction; they are recorded in
`baselines.difficulty.BOX_OPEN_MEDIUM_YAW_NOTE` for completeness.

Scene pairs drawn from the narrowed window live in `data_medium_a2/`;
`data_medium/box_open/` is untouched so campaigns B..H still refer to the
scenes they were run on.

## What works / known issues

- All 5 scripted demos succeed in their demo scenes (seed 0); `run_smoke`
  passes 5/5 including capture round-trip and back-projection checks.
- Robustness of the scripted expert across randomized scenes, phase 2b
  semantics, seeds 0-9: **nut_loosen 10/10, cap_twist 10/10, rim_grasp
  10/10, pour 10/10, box_open 10/10** (box_open with the narrowed
  `DOOR_SAMPLER_BOUNDS` + deeper handle grasp, zero re-grasps needed; with
  the env-default door sampler it was 5/10 — see the bounds section above).
  **Medium tier, seeds 1000-1009 (2026-08-16):** box_open **10/10** on the
  narrowed yaw window (7/10 on the superseded one — that is why it was
  narrowed).  `pour` is a special case under the orientation-aware
  criterion: the expert lifts and tilts on 10/10 but pours in the
  demonstrated *body-relative* direction on only 1/10, because it tilts
  about the **world** x axis instead of retargeting the demo — it is a
  from-scratch script, not a transfer.  The right feasibility reference for
  `pour` is the demo retargeted with the ground-truth transform, which
  passes **10/10** (angle to the demonstrated pour direction 1.4-4.2 deg);
  the same transform composed with the block's 180-deg body flip passes
  **0/10** (78.0-81.6 deg).  Snapshot:
  `results/campaign_a2/feasibility_pour.json`.
- `pour` uses an elongated block (`envs.PourLift`, 8.0 x 2.2 x 4.4 cm; all
  names kept as "cube"), not a cup (no such free-standing asset in
  robosuite 1.4.1); the long axis gives the vessel an identifiable frame
  (paper Sec. V scope — campaign A showed the original near-cubic Lift cube
  is ~90-deg yaw symmetric and out of scope), and the rotation waypoints
  carry the rotation-dominant character of the task.  The expert grasps
  ACROSS the long axis (bar grasp, nearest 180-deg equivalent yaw).  The
  success check verifies the block was lifted AND tilted **the demonstrated
  way** (orientation-aware, 45 deg tolerance — see "pour success criterion"
  above); the superseded check only verified that it was still held lifted
  after the 75 deg tilt-and-return, which a 180-deg-flipped execution
  satisfies just as well.
- The `sideview` camera does not exist in the PickPlace bins arena
  (`frontview` substituted automatically), and the Door env's `agentview`
  faces the robot's back — use the secondary camera for `box_open`.
- `env.step` count per demo: ~220-630 control steps (11-32 s sim time at
  20 Hz); wall clock ~4-13 s per demo on the development machine.
- **World-fixed goals — FIXED in phase 2b** (history: phase 2, see
  results/phase2_quick.md, found that the robosuite-native success of
  `nut_loosen`/`cap_twist`/`rim_grasp` was "object on a world-fixed peg/bin",
  which no object-relative one-shot method can hit by construction and which
  does not match the paper's object-attached goals).  The success checkers
  were redefined object-centrically (see "Success semantics" above) and the
  three scripted demos changed accordingly: nut_loosen = grasp + lift-off
  (was carry-to-peg), cap_twist = in-place ~45 deg twist about the nut
  center (was carry-and-insert onto the square peg), rim_grasp = rim grasp +
  lift (was transport-to-bin).  Old results under the world-fixed semantics
  are preserved in results/phase2_quick.*; phase 2b results are in
  results/phase2b_quick.*.
