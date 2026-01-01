"""Difficulty tiers for the Phase-4 evaluation (config module ONLY -- no env
edits; consumed later by the eval runner).

Motivation: the Phase-2b 10-seed eval hit a ceiling (overall success 0.93,
full == noreg; results/phase2b_part_nodoor.md), so the easy randomization
regime cannot separate methods.  Three tiers:

  * ``easy``   -- the CURRENT samplers (env defaults + the existing simtasks
                  widenings: pour +-12 cm, box_open narrowed feasible door
                  bounds).  Sampler overrides are all None.
  * ``medium`` -- wider xy + full yaw (where physically meaningful; see
                  per-task notes below).
  * ``hard``   -- medium placement + SENSOR noise applied to the captures:
                  Gaussian depth noise sigma = 3 mm and 2 px binary erosion
                  of the instance mask.

Sampler override dicts mirror robosuite's UniformRandomSampler kwargs
(x_range, y_range, rotation, rotation_axis, reference_pos, z_offset) so the
Phase-4 runner can build the sampler directly:

    UniformRandomSampler(name=..., **sampler_override(tier, task))

``rotation=None`` means uniform full yaw (robosuite convention).  A per-task
override of ``None`` means "keep the tier below" (easy = current sampler).

Per-task medium notes (ranges chosen inside the UR5e reach envelope around
the fixed demo scene; Phase 4 must feasibility-check them once with the
scripted expert / reachability probe before the big run):

  * nut_loosen / cap_twist: robosuite's default NutAssembly x-range is a
    5 mm sliver ([-0.115, -0.11]); medium widens x/y by ~5 cm on each side.
    Yaw was already full.
  * rim_grasp: the default PickPlace sampler already covers the whole bin
    (x +-0.145, y +-0.195 around bin1) with full yaw -- the bin walls make
    wider xy physically impossible, so medium == easy for placement (stated
    here so nobody mistakes it for an omission); hard still adds sensor
    noise, which is the separating axis for this task.
  * pour: easy already widened to +-12 cm; medium +-15 cm, full yaw.
  * box_open: full yaw is physically meaningless for the articulated door +
    UR5e reach; medium widens the feasible window found in Phase 2b
    (x [0.07,0.09], y [-0.01,0.01], yaw [-pi/2-0.25, -pi/2]) in TRANSLATION
    to x [0.05,0.11], y [-0.04,0.04].  Its yaw window was NARROWED on
    2026-08-16 from [-112.92, -81.41] deg to **[-112.92, -103.0] deg**: the
    old upper end swept the door through the UR5e-infeasible yaw pocket at
    [-102, -93] deg, where even the ground-truth-scripted expert manages only
    7/10, so that column measured REACHABILITY rather than retargeting
    accuracy.  On the narrowed (strict-subset) window the expert scores
    10/10 — the same >= 9/10 standard used for the other tasks.  Full
    derivation and the measured window ladder: BOX_OPEN_MEDIUM_YAW_NOTE
    below.
"""
import numpy as np

# ---------------------------------------------------------------------------
# medium-tier placement overrides (robosuite UniformRandomSampler kwargs)
# ---------------------------------------------------------------------------

MEDIUM_SAMPLER = {
    "nut_loosen": {  # env default: x [-0.115,-0.11], y [-0.225,-0.11]
        "x_range": [-0.165, -0.06],
        "y_range": [-0.275, -0.06],
        "rotation": None,                     # full yaw (as default)
        "rotation_axis": "z",
        "reference_pos": (0.0, 0.0, 0.82),    # NutAssembly table_offset
        "z_offset": 0.02,
    },
    "cap_twist": {   # env default: x [-0.115,-0.11], y [0.11,0.225]
        "x_range": [-0.165, -0.06],
        "y_range": [0.06, 0.275],
        "rotation": None,
        "rotation_axis": "z",
        "reference_pos": (0.0, 0.0, 0.82),
        "z_offset": 0.02,
    },
    # default sampler already bin-wide with full yaw; walls forbid wider xy.
    # None = keep the current sampler (see module docstring).
    "rim_grasp": None,
    "pour": {        # easy tier: +-0.12 (simtasks widened); default +-0.03
        "x_range": [-0.15, 0.15],
        "y_range": [-0.15, 0.15],
        "rotation": None,
        "rotation_axis": "z",
        "reference_pos": (0.0, 0.0, 0.8),     # Lift table_offset
        "z_offset": 0.01,
    },
    # box_open medium-tier yaw window NARROWED 2026-08-16 (see
    # BOX_OPEN_MEDIUM_YAW_NOTE below): the superseded upper bound
    # (-pi/2 + 0.15 = -81.4 deg) swept the door through the UR5e-infeasible
    # yaw pocket at [-102, -93] deg, so the medium tier was reading out
    # REACHABILITY rather than retargeting accuracy.  Translation box
    # unchanged (4 cm larger in x and 5 cm larger in y than the easy tier).
    "box_open": {    # easy tier: DOOR_SAMPLER_BOUNDS (feasible window)
        "x_range": [0.05, 0.11],
        "y_range": [-0.04, 0.04],
        "rotation": (-np.pi / 2.0 - 0.40, np.deg2rad(-103.0)),
        "rotation_axis": "z",
        "reference_pos": (-0.2, -0.35, 0.8),  # Door table_offset
    },
}

# ---------------------------------------------------------------------------
# box_open medium-tier yaw window: why it is [-112.92, -103.0] deg
# ---------------------------------------------------------------------------
BOX_OPEN_MEDIUM_YAW_NOTE = """
The medium tier exists to separate METHODS by retargeting accuracy.  The
superseded box_open medium window (x [0.05,0.11], y [-0.04,0.04], yaw
[-pi/2-0.40, -pi/2+0.15] = [-112.92, -81.41] deg) deliberately re-included
door yaws that phase 2b had already measured as controller-marginal: for
door yaw in ~[-102, -93] deg the OSC controller stalls 45-90 mm short of the
handle pre-grasp pose (UR5e workspace/IK limit) and the fingers close beside
the bar.  On that window the SCRIPTED EXPERT -- which is scripted from
ground truth and therefore has zero retargeting error -- scores only
**7/10** on the medium-tier target scenes of seeds 1000-1009 (failures at
door yaw -94.4 and -96.6 deg, hinge never moved).  Every method was
therefore low on that column (best 31/60 in Campaign A) largely because the
arm could not reach the handle, which is not what this benchmark measures.

The window was narrowed to the region where the expert is reliable, using
the SAME standard as the other tasks (>= 9/10 scripted-expert successes over
the 10 medium-tier target scenes of seeds 1000-1009, run directly in those
scenes).  Measured (10 seeds each, full scripted demo, translation box fixed
at x [0.05,0.11], y [-0.04,0.04]):

    yaw window (deg)     scripted expert
    [-112.92, -81.41]    7/10   <- superseded medium window
    [-113, -103]        10/10
    [-110, -103]        10/10
    [-107, -103]        10/10
    [-120, -103]        10/10
    [-125, -103]        10/10
    [-130, -103]        10/10
    [-130, -120]        10/10
    [-130, -115]        10/10
    [-112.92, -103]     10/10   <- ADOPTED (widest verified STRICT SUBSET
                                   of the superseded window)

Adopted: **x [0.05, 0.11], y [-0.04, 0.04], yaw [-112.92, -103.0] deg**, on
which the scripted expert scores **10/10** (hinge 0.35-0.40 rad, zero
re-grasps).  It is a strict subset of the superseded window -- the fix only
removes placements, it never adds any -- and it still keeps MORE
translational variation than the easy tier (6 x 8 cm vs 4 x 3 cm,
envs.DOOR_SAMPLER_BOUNDS) with a 9.9 deg yaw span vs the easy tier's 5 deg.
Wider feasible windows exist below -113 deg (down to -130 deg, all 10/10),
but they lie outside the superseded medium window and would change the tier
in more than one direction; they are recorded above for completeness.

Scene pairs drawn from the narrowed window live in `data_medium_a2/`;
`data_medium/box_open/` is left untouched so campaigns B..H keep referring
to the scenes they were run on.
"""

# hard-tier sensor noise (applied to captures at perception time)
HARD_SENSOR_NOISE = {
    "depth_sigma_m": 0.003,   # Gaussian, per valid-depth pixel
    "mask_erosion_px": 2,     # binary erosion iterations (3x3 structuring)
}

TIERS = {
    "easy": {
        "sampler": {task: None for task in MEDIUM_SAMPLER},
        "sensor_noise": None,
    },
    "medium": {
        "sampler": MEDIUM_SAMPLER,
        "sensor_noise": None,
    },
    "hard": {
        "sampler": MEDIUM_SAMPLER,
        "sensor_noise": HARD_SENSOR_NOISE,
    },
}


def sampler_override(tier, task):
    """Sampler kwargs override for (tier, task); None = keep the current
    (easy) sampler."""
    return TIERS[tier]["sampler"].get(task)


def sensor_noise_params(tier):
    """Sensor-noise dict for a tier (None for easy/medium)."""
    return TIERS[tier]["sensor_noise"]


# ---------------------------------------------------------------------------
# sensor-noise application (pure numpy/scipy; used by baselines.common and
# by the Phase-4 runner -- captures on disk are never modified)
# ---------------------------------------------------------------------------

def erode_mask(mask, n_px):
    """Binary erosion with a 3x3 structuring element, n_px iterations."""
    from scipy import ndimage
    m = np.asarray(mask, dtype=bool)
    if n_px <= 0:
        return m
    return ndimage.binary_erosion(m, structure=np.ones((3, 3), dtype=bool),
                                  iterations=int(n_px), border_value=0)


def apply_sensor_noise(depth, mask, params, rng):
    """Apply tier sensor noise to one (depth, mask) pair.

    Gaussian depth noise (only on valid-depth pixels) + mask erosion.
    Returns NEW arrays; inputs are not modified.
    """
    depth = np.asarray(depth, dtype=np.float64)
    sigma = float(params.get("depth_sigma_m", 0.0))
    noisy = depth.copy()
    if sigma > 0:
        valid = np.isfinite(depth) & (depth > 0)
        noisy[valid] = depth[valid] + rng.normal(0.0, sigma,
                                                 size=int(valid.sum()))
    eroded = erode_mask(mask, int(params.get("mask_erosion_px", 0)))
    return noisy, eroded
