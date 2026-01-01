"""Shared helpers for the baseline test modules.

Uses the frozen capture snapshots in `baselines/testdata/` -- the only binary
data shipped with this repository -- plus the synthetic-cloud machinery of
tests/synth.py.  No robosuite/mujoco import anywhere, which is the point:
the baselines are tested on real captures without starting a simulator.
Only the pairs the tests actually exercise are shipped (cap_twist and
nut_loosen); regenerate any other pair with
`simtasks.scene_pairs.generate_pair` if you need it.
"""
import functools
import os

import numpy as np

from alkbench import compute_candidates

from baselines import common
from tests import synth

TESTDATA = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "baselines", "testdata")

TESTDATA_PAIRS = [("cap_twist", 0), ("nut_loosen", 0)]


@functools.lru_cache(maxsize=None)
def load_testdata_pair(task="cap_twist", seed=0, **ctx_overrides_frozen):
    """(demo_capture, target_capture, ctx, pair) for a snapshot pair.
    Cached per call signature; callers that mutate ctx state (noise_seed,
    registration) should copy or re-request with overrides."""
    pair_dir = os.path.join(TESTDATA, task, str(seed))
    return common.context_from_pair(pair_dir,
                                    **dict(ctx_overrides_frozen))


def fresh_pair(task="cap_twist", seed=0, **ctx_overrides):
    """Uncached variant returning a fresh MethodContext."""
    pair_dir = os.path.join(TESTDATA, task, str(seed))
    return common.context_from_pair(pair_dir, **ctx_overrides)


# ---------------------------------------------------------------------------
# synthetic scenes (camera frame == world frame, extrinsic None)
# ---------------------------------------------------------------------------

def near_symmetric_cloud(n=4000, seed=0, bump_frac=0.05,
                         bump_offset=(0.065, 0.014, 0.0)):
    """Elongated cylinder along x with a SMALL off-axis bump: nearly
    invariant under a 180-degree yaw flip (the nut/can regime of the paper's
    paper Section 5.9).  Camera-frame points around z ~ 0.5 m."""
    rng = np.random.RandomState(seed)
    n_bump = int(n * bump_frac)
    n_cyl = n - n_bump
    x = rng.uniform(-0.09, 0.09, n_cyl)
    theta = rng.uniform(0, 2 * np.pi, n_cyl)
    r = 0.015 * np.sqrt(rng.uniform(0, 1, n_cyl))
    cyl = np.stack([x, r * np.cos(theta), r * np.sin(theta)], axis=1)
    bump = rng.normal(scale=0.006, size=(n_bump, 3)) + np.asarray(bump_offset)
    pts = np.concatenate([cyl, bump])
    pts += np.array([0.0, 0.0, 0.5])
    return pts


def percept_from_points(points, k=8, seed=0):
    """Render a synthetic cloud to (mask, depth) and run the candidate
    pipeline (camera frame, no extrinsic) -- a minimal stand-in for the
    percept dicts of baselines.common.perceive."""
    mask, depth = synth.render(points)
    cands = compute_candidates(mask, depth, synth.FX, synth.FY,
                               synth.CX, synth.CY, extrinsic=None,
                               k=k, seed=seed)
    return {"cands": cands, "camera": "synth", "n_points":
            int(cands.points3d.shape[0])}


def yaw_about_point(angle_deg, center):
    """SE(3): rotation about the +z axis line through ``center``."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("z", angle_deg, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    c = np.asarray(center, dtype=np.float64)
    T[:3, 3] = c - R @ c
    return T


def rot_err_deg(T_est, T_true):
    from alkbench import rotation_angle_deg
    T_est = np.asarray(T_est, dtype=np.float64)
    T_true = np.asarray(T_true, dtype=np.float64)
    return float(rotation_angle_deg(T_est[:3, :3] @ T_true[:3, :3].T))
