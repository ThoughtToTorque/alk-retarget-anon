"""Tests for conditioning-adaptive bounded registration (paper Section 3.7) and the
slender depth-consistency auto-prior.  All adaptive behavior is behind the
default-off `adaptive` flag; these tests also pin the fall-back path so the
flag cannot change default results."""

import numpy as np
from scipy.spatial.transform import Rotation

from alkbench import (bounded_registration, adaptive_bounds,
                      adaptive_registration, transform_points,
                      rotation_angle_deg, build_alk, lateral_extent_ratio,
                      SLENDER_RATIO_THRESHOLD, SIGMA_RATIO_THRESHOLD)


def _rot_err_deg(Ta, Tb):
    return rotation_angle_deg(np.asarray(Ta)[:3, :3].T @ np.asarray(Tb)[:3, :3])


def thin_box_cloud(n=3000, seed=0, dims=(0.080, 0.022, 0.010)):
    """Uniform cloud in a thin elongated box (pour-vessel-like), long axis x,
    centered at an off-origin position (rotation-center convention matters)."""
    rng = np.random.RandomState(seed)
    d = np.asarray(dims)
    pts = rng.uniform(-0.5, 0.5, (n, 3)) * d
    return pts + np.array([0.15, -0.05, 0.83])


def slender_alk():
    """Nearly rank-1 ALK: long axial pair + laterals close to the axis line
    (pour-like conditioning, sigma_ratio well below threshold)."""
    return np.array([
        [0.037, 0.000, 0.000],
        [-0.037, 0.000, 0.000],
        [0.002, 0.006, 0.002],
        [-0.002, -0.006, -0.002],
    ]) + np.array([0.15, -0.05, 0.83])


def spread_alk():
    """Well-conditioned ALK (laterals far off-axis)."""
    return np.array([
        [0.05, 0.00, 0.00],
        [-0.05, 0.00, 0.00],
        [0.00, 0.04, 0.01],
        [0.00, -0.04, -0.02],
    ]) + np.array([0.15, -0.05, 0.83])


# ---------------------------------------------------------------------------
# adaptive_bounds
# ---------------------------------------------------------------------------

def test_adaptive_bounds_triggers_on_slender_alk():
    b = adaptive_bounds(slender_alk())
    assert b["adaptive_triggered"]
    assert b["sigma_ratio"] < SIGMA_RATIO_THRESHOLD
    assert b["axis_rot_bound_deg"] == 90.0
    assert b["rot_bound_deg"] == 15.0
    assert b["trans_bound_m"] == 0.02
    # axis = c1 -> c2 direction (x, up to sign)
    assert abs(abs(float(b["axis_demo"] @ np.array([1.0, 0, 0]))) - 1.0) < 1e-2
    assert b["sigma23"] == b["singular_values"][1] + b["singular_values"][2]


def test_adaptive_bounds_unchanged_on_well_conditioned_alk():
    b = adaptive_bounds(spread_alk())
    assert not b["adaptive_triggered"]
    assert b["sigma_ratio"] > SIGMA_RATIO_THRESHOLD
    # all bounds stay at base
    assert b["axis_rot_bound_deg"] == 15.0
    assert b["rot_bound_deg"] == 15.0


def test_adaptive_bounds_campaign_a_demo_values():
    """The five Campaign-A demo ALK sigma ratios straddle the threshold as
    calibrated: only pour (0.359) triggers; the others (>= 0.434) do not."""
    assert 0.359 < SIGMA_RATIO_THRESHOLD < 0.434


# ---------------------------------------------------------------------------
# adaptive vs fixed registration on the diagnosed failure mode
# ---------------------------------------------------------------------------

def _forty_deg_setup(seed=0):
    """Thin box, true rotation 40 deg about its (world-x) principal axis;
    T_init = identity, i.e. a 40-deg init residual about the ALK axis, far
    outside the fixed +-15 deg bound."""
    src = thin_box_cloud(seed=seed)
    center = src.mean(axis=0)
    R = Rotation.from_rotvec(np.radians(40.0) * np.array([1.0, 0, 0])).as_matrix()
    T_true = np.eye(4)
    T_true[:3, :3] = R
    T_true[:3, 3] = center - R @ center + np.array([0.004, -0.003, 0.002])
    tgt = transform_points(T_true, src)
    return src, tgt, T_true


def test_fixed_bounds_cannot_recover_40deg_axis_rotation():
    src, tgt, T_true = _forty_deg_setup()
    res = bounded_registration(src, tgt, np.eye(4), seed=0)
    assert _rot_err_deg(res["T"], T_true) > 20.0  # stuck at the +-15 bound


def test_adaptive_recovers_40deg_axis_rotation():
    src, tgt, T_true = _forty_deg_setup()
    res = adaptive_registration(src, tgt, np.eye(4), slender_alk(), seed=0)
    assert res["adaptive"]["adaptive_triggered"]
    assert abs(res["coarse_axis_theta_deg"]) >= 30.0  # coarse stage engaged
    assert _rot_err_deg(res["T"], T_true) < 5.0
    assert res["chamfer_final"] <= res["chamfer_init"]


def test_adaptive_noop_when_well_conditioned():
    """Non-slender demo ALK: adaptive_registration must return exactly the
    plain bounded_registration result (bounds unchanged)."""
    rng = np.random.RandomState(4)
    src = thin_box_cloud(seed=2, dims=(0.08, 0.06, 0.04))
    center = src.mean(axis=0)
    R = Rotation.from_euler("xyz", [4.0, -6.0, 8.0], degrees=True).as_matrix()
    T_true = np.eye(4)
    T_true[:3, :3] = R
    T_true[:3, 3] = center - R @ center + np.array([0.006, 0.004, -0.005])
    tgt = transform_points(T_true, src) + rng.normal(scale=5e-4,
                                                     size=src.shape)
    res_a = adaptive_registration(src, tgt, np.eye(4), spread_alk(), seed=0)
    res_f = bounded_registration(src, tgt, np.eye(4), seed=0)
    assert not res_a["adaptive"]["adaptive_triggered"]
    assert res_a["coarse_axis_theta_deg"] == 0.0
    assert res_a["axis_world"] is None
    np.testing.assert_allclose(res_a["T"], res_f["T"], atol=1e-12)
    assert res_a["chamfer_final"] == res_f["chamfer_final"]


def test_adaptive_stays_within_base_bounds_when_init_is_good():
    """Triggered adaptive on an ALREADY good init must not wander: coarse
    theta 0, refined pose within a few deg of truth."""
    src, tgt, T_true = _forty_deg_setup(seed=3)
    res = adaptive_registration(src, tgt, T_true, slender_alk(), seed=0)
    assert res["adaptive"]["adaptive_triggered"]
    assert res["coarse_axis_theta_deg"] == 0.0
    assert _rot_err_deg(res["T"], T_true) < 5.0


# ---------------------------------------------------------------------------
# slender depth-consistency auto-prior
# ---------------------------------------------------------------------------

def _slender_scene(dz=0.02):
    """Slender cloud along x with a height offset between the two lateral
    halves (the artifact the depth prior removes)."""
    rng = np.random.RandomState(7)
    n = 2000
    x = rng.uniform(-0.04, 0.04, n)
    y = rng.uniform(-0.011, 0.011, n)
    z = np.where(y >= 0, dz, 0.0) + rng.uniform(0, 0.004, n)
    pts = np.stack([x, y, z], axis=1)
    pixels = np.stack([x * 1000 + 64, y * 1000 + 64], axis=1)
    c1 = pts[np.argmax(x)]
    c2 = pts[np.argmin(x)]
    uv1 = pixels[np.argmax(x)]
    uv2 = pixels[np.argmin(x)]
    return pts, pixels, c1, c2, uv1, uv2


def test_lateral_extent_ratio_separates_slender_from_round():
    pts, _, c1, c2, _, _ = _slender_scene()
    r = lateral_extent_ratio(pts, c1, c2)
    assert r["ratio"] < SLENDER_RATIO_THRESHOLD
    rng = np.random.RandomState(1)
    ball = rng.uniform(-0.03, 0.03, (2000, 3))
    ax = ball[np.argmax(ball[:, 0])], ball[np.argmin(ball[:, 0])]
    r2 = lateral_extent_ratio(ball, ax[0], ax[1])
    assert r2["ratio"] > SLENDER_RATIO_THRESHOLD


def test_build_alk_auto_depth_prior():
    pts, pixels, c1, c2, uv1, uv2 = _slender_scene()
    alk_auto = build_alk(pts, pixels, c1, c2, uv1, uv2,
                         depth_consistent="auto")
    alk_off = build_alk(pts, pixels, c1, c2, uv1, uv2,
                        depth_consistent=False)
    # slender: auto enables the prior -> lateral z equalized
    assert abs(alk_auto[2, 2] - alk_auto[3, 2]) < 1e-12
    assert abs(alk_off[2, 2] - alk_off[3, 2]) > 0.01
    # non-slender: auto leaves the ALK exactly as depth_consistent=False
    rng = np.random.RandomState(2)
    ball = rng.uniform(-0.03, 0.03, (2000, 3))
    bpix = ball[:, :2] * 1000 + 64
    b1, b2 = ball[np.argmax(ball[:, 0])], ball[np.argmin(ball[:, 0])]
    u1, u2 = bpix[np.argmax(ball[:, 0])], bpix[np.argmin(ball[:, 0])]
    a_auto = build_alk(ball, bpix, b1, b2, u1, u2, depth_consistent="auto")
    a_off = build_alk(ball, bpix, b1, b2, u1, u2, depth_consistent=False)
    np.testing.assert_allclose(a_auto, a_off, atol=1e-15)
