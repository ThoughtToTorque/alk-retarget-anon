"""Tests for Procrustes, bounded registration and waypoint retargeting."""

import time

import numpy as np
from scipy.spatial.transform import Rotation

from alkbench import (procrustes, transform_points, rotation_angle_deg,
                      chamfer_distance, bounded_registration,
                      compute_candidates, alk_from_candidates,
                      retarget_waypoints, grasp_translation_correction, retarget)
from synth import FX, FY, CX, CY, make_cloud, render, random_se3, small_se3


def _rot_err_deg(Ta, Tb):
    return rotation_angle_deg(Ta[:3, :3].T @ Tb[:3, :3])


def test_procrustes_exact_recovery():
    for seed in range(5):
        rng = np.random.RandomState(seed)
        P = rng.uniform(-0.1, 0.1, (4, 3))
        T = random_se3(seed + 100)
        Q = transform_points(T, P)
        T_hat = procrustes(P, Q)
        assert _rot_err_deg(T_hat, T) < 1e-6
        assert np.linalg.norm(T_hat[:3, 3] - T[:3, 3]) < 1e-9
        assert np.isclose(np.linalg.det(T_hat[:3, :3]), 1.0)


def test_procrustes_on_exact_alk_correspondences():
    """ALK from the synthetic scene, target ALK = T_true(demo ALK)."""
    cloud = make_cloud(seed=1)
    mask, depth = render(cloud)
    cands = compute_candidates(mask, depth, FX, FY, CX, CY, k=8, seed=0)
    order = np.argsort(cands.centers2d[:, 0])
    alk_d = alk_from_candidates(cands, int(order[0]), int(order[-1]),
                                intrinsics=(FX, FY, CX, CY))
    T_true = random_se3(7)
    alk_t = transform_points(T_true, alk_d)
    T_map = procrustes(alk_d, alk_t)
    assert _rot_err_deg(T_map, T_true) < 1e-6
    assert np.linalg.norm(T_map[:3, 3] - T_true[:3, 3]) < 1e-9


def test_bounded_registration_reduces_chamfer_under_noise():
    rng = np.random.RandomState(3)
    A = make_cloud(n=3000, seed=2)
    T_true = random_se3(11, max_angle_deg=20.0, max_trans=0.04)
    B = transform_points(T_true, A) + rng.normal(scale=0.002, size=A.shape)
    # perturb the closed-form init within the search bounds: 8 deg about the
    # object centroid + 12 mm translation
    center = transform_points(T_true, A).mean(axis=0)
    D = small_se3(8.0, [0.008, -0.006, 0.006], axis=(0.3, 0.5, 1.0))
    D[:3, 3] += center - D[:3, :3] @ center
    T_init = D @ T_true
    t0 = time.monotonic()
    res = bounded_registration(A, B, T_init, seed=0)
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0
    assert res["chamfer_final"] < res["chamfer_init"]
    assert res["chamfer_final"] < 0.9 * res["chamfer_init"]
    # note: rotation about the slender cloud's long axis is weakly observable
    # from chamfer, so we assert chamfer reduction (the Eq. 8 objective), not
    # pose-error reduction.


def test_bounded_registration_grid_option():
    rng = np.random.RandomState(5)
    A = make_cloud(n=1500, seed=4)
    T_true = np.eye(4)
    B = transform_points(T_true, A) + rng.normal(scale=0.001, size=A.shape)
    center = A.mean(axis=0)
    D = small_se3(6.0, [0.009, 0.0, -0.007])
    D[:3, 3] += center - D[:3, :3] @ center
    T_init = D @ T_true
    res = bounded_registration(A, B, T_init, method="grid",
                               rot_bound_deg=10.0, rot_step_deg=5.0,
                               trans_bound=0.01, trans_step=0.01,
                               max_points=800, seed=0)
    assert res["chamfer_final"] <= res["chamfer_init"]


def test_chamfer_zero_on_identical_sets():
    A = make_cloud(n=500, seed=6)
    assert chamfer_distance(A, A) < 1e-12


def _random_waypoints(n, seed, center=(0.0, 0.0, 0.0), spread=0.3):
    rng = np.random.RandomState(seed)
    W = np.zeros((n, 4, 4))
    for i in range(n):
        W[i, :3, :3] = Rotation.random(random_state=rng).as_matrix()
        W[i, :3, 3] = np.asarray(center) + rng.uniform(-spread, spread, 3)
        W[i, 3, 3] = 1.0
    return W


def test_retarget_waypoints_matches_ground_truth():
    W = _random_waypoints(5, seed=8)
    T_map = random_se3(9)
    gt = np.stack([T_map @ w for w in W])
    out = retarget_waypoints(T_map, W)
    assert np.allclose(out, gt, atol=1e-12)


def test_grasp_translation_correction():
    W = _random_waypoints(4, seed=10)
    T_map = random_se3(12)
    g_demo = np.array([0.05, -0.02, 0.4])
    # exact target grasp -> zero correction
    g_tgt = transform_points(T_map, g_demo[None])[0]
    dt = grasp_translation_correction(T_map, g_demo, g_tgt)
    assert np.allclose(dt, 0.0, atol=1e-12)
    # offset target grasp -> every waypoint shifted uniformly by the offset
    delta = np.array([0.003, -0.004, 0.002])
    out = retarget(T_map, W, demo_grasp=g_demo, target_grasp=g_tgt + delta)
    base = retarget_waypoints(T_map, W)
    assert np.allclose(out[:, :3, 3] - base[:, :3, 3], delta, atol=1e-12)
    assert np.allclose(out[:, :3, :3], base[:, :3, :3])


def test_end_to_end_retarget_within_tolerance():
    """Full pipe: ALK Procrustes init + bounded refine, then retarget."""
    rng = np.random.RandomState(13)
    cloud = make_cloud(seed=3)
    mask, depth = render(cloud)
    cands = compute_candidates(mask, depth, FX, FY, CX, CY, k=8, seed=0)
    order = np.argsort(cands.centers2d[:, 0])
    alk_d = alk_from_candidates(cands, int(order[0]), int(order[-1]),
                                intrinsics=(FX, FY, CX, CY))
    T_true = random_se3(14, max_angle_deg=25.0, max_trans=0.05)
    tgt_cloud = transform_points(T_true, cloud) + rng.normal(scale=0.001, size=cloud.shape)
    # 0.5 mm noise on ALK keypoints: the lateral centroids of a slender object
    # are only ~1.5 cm apart, so keypoint noise maps almost directly into
    # rotation error about the long axis -- keep it below the registration
    # capture range (15 deg).
    alk_t = transform_points(T_true, alk_d) + rng.normal(scale=0.0005, size=alk_d.shape)
    T0 = procrustes(alk_d, alk_t)
    res = bounded_registration(cloud, tgt_cloud, T0, seed=0)
    # TCP waypoints near the object (realistic lever arms of a few cm)
    W = _random_waypoints(3, seed=15, center=cloud.mean(axis=0), spread=0.06)
    gt = np.stack([T_true @ w for w in W])
    out = retarget_waypoints(res["T"], W)
    # positions within a couple of cm, rotations within a few degrees
    assert np.linalg.norm(out[:, :3, 3] - gt[:, :3, 3], axis=1).max() < 0.03
    for i in range(len(W)):
        assert rotation_angle_deg(out[i, :3, :3].T @ gt[i, :3, :3]) < 6.0
