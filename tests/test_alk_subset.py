"""ALK point-subset alignment (Campaign F): which ALK points enter the
closed-form Procrustes sum.

The earlier three-point construction sums over three pairs and labels
C1, C3, C4; this benchmark's default sums over all four.  The tests below pin
the subset plumbing and the two properties the ablation rests on:

  * a 3-point subset is COPLANAR -> sigma3 == 0 exactly (rank 2), while the
    4-point ALK is genuinely 3D (sigma3 > 0);
  * the 2-point subset is rank-1 -> sigma2 + sigma3 == 0, roll frozen.
"""
import numpy as np
import pytest

from alkbench import (ALK_SUBSETS, alk_subset_indices, select_alk_subset,
                      align_keypoints, procrustes, two_point_alignment,
                      transform_points, adaptive_bounds)


def _alk():
    # a generic (non-degenerate, non-coplanar) ALK-shaped quadruple
    return np.array([[0.00, 0.00, 0.80],
                     [0.10, 0.02, 0.83],
                     [0.05, 0.03, 0.81],
                     [0.04, -0.02, 0.82]])


def _rigid(angle_deg=20.0, axis=(0.3, -0.5, 0.8), t=(0.02, -0.03, 0.01)):
    from scipy.spatial.transform import Rotation
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(a * np.radians(angle_deg)).as_matrix()
    T[:3, 3] = t
    return T


def test_subset_indices_and_selection():
    assert alk_subset_indices("alk4") == (0, 1, 2, 3)
    assert alk_subset_indices("alk3_c134") == (0, 2, 3)   # c2 dropped
    assert alk_subset_indices("alk3_c123") == (0, 1, 2)
    assert alk_subset_indices("alk2_c12") == (0, 1)
    with pytest.raises(KeyError):
        alk_subset_indices("nope")
    A = _alk()
    for name, idx in ALK_SUBSETS.items():
        S = select_alk_subset(A, name)
        assert S.shape == (len(idx), 3)
        assert np.array_equal(S, A[list(idx)])
    with pytest.raises(ValueError):
        select_alk_subset(A[:3], "alk4")


@pytest.mark.parametrize("name", sorted(ALK_SUBSETS))
def test_every_subset_recovers_an_exact_rigid_motion(name):
    """On noise-free correspondences every subset (even the rank-1 pair, up to
    its frozen roll) must reproduce the generating transform on its OWN
    points."""
    A = _alk()
    T = _rigid()
    B = transform_points(T, A)
    P, Q = select_alk_subset(A, name), select_alk_subset(B, name)
    T_hat = align_keypoints(P, Q)
    assert np.allclose(transform_points(T_hat, P), Q, atol=1e-9)
    if name != "alk2_c12":       # 2 points leave roll about the axis free
        assert np.allclose(T_hat, T, atol=1e-9)


def test_align_keypoints_dispatch():
    A = _alk()
    B = transform_points(_rigid(), A)
    assert np.allclose(align_keypoints(A, B), procrustes(A, B))
    assert np.allclose(align_keypoints(A[:2], B[:2]),
                       two_point_alignment(A[:2], B[:2]))


def test_three_point_subset_is_rank_deficient_by_construction():
    A = _alk()

    def sv(P):
        X = P - P.mean(axis=0)
        s = np.linalg.svd(X, compute_uv=False)
        return np.concatenate([s, np.zeros(3)])[:3]

    s4 = sv(A)
    assert s4[2] > 0                      # the 4-point ALK spans 3D
    for name in ("alk3_c134", "alk3_c123"):
        s3 = sv(select_alk_subset(A, name))
        assert s3[2] == pytest.approx(0.0, abs=1e-15)   # 3 points => coplanar
        assert s3[1] + s3[2] < s4[1] + s4[2]            # smaller sigma23
    s2 = sv(select_alk_subset(A, "alk2_c12"))
    assert s2[1] + s2[2] == pytest.approx(0.0, abs=1e-15)


def test_adaptive_bounds_cond_points_splits_axis_from_conditioning():
    """`cond_points` lets the trigger read the SOLVED subset while the widened
    axis still comes from the full ALK's c1 -> c2."""
    A = _alk()
    sub = select_alk_subset(A, "alk3_c134")
    full = adaptive_bounds(A)
    mixed = adaptive_bounds(A, cond_points=sub)
    assert np.allclose(mixed["axis_demo"], full["axis_demo"])
    assert mixed["singular_values"][2] == pytest.approx(0.0, abs=1e-15)
    assert mixed["sigma23"] < full["sigma23"]
    # default stays exactly the old behavior
    again = adaptive_bounds(A, cond_points=None)
    assert sorted(again) == sorted(full)
    for k, v in full.items():
        assert np.all(np.asarray(again[k]) == np.asarray(v))
