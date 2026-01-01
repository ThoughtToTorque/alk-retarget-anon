"""Tests for candidate extraction and ALK construction on synthetic scenes."""

import numpy as np
import pytest

from alkbench import (backproject, compute_candidates, kmeans, build_alk,
                      alk_from_candidates, halfplane_signed_distance)
from synth import FX, FY, CX, CY, make_cloud, render


def _scene(seed=0):
    pts = make_cloud(seed=seed)
    mask, depth = render(pts)
    return pts, mask, depth


def test_backproject_matches_projection():
    _, mask, depth = _scene()
    pts, pix = backproject(mask, depth, FX, FY, CX, CY)
    u = pts[:, 0] / pts[:, 2] * FX + CX
    v = pts[:, 1] / pts[:, 2] * FY + CY
    assert np.allclose(u, pix[:, 0], atol=1e-9)
    assert np.allclose(v, pix[:, 1], atol=1e-9)


def test_backproject_extrinsic():
    _, mask, depth = _scene()
    E = np.eye(4)
    E[:3, 3] = [1.0, -2.0, 3.0]
    p_cam, _ = backproject(mask, depth, FX, FY, CX, CY)
    p_world, _ = backproject(mask, depth, FX, FY, CX, CY, extrinsic=E)
    assert np.allclose(p_world, p_cam + E[:3, 3])


def test_kmeans_deterministic():
    _, mask, depth = _scene()
    _, pix = backproject(mask, depth, FX, FY, CX, CY)
    c1, l1 = kmeans(pix, k=8, seed=42)
    c2, l2 = kmeans(pix, k=8, seed=42)
    assert np.array_equal(c1, c2)
    assert np.array_equal(l1, l2)


def test_compute_candidates_deterministic_and_sane():
    cloud, mask, depth = _scene()
    a = compute_candidates(mask, depth, FX, FY, CX, CY, k=8, seed=0)
    b = compute_candidates(mask, depth, FX, FY, CX, CY, k=8, seed=0)
    assert np.array_equal(a.candidates3d, b.candidates3d)
    assert a.k == 8
    assert np.isfinite(a.candidates3d).all()
    lo, hi = cloud.min(axis=0), cloud.max(axis=0)
    assert (a.candidates3d >= lo - 1e-6).all()
    assert (a.candidates3d <= hi + 1e-6).all()
    # every cluster non-empty
    assert set(np.unique(a.labels)) == set(range(8))


def _rect_scene():
    """Constant-depth rectangle mask: 20 rows x 60 cols."""
    mask = np.zeros((64, 128), dtype=bool)
    mask[22:42, 30:90] = True
    depth = np.where(mask, 0.5, 0.0)
    return mask, depth


def test_halfplane_partition_consistent_and_swap():
    mask, depth = _rect_scene()
    pts, pix = backproject(mask, depth, FX, FY, CX, CY)
    # axial endpoints: left and right centers of the rectangle
    uv1, uv2 = np.array([30.0, 31.5]), np.array([89.0, 31.5])
    c1 = pts[pix[:, 0] < 40].mean(axis=0)
    c2 = pts[pix[:, 0] > 80].mean(axis=0)
    s = halfplane_signed_distance(pix, uv1, uv2)
    # for a horizontal line, s = (u2-u1)*(v1-v) has sign of (v1 - v)... check
    # against the explicit formula
    expect = (pix[:, 0] - uv1[0]) * (uv2[1] - uv1[1]) - (pix[:, 1] - uv1[1]) * (uv2[0] - uv1[0])
    assert np.array_equal(s, expect)
    alk = build_alk(pts, pix, c1, c2, uv1, uv2, phi3=False)
    alk_sw = build_alk(pts, pix, c1, c2, uv1, uv2, phi3=True)
    # swap bit exchanges c3 and c4, leaves c1/c2 alone
    assert np.allclose(alk[2], alk_sw[3])
    assert np.allclose(alk[3], alk_sw[2])
    assert np.allclose(alk[:2], alk_sw[:2])
    # the two lateral centroids sit on opposite sides of the axis in y
    axis_y = 0.5 * (c1[1] + c2[1])
    assert (alk[2][1] - axis_y) * (alk[3][1] - axis_y) < 0
    # partition covers all points exactly once
    assert ((s >= 0).sum() + (s < 0).sum()) == len(pix)


def test_alk_depth_consistency_prior():
    mask, depth = _rect_scene()
    pts, pix = backproject(mask, depth, FX, FY, CX, CY)
    uv1, uv2 = np.array([30.0, 31.5]), np.array([89.0, 31.5])
    c1, c2 = pts[:50].mean(axis=0), pts[-50:].mean(axis=0)
    alk = build_alk(pts, pix, c1, c2, uv1, uv2, depth_consistent=True)
    assert alk[2][2] == alk[3][2]
    alk0 = build_alk(pts, pix, c1, c2, uv1, uv2, depth_consistent=False)
    assert np.isclose(alk[2][2], 0.5 * (alk0[2][2] + alk0[3][2]))


def test_alk_from_candidates_and_validation():
    _, mask, depth = _scene()
    cands = compute_candidates(mask, depth, FX, FY, CX, CY, k=8, seed=0)
    order = np.argsort(cands.centers2d[:, 0])
    phi1, phi2 = int(order[0]), int(order[-1])
    alk = alk_from_candidates(cands, phi1, phi2, intrinsics=(FX, FY, CX, CY))
    assert alk.shape == (4, 3)
    assert np.allclose(alk[0], cands.candidates3d[phi1])
    assert np.allclose(alk[1], cands.candidates3d[phi2])
    with pytest.raises(ValueError):
        alk_from_candidates(cands, 2, 2)
