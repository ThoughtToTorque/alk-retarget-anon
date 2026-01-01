"""Synthetic scene helpers shared by the alkbench tests."""

import numpy as np
from scipy.spatial.transform import Rotation

FX = FY = 220.0
CX = CY = 64.0
H = W = 128


def make_cloud(n=4000, seed=0):
    """Asymmetric elongated cloud: cylinder along x + off-axis bump near one end.

    Points in camera frame, centered around z ~ 0.5 m. Returns (n, 3)."""
    rng = np.random.RandomState(seed)
    n_cyl = int(n * 0.8)
    n_bump = n - n_cyl
    x = rng.uniform(-0.10, 0.10, n_cyl)
    theta = rng.uniform(0, 2 * np.pi, n_cyl)
    r = 0.018 * np.sqrt(rng.uniform(0, 1, n_cyl))
    cyl = np.stack([x, r * np.cos(theta), r * np.sin(theta)], axis=1)
    bump = rng.normal(scale=0.008, size=(n_bump, 3)) + np.array([0.07, 0.025, 0.0])
    pts = np.concatenate([cyl, bump])
    pts += np.array([0.0, 0.0, 0.5])
    return pts


def render(points, h=H, w=W, fx=FX, fy=FY, cx=CX, cy=CY):
    """Point-splat z-buffer render -> (mask, depth)."""
    u = np.round(points[:, 0] / points[:, 2] * fx + cx).astype(int)
    v = np.round(points[:, 1] / points[:, 2] * fy + cy).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (points[:, 2] > 0)
    depth = np.full((h, w), np.inf)
    np.minimum.at(depth, (v[ok], u[ok]), points[ok, 2])
    mask = np.isfinite(depth)
    depth[~mask] = 0.0
    return mask, depth


def random_se3(seed, max_angle_deg=30.0, max_trans=0.05):
    """Random SE(3) transform with bounded magnitude."""
    rng = np.random.RandomState(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(rng.uniform(5.0, max_angle_deg))
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
    T[:3, 3] = rng.uniform(-max_trans, max_trans, 3)
    return T


def small_se3(rot_deg, trans, axis=(0.0, 0.0, 1.0)):
    """SE(3) with a given rotation angle about `axis` and translation vector."""
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(a * np.radians(rot_deg)).as_matrix()
    T[:3, 3] = np.asarray(trans, dtype=float)
    return T
