"""Candidate keypoint extraction: backprojection + 2D k-means, 3D centroids."""

import numpy as np


def backproject(mask, depth, fx, fy, cx, cy, extrinsic=None):
    """Backproject masked valid-depth pixels to 3D.

    Parameters
    ----------
    mask : (H, W) bool
    depth : (H, W) float, metric depth (<=0 or non-finite is invalid)
    fx, fy, cx, cy : float, pinhole intrinsics
    extrinsic : (4, 4) or None, camera-to-world; if given, points are
        returned in world frame, otherwise in camera frame.

    Returns
    -------
    points : (N, 3) float64
    pixels : (N, 2) float64, matching (u, v) pixel coordinates
    """
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth, dtype=np.float64)
    valid = mask & np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(valid)
    z = depth[v, u]
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    points = np.stack([x, y, z], axis=1)
    if extrinsic is not None:
        E = np.asarray(extrinsic, dtype=np.float64)
        points = points @ E[:3, :3].T + E[:3, 3]
    pixels = np.stack([u, v], axis=1).astype(np.float64)
    return points, pixels


def project(points, fx, fy, cx, cy):
    """Project camera-frame 3D points to (u, v) pixel coordinates. (N, 3) -> (N, 2)."""
    points = np.asarray(points, dtype=np.float64)
    u = points[:, 0] / points[:, 2] * fx + cx
    v = points[:, 1] / points[:, 2] * fy + cy
    return np.stack([u, v], axis=1)


def kmeans(X, k=8, seed=0, n_iter=100, tol=1e-8):
    """Plain Lloyd's k-means with a fixed seed (deterministic).

    Parameters
    ----------
    X : (N, D) float
    k : int
    seed : int, seeds both the init draw and nothing else.

    Returns
    -------
    centers : (k, D) float64
    labels : (N,) int
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < k:
        raise ValueError("kmeans: need at least k=%d samples, got %d" % (k, n))
    rng = np.random.RandomState(seed)
    centers = X[rng.choice(n, size=k, replace=False)].copy()
    for _ in range(n_iter):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)
        new_centers = centers.copy()
        for j in range(k):
            m = labels == j
            if m.any():
                new_centers[j] = X[m].mean(axis=0)
            else:  # re-seed empty cluster at the point farthest from its center
                new_centers[j] = X[d2.min(axis=1).argmax()]
        shift = np.abs(new_centers - centers).max()
        centers = new_centers
        if shift < tol:
            break
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return centers, d2.argmin(axis=1)


class CandidateSet(object):
    """Indexed candidate keypoints for one segmented object.

    Attributes
    ----------
    points3d : (N, 3) backprojected object points P_M
    pixels : (N, 2) matching (u, v) pixel coordinates
    labels : (N,) cluster index in [0, k)
    centers2d : (k, 2) k-means centers in pixel space
    candidates3d : (k, 3) 3D centroid of each cluster's valid-depth points
    """

    def __init__(self, points3d, pixels, labels, centers2d, candidates3d):
        self.points3d = points3d
        self.pixels = pixels
        self.labels = labels
        self.centers2d = centers2d
        self.candidates3d = candidates3d

    @property
    def k(self):
        return self.candidates3d.shape[0]


def compute_candidates(mask, depth, fx, fy, cx, cy, extrinsic=None, k=8, seed=0):
    """Full candidate pipeline: backproject, k-means in 2D pixel space,
    then average each cluster's 3D points (cluster in 2D, average in 3D).

    Returns
    -------
    CandidateSet
    """
    points, pixels = backproject(mask, depth, fx, fy, cx, cy, extrinsic=extrinsic)
    centers2d, labels = kmeans(pixels, k=k, seed=seed)
    candidates3d = np.stack([points[labels == j].mean(axis=0) for j in range(k)])
    return CandidateSet(points, pixels, labels, centers2d, candidates3d)
