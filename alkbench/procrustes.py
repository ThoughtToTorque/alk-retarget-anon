"""Orthogonal Procrustes (Kabsch) alignment of ordered 3D point pairs."""

import numpy as np
from scipy.spatial.transform import Rotation


def transform_points(T, points):
    """Apply a 4x4 homogeneous transform to (N, 3) points."""
    T = np.asarray(T, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    return points @ T[:3, :3].T + T[:3, 3]


def rotation_angle_deg(R):
    """Rotation angle of a 3x3 rotation matrix, in degrees.

    Uses the quaternion magnitude (numerically accurate near identity,
    unlike the trace/arccos formula)."""
    R = np.asarray(R, dtype=np.float64)[:3, :3]
    return np.degrees(Rotation.from_matrix(R).magnitude())


def procrustes(P, Q):
    """Rigid transform T = (R, t) in SE(3) minimizing sum ||R p_j + t - q_j||^2.

    Parameters
    ----------
    P : (n, 3) source points, n >= 3
    Q : (n, 3) target points, in correspondence with P

    Returns
    -------
    T : (4, 4) float64 with proper rotation (det R = +1)
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape or P.shape[0] < 3 or P.shape[1] != 3:
        raise ValueError("procrustes: expected matching (n>=3, 3) arrays")
    pc = P.mean(axis=0)
    qc = Q.mean(axis=0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = qc - R @ pc
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def shortest_arc_rotation(a, b):
    """Minimal rotation matrix taking direction `a` onto `b` (zero twist
    about the rotation axis).  Antiparallel case: 180 deg about an arbitrary
    perpendicular axis.  (Same convention -- and formulas -- as
    baselines.common.shortest_arc_rotation, kept here so the core library has
    no dependency on the baselines package.)"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = np.cross(a, b)
    d = float(np.dot(a, b))
    if np.linalg.norm(c) < 1e-12:
        if d > 0:
            return np.eye(3)
        p = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            p = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, p)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * np.pi).as_matrix()
    axis = c / np.linalg.norm(c)
    angle = np.arctan2(np.linalg.norm(c), d)
    return Rotation.from_rotvec(axis * angle).as_matrix()


def two_point_alignment(P, Q):
    """Closed-form SE(3) from 2 matched point pairs.

    A 2-point configuration is rank-1: rotation about the segment direction
    is unobservable.  Convention (identical to the kp_pca2 baseline): the
    shortest-arc rotation taking the source segment direction onto the target
    one -- i.e. the unobservable roll is frozen at zero -- plus the midpoint
    translation.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != (2, 3) or Q.shape != (2, 3):
        raise ValueError("two_point_alignment: expected two (2, 3) arrays")
    R = shortest_arc_rotation(P[1] - P[0], Q[1] - Q[0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = Q.mean(axis=0) - R @ P.mean(axis=0)
    return T


def align_keypoints(P, Q):
    """Closed-form SE(3) alignment of n matched keypoint pairs.

    n >= 3 -> orthogonal `procrustes`; n == 2 -> `two_point_alignment`
    (roll about the axis frozen at zero).  Used by the ALK point-subset
    ablation (Campaign F), where the subset may be a rank-1 pair.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.shape[0] == 2:
        return two_point_alignment(P, Q)
    return procrustes(P, Q)
