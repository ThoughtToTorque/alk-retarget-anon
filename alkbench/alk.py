"""ALK (Axial-Lateral Keypoint) construction (paper Section 3.4, Eq. 6)."""

import numpy as np

from alkbench.candidates import project

# Slender-object auto-threshold for the depth-consistency prior: RMS lateral
# spread about the c1->c2 axis over RMS axial spread (lateral_extent_ratio).
# Calibrated on the five Campaign-A demo clouds (per-task constants, shared
# demo): pour 0.625 | cap_twist 0.678 | rim_grasp 0.728 | nut_loosen 0.767 |
# box_open 1.107.  0.65 sits between pour (the slender elongated block whose
# lateral halfplane centroids are depth-noise dominated) and the nearest
# non-slender task.
SLENDER_RATIO_THRESHOLD = 0.65

# Which of the four ALK points enter the closed-form Procrustes sum.
# name -> row indices into the (4, 3) ALK array [c1, c2, c3, c4].
#
#   alk4       {c1,c2,c3,c4} -- this benchmark's default (sum_{j=1}^{4}).
#   alk3_c134  {c1,c3,c4}    -- the earlier three-point construction of the
#                               hardware system (sum over {c1, c3, c4})
#                               (one axial endpoint + both lateral centroids;
#                               c2 is still perceived -- it anchors the
#                               halfplane split -- but is dropped from the
#                               alignment sum).  Any 3 points are coplanar, so
#                               this configuration has sigma3 == 0 by
#                               construction (Campaign F).
#   alk3_c123  {c1,c2,c3}    -- axis + one lateral centroid (completeness).
#   alk2_c12   {c1,c2}       -- axial pair only: rank-1, roll about the axis
#                               unobservable, frozen at zero by the
#                               shortest-arc convention (cf. baseline
#                               kp_pca2, which uses the same convention on a
#                               differently matched candidate pair).
ALK_SUBSETS = {
    "alk4": (0, 1, 2, 3),
    "alk3_c134": (0, 2, 3),
    "alk3_c123": (0, 1, 2),
    "alk2_c12": (0, 1),
}


def alk_subset_indices(name):
    """Row indices of a named ALK point subset (see ALK_SUBSETS)."""
    try:
        return ALK_SUBSETS[name]
    except KeyError:
        raise KeyError("unknown ALK subset %r (have %s)"
                       % (name, sorted(ALK_SUBSETS)))


def select_alk_subset(alk, name):
    """Rows of a (4, 3) ALK array selected by a named subset."""
    A = np.asarray(alk, dtype=np.float64)
    if A.shape != (4, 3):
        raise ValueError("expected a (4, 3) ALK array, got %r" % (A.shape,))
    return A[list(alk_subset_indices(name))]


def lateral_extent_ratio(points3d, c1, c2):
    """Lateral-vs-axial extent of an object cloud about the c1->c2 axis.

    ratio = RMS distance of the centered cloud from the axis direction
    (perpendicular component) / RMS extent along the axis.  Small values
    mean a slender, nearly rank-1 object whose lateral ALK centroids carry
    little geometric signal (the conditioning regime of paper Section 3.7).

    Returns dict: lateral_rms_m, axial_rms_m, ratio, axis_len_m.
    """
    P = np.asarray(points3d, dtype=np.float64)
    c1 = np.asarray(c1, dtype=np.float64)
    c2 = np.asarray(c2, dtype=np.float64)
    axis = c2 - c1
    axis_len = float(np.linalg.norm(axis))
    if axis_len <= 0:
        raise ValueError("degenerate axis: c1 == c2")
    axis = axis / axis_len
    X = P - P.mean(axis=0)
    axial = X @ axis
    perp = X - np.outer(axial, axis)
    lateral_rms = float(np.sqrt((np.linalg.norm(perp, axis=1) ** 2).mean()))
    axial_rms = float(np.sqrt((axial ** 2).mean()))
    ratio = lateral_rms / axial_rms if axial_rms > 0 else np.inf
    return {"lateral_rms_m": lateral_rms, "axial_rms_m": axial_rms,
            "ratio": float(ratio), "axis_len_m": axis_len}


def halfplane_signed_distance(pixels, uv1, uv2):
    """Signed distance of pixels to the image-plane line through uv1 -> uv2.

    s(u, v) = (u - u1)(v2 - v1) - (v - v1)(u2 - u1)

    Returns (N,) float; s >= 0 defines halfplane S+, s < 0 defines S-.
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    u1, v1 = float(uv1[0]), float(uv1[1])
    u2, v2 = float(uv2[0]), float(uv2[1])
    return (pixels[:, 0] - u1) * (v2 - v1) - (pixels[:, 1] - v1) * (u2 - u1)


def build_alk(points3d, pixels, c1, c2, uv1, uv2, phi3=False, depth_consistent=False):
    """Build the 4-point ALK from axial endpoints and the halfplane split.

    Parameters
    ----------
    points3d : (N, 3) object points (camera frame if depth_consistent is used)
    pixels : (N, 2) matching pixel coordinates
    c1, c2 : (3,) axial endpoint 3D centroids (candidates phi1, phi2)
    uv1, uv2 : (2,) image-plane projections of c1, c2 (line anchors)
    phi3 : bool, lateral swap bit -- swaps c3/c4
    depth_consistent : bool, slender-object prior: c3_z <- c4_z <- (c3_z + c4_z)/2;
        or "auto": enable iff lateral_extent_ratio(points3d, c1, c2)["ratio"]
        < SLENDER_RATIO_THRESHOLD

    Returns
    -------
    (4, 3) float64 array [c1, c2, c3, c4]
    """
    points3d = np.asarray(points3d, dtype=np.float64)
    if isinstance(depth_consistent, str):
        if depth_consistent != "auto":
            raise ValueError("depth_consistent must be bool or 'auto'")
        depth_consistent = bool(
            lateral_extent_ratio(points3d, c1, c2)["ratio"]
            < SLENDER_RATIO_THRESHOLD)
    s = halfplane_signed_distance(pixels, uv1, uv2)
    plus = s >= 0
    minus = ~plus
    if not plus.any() or not minus.any():
        raise ValueError("degenerate halfplane split: one side is empty")
    c3 = points3d[plus].mean(axis=0)
    c4 = points3d[minus].mean(axis=0)
    if depth_consistent:
        z = 0.5 * (c3[2] + c4[2])
        c3 = c3.copy()
        c4 = c4.copy()
        c3[2] = z
        c4[2] = z
    if phi3:
        c3, c4 = c4, c3
    return np.stack([np.asarray(c1, dtype=np.float64),
                     np.asarray(c2, dtype=np.float64), c3, c4])


def alk_from_candidates(cands, phi1, phi2, phi3=False, depth_consistent=False,
                        intrinsics=None):
    """Build ALK from a CandidateSet and discrete choices.

    Parameters
    ----------
    cands : CandidateSet
    phi1, phi2 : int, candidate indices (0-based) of the axial endpoints
    phi3 : bool, lateral swap bit
    depth_consistent : bool, slender-object depth prior
    intrinsics : (fx, fy, cx, cy) or None. If given, the halfplane line is
        anchored at the exact projections of c1, c2 (requires camera-frame
        points); otherwise the clusters' 2D k-means centers are used.

    Returns
    -------
    (4, 3) ALK array [c1, c2, c3, c4]
    """
    if phi1 == phi2:
        raise ValueError("phi1 and phi2 must differ")
    c1 = cands.candidates3d[phi1]
    c2 = cands.candidates3d[phi2]
    if intrinsics is not None:
        fx, fy, cx, cy = intrinsics
        uv = project(np.stack([c1, c2]), fx, fy, cx, cy)
        uv1, uv2 = uv[0], uv[1]
    else:
        uv1 = cands.centers2d[phi1]
        uv2 = cands.centers2d[phi2]
    return build_alk(cands.points3d, cands.pixels, c1, c2, uv1, uv2,
                     phi3=phi3, depth_consistent=depth_consistent)
