"""Waypoint retargeting: T_k^t = T_map @ T_k^d, plus grasp-point translation
correction (paper Section 3.8)."""

import numpy as np

from alkbench.procrustes import transform_points


def retarget_waypoints(T_map, waypoints):
    """Apply T_map to demonstration TCP waypoints.

    Parameters
    ----------
    T_map : (4, 4)
    waypoints : (n, 4, 4) or list of 4x4 poses

    Returns
    -------
    (n, 4, 4) float64
    """
    T_map = np.asarray(T_map, dtype=np.float64)
    W = np.asarray(waypoints, dtype=np.float64)
    if W.ndim == 2:
        W = W[None]
    return np.einsum("ij,njk->nik", T_map, W)


def grasp_translation_correction(T_map, demo_grasp, target_grasp):
    """Uniform translation Delta_t = g_target - T_map(g_demo).

    Parameters
    ----------
    T_map : (4, 4)
    demo_grasp : (3,) grasp point in the demo scene
    target_grasp : (3,) grasp point in the target scene

    Returns
    -------
    (3,) float64 translation correction
    """
    mapped = transform_points(T_map, np.asarray(demo_grasp, dtype=np.float64)[None])[0]
    return np.asarray(target_grasp, dtype=np.float64) - mapped


def retarget(T_map, waypoints, demo_grasp=None, target_grasp=None):
    """Retarget waypoints; if both grasp points are given, apply the uniform
    grasp-point translation correction to all waypoints.

    Returns
    -------
    (n, 4, 4) retargeted waypoints
    """
    W = retarget_waypoints(T_map, waypoints)
    if demo_grasp is not None and target_grasp is not None:
        dt = grasp_translation_correction(T_map, demo_grasp, target_grasp)
        W = W.copy()
        W[:, :3, 3] += dt
    return W
