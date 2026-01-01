"""Keypoint-anchored bounded registration: local search around T_map
minimizing the averaged symmetric Chamfer distance (paper Eq. 8)."""

import itertools
import warnings

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from alkbench.procrustes import transform_points


def chamfer_distance(A, B, tree_b=None):
    """Symmetric Chamfer distance: mean of the two directed mean-NN distances.

    d_ch(A, B) = 1/2 * ( mean_a min_b ||a-b|| + mean_b min_a ||b-a|| )
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    tb = tree_b if tree_b is not None else cKDTree(B)
    d_ab = tb.query(A)[0].mean()
    d_ba = cKDTree(A).query(B)[0].mean()
    return 0.5 * (d_ab + d_ba)


def _subsample(P, max_points, seed):
    P = np.asarray(P, dtype=np.float64)
    if P.shape[0] <= max_points:
        return P
    rng = np.random.RandomState(seed)
    return P[rng.choice(P.shape[0], size=max_points, replace=False)]


def _grid(bound, step):
    n = int(round(bound / step))
    return np.arange(-n, n + 1) * step


def _delta_transform(rot_deg, trans, center):
    """SE(3) perturbation: rotate by euler-xyz `rot_deg` about `center`, then
    translate by `trans`. Returns 4x4."""
    R = Rotation.from_euler("xyz", rot_deg, degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center - R @ center + np.asarray(trans, dtype=np.float64)
    return T


def _axis_delta_transform(axis, theta_deg, center):
    """SE(3) perturbation: rotate by `theta_deg` about the world-frame line
    through `center` with direction `axis` (unit). Returns 4x4. Same
    rotation-center convention as `_delta_transform`."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    R = Rotation.from_rotvec(axis * np.radians(theta_deg)).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center - R @ center
    return T


# Conditioning-adaptive registration (paper Section 3.7):
# the Procrustes rotation-error bound scales with 1/(sigma2+sigma3) of the
# centered demo ALK configuration.  When sigma2+sigma3 is small RELATIVE to
# sigma1 (a nearly rank-1, slender configuration), the poorly observed DoF
# is the rotation ABOUT the ALK principal axis (the c1-c2 direction): the
# lateral centroids barely constrain it, so the ALK init can carry a
# rotation residual far beyond the fixed +-15 deg registration bound.
# `adaptive_bounds` detects that regime from the demo ALK alone and widens
# ONLY the rotation bound about that axis; all other DoF keep base bounds.
#
# Threshold calibration (demo ALK of the five Campaign-A tasks; the demo is
# shared across seeds so these are per-task constants):
#   sigma_ratio (sigma2+sigma3)/sigma1: pour 0.359 | cap_twist 0.434 |
#   nut_loosen 0.456 | rim_grasp 0.498 | box_open 0.504.
# 0.40 sits between pour (worst-conditioned, the diagnosed failure) and the
# nearest well-behaved task with margin on both sides.  The relative
# criterion is preferred over absolute sigma23 because the absolute value
# is scale-dependent (rim_grasp's small can has sigma23 = 0.024 yet is
# well-conditioned relative to its size).
SIGMA_RATIO_THRESHOLD = 0.40


def adaptive_bounds(alk_demo, base_theta_deg=15.0, base_t_mm=20.0,
                    sigma_ratio_threshold=SIGMA_RATIO_THRESHOLD,
                    axis_bound_deg=90.0, cond_points=None):
    """Conditioning-adaptive per-axis registration bounds from the demo ALK.

    Computes the singular values sigma1 >= sigma2 >= sigma3 of the centered
    demo ALK configuration.  When (sigma2+sigma3)/sigma1 falls below
    `sigma_ratio_threshold`, the rotation bound about the ALK principal
    axis (unit vector c1 -> c2, demo world frame) is widened to
    `axis_bound_deg`; every other bound stays at base.

    `cond_points` (default None -> `alk_demo`): the configuration whose
    singular values drive the trigger, when it differs from the one that
    defines the axis.  Used by the ALK point-subset ablation (Campaign F):
    there the Procrustes sum runs over a SUBSET of the ALK (e.g. the paper's
    {c1,c3,c4}), so the conditioning that governs the closed-form rotation
    error is the subset's, while the axial axis is still c1 -> c2 of the full
    ALK.

    Returns
    -------
    dict with keys:
      adaptive_triggered : bool
      singular_values    : [sigma1, sigma2, sigma3]
      sigma23            : sigma2 + sigma3 (absolute, meters)
      sigma_ratio        : (sigma2 + sigma3) / sigma1
      axis_demo          : (3,) unit c1->c2 axis (demo world frame)
      axis_rot_bound_deg : widened (or base) bound about that axis
      rot_bound_deg      : base rotation bound for the other axes
      trans_bound_m      : base translation bound
    """
    A = np.asarray(alk_demo, dtype=np.float64)
    C = A if cond_points is None else np.asarray(cond_points,
                                                dtype=np.float64)
    X = C - C.mean(axis=0)
    s = np.linalg.svd(X, compute_uv=False)
    s = np.concatenate([s, np.zeros(3)])[:3]
    sigma23 = float(s[1] + s[2])
    sigma_ratio = float(sigma23 / s[0]) if s[0] > 0 else 0.0
    axis = A[1] - A[0]
    n = float(np.linalg.norm(axis))
    axis = axis / n if n > 0 else np.array([1.0, 0.0, 0.0])
    triggered = bool(sigma_ratio < sigma_ratio_threshold)
    return {
        "adaptive_triggered": triggered,
        "singular_values": [float(v) for v in s],
        "sigma23": sigma23,
        "sigma_ratio": sigma_ratio,
        "sigma_ratio_threshold": float(sigma_ratio_threshold),
        "axis_demo": axis,
        "axis_rot_bound_deg": float(axis_bound_deg if triggered
                                    else base_theta_deg),
        "rot_bound_deg": float(base_theta_deg),
        "trans_bound_m": float(base_t_mm) * 1e-3,
    }


def adaptive_registration(src, tgt, T_init, alk_demo,
                          base_theta_deg=15.0, base_t_mm=20.0,
                          sigma_ratio_threshold=SIGMA_RATIO_THRESHOLD,
                          axis_bound_deg=90.0, coarse_step_deg=15.0,
                          max_points=2000, seed=0, cond_points=None):
    """Conditioning-adaptive bounded registration (default-off pipeline flag).

    Falls back to plain `bounded_registration` with base bounds when the
    demo ALK is well conditioned (`adaptive_bounds` not triggered).  When
    triggered, runs a coarse-to-fine search of the rotation about the ALK
    principal axis:

      1. coarse: rotations of `coarse_step_deg` steps in
         [-axis_bound_deg, +axis_bound_deg] about the world-frame axis
         R_init @ axis_demo, composed about the centroid of T_init(src)
         (the same rotation-center convention as `bounded_registration`),
         scored on the symmetric Chamfer objective;
      2. fine: standard `bounded_registration` (base +-theta/+-t bounds)
         around the best coarse pose.

    Returns the `bounded_registration` dict plus:
      adaptive             : the `adaptive_bounds` dict
      coarse_axis_theta_deg: best coarse rotation about the axis (0 when
                             not triggered)
      axis_world           : (3,) widened axis in world frame (None when
                             not triggered)

    `cond_points` (default None -> `alk_demo`) is forwarded to
    `adaptive_bounds`: the trigger reads this configuration's singular values
    while the widened axis still comes from `alk_demo` (ALK point-subset
    ablation, Campaign F).
    """
    bounds = adaptive_bounds(alk_demo, base_theta_deg=base_theta_deg,
                             base_t_mm=base_t_mm,
                             sigma_ratio_threshold=sigma_ratio_threshold,
                             axis_bound_deg=axis_bound_deg,
                             cond_points=cond_points)
    if not bounds["adaptive_triggered"]:
        out = bounded_registration(src, tgt, T_init,
                                   rot_bound_deg=base_theta_deg,
                                   trans_bound=float(base_t_mm) * 1e-3,
                                   max_points=max_points, seed=seed)
        out["adaptive"] = bounds
        out["coarse_axis_theta_deg"] = 0.0
        out["axis_world"] = None
        return out

    T_init = np.asarray(T_init, dtype=np.float64)
    axis_world = T_init[:3, :3] @ bounds["axis_demo"]
    axis_world = axis_world / np.linalg.norm(axis_world)

    src_s = _subsample(src, max_points, seed)
    tgt_s = _subsample(tgt, max_points, seed)
    A0 = transform_points(T_init, src_s)
    center = A0.mean(axis=0)
    tree_tgt = cKDTree(tgt_s)

    thetas = _grid(axis_bound_deg, coarse_step_deg)
    best_theta = 0.0
    best_cost = np.inf
    chamfer_init = None
    for th in thetas:
        D = _axis_delta_transform(axis_world, th, center)
        c = chamfer_distance(transform_points(D, A0), tgt_s, tree_b=tree_tgt)
        if th == 0.0:
            chamfer_init = c
        if c < best_cost:
            best_cost = c
            best_theta = float(th)

    T_coarse = _axis_delta_transform(axis_world, best_theta,
                                     center) @ T_init
    out = bounded_registration(src, tgt, T_coarse,
                               rot_bound_deg=base_theta_deg,
                               trans_bound=float(base_t_mm) * 1e-3,
                               max_points=max_points, seed=seed)
    out["chamfer_init"] = float(chamfer_init)  # chamfer at T_init (theta=0)
    out["adaptive"] = bounds
    out["coarse_axis_theta_deg"] = best_theta
    out["axis_world"] = axis_world
    return out


def bounded_registration(src, tgt, T_init,
                         rot_bound_deg=15.0, rot_step_deg=5.0,
                         trans_bound=0.02, trans_step=0.01,
                         method="coord", n_passes=2,
                         max_points=2000, seed=0):
    """Refine T_init within a bounded SE(3) neighborhood by minimizing the
    symmetric Chamfer distance between transformed `src` and `tgt`.

    Perturbations rotate about the centroid of T_init(src), so the rotation
    and translation bounds stay meaningful for off-origin objects.

    method="coord" (default): axis-by-axis coordinate descent over the 6 DoF
    grids, repeated `n_passes` times: 3 rotation axes x 6 non-incumbent grid
    values + 3 translation axes x 4, i.e. 30 evaluations per pass plus the
    initial one, 61 with the default n_passes=2 (measured median 63; well
    under 1 s). method="grid": exhaustive product
    grid; with the default steps that is 7^3 * 5^3 = 42875 poses (minutes),
    so pass coarser steps -- a warning is raised above 10000 poses.

    Returns
    -------
    dict with keys "T" (refined 4x4), "chamfer_init", "chamfer_final".
    """
    src_s = _subsample(src, max_points, seed)
    tgt_s = _subsample(tgt, max_points, seed)
    A0 = transform_points(T_init, src_s)
    center = A0.mean(axis=0)
    tree_tgt = cKDTree(tgt_s)

    def cost(params):
        T = _delta_transform(params[:3], params[3:], center)
        return chamfer_distance(transform_points(T, A0), tgt_s, tree_b=tree_tgt)

    rot_vals = _grid(rot_bound_deg, rot_step_deg)
    trans_vals = _grid(trans_bound, trans_step)
    best = np.zeros(6)
    best_cost = cost(best)
    chamfer_init = best_cost

    if method == "coord":
        for _ in range(n_passes):
            for dim in range(6):
                vals = rot_vals if dim < 3 else trans_vals
                for v in vals:
                    if v == best[dim]:
                        continue
                    cand = best.copy()
                    cand[dim] = v
                    c = cost(cand)
                    if c < best_cost:
                        best_cost = c
                        best = cand
    elif method == "grid":
        n_poses = len(rot_vals) ** 3 * len(trans_vals) ** 3
        if n_poses > 10000:
            warnings.warn("exhaustive grid has %d poses; consider coarser "
                          "steps or method='coord'" % n_poses)
        for combo in itertools.product(rot_vals, rot_vals, rot_vals,
                                       trans_vals, trans_vals, trans_vals):
            cand = np.array(combo)
            c = cost(cand)
            if c < best_cost:
                best_cost = c
                best = cand
    else:
        raise ValueError("method must be 'coord' or 'grid'")

    T_refined = _delta_transform(best[:3], best[3:], center) @ np.asarray(T_init, dtype=np.float64)
    return {"T": T_refined, "chamfer_init": chamfer_init, "chamfer_final": best_cost}
