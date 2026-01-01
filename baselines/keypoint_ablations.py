"""B4 -- keypoint-form ablations (docs/CODE_MAP.md B4).

All variants share the EXACT pipeline stack -- closed-form alignment of a
small keypoint configuration, then the same bounded Chamfer registration on
the same world-frame object clouds -- and differ ONLY in how the keypoint
configuration is constructed:

  * ``alk``        -- the paper's Axial-Lateral Keypoints (reference; oracle
                      discrete answers, identical to the pipeline).
  * ``fps4``       -- 4 farthest-point-sampled candidates from the k-means
                      pool; correspondence given by the oracle GT
                      nearest-neighbor match (its BEST-case condition).
  * ``random4``    -- 4 random distinct candidates (seeded via
                      ctx.random4_seed; Phase-4 averages over several seeds);
                      oracle GT-NN correspondence.
  * ``pca2``       -- only the two PCA-axis endpoint candidates.  A 2-point
                      configuration is rank-1, so rotation about the axis is
                      UNDERDETERMINED.  Convention: the closed-form init is
                      the SHORTEST-ARC rotation mapping the demo axis onto
                      the target axis (zero roll about the axis) plus the
                      midpoint translation -- i.e. the unobservable lateral
                      DoF is frozen at 0, which is exactly the missing-DoF
                      failure the ablation is meant to expose.
  * ``dense_init`` -- "Dense-ICP-init": the closed-form Procrustes on
                      keypoints is REPLACED by centroid + first-PCA-axis
                      alignment of the dense clouds (axis sign disambiguated
                      by whichever of the two gives the lower Chamfer), then
                      the same bounded registration.  No keypoint
                      correspondences at all.

Every result records the conditioning of the CENTERED demo keypoint
configuration -- singular values (s1 >= s2 >= s3) and the theory-linked
number s2 + s3 (Proposition 3: the Procrustes rotation-error bound scales
like 1/(s2+s3)).  For ``pca2`` the number is exactly 0 (rank-1
configuration); for ``dense_init`` the conditioning is reported for its
EFFECTIVE configuration (centroid +- axis), which is likewise rank-1 -> 0:
the init constrains no rotation about the PCA axis and only the bounded
registration can recover it.
"""
import numpy as np

from alkbench import (alk_from_candidates, bounded_registration,
                      chamfer_distance, procrustes, transform_points)
from alkbench.registration import _subsample
from pipeline import oracle  # pure numpy/scipy (no robosuite)

from baselines import common


# ---------------------------------------------------------------------------
# closed-form helpers
# ---------------------------------------------------------------------------

def two_point_alignment(P, Q):
    """Closed-form SE(3) from 2 matched points: shortest-arc rotation of the
    segment direction (zero roll about the axis -- documented convention) +
    midpoint translation."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    R = common.shortest_arc_rotation(P[1] - P[0], Q[1] - Q[0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = Q.mean(axis=0) - R @ P.mean(axis=0)
    return T


def _first_pca_axis(points):
    X = np.asarray(points, dtype=np.float64)
    c = X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X - c, full_matrices=False)
    return c, Vt[0], s[0]


def dense_pca_centroid_init(src, tgt, max_points=2000, seed=0):
    """Centroid + first-PCA-axis alignment of two dense clouds; the target
    axis sign is chosen by the lower resulting Chamfer distance (a real
    dense pipeline has no oracle for the 180-degree flip)."""
    src_s = _subsample(src, max_points, seed)
    tgt_s = _subsample(tgt, max_points, seed)
    c_s, a_s, s1_s = _first_pca_axis(src_s)
    c_t, a_t, _ = _first_pca_axis(tgt_s)
    best = None
    for sign in (1.0, -1.0):
        R = common.shortest_arc_rotation(a_s, sign * a_t)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = c_t - R @ c_s
        ch = chamfer_distance(transform_points(T, src_s), tgt_s)
        if best is None or ch < best[0]:
            best = (ch, T, sign)
    ch, T, sign = best
    # effective (rank-1) configuration for the conditioning report
    scale = s1_s / np.sqrt(max(src_s.shape[0], 1))
    eff = np.stack([c_s - a_s * scale, c_s + a_s * scale])
    return T, {"axis_sign": sign, "chamfer_init_m": float(ch)}, eff


# ---------------------------------------------------------------------------
# constructions: each returns dict(P, Q, T0, cond_points, info)
#   P/Q      : matched demo/target keypoints (None for dense_init)
#   T0       : closed-form init (None -> procrustes(P, Q))
#   cond_points : demo configuration whose conditioning is reported
# ---------------------------------------------------------------------------

def construct_alk(p_d, p_t, ctx):
    T_gt = ctx.require_T_gt("kp_alk")
    d_cands, t_cands = p_d["cands"], p_t["cands"]
    d1, d2 = oracle.demo_axial_choice(d_cands)
    demo_alk = alk_from_candidates(d_cands, d1, d2, phi3=False)
    t1, t2, _, _ = oracle.target_axial_choice(
        t_cands, d_cands.candidates3d[d1], d_cands.candidates3d[d2], T_gt)
    target_alk = alk_from_candidates(t_cands, t1, t2, phi3=False)
    phi3, _, _ = oracle.lateral_swap_choice(target_alk, demo_alk, T_gt)
    if phi3:
        target_alk = target_alk.copy()
        target_alk[[2, 3]] = target_alk[[3, 2]]
    info = {"demo_phi": [d1, d2], "target_phi": [t1, t2],
            "phi3": int(phi3)}
    return {"P": demo_alk, "Q": target_alk, "T0": None,
            "cond_points": demo_alk, "info": info}


def _matched_candidates(p_d, p_t, ctx, sel, name):
    T_gt = ctx.require_T_gt(name)
    P = p_d["cands"].candidates3d[sel]
    match, dists = common.gt_nn_match(P, p_t["cands"].candidates3d, T_gt)
    Q = p_t["cands"].candidates3d[match]
    info = {"demo_indices": [int(i) for i in sel],
            "target_indices": [int(j) for j in match],
            "match_dist_m": [float(d) for d in dists]}
    return {"P": P, "Q": Q, "T0": None, "cond_points": P, "info": info}


def construct_fps4(p_d, p_t, ctx):
    n = min(4, p_d["cands"].k, p_t["cands"].k)
    sel = common.farthest_point_indices(p_d["cands"].candidates3d, n)
    return _matched_candidates(p_d, p_t, ctx, sel, "kp_fps4")


def construct_random4(p_d, p_t, ctx):
    n = min(4, p_d["cands"].k, p_t["cands"].k)
    rng = np.random.RandomState(ctx.random4_seed)
    sel = rng.choice(p_d["cands"].k, size=n, replace=False)
    out = _matched_candidates(p_d, p_t, ctx, [int(i) for i in sel],
                              "kp_random4")
    out["info"]["random_seed"] = ctx.random4_seed
    return out


def construct_pca2(p_d, p_t, ctx):
    d1, d2 = oracle.demo_axial_choice(p_d["cands"])
    out = _matched_candidates(p_d, p_t, ctx, [d1, d2], "kp_pca2")
    out["T0"] = two_point_alignment(out["P"], out["Q"])
    return out


def construct_dense_init(p_d, p_t, ctx):
    T0, info, eff = dense_pca_centroid_init(
        p_d["cands"].points3d, p_t["cands"].points3d, seed=ctx.kmeans_seed)
    return {"P": None, "Q": None, "T0": T0, "cond_points": eff,
            "info": info}


CONSTRUCTIONS = {
    "alk": construct_alk,
    "fps4": construct_fps4,
    "random4": construct_random4,
    "pca2": construct_pca2,
    "dense_init": construct_dense_init,
}


# ---------------------------------------------------------------------------
# shared stack
# ---------------------------------------------------------------------------

def run_ablation(demo_capture, target_capture, ctx, construction):
    """Registry entry point: keypoint construction -> closed-form init ->
    (optional, on by default) bounded Chamfer registration.

    Returns the uniform result dict; ``conditioning`` carries
    {singular_values, sigma23} of the centered demo configuration."""
    if construction not in CONSTRUCTIONS:
        raise KeyError("unknown construction %r (have %s)"
                       % (construction, sorted(CONSTRUCTIONS)))
    p_d = ctx.percept(demo_capture, "demo")
    p_t = ctx.percept(target_capture, "target")
    built = CONSTRUCTIONS[construction](p_d, p_t, ctx)
    T0 = built["T0"]
    if T0 is None:
        T0 = procrustes(built["P"], built["Q"])
    src = p_d["cands"].points3d
    tgt = p_t["cands"].points3d
    if ctx.registration:
        reg = bounded_registration(src, tgt, T0, seed=ctx.kmeans_seed)
        T_map = reg["T"]
        ch_before = float(reg["chamfer_init"])
        ch_after = float(reg["chamfer_final"])
    else:
        T_map = T0
        src_s = _subsample(src, 2000, ctx.kmeans_seed)
        tgt_s = _subsample(tgt, 2000, ctx.kmeans_seed)
        ch_before = ch_after = float(chamfer_distance(
            transform_points(T0, src_s), tgt_s))
    return common.base_result(
        "kp_%s" % construction, T_map,
        T_init=np.asarray(T0).tolist(),
        chamfer_before_m=ch_before,
        chamfer_after_m=ch_after,
        conditioning=common.conditioning(built["cond_points"]),
        registration=bool(ctx.registration),
        construction=construction,
        info=built["info"],
    )
