"""B1 -- Mask+ICP baseline (docs/CODE_MAP.md B1).

Point-to-point ICP between the demo and target WORLD-frame object clouds
(the same perception output all methods share): scipy cKDTree nearest
neighbors + Kabsch re-fit per iteration, max 50 iterations, convergence
threshold 1e-6 on the mean matched distance.  Two init variants, both
reported per SPEC:

  * ``init="identity"``  -- T_init = I (the paper's geometry baseline)
  * ``init="centroid"``  -- T_init = pure translation aligning centroids

Expected behavior (reproduced in tests/test_baseline_icp.py): converges on
asymmetric objects, but on near-symmetric objects (nut, can) the NN
correspondences lock onto the wrong symmetry branch and ICP converges to a
biased pose -- the wrong-symmetry-branch failure the paper measures at scale (Sections 5.2 and 5.9).
"""
import numpy as np
from scipy.spatial import cKDTree

from alkbench import chamfer_distance, procrustes, transform_points
from alkbench.registration import _subsample

from baselines import common


def icp_point_to_point(src, tgt, T_init=None, max_iter=50, tol=1e-6,
                       max_points=2000, seed=0):
    """Classic point-to-point ICP.

    Parameters
    ----------
    src, tgt : (N,3)/(M,3) clouds (same world frame)
    T_init : (4,4) or None (identity)
    max_iter, tol : iteration cap / convergence threshold on the change of
        the mean matched NN distance
    max_points : per-cloud subsampling cap (seeded, deterministic)

    Returns
    -------
    dict: T (4,4), n_iter, converged, rmse (final mean matched distance),
          rmse_history
    """
    src_s = _subsample(src, max_points, seed)
    tgt_s = _subsample(tgt, max_points, seed)
    T = np.eye(4) if T_init is None else np.asarray(T_init, dtype=np.float64)
    tree = cKDTree(tgt_s)
    prev = np.inf
    history = []
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        moved = transform_points(T, src_s)
        d, j = tree.query(moved)
        # re-fit FROM THE ORIGINAL source each iteration (numerically stable,
        # avoids compounding rotations)
        T = procrustes(src_s, tgt_s[j])
        err = float(d.mean())
        history.append(err)
        if abs(prev - err) < tol:
            converged = True
            break
        prev = err
    return {"T": T, "n_iter": n_iter, "converged": converged,
            "rmse": history[-1] if history else float("nan"),
            "rmse_history": history}


def centroid_init(src, tgt):
    T = np.eye(4)
    T[:3, 3] = (np.asarray(tgt, dtype=np.float64).mean(axis=0)
                - np.asarray(src, dtype=np.float64).mean(axis=0))
    return T


def run_icp(demo_capture, target_capture, ctx, init="identity"):
    """Registry entry point.  Returns the uniform result dict with T_map =
    the ICP pose, plus chamfer + ICP diagnostics."""
    p_d = ctx.percept(demo_capture, "demo")
    p_t = ctx.percept(target_capture, "target")
    src = p_d["cands"].points3d
    tgt = p_t["cands"].points3d
    if init == "identity":
        T0 = None
    elif init == "centroid":
        T0 = centroid_init(src, tgt)
    else:
        raise ValueError("init must be 'identity' or 'centroid'")
    res = icp_point_to_point(src, tgt, T_init=T0, seed=ctx.kmeans_seed)
    src_s = _subsample(src, 2000, ctx.kmeans_seed)
    tgt_s = _subsample(tgt, 2000, ctx.kmeans_seed)
    ch = float(chamfer_distance(transform_points(res["T"], src_s), tgt_s))
    return common.base_result(
        "icp_%s" % init, res["T"],
        chamfer_after_m=ch,
        icp_rmse_m=res["rmse"],
        icp_n_iter=res["n_iter"],
        icp_converged=res["converged"],
        init=init,
    )
