"""B3 -- ReKep-style baseline: keypoint-relative constraint optimization
(docs/CODE_MAP.md B3).

Pipeline: K candidate keypoints are selected from OUR k-means candidate pool
(fair -- all methods see the same perception); the demonstration records the
keypoint-TCP relative constraints (world-frame offset vectors from the demo
pre-grasp TCP to every selected keypoint); the target pose is obtained by
minimizing the violation of those relative constraints with
scipy.optimize.minimize(method="SLSQP") over SE(3) (rotation-vector + t
parameterization).  NO bounded Chamfer registration is applied (faithful to
ReKep, which never refines against dense geometry).

Honest simplifications w.r.t. the actual ReKep system (Huang et al., 2024)
-- to be stated verbatim in the paper's adaptation notes:

1. **A single rigid SE(3) T_map is optimized instead of per-stage
   end-effector poses.**  ReKep solves each subgoal pose independently; here
   all methods must emit one T_map that the SHARED executor applies to the
   demo waypoints, which is what makes the comparison fair (identical motion
   primitives, gripper schedule and success checker).  For our single-object,
   single-grasp tasks the two coincide: every demo waypoint moves rigidly
   with the object.
2. **Constraints are quadratic "preserve the demo keypoint-TCP relative
   vectors" costs recorded from the single demo, not VLM-generated Python
   constraint functions.**  In the one-shot setting there is no constraint
   author but the demo itself; this is the standard one-shot adaptation.
3. **Keypoint selection/correspondence is oracle (GT nearest-neighbor,
   injective) or spread-based (FPS), not a VLM + point tracker.**  This
   gives ReKep-style its BEST-case correspondences; reported results are
   therefore an upper bound on the adapted method.
4. **SLSQP is a local optimizer started from a centroid-aligned,
   zero-rotation init; no global search.**  ReKep likewise solves local
   nonlinear programs from heuristic inits (with a sampling outer loop we
   drop since the cost here is smooth).
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from alkbench import chamfer_distance, transform_points
from alkbench.registration import _subsample

from baselines import common


def record_demo_constraints(keypoints, tcp):
    """Demo-side constraint record: world-frame offsets keypoint - TCP."""
    P = np.asarray(keypoints, dtype=np.float64)
    return P - np.asarray(tcp, dtype=np.float64)[None]


def solve_se3_keypoint_constraints(demo_keypoints, target_keypoints,
                                   demo_tcp, weights=None, x0=None):
    """Minimize sum_i w_i || (q_i - T(tcp_d)) - R v_i ||^2 over T in SE(3),
    where v_i = p_i - tcp_d are the demo keypoint-TCP relative vectors.

    (Algebraically this equals sum_i w_i ||T(p_i) - q_i||^2; it is written in
    the keypoint-relative form to mirror ReKep's constraint formulation.)

    Returns dict: T (4,4), cost_init, cost_final, opt (scipy result summary).
    """
    P = np.asarray(demo_keypoints, dtype=np.float64)
    Q = np.asarray(target_keypoints, dtype=np.float64)
    tcp = np.asarray(demo_tcp, dtype=np.float64)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("keypoint arrays must be matching (n, 3)")
    w = np.ones(P.shape[0]) if weights is None else np.asarray(
        weights, dtype=np.float64)
    V = record_demo_constraints(P, tcp)

    def cost(x):
        R = Rotation.from_rotvec(x[:3]).as_matrix()
        tcp_t = R @ tcp + x[3:]
        resid = (Q - tcp_t[None]) - V @ R.T
        return float((w[:, None] * resid ** 2).sum())

    if x0 is None:
        x0 = np.zeros(6)
        x0[3:] = Q.mean(axis=0) - P.mean(axis=0)  # centroid-aligned init
    c0 = cost(x0)
    res = minimize(cost, x0, method="SLSQP",
                   options={"maxiter": 200, "ftol": 1e-12})
    T = common.se3_from_rotvec(res.x[:3], res.x[3:])
    return {
        "T": T,
        "cost_init": c0,
        "cost_final": float(res.fun),
        "opt": {"success": bool(res.success), "n_iter": int(res.nit),
                "message": str(res.message)},
    }


def select_keypoints(demo_percept, target_percept, T_gt, n_keypoints=4):
    """Select K demo keypoints from the k-means candidate pool (FPS for
    spread -- the stand-in for 'semantically distinctive' VLM picks) and
    match them to target candidates with the oracle GT-NN correspondence.

    Returns (demo_kps (K,3), target_kps (K,3), info dict)."""
    d_cands = demo_percept["cands"].candidates3d
    t_cands = target_percept["cands"].candidates3d
    n = min(int(n_keypoints), d_cands.shape[0], t_cands.shape[0])
    sel = common.farthest_point_indices(d_cands, n)
    P = d_cands[sel]
    match, dists = common.gt_nn_match(P, t_cands, T_gt)
    Q = t_cands[match]
    info = {"demo_indices": [int(i) for i in sel],
            "target_indices": [int(j) for j in match],
            "match_dist_m": [float(d) for d in dists]}
    return P, Q, info


def run_rekep(demo_capture, target_capture, ctx):
    """Registry entry point."""
    T_gt = ctx.require_T_gt("rekep (oracle keypoint correspondence)")
    p_d = ctx.percept(demo_capture, "demo")
    p_t = ctx.percept(target_capture, "target")
    P, Q, sel_info = select_keypoints(p_d, p_t, T_gt,
                                      n_keypoints=ctx.rekep_n_keypoints)
    tcp = ctx.keyframe_tcp("pre_grasp")
    sol = solve_se3_keypoint_constraints(P, Q, tcp)
    src_s = _subsample(p_d["cands"].points3d, 2000, ctx.kmeans_seed)
    tgt_s = _subsample(p_t["cands"].points3d, 2000, ctx.kmeans_seed)
    ch = float(chamfer_distance(transform_points(sol["T"], src_s), tgt_s))
    return common.base_result(
        "rekep", sol["T"],
        chamfer_after_m=ch,
        cost_init=sol["cost_init"],
        cost_final=sol["cost_final"],
        opt=sol["opt"],
        n_keypoints=int(P.shape[0]),
        keypoints=sel_info,
        conditioning=common.conditioning(P),
    )
