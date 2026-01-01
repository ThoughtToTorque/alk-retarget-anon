"""Oracle answers to the DISCRETE questions of the pipeline (phi1..phi5).

The oracle stands in for the VLM and answers ONLY the discrete index
questions, using the ground-truth relative object transform
T_gt = pose_target o pose_demo^-1 (from the saved GT poses in pair.json).
Everything continuous (candidate 3D positions, ALK centroids, Procrustes,
bounded registration, grasp-region centroids) still runs on the noisy
rendered geometry, so the pipeline remains meaningful: a real VLM can later
be scored against the recorded oracle answers, and the geometric stages are
identical either way.

Conventions
-----------
* Demo axial endpoints (phi1, phi2): the two candidates spanning the object's
  dominant extent (first PCA axis of the demo candidates).  Fixed convention
  for which one is "endpoint 1": the PCA axis sign is chosen so that its dot
  product with AXIS_REF (mostly world +x, small +y/+z tie-breakers) is
  positive; endpoint 1 = candidate at the + end.
* Demo lateral split: phi3 = False by definition (demo is the reference).
* Target endpoints: candidate nearest to T_gt applied to the corresponding
  demo endpoint centroid (phi2 excludes phi1's index).
* Target phi3: the lateral assignment (identity vs swap) whose centroids best
  match the T_gt-mapped demo lateral centroids.
* Grasp region (phi4 coarse, k=5; phi5 fine, k=5 within the coarse region):
  region whose 3D centroid is nearest to the T_gt-mapped demo grasp point.
  The demo grasp point is the demo TCP position at the pre_grasp keyframe,
  projected onto the nearest demo object point.
  NOTE: only the REGION CHOICE is discrete/oracle.  The continuous target
  grasp point used by the runner's translation correction is the projection
  of the T_map-mapped demo grasp point onto the chosen fine region's points
  (T_map is the pipeline's own estimate, so no GT leaks).  The fine-region
  centroid is recorded for reference; using it directly as the grasp point
  injects the region quantization (up to several cm on large instances such
  as the door) straight into the correction.
"""
import numpy as np

from alkbench import (alk_from_candidates, transform_points,
                      lateral_extent_ratio, SLENDER_RATIO_THRESHOLD)
from alkbench.candidates import kmeans
from scipy.spatial.transform import Rotation

# deterministic sign convention for the demo PCA axis (dominant +x with tiny
# +y/+z tie-breakers for near-orthogonal axes)
AXIS_REF = np.array([1.0, 1e-2, 1e-4])


def pose_matrix(pose):
    """{"pos": [...], "quat_xyzw": [...]} -> 4x4 world pose."""
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat(pose["quat_xyzw"]).as_matrix()
    M[:3, 3] = np.asarray(pose["pos"], dtype=np.float64)
    return M


def gt_relative_transform(demo_pose, target_pose):
    """T_gt = pose_t o pose_d^-1 (both {"pos","quat_xyzw"} dicts)."""
    return pose_matrix(target_pose) @ np.linalg.inv(pose_matrix(demo_pose))


def demo_axial_choice(cands):
    """(phi1, phi2) on the demo side: extremal candidates along the first PCA
    axis of the demo candidates, sign fixed by AXIS_REF."""
    C = np.asarray(cands.candidates3d, dtype=np.float64)
    X = C - C.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    u = Vt[0]
    if float(u @ AXIS_REF) < 0:
        u = -u
    proj = X @ u
    phi1 = int(np.argmax(proj))
    phi2 = int(np.argmin(proj))
    if phi1 == phi2:  # fully degenerate cloud; pick any distinct pair
        phi2 = (phi1 + 1) % C.shape[0]
    return phi1, phi2


def target_axial_choice(target_cands, demo_c1, demo_c2, T_gt):
    """Target (phi1, phi2): candidates nearest to the T_gt-mapped demo
    endpoints. Returns (phi1, phi2, dist1, dist2)."""
    C = np.asarray(target_cands.candidates3d, dtype=np.float64)
    m1 = transform_points(T_gt, np.asarray(demo_c1)[None])[0]
    m2 = transform_points(T_gt, np.asarray(demo_c2)[None])[0]
    d1 = np.linalg.norm(C - m1, axis=1)
    phi1 = int(np.argmin(d1))
    d2 = np.linalg.norm(C - m2, axis=1)
    d2m = d2.copy()
    d2m[phi1] = np.inf
    phi2 = int(np.argmin(d2m))
    return phi1, phi2, float(d1[phi1]), float(d2[phi2])


def lateral_swap_choice(target_alk_noswap, demo_alk, T_gt):
    """phi3 on the target side: swap iff the crossed assignment of the target
    lateral centroids to the T_gt-mapped demo lateral centroids is closer."""
    m = transform_points(T_gt, np.asarray(demo_alk)[2:4])
    c3, c4 = np.asarray(target_alk_noswap)[2], np.asarray(target_alk_noswap)[3]
    cost_id = np.linalg.norm(c3 - m[0]) + np.linalg.norm(c4 - m[1])
    cost_sw = np.linalg.norm(c3 - m[1]) + np.linalg.norm(c4 - m[0])
    return bool(cost_sw < cost_id), float(cost_id), float(cost_sw)


def demo_grasp_point(demo_cands, pre_grasp_tcp_pos):
    """Demo grasp point: demo TCP position at the pre_grasp keyframe,
    projected onto the nearest demo object point (world frame)."""
    P = np.asarray(demo_cands.points3d, dtype=np.float64)
    tcp = np.asarray(pre_grasp_tcp_pos, dtype=np.float64)
    i = int(np.argmin(np.linalg.norm(P - tcp, axis=1)))
    return P[i], float(np.linalg.norm(P[i] - tcp))


def grasp_region_choice(target_cands, mapped_grasp, k_coarse=5, k_fine=5,
                        seed=0):
    """phi4 (coarse, k=5 regions over the target object pixels) and phi5
    (fine, k=5 sub-regions within the chosen coarse region): region whose 3D
    centroid is nearest to `mapped_grasp` (= T_gt-mapped demo grasp point).

    Returns (phi4, phi5, target_grasp_point(3,), info dict).
    """
    pix = np.asarray(target_cands.pixels, dtype=np.float64)
    pts = np.asarray(target_cands.points3d, dtype=np.float64)
    mapped = np.asarray(mapped_grasp, dtype=np.float64)

    k_c = min(k_coarse, pts.shape[0])
    _, labels = kmeans(pix, k=k_c, seed=seed)
    coarse_cent = np.stack([pts[labels == j].mean(axis=0) for j in range(k_c)])
    phi4 = int(np.argmin(np.linalg.norm(coarse_cent - mapped, axis=1)))

    sub_pts = pts[labels == phi4]
    sub_pix = pix[labels == phi4]
    k_f = min(k_fine, sub_pts.shape[0])
    if k_f >= 2:
        _, sub_labels = kmeans(sub_pix, k=k_f, seed=seed)
        fine_cent = np.stack([sub_pts[sub_labels == j].mean(axis=0)
                              for j in range(k_f)])
        phi5 = int(np.argmin(np.linalg.norm(fine_cent - mapped, axis=1)))
        g_t = fine_cent[phi5]
        region_pts = sub_pts[sub_labels == phi5]
    else:
        phi5 = 0
        g_t = coarse_cent[phi4]
        region_pts = sub_pts if sub_pts.shape[0] else pts
    info = {
        "k_coarse": int(k_c),
        "k_fine": int(k_f),
        "coarse_err_m": float(np.linalg.norm(coarse_cent[phi4] - mapped)),
        "fine_err_m": float(np.linalg.norm(g_t - mapped)),
        "region_n_points": int(region_pts.shape[0]),
    }
    return phi4, phi5, g_t, region_pts, info


def solve(demo_percept, target_percept, T_gt, demo_pre_grasp_tcp,
          seed=0, depth_consistent=False):
    """Full oracle pass: discrete answers + the ALKs / grasp points the
    continuous stages need.

    Parameters
    ----------
    demo_percept, target_percept : dicts from perception.perceive
    T_gt : (4,4) ground-truth relative object transform demo -> target
    demo_pre_grasp_tcp : (3,) demo TCP position at the pre_grasp keyframe
    depth_consistent : bool, or "auto" -- enable the slender depth prior iff
        the DEMO cloud's lateral_extent_ratio about the chosen axial
        endpoints is below SLENDER_RATIO_THRESHOLD (applied to BOTH ALKs so
        the correspondence stays consistent; the measured ratio is exposed
        in the returned "slender" dict)

    Returns
    -------
    dict with keys:
      demo_choice / target_choice : {"phi1","phi2","phi3","phi4","phi5"}
      demo_alk / target_alk : (4,3) arrays
      demo_grasp / target_grasp_centroid : (3,) points
      target_grasp_region_points : (m,3) points of the chosen fine region
      diagnostics : per-question oracle distances (for later VLM scoring)
    """
    d_cands = demo_percept["cands"]
    t_cands = target_percept["cands"]

    d1, d2 = demo_axial_choice(d_cands)
    slender = None
    dc = depth_consistent
    if isinstance(dc, str):
        if dc != "auto":
            raise ValueError("depth_consistent must be bool or 'auto'")
        slender = lateral_extent_ratio(d_cands.points3d,
                                       d_cands.candidates3d[d1],
                                       d_cands.candidates3d[d2])
        slender["threshold"] = float(SLENDER_RATIO_THRESHOLD)
        dc = bool(slender["ratio"] < SLENDER_RATIO_THRESHOLD)
        slender["enabled"] = dc
    demo_alk = alk_from_candidates(d_cands, d1, d2, phi3=False,
                                   depth_consistent=dc)

    t1, t2, dist1, dist2 = target_axial_choice(
        t_cands, d_cands.candidates3d[d1], d_cands.candidates3d[d2], T_gt)
    target_alk0 = alk_from_candidates(t_cands, t1, t2, phi3=False,
                                      depth_consistent=dc)
    phi3, cost_id, cost_sw = lateral_swap_choice(target_alk0, demo_alk, T_gt)
    target_alk = target_alk0.copy()
    if phi3:
        target_alk[[2, 3]] = target_alk[[3, 2]]

    g_d, g_d_proj_dist = demo_grasp_point(d_cands, demo_pre_grasp_tcp)
    mapped_grasp = transform_points(T_gt, g_d[None])[0]
    phi4, phi5, g_t, region_pts, grasp_info = grasp_region_choice(
        t_cands, mapped_grasp, seed=seed)

    return {
        "demo_choice": {"phi1": d1, "phi2": d2, "phi3": 0,
                        "phi4": None, "phi5": None},
        "target_choice": {"phi1": t1, "phi2": t2, "phi3": int(phi3),
                          "phi4": phi4, "phi5": phi5},
        "demo_alk": demo_alk,
        "target_alk": target_alk,
        "demo_grasp": g_d,
        "target_grasp_centroid": g_t,
        "target_grasp_region_points": region_pts,
        "depth_consistent": bool(dc),
        "slender": slender,
        "diagnostics": {
            "target_endpoint_dist_m": [dist1, dist2],
            "phi3_cost_identity": cost_id,
            "phi3_cost_swap": cost_sw,
            "demo_grasp_projection_dist_m": g_d_proj_dist,
            **grasp_info,
        },
    }
