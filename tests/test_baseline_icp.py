"""B1 Mask+ICP tests: convergence on asymmetric geometry, symmetry bias on
near-symmetric geometry (the paper's Section 5.9 observation in miniature: ICP locks
onto the wrong flip while ALK, whose endpoints are matched by the oracle
discrete answers, recovers the yaw)."""
import numpy as np

from alkbench import procrustes, transform_points

from baselines import common, icp
from baselines.keypoint_ablations import construct_alk
from tests import synth
from tests.baseline_helpers import (fresh_pair, near_symmetric_cloud,
                                    percept_from_points, rot_err_deg,
                                    yaw_about_point)


def test_icp_converges_on_asymmetric_cloud():
    # two independent samples of the same asymmetric shape (bumped cylinder)
    src = synth.make_cloud(n=3000, seed=0)
    tgt_shape = synth.make_cloud(n=3000, seed=1)
    T_true = yaw_about_point(25.0, src.mean(axis=0))
    T_true[:3, 3] += np.array([0.03, -0.02, 0.01])
    tgt = transform_points(T_true, tgt_shape)

    res = icp.icp_point_to_point(src, tgt,
                                 T_init=icp.centroid_init(src, tgt))
    assert res["converged"]
    assert rot_err_deg(res["T"], T_true) < 5.0
    mapped = transform_points(res["T"], src.mean(axis=0)[None])[0]
    gt_mapped = transform_points(T_true, src.mean(axis=0)[None])[0]
    assert np.linalg.norm(mapped - gt_mapped) < 0.01


def test_icp_identity_init_converges_for_small_motion():
    src = synth.make_cloud(n=3000, seed=0)
    T_true = yaw_about_point(10.0, src.mean(axis=0))
    T_true[:3, 3] += np.array([0.01, 0.005, 0.0])
    tgt = transform_points(T_true, synth.make_cloud(n=3000, seed=1))
    res = icp.icp_point_to_point(src, tgt)
    assert rot_err_deg(res["T"], T_true) < 5.0


def test_icp_biased_on_near_symmetric_cloud_vs_alk():
    """170-degree yaw of a nearly flip-symmetric object: ICP (identity init)
    falls into the flipped basin (rotation error near 180 deg), while the
    ALK construction with oracle endpoint matching recovers the yaw."""
    demo_pts = near_symmetric_cloud(seed=0)
    T_true = yaw_about_point(170.0, demo_pts.mean(axis=0))
    target_pts = transform_points(T_true, near_symmetric_cloud(seed=1))

    # perception through the same render->cluster pipeline for both methods
    p_demo = percept_from_points(demo_pts)
    p_target = percept_from_points(target_pts)
    src = p_demo["cands"].points3d
    tgt = p_target["cands"].points3d

    res_icp = icp.icp_point_to_point(src, tgt)
    icp_err = rot_err_deg(res_icp["T"], T_true)

    ctx = common.MethodContext(task="synthetic", T_gt=T_true)
    built = construct_alk(p_demo, p_target, ctx)
    T_alk = procrustes(built["P"], built["Q"])
    alk_err = rot_err_deg(T_alk, T_true)

    assert icp_err > 90.0, "ICP unexpectedly escaped the symmetric basin"
    assert alk_err < 25.0
    assert alk_err < icp_err


def test_run_icp_on_testdata_returns_valid_result():
    demo_cap, target_cap, ctx, pair = fresh_pair("cap_twist", 0)
    res = icp.run_icp(demo_cap, target_cap, ctx, init="centroid")
    assert common.is_valid_se3(np.asarray(res["T_map"]))
    assert res["chamfer_after_m"] < 0.05
    assert res["icp_n_iter"] >= 1
