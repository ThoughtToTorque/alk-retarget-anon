"""B3 ReKep-style tests: the SLSQP SE(3) solver recovers an exact synthetic
transform, and the full method returns a sane map on real captures."""
import numpy as np

from alkbench import transform_points

from baselines import common, rekep_style
from tests import synth
from tests.baseline_helpers import fresh_pair, rot_err_deg


def test_optimizer_recovers_exact_transform():
    rng = np.random.RandomState(3)
    P = rng.uniform(-0.08, 0.08, size=(5, 3)) + np.array([0.1, -0.2, 0.8])
    tcp = P.mean(axis=0) + np.array([0.0, 0.0, 0.05])
    T_true = synth.random_se3(seed=7, max_angle_deg=40.0, max_trans=0.08)
    Q = transform_points(T_true, P)

    sol = rekep_style.solve_se3_keypoint_constraints(P, Q, tcp)
    assert sol["opt"]["success"]
    assert sol["cost_final"] < 1e-9
    assert rot_err_deg(sol["T"], T_true) < 0.1
    assert np.linalg.norm(sol["T"][:3, 3] - T_true[:3, 3]) < 1e-3


def test_optimizer_reduces_cost_under_noise():
    rng = np.random.RandomState(11)
    P = rng.uniform(-0.06, 0.06, size=(4, 3)) + np.array([0.0, 0.1, 0.5])
    tcp = P.mean(axis=0)
    T_true = synth.random_se3(seed=2, max_angle_deg=30.0, max_trans=0.05)
    Q = transform_points(T_true, P) + rng.normal(scale=0.003, size=(4, 3))
    sol = rekep_style.solve_se3_keypoint_constraints(P, Q, tcp)
    assert sol["cost_final"] < sol["cost_init"]
    assert rot_err_deg(sol["T"], T_true) < 15.0


def test_run_rekep_on_testdata():
    demo_cap, target_cap, ctx, pair = fresh_pair("cap_twist", 0)
    res = rekep_style.run_rekep(demo_cap, target_cap, ctx)
    T = np.asarray(res["T_map"])
    assert common.is_valid_se3(T)
    assert res["cost_final"] <= res["cost_init"] + 1e-12
    assert res["n_keypoints"] == 4
    assert res["conditioning"]["sigma23"] > 0
    # oracle-matched keypoints from the same pool: the fit should be in the
    # right ballpark of the GT map (loose bounds; execution-grade accuracy
    # is Phase 4's business)
    inst = pair["target_instance"]
    rot = rot_err_deg(T, ctx.T_gt)
    mapped = transform_points(
        T, np.asarray(pair["demo_object_poses"][inst]["pos"])[None])[0]
    trans = np.linalg.norm(
        mapped - np.asarray(pair["target_object_poses"][inst]["pos"]))
    assert rot < 30.0
    assert trans < 0.05
