"""B4 keypoint-form ablation tests: every construction yields a valid SE(3)
plus the theory-linked conditioning number; rank-1 constructions (pca2,
dense_init) report sigma2+sigma3 = 0; the shared bounded registration does
not increase Chamfer."""
import numpy as np
import pytest

from baselines import common
from baselines import keypoint_ablations as ka
from tests.baseline_helpers import fresh_pair

ALL = sorted(ka.CONSTRUCTIONS.keys())


@pytest.mark.parametrize("construction", ALL)
def test_construction_valid_se3_and_conditioning(construction):
    demo_cap, target_cap, ctx, _ = fresh_pair("cap_twist", 0,
                                              registration=False)
    res = ka.run_ablation(demo_cap, target_cap, ctx, construction)
    T = np.asarray(res["T_map"])
    assert common.is_valid_se3(T)
    cond = res["conditioning"]
    s = cond["singular_values"]
    assert len(s) == 3 and s[0] >= s[1] >= s[2] >= 0
    assert cond["sigma23"] >= 0
    assert np.isclose(cond["sigma23"], s[1] + s[2])
    assert res["chamfer_after_m"] < 0.10


def test_rank1_constructions_have_zero_conditioning():
    demo_cap, target_cap, ctx, _ = fresh_pair("nut_loosen", 0,
                                              registration=False)
    for construction in ("pca2", "dense_init"):
        res = ka.run_ablation(demo_cap, target_cap, ctx, construction)
        assert res["conditioning"]["sigma23"] < 1e-9, construction
    for construction in ("alk", "fps4", "random4"):
        res = ka.run_ablation(demo_cap, target_cap, ctx, construction)
        assert res["conditioning"]["sigma23"] > 1e-4, construction


def test_bounded_registration_does_not_increase_chamfer():
    demo_cap, target_cap, ctx, _ = fresh_pair("cap_twist", 0,
                                              registration=True)
    res = ka.run_ablation(demo_cap, target_cap, ctx, "alk")
    assert res["chamfer_after_m"] <= res["chamfer_before_m"] + 1e-12


def test_random4_is_seeded():
    a = fresh_pair("cap_twist", 0, registration=False, random4_seed=1)
    b = fresh_pair("cap_twist", 0, registration=False, random4_seed=1)
    c = fresh_pair("cap_twist", 0, registration=False, random4_seed=2)
    res_a = ka.run_ablation(a[0], a[1], a[2], "random4")
    res_b = ka.run_ablation(b[0], b[1], b[2], "random4")
    res_c = ka.run_ablation(c[0], c[1], c[2], "random4")
    assert res_a["info"]["demo_indices"] == res_b["info"]["demo_indices"]
    assert np.allclose(res_a["T_map"], res_b["T_map"])
    assert res_a["info"]["demo_indices"] != res_c["info"]["demo_indices"]


def test_two_point_alignment_convention():
    """pca2 convention: shortest-arc rotation (zero roll about the axis) +
    midpoint translation."""
    P = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    Q = np.array([[0.1, 0.1, 0.0], [0.1, 0.3, 0.0]])  # x-axis -> y-axis
    T = ka.two_point_alignment(P, Q)
    assert common.is_valid_se3(T)
    from alkbench import transform_points
    assert np.allclose(transform_points(T, P), Q, atol=1e-12)
    # zero roll: the minimal rotation for x->y is exactly 90 deg about z
    from scipy.spatial.transform import Rotation
    rv = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    assert np.allclose(rv, [0, 0, np.pi / 2], atol=1e-12)
