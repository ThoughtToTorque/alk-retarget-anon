"""runner_hooks registry tests: uniform iteration surface for the Phase-4
eval runner."""
import numpy as np
import pytest

from baselines import common, runner_hooks as hooks
from tests.baseline_helpers import fresh_pair

EXPECTED = {
    "icp", "icp_centroid",
    "moka_oracle", "moka_oracle_px0", "moka_oracle_px5",
    "moka_oracle_px10", "moka_oracle_px20", "moka_vlm",
    "rekep",
    "kp_alk", "kp_fps4", "kp_random4", "kp_pca2", "kp_dense_init",
}


def test_registry_contents():
    names = hooks.list_methods()
    assert set(names) == EXPECTED
    assert len(names) == len(set(names))
    for n in names:
        assert callable(hooks.get_method(n))


def test_unknown_method_raises():
    with pytest.raises(KeyError):
        hooks.get_method("nope")


def _check_normalized(res, name):
    assert res["method"] == name
    T = np.asarray(res["T_map"])
    assert T.shape == (4, 4)
    assert common.is_valid_se3(T)
    assert isinstance(res["T_map"], list)  # JSON-ready
    assert res["time_s"] >= 0


@pytest.mark.parametrize("name", ["icp_centroid", "moka_oracle_px0",
                                  "rekep", "kp_alk", "kp_pca2"])
def test_run_method_uniform_result(name):
    demo_cap, target_cap, ctx, _ = fresh_pair("cap_twist", 0,
                                              registration=False)
    res = hooks.run_method(name, demo_cap, target_cap, ctx)
    _check_normalized(res, name)


def test_kp_alk_matches_gt_map_on_easy_pair():
    """End-to-end sanity on the snapshot pair: the ALK reference method's
    T_map should be close to GT (phase-2b measured ~8 deg / ~9 mm for
    cap_twist noreg)."""
    demo_cap, target_cap, ctx, pair = fresh_pair("cap_twist", 0,
                                                 registration=False)
    res = hooks.run_method("kp_alk", demo_cap, target_cap, ctx)
    inst = pair["target_instance"]
    rot, trans = hooks.map_errors(
        np.asarray(res["T_map"]), ctx.T_gt,
        pair["demo_object_poses"][inst]["pos"],
        pair["target_object_poses"][inst]["pos"])
    assert rot < 20.0
    assert trans < 0.03


def test_conditioning_and_chamfer_exposed_for_theory_analysis():
    demo_cap, target_cap, ctx, _ = fresh_pair("nut_loosen", 0,
                                              registration=False)
    for name in ("kp_alk", "kp_fps4", "kp_random4", "kp_pca2",
                 "kp_dense_init"):
        res = hooks.run_method(name, demo_cap, target_cap, ctx)
        assert "sigma23" in res["conditioning"]
        assert "chamfer_after_m" in res
