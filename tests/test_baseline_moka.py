"""B2 MOKA-style tests: translation-only template anchoring, depth lifting,
and the coupling claim -- pixel noise monotonically degrades 3D anchor
accuracy."""
import os

import numpy as np
import pytest

from baselines import common, moka_style
from tests.baseline_helpers import fresh_pair

SIGMAS = [0.0, 5.0, 10.0, 20.0]


def _mean_anchor_err(demo_cap, target_cap, ctx, sigma, n_seeds=25):
    errs = []
    for s in range(n_seeds):
        ctx.noise_seed = s
        res = moka_style.run_moka_oracle(demo_cap, target_cap, ctx,
                                         sigma_px=sigma)
        errs.append(res["anchor_err_m"])
    return float(np.mean(errs))


def test_pixel_noise_monotonically_degrades_anchor():
    demo_cap, target_cap, ctx, _ = fresh_pair("cap_twist", 0)
    means = [_mean_anchor_err(demo_cap, target_cap, ctx, s) for s in SIGMAS]
    # strictly worse end to end, non-decreasing within noise tolerance
    assert means[-1] > means[0]
    for a, b in zip(means, means[1:]):
        assert b >= a - 0.002, "noise did not degrade accuracy: %r" % (means,)


def test_zero_noise_mark_hits_gt_anchor():
    demo_cap, target_cap, ctx, _ = fresh_pair("cap_twist", 0)
    res = moka_style.run_moka_oracle(demo_cap, target_cap, ctx, sigma_px=0.0)
    assert res["pixel_err_px"] == 0.0
    # only pixel rounding + rendered-depth quantization remain
    assert res["anchor_err_m"] < 0.03


def test_t_map_is_translation_only():
    demo_cap, target_cap, ctx, _ = fresh_pair("nut_loosen", 0)
    res = moka_style.run_moka_oracle(demo_cap, target_cap, ctx, sigma_px=5.0)
    T = np.asarray(res["T_map"])
    assert common.is_valid_se3(T)
    assert np.allclose(T[:3, :3], np.eye(3))
    assert np.allclose(np.asarray(res["target_grasp"])
                       - np.asarray(res["demo_grasp"]), T[:3, 3])


def test_lift_pixel_falls_back_to_nearest_valid_depth():
    depth = np.full((32, 32), 0.5)
    depth[10:20, 10:20] = 0.0  # invalid hole
    K = np.array([[40.0, 0, 16.0], [0, 40.0, 16.0], [0, 0, 1.0]])
    p, uv_used, fell_back = moka_style.lift_pixel(
        (14, 14), depth, K, np.eye(4))
    assert fell_back
    assert depth[int(uv_used[1]), int(uv_used[0])] > 0
    assert np.isfinite(p).all()

    with pytest.raises(ValueError):
        moka_style.lift_pixel((14, 14), np.zeros((32, 32)), K, np.eye(4))


@pytest.mark.skipif(os.environ.get("ALK_VLM_TEST") != "1",
                    reason="live VLM test; set ALK_VLM_TEST=1 with a local "
                           "server running (see docs/VLM_SETUP.md)")
def test_moka_vlm_live():
    demo_cap, target_cap, ctx, _ = fresh_pair("cap_twist", 0)
    res = moka_style.run_moka_vlm(demo_cap, target_cap, ctx)
    assert common.is_valid_se3(np.asarray(res["T_map"]))
    assert "pixel_err_px" in res  # T_gt available -> scored


def test_moka_requires_t_gt():
    demo_cap, target_cap, ctx, _ = fresh_pair("cap_twist", 0)
    ctx.T_gt = None
    with pytest.raises(ValueError):
        moka_style.run_moka_oracle(demo_cap, target_cap, ctx, sigma_px=0.0)
