"""Difficulty tier config tests: tier structure, sensor-noise application,
and the hard-tier path through perception."""
import numpy as np

from baselines import common, difficulty
from tests.baseline_helpers import fresh_pair


def test_tier_structure():
    assert set(difficulty.TIERS) == {"easy", "medium", "hard"}
    tasks = {"nut_loosen", "rim_grasp", "pour", "box_open", "cap_twist"}
    for tier in difficulty.TIERS.values():
        assert set(tier["sampler"]) == tasks
    # easy: no overrides anywhere
    assert all(v is None for v in difficulty.TIERS["easy"]["sampler"].values())
    assert difficulty.TIERS["easy"]["sensor_noise"] is None
    assert difficulty.TIERS["medium"]["sensor_noise"] is None
    hard = difficulty.TIERS["hard"]["sensor_noise"]
    assert hard["depth_sigma_m"] == 0.003
    assert hard["mask_erosion_px"] == 2
    # hard shares medium placement
    assert difficulty.TIERS["hard"]["sampler"] is \
        difficulty.TIERS["medium"]["sampler"]


def test_medium_widens_or_keeps_placement():
    # spot-check the sampler override schema for the Phase-4 runner
    ov = difficulty.sampler_override("medium", "nut_loosen")
    assert ov["x_range"][0] < -0.115 and ov["x_range"][1] > -0.11
    assert ov["rotation"] is None  # full yaw
    assert difficulty.sampler_override("medium", "rim_grasp") is None
    assert difficulty.sampler_override("easy", "pour") is None
    door = difficulty.sampler_override("hard", "box_open")
    assert door["rotation"][0] < -np.pi / 2 - 0.25  # wider than easy window


def test_apply_sensor_noise():
    rng = np.random.RandomState(0)
    depth = np.full((64, 64), 0.7)
    depth[0, 0] = 0.0  # invalid pixel must stay untouched
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:40, 20:40] = True
    noisy, eroded = difficulty.apply_sensor_noise(
        depth, mask, difficulty.HARD_SENSOR_NOISE, rng)
    # depth: sigma ~ 3 mm on valid pixels, invalid untouched, input intact
    dd = (noisy - depth)[depth > 0]
    assert 0.002 < dd.std() < 0.004
    assert noisy[0, 0] == 0.0
    assert depth[30, 30] == 0.7
    # mask: strict 2 px erosion of a 20x20 square -> 16x16
    assert eroded.sum() == 16 * 16
    assert not (eroded & ~mask).any()
    # zero-noise params are a no-op
    same_d, same_m = difficulty.apply_sensor_noise(
        depth, mask, {"depth_sigma_m": 0.0, "mask_erosion_px": 0}, rng)
    assert np.array_equal(same_d, depth) and np.array_equal(same_m, mask)


def test_hard_tier_perception_path():
    demo_cap, _, ctx, _ = fresh_pair("cap_twist", 0)
    clean = common.perceive(demo_cap, "cap_twist")
    noisy = common.perceive(demo_cap, "cap_twist",
                            sensor_noise=difficulty.HARD_SENSOR_NOISE,
                            noise_seed=0)
    assert noisy["mask_pixels"] < clean["mask_pixels"]
    assert noisy["n_points"] >= 8  # still clusterable
    # degradation is bounded: the object is still roughly where it should be
    assert noisy["centroid_err_m"] < 0.10
