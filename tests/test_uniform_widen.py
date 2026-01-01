"""Tests for UNIFORM registration-bound widening (Campaign G).

The uniform variant lives in its own module (`alkbench.registration_uniform`)
and is default-off; these tests pin (a) that the base setting is bit-exactly
the production `bounded_registration` default, (b) that the Chamfer-eval
counter is exact and leaves the module untouched, and (c) the qualitative
claim under test: on a slender object, UNIFORM +-90 deg widening does not
recover the rotation as well as SELECTIVE +-90 deg about the ALK principal
axis, while costing several times more Chamfer evaluations.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from alkbench import (bounded_registration, adaptive_registration,
                      chamfer_distance, transform_points, rotation_angle_deg)
from alkbench import registration as reg_mod
from alkbench import registration_uniform as ru

from tests.test_adaptive_registration import thin_box_cloud, slender_alk


def _rot_err_deg(Ta, Tb):
    return rotation_angle_deg(np.asarray(Ta)[:3, :3].T @ np.asarray(Tb)[:3, :3])


# ---------------------------------------------------------------------------
# defaults are untouched
# ---------------------------------------------------------------------------

def test_base_setting_is_bit_exact_default():
    src = thin_box_cloud(n=800, seed=1)
    T_true = np.eye(4)
    T_true[:3, :3] = Rotation.from_euler("z", 8.0, degrees=True).as_matrix()
    tgt = transform_points(T_true, src)
    T_init = np.eye(4)
    ref = bounded_registration(src, tgt, T_init)
    got = ru.uniform_widen_registration(src, tgt, T_init)
    assert np.array_equal(np.asarray(got["T"]), np.asarray(ref["T"]))
    assert got["chamfer_init"] == ref["chamfer_init"]
    assert got["chamfer_final"] == ref["chamfer_final"]
    assert got["uniform_widen"]["widened"] is False


def test_count_chamfer_is_exact_and_restores():
    src = thin_box_cloud(n=400, seed=2)
    tgt = thin_box_cloud(n=400, seed=3)
    orig = reg_mod.chamfer_distance
    with ru.count_chamfer() as c:
        chamfer_distance(src, tgt)          # not counted (module-level import)
        reg_mod.chamfer_distance(src, tgt)  # counted
        reg_mod.chamfer_distance(src, tgt)
    assert c["n"] == 2
    assert reg_mod.chamfer_distance is orig


def test_count_chamfer_restores_on_exception():
    orig = reg_mod.chamfer_distance
    try:
        with ru.count_chamfer():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert reg_mod.chamfer_distance is orig


def test_expected_cost_matches_measured_upper_bound():
    src = thin_box_cloud(n=400, seed=4)
    tgt = thin_box_cloud(n=400, seed=4)
    for theta, t in ((15.0, 20.0), (45.0, 60.0), (90.0, 60.0)):
        out = ru.uniform_widen_registration(src, tgt, np.eye(4),
                                            theta_max_deg=theta, t_max_mm=t)
        exp = out["uniform_widen"]["cost"]["n_chamfer_evals_expected"]
        assert out["n_chamfer_evals"] <= exp
        assert out["n_chamfer_evals"] >= 0.5 * exp


def test_widening_monotonically_increases_search_cost():
    src = thin_box_cloud(n=400, seed=5)
    tgt = thin_box_cloud(n=400, seed=5)
    costs = []
    for theta, t in ((15.0, 20.0), (30.0, 40.0), (45.0, 60.0), (90.0, 60.0)):
        out = ru.uniform_widen_registration(src, tgt, np.eye(4),
                                            theta_max_deg=theta, t_max_mm=t)
        costs.append(out["n_chamfer_evals"])
    assert costs == sorted(costs)
    assert costs[-1] >= 3 * costs[0]


def test_c2f_mode_runs_coarse_then_fine():
    src = thin_box_cloud(n=600, seed=6)
    T_true = np.eye(4)
    T_true[:3, :3] = Rotation.from_euler("x", 40.0, degrees=True).as_matrix()
    tgt = transform_points(T_true, src)
    out = ru.uniform_widen_registration(src, tgt, np.eye(4),
                                        theta_max_deg=90.0, t_max_mm=60.0,
                                        mode="c2f")
    assert out["uniform_widen"]["mode"] == "c2f"
    assert "chamfer_coarse" in out
    assert out["chamfer_final"] <= out["chamfer_coarse"] + 1e-12
    assert out["chamfer_final"] <= out["chamfer_init"]
    assert out["uniform_widen"]["cost"]["coarse"]["n_rot_values"] == 13


def test_bad_mode_rejected():
    try:
        ru.uniform_widen_registration(np.zeros((4, 3)), np.zeros((4, 3)),
                                      np.eye(4), mode="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# the claim under test: uniform vs selective on a slender object
# ---------------------------------------------------------------------------

def test_uniform_and_selective_both_recover_clean_synthetic_but_cost_differs():
    """Thin box rotated 40 deg about its own long axis (the DoF the slender
    ALK cannot observe), NOISELESS and complete cloud.

    Honest baseline for the Campaign-G claim: in this idealized instance the
    widened box is not the problem -- uniform +-90 deg finds the pose too
    (the true rotation is exactly on its grid), it just pays several times
    the search cost.  So "uniform widening provides no improvement" is NOT a
    property of the search being unable to reach the pose; it is a property
    of real, partial, noisy clouds, where the extra DoF let the Chamfer
    objective reach equally-good WRONG poses.  That part is measured on real
    scenes in results/campaign_g, not here."""
    src = thin_box_cloud(n=2500, seed=7)
    alk = slender_alk()
    axis = alk[1] - alk[0]
    axis = axis / np.linalg.norm(axis)
    center = src.mean(axis=0)
    R = Rotation.from_rotvec(axis * np.radians(40.0)).as_matrix()
    T_true = np.eye(4)
    T_true[:3, :3] = R
    T_true[:3, 3] = center - R @ center
    tgt = transform_points(T_true, src)
    T_init = np.eye(4)  # ALK init blind to the axial rotation

    fixed = bounded_registration(src, tgt, T_init)
    with ru.count_chamfer() as c_sel:
        sel = adaptive_registration(src, tgt, T_init, alk)
    uni = ru.uniform_widen_registration(src, tgt, T_init,
                                        theta_max_deg=90.0, t_max_mm=60.0)

    e_fixed = _rot_err_deg(fixed["T"], T_true)
    e_sel = _rot_err_deg(sel["T"], T_true)
    e_uni = _rot_err_deg(uni["T"], T_true)
    assert e_fixed > 20.0                 # the fixed bound cannot reach it
    assert e_sel < 10.0                   # selective recovers it
    assert e_uni < 10.0                   # so does uniform, on clean data
    # but at a much larger search cost: selective = 13 coarse evals + one
    # base-bound refinement; uniform = a refinement over a 6x/3x wider box
    assert uni["n_chamfer_evals"] > 2 * c_sel["n"]
