"""Orientation-aware `pour` success criterion (2026-08-16 revision).

The superseded criterion was grasp-and-hold-through-the-tilt (env-native
Lift check at the end of the rollout), which cannot distinguish an execution
whose object transform is off by the vessel's 180-deg body flip: same grasp,
same tilt magnitude, opposite tilt sense.  These tests pin the construction
(reference direction read off the fixed demonstration), the tolerance, and
the two margins that make the tolerance a decision rather than a knob.

Pure numpy — no mujoco needed except the one test that reads the recorded
demo keyframes off disk.
"""
import os

import numpy as np
import pytest

from simtasks import envs

SIMBENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(SIMBENCH, "data", "pour", "0", "demo")


def _rx(deg):
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _rz(deg):
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _angle(a, b):
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)), -1, 1))))


def test_pinned_reference_matches_the_recorded_demo():
    """POUR_TILT_REF_BODY is exactly what the fixed demo recording says."""
    if not os.path.exists(os.path.join(DEMO_DIR, "keyframes")):
        pytest.skip("canonical pour demo recording not present")
    up, ref = envs.pour_reference_from_demo(DEMO_DIR)
    assert np.allclose(up, envs.POUR_UP_AXIS_BODY, atol=1e-6)
    assert np.allclose(ref, envs.POUR_TILT_REF_BODY, atol=1e-6)
    # the demonstration carries the up-axis well past the tolerance
    assert _angle(up, ref) > envs.POUR_ORI_TOL_DEG + 30.0


def test_tolerance_pin():
    assert envs.POUR_ORI_TOL_DEG == 45.0


def test_reference_is_the_demonstrated_tilt():
    """81 deg down from vertical == the demo's achieved pour tilt."""
    ang = _angle(np.asarray(envs.POUR_UP_AXIS_BODY),
                 np.asarray(envs.POUR_TILT_REF_BODY))
    assert 75.0 <= ang <= 85.0


def test_flip_is_rejected_by_a_wide_margin_at_every_instant():
    """An execution off by the block's 180-deg body flip tilts the vessel the
    opposite way.  Its up-axis is ~162 deg from the reference at the apex and
    never closer than ~81 deg at ANY instant of the rollout — so the 45 deg
    tolerance rejects it with >= 30 deg of margin, while a correct pour is
    exact.

    Simulated analytically: the demo's body-frame rotation from rest to apex
    is B, the flipped one is Rz(180) @ B (the extra world-z half-turn the
    flipped object transform injects into the retargeted trajectory).
    """
    up = np.asarray(envs.POUR_UP_AXIS_BODY, dtype=float)
    ref = np.asarray(envs.POUR_TILT_REF_BODY, dtype=float)
    tol = envs.POUR_ORI_TOL_DEG

    # sweep the demonstrated tilt from 0 to its apex, correct and flipped
    worst_correct, best_flip = 0.0, 180.0
    for frac in np.linspace(0.0, 1.0, 41):
        # interpolate the body-frame rotation that carries `up` to `ref`
        axis = np.cross(up, ref)
        axis /= np.linalg.norm(axis)
        th = np.deg2rad(_angle(up, ref) * frac)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        B = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
        worst_correct = max(worst_correct, _angle(B @ up, ref))
        best_flip = min(best_flip, _angle(_rz(180.0) @ B @ up, ref))
    assert worst_correct <= 81.5           # only the untilted start is far
    assert _angle(B @ up, ref) < 0.02      # apex of a correct pour: exact
    assert best_flip >= tol + 30.0         # flip never gets close
    assert _angle(_rz(180.0) @ B @ up, ref) > 160.0   # flipped apex


def test_yaw_error_budget():
    """The 45 deg tolerance corresponds to ~47 deg of estimated-yaw error on
    the demonstrated 81 deg tilt: generous for a correct pour, decisive
    against the flip."""
    ref = np.asarray(envs.POUR_TILT_REF_BODY, dtype=float)
    passing = [d for d in range(0, 181)
               if _angle(_rz(d) @ ref, ref) <= envs.POUR_ORI_TOL_DEG]
    assert max(passing) >= 45
    assert max(passing) <= 55
    assert 180 not in passing


def test_success_fn_is_wired_and_needs_the_tracker():
    spec = envs.TASKS["pour"]
    assert spec.success_fn is envs._pour_success
    assert "45" in spec.description

    class _FakeEnv(object):
        _simtasks_init_obj_pose = {"pos": [0, 0, 0.82],
                                   "quat_xyzw": [0, 0, 0, 1], "yaw": 0.0}
        _simtasks_pour_track = []

        def _check_success(self):
            return True

    # lifted but nothing recorded -> not a pour
    assert envs._pour_success(_FakeEnv()) is False
