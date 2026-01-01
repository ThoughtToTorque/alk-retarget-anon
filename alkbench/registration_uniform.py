"""Uniform bound widening for the bounded Chamfer registration (Campaign G).

DEFAULT OFF everywhere: this module is purely ADDITIVE -- it imports
`alkbench.registration` and does not modify it, so the production default
path (`bounded_registration(src, tgt, T0)` with +-15 deg / +-20 mm) is
byte-identical whether or not this file exists.

Why it exists
-------------
The authors' original paper states, in the hyperparameter-sensitivity
subsection: "Registration bounds dtheta_max = 15 deg, dt_max = 20 mm cover
typical keypoint errors; wider bounds provided no improvement."  Our
conditioning-adaptive registration (`alkbench.registration.
adaptive_registration`, Campaign A-adaptive) widens the rotation bound to
+-90 deg -- but only about ONE axis, the ALK principal axis, and only when
the demo ALK is ill conditioned.  Read literally the two statements
conflict.  The hypothesis this module exists to test is that the authors
widened bounds UNIFORMLY (all three rotation axes and the translation
together), which enlarges the search space isotropically and lets the
Chamfer objective reach poses that fit as well or better while being
geometrically wrong -- no net gain -- whereas SELECTIVE widening along the
single provably-unobservable direction does help.

`uniform_widen_registration` is the uniform variant: same code, same
objective, same step sizes as `bounded_registration`, only larger bounds
(`mode="plain"`), plus an optional coarse-to-fine schedule
(`mode="c2f"`) that mirrors the adaptive path's search strategy so that
uniform-vs-selective is compared at equal angular resolution rather than at
equal grid size.

Search cost is part of the answer (wider bounds are also slower), so every
call reports the number of Chamfer evaluations actually performed and the
wall-clock time of the registration call.
"""

import contextlib
import time

import numpy as np

from alkbench import registration as _reg

__all__ = [
    "BASE_THETA_DEG", "BASE_T_MM", "BASE_ROT_STEP_DEG", "BASE_TRANS_STEP_MM",
    "count_chamfer", "grid_values", "expected_cost",
    "uniform_widen_registration",
]

# the production defaults (alkbench.registration.bounded_registration
# signature); "base" everywhere below means exactly these numbers
BASE_THETA_DEG = 15.0
BASE_T_MM = 20.0
BASE_ROT_STEP_DEG = 5.0
BASE_TRANS_STEP_MM = 10.0

# coarse-to-fine defaults: 15 deg rotation steps are exactly the coarse step
# of `adaptive_registration`, so `mode="c2f"` differs from the selective
# adaptive path ONLY in WHICH degrees of freedom are widened.
COARSE_ROT_STEP_DEG = 15.0
COARSE_TRANS_STEP_MM = 20.0


@contextlib.contextmanager
def count_chamfer():
    """Count Chamfer evaluations inside `alkbench.registration`.

    Temporarily wraps the module-global `chamfer_distance` that
    `bounded_registration` resolves at call time with a counting proxy.
    Single-threaded use only; restored on exit (also on exception).
    """
    counter = {"n": 0}
    orig = _reg.chamfer_distance

    def proxy(*a, **kw):
        counter["n"] += 1
        return orig(*a, **kw)

    _reg.chamfer_distance = proxy
    try:
        yield counter
    finally:
        _reg.chamfer_distance = orig


def grid_values(bound, step):
    """Number of grid values `alkbench.registration._grid` produces."""
    return 2 * int(round(float(bound) / float(step))) + 1


def expected_cost(theta_max_deg=BASE_THETA_DEG, t_max_mm=BASE_T_MM,
                  rot_step_deg=BASE_ROT_STEP_DEG,
                  trans_step_mm=BASE_TRANS_STEP_MM, n_passes=2):
    """Analytic Chamfer-evaluation count of one coordinate-descent search.

    `bounded_registration(method="coord")` evaluates the identity pose once
    and then, for each of `n_passes` passes and each of the 6 DoF, every grid
    value except the one currently held -- an upper bound of
    1 + n_passes * (3*(n_rot-1) + 3*(n_trans-1)) evaluations (exact when the
    incumbent never moves off a grid value already visited in that pass).
    """
    n_rot = grid_values(theta_max_deg, rot_step_deg)
    n_trans = grid_values(t_max_mm, trans_step_mm)
    return {
        "n_rot_values": n_rot,
        "n_trans_values": n_trans,
        "n_poses_in_box": n_rot ** 3 * n_trans ** 3,
        "n_chamfer_evals_expected": 1 + n_passes * (3 * (n_rot - 1)
                                                    + 3 * (n_trans - 1)),
    }


def uniform_widen_registration(src, tgt, T_init,
                               theta_max_deg=BASE_THETA_DEG,
                               t_max_mm=BASE_T_MM,
                               mode="plain",
                               rot_step_deg=BASE_ROT_STEP_DEG,
                               trans_step_mm=BASE_TRANS_STEP_MM,
                               coarse_rot_step_deg=COARSE_ROT_STEP_DEG,
                               coarse_trans_step_mm=COARSE_TRANS_STEP_MM,
                               n_passes=2, max_points=2000, seed=0):
    """Bounded Chamfer registration with UNIFORMLY widened bounds.

    Uniform = the same bound is applied to all three rotation axes and the
    same bound to all three translation axes (contrast
    `alkbench.registration.adaptive_registration`, which widens the rotation
    bound about ONE axis only).

    Parameters
    ----------
    theta_max_deg, t_max_mm : the widened bounds.  At the base values
        (15 deg, 20 mm) with `mode="plain"` and default steps this call is
        EXACTLY `bounded_registration(src, tgt, T_init)` -- same code path,
        same result, bit for bit.
    mode : "plain" widens the bounds and changes nothing else (the literal
        one-parameter change the sensitivity study describes).
        "c2f" runs a coarse pass over the widened box at
        `coarse_rot_step_deg` / `coarse_trans_step_mm` and then refines
        inside the BASE box around the coarse optimum -- the same
        coarse-to-fine schedule (and the same 15 deg coarse rotation step)
        as the selective adaptive path.
    n_passes, max_points, seed : forwarded to `bounded_registration`.

    Returns
    -------
    dict with the `bounded_registration` keys ("T", "chamfer_init",
    "chamfer_final") plus:
      uniform_widen : {mode, theta_max_deg, t_max_mm, rot_step_deg,
                       trans_step_mm, base_theta_deg, base_t_mm,
                       widened (bool), cost {...}}
      n_chamfer_evals : Chamfer evaluations actually performed
      reg_time_s      : wall-clock seconds of the registration call
      chamfer_coarse  : (c2f only) Chamfer after the coarse pass
    """
    if mode not in ("plain", "c2f"):
        raise ValueError("mode must be 'plain' or 'c2f', got %r" % (mode,))
    theta_max_deg = float(theta_max_deg)
    t_max_mm = float(t_max_mm)
    widened = bool(theta_max_deg > BASE_THETA_DEG or t_max_mm > BASE_T_MM
                   or mode != "plain")

    t0 = time.time()
    with count_chamfer() as counter:
        if mode == "plain":
            out = _reg.bounded_registration(
                src, tgt, T_init,
                rot_bound_deg=theta_max_deg, rot_step_deg=rot_step_deg,
                trans_bound=t_max_mm * 1e-3,
                trans_step=trans_step_mm * 1e-3,
                n_passes=n_passes, max_points=max_points, seed=seed)
            cost = expected_cost(theta_max_deg, t_max_mm, rot_step_deg,
                                 trans_step_mm, n_passes)
        else:
            coarse = _reg.bounded_registration(
                src, tgt, T_init,
                rot_bound_deg=theta_max_deg,
                rot_step_deg=coarse_rot_step_deg,
                trans_bound=t_max_mm * 1e-3,
                trans_step=coarse_trans_step_mm * 1e-3,
                n_passes=n_passes, max_points=max_points, seed=seed)
            fine = _reg.bounded_registration(
                src, tgt, coarse["T"],
                rot_bound_deg=BASE_THETA_DEG,
                rot_step_deg=BASE_ROT_STEP_DEG,
                trans_bound=BASE_T_MM * 1e-3,
                trans_step=BASE_TRANS_STEP_MM * 1e-3,
                n_passes=n_passes, max_points=max_points, seed=seed)
            out = {"T": fine["T"],
                   "chamfer_init": float(coarse["chamfer_init"]),
                   "chamfer_final": float(fine["chamfer_final"])}
            out["chamfer_coarse"] = float(coarse["chamfer_final"])
            c_coarse = expected_cost(theta_max_deg, t_max_mm,
                                     coarse_rot_step_deg,
                                     coarse_trans_step_mm, n_passes)
            c_fine = expected_cost(BASE_THETA_DEG, BASE_T_MM,
                                   BASE_ROT_STEP_DEG, BASE_TRANS_STEP_MM,
                                   n_passes)
            cost = {"coarse": c_coarse, "fine": c_fine,
                    "n_chamfer_evals_expected":
                        c_coarse["n_chamfer_evals_expected"]
                        + c_fine["n_chamfer_evals_expected"]}
        n_evals = int(counter["n"])
    out["reg_time_s"] = float(time.time() - t0)
    out["n_chamfer_evals"] = n_evals
    out["uniform_widen"] = {
        "mode": mode,
        "theta_max_deg": theta_max_deg,
        "t_max_mm": t_max_mm,
        "rot_step_deg": float(rot_step_deg),
        "trans_step_mm": float(trans_step_mm),
        "coarse_rot_step_deg": float(coarse_rot_step_deg),
        "coarse_trans_step_mm": float(coarse_trans_step_mm),
        "base_theta_deg": BASE_THETA_DEG,
        "base_t_mm": BASE_T_MM,
        "widened": widened,
        "cost": cost,
    }
    return out
