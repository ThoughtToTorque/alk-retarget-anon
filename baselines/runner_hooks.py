"""Method registry: name -> callable(demo_capture, target_capture, ctx).

The thin adapter between the baseline methods (B1-B4) and the evaluation
code.  Nothing here imports robosuite; the Phase-4 runner imports this module
and iterates methods uniformly:

    from baselines import runner_hooks as hooks

    demo_cap, target_cap, ctx, pair = hooks.common.context_from_pair(pair_dir)
    for name in hooks.list_methods():
        result = hooks.run_method(name, demo_cap, target_cap, ctx)
        # result["T_map"] is a 4x4 (list) world-frame demo->target object
        # transform; feed it to pipeline.retarget_runner's execution path
        # (retarget_waypoint_dicts + simtasks.motion.execute_waypoints).

Registry API
------------
* ``REGISTRY``            -- OrderedDict name -> fn(demo, target, ctx)
* ``register(name)``      -- decorator adding an entry
* ``list_methods()``      -- registered names, registration order
* ``get_method(name)``    -- the raw callable
* ``run_method(name, demo_capture, target_capture, ctx)`` -- call + result
  normalization.  Every returned dict has: ``method`` (registry name),
  ``T_map`` (4x4 nested list), ``time_s``, plus method-specific aux metrics
  (``chamfer_after_m`` everywhere it is meaningful, ``conditioning``
  {singular_values, sigma23} for the keypoint methods, pixel errors for
  MOKA, optimizer stats for ReKep).  MOKA results additionally carry
  ``demo_grasp`` / ``target_grasp`` (the anchor is part of the method);
  whether to apply the pipeline's grasp-point correction on top of the other
  methods' T_map is the eval runner's policy decision, not encoded here.
* ``map_errors(T_est, T_gt, demo_obj_pos, target_obj_pos)`` -- the same
  rotation/translation error metric as pipeline.retarget_runner, duplicated
  here without the robosuite-importing dependency chain.

Naming: MOKA oracle variants are registered per SPEC noise level
(``moka_oracle_px{0,5,10,20}``) plus a ``moka_oracle`` entry that reads
sigma from ctx.sigma_px; keypoint ablations are ``kp_{alk,fps4,random4,
pca2,dense_init}``.
"""
import time
from collections import OrderedDict

import numpy as np

from alkbench import rotation_angle_deg, transform_points

from baselines import common, icp, keypoint_ablations, moka_style, \
    rekep_style

REGISTRY = OrderedDict()


def register(name):
    def deco(fn):
        if name in REGISTRY:
            raise KeyError("method %r already registered" % name)
        REGISTRY[name] = fn
        return fn
    return deco


def list_methods():
    return list(REGISTRY.keys())


def get_method(name):
    return REGISTRY[name]


def run_method(name, demo_capture, target_capture, ctx):
    """Run one registered method and normalize its result dict."""
    fn = get_method(name)
    t0 = time.time()
    result = fn(demo_capture, target_capture, ctx)
    result["method"] = name  # registry name wins over the module-level one
    T = np.asarray(result["T_map"], dtype=np.float64)
    if not common.is_valid_se3(T):
        raise ValueError("method %r returned an invalid SE(3) T_map" % name)
    result["T_map"] = T.tolist()
    result["time_s"] = round(time.time() - t0, 3)
    return result


def map_errors(T_est, T_gt, demo_obj_pos, target_obj_pos):
    """Rotation error (deg) + translation error at the object center (m),
    identical metric to pipeline.retarget_runner.map_errors."""
    T_est = np.asarray(T_est, dtype=np.float64)
    T_gt = np.asarray(T_gt, dtype=np.float64)
    rot = float(rotation_angle_deg(T_est[:3, :3] @ T_gt[:3, :3].T))
    mapped = transform_points(T_est,
                              np.asarray(demo_obj_pos, dtype=np.float64)[None])[0]
    trans = float(np.linalg.norm(mapped - np.asarray(target_obj_pos,
                                                     dtype=np.float64)))
    return rot, trans


# ---------------------------------------------------------------------------
# registrations
# ---------------------------------------------------------------------------

# B1 -- Mask+ICP, both init variants (SPEC reports both)
register("icp")(
    lambda d, t, c: icp.run_icp(d, t, c, init="identity"))
register("icp_centroid")(
    lambda d, t, c: icp.run_icp(d, t, c, init="centroid"))

# B2 -- MOKA-style: oracle mark at the SPEC noise levels + ctx-driven +
# real-VLM variant
register("moka_oracle")(moka_style.run_moka_oracle)  # sigma from ctx.sigma_px
for _s in (0, 5, 10, 20):
    register("moka_oracle_px%d" % _s)(
        (lambda s: lambda d, t, c: moka_style.run_moka_oracle(
            d, t, c, sigma_px=float(s)))(_s))
register("moka_vlm")(moka_style.run_moka_vlm)

# B3 -- ReKep-style constraint optimization
register("rekep")(rekep_style.run_rekep)

# B4 -- keypoint-form ablations (shared Procrustes + bounded registration)
for _c in ("alk", "fps4", "random4", "pca2", "dense_init"):
    register("kp_%s" % _c)(
        (lambda cn: lambda d, t, c: keypoint_ablations.run_ablation(
            d, t, c, cn))(_c))
