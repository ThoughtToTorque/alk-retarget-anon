"""baselines: Phase-3 baseline & ablation methods (see docs/CODE_MAP.md).

Every method maps (demo capture, target capture, discrete answers) to a
world-frame 4x4 T_map that plugs into pipeline/retarget_runner's execution
path.  Entry point for evaluation code: baselines.runner_hooks.

Modules (import them directly; this package init stays import-light so the
namespace works without scipy/imageio at import time):
  common             shared capture loading / perception / SE(3) helpers
  icp                B1 Mask+ICP (identity + centroid-aligned inits)
  moka_style         B2 MOKA-style pixel marking (oracle noise + real VLM)
  rekep_style        B3 ReKep-style keypoint-constraint optimization (SLSQP)
  keypoint_ablations B4 keypoint-form ablations (+ conditioning numbers)
  difficulty         randomization tiers (easy/medium/hard) config
  runner_hooks       method registry consumed by the Phase-4 runner
"""
