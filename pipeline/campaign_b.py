"""Phase-4 Campaign B runner: sensor-noise sensitivity (medium-tier scenes).

Design (docs/REPRODUCE.md, Campaign B; frozen before the run):

  methods {ours_full, ours_noreg, icp, kp_fps4}
  x 5 tasks x medium-tier scene pairs (existing data_medium, seeds 1000..1029)
  x noise grid depth_sigma in {0, 1.5, 3, 6} mm x mask_erosion in {0, 2, 4} px

Noise model = baselines.difficulty.apply_sensor_noise (Gaussian depth noise
on valid pixels + binary mask erosion), applied at PERCEPTION time to BOTH
the demo and the target capture (same sensor); captures on disk are never
modified.  Placement stays medium tier -- the noise grid generalizes the
"hard" tier (which is exactly the (3 mm, 2 px) cell).

Noise wiring (per method family):

  * registry baselines (icp, kp_fps4): the existing hook --
    baselines.common.MethodContext(sensor_noise=..., noise_seed=...) ->
    baselines.common.perceive applies apply_sensor_noise before
    backprojection.
  * ours_full / ours_noreg: pipeline.retarget_runner.run_pair perceives via
    the module attribute ``pipeline.perception.perceive``; this runner
    temporarily wraps THAT attribute (``noisy_perception`` context manager)
    so the same apply_sensor_noise(depth, mask, params,
    RandomState(noise_seed)) is applied to the picked camera before the
    original perceive runs.  retarget_runner itself is untouched (it is
    being modified concurrently for the adaptive campaign).

  Both paths compute the ORIGINAL instance mask first and call
  apply_sensor_noise with a fresh RandomState(noise_seed), so all four
  methods see the IDENTICAL noisy depth/mask realization on a given
  (task, seed, cell) -- the paired design extends to the noise itself.

Noise RNG seed: crc32("<task>|<seed>|<cell>") per (task, seed, cell) --
reproducible, distinct across cells and seeds, and shared demo/target
(fresh RandomState per perceive call, matching the existing hard-tier
convention in baselines.common.perceive).

The (0 mm, 0 px) cell is identical to Campaign A's medium tier; its
rollouts are SYMLINKED from results/campaign_a (--link-zero) instead of
recomputed.

Per-rollout JSON: <out-root>/<task>/<cell>/<seed>/rollout_<method>.json
with the Campaign-A record schema plus {"cell", "sensor_noise",
"noise_seed"}.  Resumable: existing files are skipped.

Usage (one process per task, like Campaign A):

  # one-time: link the zero-noise cell from campaign_a
  python -m pipeline.campaign_b --task pour --link-zero
  # the sweep
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m pipeline.campaign_b \
      --task pour --seed-start 1000 --seed-end 1029
"""
import argparse
import json
import os
import time
import traceback
import zlib

import numpy as np

from simtasks import envs
from pipeline import campaign, oracle, perception
from pipeline import retarget_runner as rr
from baselines import common as bcommon
from baselines import difficulty

SIMBENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_ROOT = os.path.join(SIMBENCH, "results", "campaign_b")
CAMPAIGN_A_ROOT = os.path.join(SIMBENCH, "results", "campaign_a")
DATA_ROOT = os.path.join(SIMBENCH, "data_medium")   # medium-tier pairs
TIER = "medium"

METHODS = ["ours_full", "ours_noreg", "icp", "kp_fps4"]

DEPTH_SIGMAS_MM = (0.0, 1.5, 3.0, 6.0)
MASK_EROSIONS_PX = (0, 2, 4)


def cell_name(sigma_mm, erode_px):
    return "s%g_e%d" % (sigma_mm, erode_px)


# grid order: erosion-major, sigma-minor; (0,0) first
CELLS = [(s, e) for e in MASK_EROSIONS_PX for s in DEPTH_SIGMAS_MM]
CELL_NAMES = [cell_name(s, e) for s, e in CELLS]


def cell_params(sigma_mm, erode_px):
    """difficulty-style sensor-noise dict; None for the zero cell (exact
    parity with the Campaign-A medium tier, which passed sensor_noise=None).
    """
    if sigma_mm == 0 and erode_px == 0:
        return None
    return {"depth_sigma_m": float(sigma_mm) * 1e-3,
            "mask_erosion_px": int(erode_px)}


def noise_seed_for(task, seed, sigma_mm, erode_px):
    """Deterministic noise RNG seed per (task, seed, cell)."""
    key = "%s|%d|%s" % (task, int(seed), cell_name(sigma_mm, erode_px))
    return zlib.crc32(key.encode("ascii")) & 0x7FFFFFFF


def result_path(out_root, task, cell, seed, method):
    return os.path.join(out_root, task, cell, str(seed),
                        "rollout_%s.json" % method)


# ---------------------------------------------------------------------------
# perception-noise injection for the ours_* path (no retarget_runner edits)
# ---------------------------------------------------------------------------

def _noisy_camera(cap, task, cam, params, noise_seed):
    """Return (noisy_depth, rewritten_seg) for one camera of a capture.

    Replicates baselines.common.perceive's noise application exactly:
    original instance mask -> apply_sensor_noise(depth, mask, params,
    RandomState(noise_seed)); the seg is rewritten (int64 copy, eroded-away
    target pixels set to -1) so that ``seg == inst_id`` afterwards yields
    exactly the eroded mask.
    """
    cd = cap["cameras"][cam]
    inst = bcommon.TASK_INFO[task]["target_instance"]
    inst_id = perception.instance_id_for(cap["meta"], inst)
    seg = np.asarray(cd["seg"])
    mask = seg == inst_id
    depth2, mask2 = difficulty.apply_sensor_noise(
        cd["depth"], mask, params, np.random.RandomState(noise_seed))
    seg2 = seg.astype(np.int64, copy=True)
    seg2[mask & ~mask2] = -1
    return depth2, seg2


class noisy_perception(object):
    """Context manager: wrap pipeline.perception.perceive so every call sees
    sensor noise on its picked camera.  No-op when params is falsy."""

    def __init__(self, params, noise_seed):
        self.params = params
        self.noise_seed = int(noise_seed)
        self._orig = None

    def __enter__(self):
        if not self.params:
            return self
        self._orig = perception.perceive
        params, nseed, orig = self.params, self.noise_seed, perception.perceive

        def wrapped(cap, task, k=8, seed=0, camera=None):
            cam = camera or perception.pick_camera(cap, task)
            depth2, seg2 = _noisy_camera(cap, task, cam, params, nseed)
            cd2 = dict(cap["cameras"][cam])
            cd2["depth"] = depth2
            cd2["seg"] = seg2
            cams2 = dict(cap["cameras"])
            cams2[cam] = cd2
            cap2 = dict(cap)
            cap2["cameras"] = cams2
            return orig(cap2, task, k=k, seed=seed, camera=cam)

        perception.perceive = wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig is not None:
            perception.perceive = self._orig
            self._orig = None
        return False


# ---------------------------------------------------------------------------
# zero-noise cell: symlink Campaign A's medium-tier rollouts
# ---------------------------------------------------------------------------

def link_zero_cell(task, seeds, methods=METHODS, out_root=DEFAULT_OUT_ROOT,
                   campaign_a_root=CAMPAIGN_A_ROOT):
    """Symlink campaign_a/<task>/<seed>/rollout_<m>.json into the (0,0)
    cell (identical protocol: medium placement, no sensor noise)."""
    cell = cell_name(0, 0)
    n_new = n_missing = 0
    for seed in seeds:
        dst_dir = os.path.join(out_root, task, cell, str(seed))
        for m in methods:
            src = os.path.join(campaign_a_root, task, str(seed),
                               "rollout_%s.json" % m)
            dst = os.path.join(dst_dir, "rollout_%s.json" % m)
            if not os.path.exists(src):
                print("MISSING campaign_a source: %s" % src)
                n_missing += 1
                continue
            if os.path.lexists(dst):
                continue
            os.makedirs(dst_dir, exist_ok=True)
            os.symlink(os.path.relpath(src, dst_dir), dst)
            n_new += 1
    print("[%s] zero cell: %d new symlinks, %d missing sources"
          % (task, n_new, n_missing))
    return n_new, n_missing


# ---------------------------------------------------------------------------
# task loop
# ---------------------------------------------------------------------------

def run_task(task, seeds, cells, methods, out_root):
    """One process per task; env reused across seeds/cells/methods."""
    campaign.ensure_demo_assets(task, DATA_ROOT)
    env = campaign.make_tier_env(task, TIER)
    demo = None
    demo_cap = None
    n_done = 0
    try:
        for seed in seeds:
            todo = [(s, e, m) for (s, e) in cells for m in methods
                    if not os.path.exists(result_path(
                        out_root, task, cell_name(s, e), seed, m))]
            if not todo:
                continue
            pair = rr.ensure_pair(task, seed, env=env, data_root=DATA_ROOT)
            if demo is None:
                demo = rr.load_demo(task, DATA_ROOT)
                demo_cap = bcommon.load_capture(
                    os.path.join(rr.demo_dir_for(task, DATA_ROOT), "scene"))
            target_cap = bcommon.load_capture(
                os.path.join(DATA_ROOT, task, str(seed), "target", "scene"))
            inst = pair["target_instance"]
            T_gt = oracle.gt_relative_transform(
                pair["demo_object_poses"][inst],
                pair["target_object_poses"][inst])

            for sigma_mm, erode_px in cells:
                cell = cell_name(sigma_mm, erode_px)
                cell_todo = [m for (s, e, m) in todo
                             if (s, e) == (sigma_mm, erode_px)]
                if not cell_todo:
                    continue
                params = cell_params(sigma_mm, erode_px)
                nseed = noise_seed_for(task, seed, sigma_mm, erode_px)
                # one ctx per (pair, cell): noisy perception cached across
                # the registry methods of this cell only
                ctx = bcommon.MethodContext(
                    task=task, T_gt=T_gt, keyframes=demo["keyframes"],
                    waypoints=demo["waypoints"], sigma_px=0.0,
                    noise_seed=nseed, random4_seed=seed,
                    sensor_noise=params)
                for name in cell_todo:
                    t0 = time.time()
                    try:
                        if name in campaign.OURS_KW:
                            with noisy_perception(params, nseed):
                                r = campaign.run_ours_variant(
                                    name, task, seed, env, DATA_ROOT, TIER)
                        else:
                            r = campaign.run_registry_method(
                                name, task, seed, pair, demo, demo_cap,
                                target_cap, ctx, env, TIER)
                    except Exception as e:
                        traceback.print_exc()
                        r = {"task": task, "seed": seed, "method": name,
                             "variant": name, "tier": TIER, "success": False,
                             "failure_stage": "exception", "error": repr(e),
                             "time_s": round(time.time() - t0, 1)}
                    r["cell"] = cell
                    r["sensor_noise"] = params or {"depth_sigma_m": 0.0,
                                                   "mask_erosion_px": 0}
                    r["noise_seed"] = nseed
                    campaign._write_json(
                        result_path(out_root, task, cell, seed, name), r)
                    n_done += 1
                    print("[%s %d %s %-10s] success=%s stage=%-12s rot=%s "
                          "trans=%s (%.1fs)"
                          % (task, seed, cell, name, r.get("success"),
                             str(r.get("failure_stage")),
                             ("%.1fdeg" % r["rot_err_deg"])
                             if r.get("rot_err_deg") is not None else "-",
                             ("%.1fmm" % (1e3 * r["trans_err_m"]))
                             if r.get("trans_err_m") is not None else "-",
                             time.time() - t0), flush=True)
    finally:
        env.close()
    return n_done


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=sorted(envs.TASKS.keys()))
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--seed-end", type=int, default=1029, help="inclusive")
    p.add_argument("--methods", nargs="*", default=METHODS)
    p.add_argument("--cells", nargs="*", default=None,
                   help="cell names (e.g. s1.5_e2); default: all 12 grid "
                        "cells (the linked zero cell is skipped by resume)")
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--link-zero", action="store_true",
                   help="only symlink the (0,0) cell from campaign_a")
    args = p.parse_args(argv)

    seeds = list(range(args.seed_start, args.seed_end + 1))
    for m in args.methods:
        if m not in campaign.ALL_METHODS:
            raise SystemExit("unknown method %r" % m)
    if args.link_zero:
        link_zero_cell(args.task, seeds, methods=args.methods,
                       out_root=args.out_root)
        return 0
    if args.cells:
        by_name = dict(zip(CELL_NAMES, CELLS))
        for c in args.cells:
            if c not in by_name:
                raise SystemExit("unknown cell %r (have %s)"
                                 % (c, CELL_NAMES))
        cells = [by_name[c] for c in args.cells]
    else:
        cells = list(CELLS)
    t0 = time.time()
    n = run_task(args.task, seeds, cells, args.methods, args.out_root)
    print("done: %s, %d new rollouts in %.1f min"
          % (args.task, n, (time.time() - t0) / 60.0), flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
