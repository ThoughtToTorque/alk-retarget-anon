"""(demo_scene, target_scene) pair generator.

Reproduces the paper's experimental unit: ONE demonstration scene per task
(fixed, feasible for the scripted expert) and a TARGET scene in which the
object placement (position + yaw) is re-randomized by the env's placement
sampler under a per-pair seed.

Layout (under <repo>/data by default; ALK_DATA_ROOT overrides):
  <task>/<seed>/pair.json          seeds + ground-truth poses of both scenes
  <task>/<seed>/demo/scene/        capture of the demo scene initial state
  <task>/<seed>/target/scene/      capture of the target scene initial state
  <task>/<seed>/demo/              (after scripted_demo.run_demo) keyframes/,
                                   trajectory.npz, waypoints.json, demo_meta.json

Determinism: scene <-> seed via envs.reset_with_seed (global numpy RNG drives
robosuite's placement samplers), so a scene can always be re-instantiated
from its recorded seed.
"""
import json
import os

import numpy as np

try:
    from . import capture, envs
except ImportError:
    import capture
    import envs

# Scene pairs are NOT shipped with this repository: they are regenerated
# deterministically from their seeds (see `generate_pair` below), so this is
# just a cache directory.  Override with the ALK_DATA_ROOT env var or the
# --data-root flag of the runners.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("ALK_DATA_ROOT",
                           os.path.join(REPO_ROOT, "data"))

# Fixed demonstration-scene seed per task (chosen so the scripted expert
# succeeds; the paper likewise records a single successful human demo).
DEMO_SEEDS = {
    "nut_loosen": 0,
    "rim_grasp": 0,
    "pour": 0,
    "box_open": 0,
    "cap_twist": 0,
}

# Offset so target-scene seeds never collide with demo-scene seeds.
TARGET_SEED_BASE = 500000


def target_seed_for(task, seed):
    # NOTE: must be deterministic across processes (python randomizes str
    # hash() per process), hence the fixed task index table
    task_index = sorted(envs.TASKS.keys()).index(task)
    return TARGET_SEED_BASE + task_index * 10000 + seed


def success_checker(task):
    """Hook for later evaluation: callable env -> bool."""
    return envs.success_checker(task)


def _capture_state(env, out_dir, task):
    spec = envs.TASKS[task]
    meta = capture.capture_scene(env, out_dir,
                                 extra_state=spec.extra_state(env))
    return meta


def generate_pair(task, seed, out_root=None, demo_seed=None,
                  camera_size=envs.DEFAULT_CAM_SIZE):
    """Generate and save one (demo_scene, target_scene) pair.

    Returns the pair-info dict (also written to pair.json)."""
    out_root = out_root or DATA_ROOT
    pair_dir = os.path.join(out_root, task, str(seed))
    demo_seed = DEMO_SEEDS[task] if demo_seed is None else demo_seed
    tgt_seed = target_seed_for(task, seed)

    env = envs.make_env(task, camera_size=camera_size)
    try:
        envs.reset_with_seed(env, demo_seed)
        demo_meta = _capture_state(env, os.path.join(pair_dir, "demo", "scene"),
                                   task)
        envs.reset_with_seed(env, tgt_seed)
        target_meta = _capture_state(env,
                                     os.path.join(pair_dir, "target", "scene"),
                                     task)
    finally:
        env.close()

    spec = envs.TASKS[task]
    info = {
        "task": task,
        "env_name": spec.env_name,
        "seed": seed,
        "demo_seed": demo_seed,
        "target_seed": tgt_seed,
        "target_instance": spec.target_instance,
        "demo_object_poses": demo_meta["objects"],
        "target_object_poses": target_meta["objects"],
    }
    with open(os.path.join(pair_dir, "pair.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info


def load_pair(task, seed, out_root=None):
    out_root = out_root or DATA_ROOT
    pair_dir = os.path.join(out_root, task, str(seed))
    with open(os.path.join(pair_dir, "pair.json")) as f:
        info = json.load(f)
    return {
        "info": info,
        "demo_scene": capture.load_capture(os.path.join(pair_dir, "demo",
                                                        "scene")),
        "target_scene": capture.load_capture(os.path.join(pair_dir, "target",
                                                          "scene")),
    }


def make_env_for(pair_info, which, camera_size=envs.DEFAULT_CAM_SIZE):
    """Re-instantiate the demo or target scene of a pair as a live env
    ('which' in {'demo', 'target'}).  Caller must close() it."""
    env = envs.make_env(pair_info["task"], camera_size=camera_size)
    seed = pair_info["demo_seed"] if which == "demo" else pair_info["target_seed"]
    envs.reset_with_seed(env, seed)
    return env


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=sorted(envs.TASKS.keys()))
    p.add_argument("seed", type=int)
    p.add_argument("--out-root", default=None)
    args = p.parse_args()
    info = generate_pair(args.task, args.seed, out_root=args.out_root)
    print(json.dumps(info, indent=2))
