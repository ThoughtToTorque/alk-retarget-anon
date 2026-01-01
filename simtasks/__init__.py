"""simtasks: simulation task suite for the one-shot retargeting benchmark.

See TASKS.md for the task/env mapping and data conventions.
"""
from . import capture, envs, scene_pairs, scripted_demo  # noqa: F401
from .envs import TASKS, make_env, reset_with_seed, success_checker  # noqa: F401
from .scene_pairs import generate_pair, load_pair, make_env_for  # noqa: F401
from .scripted_demo import run_demo  # noqa: F401
