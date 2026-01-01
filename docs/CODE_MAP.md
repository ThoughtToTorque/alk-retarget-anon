# Code map

This is the file-and-function level walkthrough for someone who wants to
*modify* the method or the benchmark. For the paper-table-to-command mapping
see [REPRODUCE.md](REPRODUCE.md); for the optional real-VLM path see
[VLM_SETUP.md](VLM_SETUP.md).

## The idea in one paragraph

One RGB-D demonstration gives a masked object point cloud plus three TCP
keyframes. A new scene gives a second masked cloud of the same object at an
unknown pose. Retargeting the demonstration means finding one rigid transform
`T_map` that carries the demo object onto the target object *semantically*
(rim to rim, handle to handle), not merely with minimal geometric distance —
and geometry alone cannot tell you that, because near-symmetric objects have
several equally good geometric alignments. The method therefore splits the
problem: a **small discrete** part (which end of the object is the "head",
which side is which — five integer variables `Φ = (φ1..φ5)`) is answered
semantically, by a VLM in deployment and by the simulator oracle in the
large-N evaluation; and a **continuous** part (the actual pose) is then
solved in closed form by Procrustes on four *anchored landmark keypoints*
(ALK) built deterministically from `Φ` and the geometry, followed by a
*bounded* Chamfer refinement that can polish the pose but cannot jump to a
different symmetry branch. The demo TCP waypoints are mapped through `T_map`,
corrected at the grasp point, and executed.

## Data flow

Each arrow names the file and function that does the work. The single-rollout
entry point that strings all of this together is
`pipeline/retarget_runner.py::run_pair`.

```
                                   simtasks/scene_pairs.py::generate_pair
   seed  ────────────────────────►  (demo scene, target scene)
                                   simtasks/envs.py::reset_with_seed
```

| # | stage | file :: function |
|---|---|---|
| 1 | **scene pair from a seed** — one fixed demo scene per task, target scene re-randomised under the tier's placement sampler | `simtasks/scene_pairs.py::generate_pair`, `target_seed_for`; `simtasks/envs.py::make_env`, `reset_with_seed` (seeds the global numpy RNG that drives robosuite's placement samplers, then settles), `baselines/difficulty.py::TIERS` for the easy/medium/hard placement windows |
| 2 | **capture** — RGB + depth + ground-truth instance segmentation + intrinsics/extrinsics per camera, written to disk | `simtasks/capture.py::capture_scene` → `load_capture`; camera matrices from `camera_params` |
| 3 | **the demonstration** — scripted expert acts in the demo scene, recording 3 keyframes (approach / pre_grasp / post_action) and the commanded TCP waypoints | `simtasks/scripted_demo.py::run_demo` (per-task `demo_nut_loosen`, `demo_cap_twist`, `demo_rim_grasp`, `demo_pour`, `demo_box_open`), keyframes via `KeyframeRecorder`; primitives in `simtasks/motion.py::Executor` |
| 4 | **segmentation → candidates** — the object mask (GT instance seg stands in for GroundingDINO+SAM), masked valid-depth pixels back-projected to the world frame, then 2-D k-means with `k=8` and 3-D cluster centroids | `pipeline/perception.py::perceive` → `alkbench/candidates.py::compute_candidates` (`backproject`, hand-rolled `kmeans`), returns a `CandidateSet` with `points3d`, `pixels`, `labels`, `centers2d`, `candidates3d` |
| 5 | **discrete answers Φ** — `φ1,φ2` axial endpoint candidate ids, `φ3` lateral swap bit, `φ4,φ5` coarse/fine grasp region | oracle: `pipeline/oracle.py::solve` (`demo_axial_choice`, `target_axial_choice`, `lateral_swap_choice`, `demo_grasp_point`, `grasp_region_choice`); VLM: `alkbench/discrete.py::VLMSolver.solve` on a numbered-marker image rendered by `draw_candidate_markup` + `encode_png` (pure numpy + zlib, no PIL) |
| 6 | **ALK construction** — `C1,C2` = the 3-D centroids of the two chosen axial clusters; the image-plane line through their projections splits the mask into two half-planes whose 3-D centroids are `C3,C4` (order set by `φ3`) | `alkbench/alk.py::alk_from_candidates` → `build_alk`, `halfplane_signed_distance`; slenderness / depth-consistency prior via `lateral_extent_ratio`, `SLENDER_RATIO_THRESHOLD` |
| 7 | **closed-form pose** — Kabsch/SVD Procrustes on the two ordered ALK quadruples | `alkbench/procrustes.py::procrustes` (subset ablations: `align_keypoints` on `alkbench/alk.py::select_alk_subset`, e.g. the earlier three-point construction `{C1,C3,C4}` = `alk3_c134`) |
| 8 | **bounded registration** — coordinate-descent search over Δθ ≤ 15° (step 5°) and Δt ≤ 20 mm (step 10 mm) around `T0`, minimising the symmetric Chamfer distance between the two dense clouds. Bounded on purpose: it refines, it cannot switch symmetry branch | `alkbench/registration.py::bounded_registration`, `chamfer_distance`; the conditioning-adaptive variant that widens the rotation search about the ALK principal axis when the demo ALK is nearly rank-1 is `adaptive_registration` + `adaptive_bounds` (`SIGMA_RATIO_THRESHOLD`) |
| 9 | **waypoint retargeting + grasp correction** — `T_map` applied to each demo waypoint as a 4×4 (orientation rotated too), plus a translation correction that puts the mapped grasp point on the oracle-chosen fine grasp region | `alkbench/retarget.py::retarget` (`retarget_waypoints`, `grasp_translation_correction`); the dict-level wrapper is `pipeline/retarget_runner.py::retarget_waypoint_dicts` |
| 10 | **execution** — target scene reset to its seed, retargeted waypoints run through the shared OSC primitives with the demo's gripper schedule | `simtasks/motion.py::execute_waypoints` (+ `Executor`), grasp check `pipeline/retarget_runner.py::_grasp_check` |
| 11 | **success check** — object-centric, computed from GT sim state relative to the object's *settled initial* pose (no world-fixed goals; see `simtasks/TASKS.md`) | `simtasks/envs.py::success_checker` → `_nut_loosen_success`, `_cap_twist_success`, `_rim_grasp_success`, `_pour_success` (orientation-aware), env-native hinge check for `box_open` |
| 12 | **record** — one JSON per rollout: `T_init`, `T_map`, Chamfer before/after, rot/trans error vs GT, the oracle answers and diagnostics, success, one-level failure-stage guess | `pipeline/retarget_runner.py::_save_result`; tidy rows and CSV via `stats/aggregate.py::row_from_record`, `load_rollouts` |
| 13 | **statistics** — success rate with Wilson 95% CI, exact paired McNemar between methods on shared seeds, Fisher as the unpaired fallback, Holm–Bonferroni across families | `stats/tests.py::wilson_ci`, `mcnemar_exact`, `mcnemar_from_vectors`, `fisher_exact`, `paired_bootstrap_diff`, `holm_bonferroni`; tables in `stats/report.py`, `stats/make_tables.py`, `stats/campaign_*_tables.py`; design power in `stats/power.py` |

## Package map

| package | what it is | may import a simulator? |
|---|---|---|
| `alkbench/` | the method itself: candidates, ALK, Procrustes, bounded/adaptive registration, waypoint retargeting, the discrete-variable solver interface (oracle and VLM). Pure numpy + scipy | **no** — deliberately simulator-free, so the same code can drive a real robot |
| `simtasks/` | the 5 robosuite task environments, object-centric success checkers, scene capture, the scripted expert, the seed → scene-pair generator | yes |
| `baselines/` | the comparison methods and ablations, all behind one registry so the evaluator treats them uniformly | no (they consume saved captures) |
| `pipeline/` | perception stage, the oracle, the single-rollout closed loop, and the campaign drivers | yes |
| `stats/` | Wilson CIs, exact paired tests, power analysis, table generation | no |
| `tests/` | pytest suite: geometry unit tests, baseline contracts, statistics against closed forms | no simulator except one skipped-by-default test |

`alkbench/NOTES.md` records the implementation choices behind the reported
numbers (hand-rolled k-means, the Chamfer convention, the coordinate-descent
search, the VLM prompt, …).
`simtasks/TASKS.md` documents the task ↔ robosuite mapping, the object-centric
success semantics and the placement windows.

## Baselines and ablations

All of them are evaluated through the *same* executor, the same demo waypoint
template, and the same success checker; they differ only in how `T_map` is
obtained. That is what makes the comparison fair, and it is enforced
structurally: every method is a `callable(demo_capture, target_capture, ctx)
-> {"T_map": 4x4, ...}` registered in `baselines/runner_hooks.py::REGISTRY`
and dispatched by `run_method`.

| registry name(s) | method | file |
|---|---|---|
| `icp`, `icp_centroid` | B1 point-to-point ICP on the shared object clouds (cKDTree NN + Kabsch, ≤50 iters), identity and centroid-aligned inits. Locks onto the wrong symmetry branch on the nut and the can — the wrong-branch failure of paper Sections 5.2 and 5.9 | `baselines/icp.py` |
| `moka_oracle`, `moka_oracle_px{0,5,10,20}`, `moka_vlm` | B2 MOKA-style: the semantic answer *is* a pixel coordinate, lifted to 3-D by the depth at that pixel, with no registration refinement. The oracle variants inject controlled pixel noise σ ∈ {0,5,10,20} px, which quantifies the coupling MOKA's design has and this method does not | `baselines/moka_style.py`, marker rendering and real-VLM prompting in `baselines/moka_marks.py` |
| `rekep` | B3 ReKep-style: keypoint relational constraints solved for an SE(3) pose with SLSQP, no bounded registration | `baselines/rekep_style.py`; the faithful real-VLM + DINOv2-proposal version is `baselines/rekep_real.py` with the proposal worker `baselines/rekep_keypoint_proposal.py` (torch, separate interpreter — see VLM_SETUP.md) |
| `kp_alk`, `kp_fps4`, `kp_random4`, `kp_pca2`, `kp_dense_init` | B4 keypoint-*form* ablations: same Procrustes + bounded registration, only the 4-point construction changes (ALK / farthest-point / random / PCA 2-point / centroid+PCA init). Each records the `(σ2+σ3)` conditioning number, which is what ties the empirical rotation error to the error bound | `baselines/keypoint_ablations.py` |
| `ours_full`, `ours_noreg`, `ours_nocorr` | B5 system ablations, run through `retarget_runner.run_pair` flags rather than the registry | `pipeline/campaign.py::OURS_METHODS` |
| `alk4`, `alk3_c134`, `alk3_c123`, `alk2_c12`, `*_adaptive` | which ALK points enter the closed-form Procrustes sum; `alk3_c134` is the paper's original construction | `pipeline/campaign.py::ALK_SUBSET_METHODS`, `alkbench/alk.py::ALK_SUBSETS` |

## How to add a task

1. **Environment + spec.** Add a `TaskSpec` entry to `simtasks/envs.py::TASKS`
   (`env_name`, `target_instance` — the instance-segmentation name of the
   object the method acts on —, `sanity_tol` for the perception centroid
   check, `extra_state` if the env has articulated state to record). If the
   robosuite env needs modifying, subclass it in `envs.py` (see `PourLift`).
2. **Success checker.** Write `_yourtask_success(env)` reading GT sim state
   *relative to the object's settled initial pose* (`reset_with_seed` stores
   it as `env._simtasks_init_obj_pose`) and register it in
   `success_checker`. Object-centric, never a world-fixed goal — otherwise no
   object-relative one-shot method can hit it by construction. Rationale in
   `simtasks/TASKS.md`.
3. **Placement sampler.** Give the task a demo seed in
   `simtasks/scene_pairs.py::DEMO_SEEDS` and, if the env default is
   unsuitable, a tier override in `baselines/difficulty.py`. Verify the
   window is *feasible*: `python -m pipeline.campaign --task yourtask
   --report-feasibility --seed-start 1000 --seed-end 1009`.
4. **Scripted expert.** Write `demo_yourtask(env, spec, ex, kf)` in
   `simtasks/scripted_demo.py` using the `Executor` primitives, record the
   three keyframes with `kf`, and add it to the dispatch in `run_demo`. It
   must succeed in the demo scene — the whole pipeline is one-shot off this
   single recording.
5. **Camera.** If the default `agentview` is uninformative (as for the Door
   env), add the task to `pipeline/perception.py::CAMERA_PRIORITY`.
6. **Grasp geoms.** Add the task's contact geoms to
   `pipeline/retarget_runner.py::object_geoms` so the grasp check works.
7. **Check it.** `python -m simtasks.run_smoke` (capture round-trip +
   back-projection + all scripted demos), then
   `python demo/quickstart.py --task yourtask`.

## How to add a baseline

1. Write `run_yourmethod(demo_capture, target_capture, ctx)` in a new module
   under `baselines/`. Read the perception output through
   `ctx.percept(capture, role)` so you consume *exactly* the clouds and
   candidates every other method consumes, and the demo keyframes/waypoints
   through `ctx.keyframe_tcp(name)` / `ctx.waypoints`. `ctx` is
   `baselines/common.py::MethodContext`; build one outside the simulator with
   `common.context_from_pair(pair_dir)`.
2. Return `common.base_result("yourmethod", T_map, **aux)`. `T_map` must be a
   valid SE(3) — `runner_hooks.run_method` validates it and will raise
   otherwise. Put anything you want in the tables into `aux`
   (`chamfer_after_m`, `conditioning`, optimizer stats …); `stats/aggregate.py`
   flattens the keys it knows.
3. If your method needs ground truth (an oracle variant does), take it from
   `ctx.require_T_gt("what for")` — that call documents the dependency and
   fails loudly when `T_gt` is absent, so oracle-only methods cannot silently
   leak GT into a deployment path.
4. Register it: `register("yourmethod")(yourmodule.run_yourmethod)` in
   `baselines/runner_hooks.py`, and add the name to
   `pipeline/campaign.py::REGISTRY_METHODS` if it should run in the default
   campaign method set.
5. Add a contract test in `tests/` modelled on `tests/test_baseline_icp.py`:
   the frozen captures in `baselines/testdata/` let you assert on real data
   without a simulator.
6. Run it: `python demo/mini_benchmark.py --methods ours_full yourmethod
   --seeds 6`.

## Conventions worth knowing before you edit

* **Poses** are 4×4 homogeneous matrices in the **world** frame throughout;
  quaternions at the robosuite boundary are `xyzw`
  (`simtasks/motion.py::pose_to_matrix` / `matrix_to_pose`).
* **Cameras** use OpenCV axes; `capture_scene` stores `K` and `T_world_cam`
  per camera and `alkbench/candidates.py::backproject` consumes them
  directly. The conventions were verified empirically — do not "fix" a sign.
* **Determinism.** `reset_with_seed` seeds the *global* numpy RNG because that
  is what robosuite's placement samplers read; `target_seed_for` uses a fixed
  task-index table rather than `hash()` (which python randomises per
  process). Scene geometry is independent of the render resolution, so the
  same seed gives the same scene at any `--cam-size` — but captures of
  different resolutions must not be mixed inside one pair directory, which
  `pipeline/campaign.py::check_pair_res` enforces.
* **Scene data is never shipped or archived as an input.** Everything under a
  data root is a cache regenerated from seeds. Delete it freely.
* **"Phase 2 / 3 / 4" in docstrings** are development-stage labels from the
  order the code was built: phase 2 = the task suite and the single-rollout
  closed loop, phase 3 = the baselines, phase 4 = the large-N campaigns and
  the statistics. They carry no meaning beyond that; "phase 2b" marks the
  point where the success checkers were redefined object-centrically
  (`simtasks/TASKS.md`).
* **Rollout JSONs are append-only in spirit.** New flags add new keys and
  leave the default path byte-identical, so records from different campaign
  generations stay comparable. Keep it that way when you add options.
