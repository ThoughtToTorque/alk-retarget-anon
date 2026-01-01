# Reproducing the paper

This file maps what is in the paper to the command that produces it, with the
honest runtime cost. Read the two sections at the top first — they decide
whether you want to run anything at all.

## What is in this repository, and what is not

**In this repository (everything needed to *run* the benchmark):**

* the method (`alkbench/`), the 5 task environments and the scene-pair
  generator (`simtasks/`), the baselines and ablations (`baselines/`), the
  rollout runner and every campaign driver (`pipeline/`), the statistics and
  table generators (`stats/`), the test suite (`tests/`), the two demos
  (`demo/`);
* `baselines/testdata/` — three frozen scene captures (7 MB), the only binary
  data shipped. The pytest suite uses them to test baselines on real captures
  without starting a simulator (`tests/baseline_helpers.py`); they are kept
  for that reason and nothing else.

**Not in this repository:**

| what | why not | where it is |
|---|---|---|
| scene pairs (`data*/`, ~2.5 GB across resolutions and tiers) | they are a *deterministic function of a seed*. Shipping them would be shipping a cache | regenerated on demand by `simtasks/scene_pairs.py::generate_pair`; nothing you run needs a download |
| per-rollout records and generated tables (`results/`, ~340 MB) | they are the evidence, not the code; a code repository is the wrong place for them | release asset `alk-retarget-records-v1.tar.gz` on the [`records-v1` release](https://github.com/ThoughtToTorque/alk-retarget-anon); `tar -xzf` at the repository root recreates `results/` |
| real-robot experiments | documented in the paper | this repository covers the simulation study |
| VLM weights and server | 30–40 GB of third-party model weights | pulled from Hugging Face by you; see [VLM_SETUP.md](VLM_SETUP.md) |

Docstrings across the code refer to paths like `results/campaign_j/...` and
`data_medium_a2/`. Those are paths *inside the record deposit / the
regenerated cache*, not files in this checkout. Where a driver genuinely
consumes a cached record as an input (Campaign K reads Campaign J's cached VLM
answers; several `report` phases read their own `run` phase's output) it is
called out in the table below.

## Runtime cost, stated plainly

One rollout on the development machine (RTX PRO 6000, single process, EGL
headless, 256 px captures) takes **≈6.5 s** wall clock, essentially all of it
MuJoCo stepping. The full campaign set is on the order of **26,000 rollouts**
— about **46 hours of single-process wall clock**, and the real campaigns were
run as several concurrent per-task processes over several days. Nothing here
is a five-minute reproduction, which is exactly why `demo/mini_benchmark.py`
exists.

Every driver is **resumable**: an existing `rollout_*.json` is skipped, so an
interrupted run is restarted with the same command.

## Prerequisites for any campaign command

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl   # headless rendering
```

Run every command from the repository root. Scene-pair caches and record trees
are created as needed; use `--data-root` / `--out-root` to put them somewhere
with space.

## Campaign → command map

Rollout counts are computed from the design (tasks × seeds × methods), not
measured; they are the number of records a completed campaign contains.

| campaign | what it establishes | driver | design | ≈rollouts |
|---|---|---|---|---|
| **A** | the headline paired comparison: full method vs 3 system ablations and 8 baseline/ablation methods, 5 tasks, medium tier | `pipeline/campaign.py` | 5 tasks × 60 seeds (1000–1059) × 11 methods | 3300 |
| **A-adaptive** | conditioning-adaptive registration (widened rotation search about the ALK principal axis) vs the production default | `pipeline/campaign.py --adaptive` with its own `--out-root` | 5 × 60 × `ours_*` | 900 |
| **B** | sensor-noise sensitivity: depth σ ∈ {0, 1.5, 3, 6} mm × mask erosion ∈ {0, 2, 4} px, applied at perception time to both scenes | `pipeline/campaign_b.py` | 4 methods × 5 tasks × 30 seeds × 12 noise cells | 7200 |
| **C** | conditional decoupling: inject discrete-answer errors at rate ρ ∈ {0, .1, .2, .5, 1} and measure how success degrades | `pipeline/campaign_c.py` | 5 tasks × 40 seeds × 4 non-zero ρ (ρ=0 reuses A) | 800 |
| **D** | real-VLM discrete error rate, raw and symmetry-corrected, plus end-to-end success with VLM answers, plus the MOKA pixel-regression contrast | `pipeline/campaign_d.py` | 4 tasks × 30 seeds × arms | ≈500 |
| **E** | symmetry quantification (re-scores A's rotation errors under each object's symmetry group — no new rollouts) and the cross-category stress test | `pipeline/campaign_e.py` | re-score + 2 × 30 cross-category | ≈120 |
| **F** | which ALK points should enter the closed-form sum (`alk4` vs the earlier three-point construction `alk3_c134` vs `alk3_c123`, `alk2_c12`) | `pipeline/campaign.py --methods alk4 alk3_c134 ...` | 5 × 60 × 6 subset variants | 1800 |
| **G** | uniform vs selective (conditioning-adaptive) bound widening at matched search budget | `pipeline/campaign_g.py` | 5 × 60 × 6 settings (base reused from A) | 1800 |
| **H** | capture-resolution sensitivity (256 / 384 / 512 px) | `pipeline/campaign.py --cam-size {384,512}` with per-size `--data-root` and `--out-root` | 5 × 60 × methods × 2 extra resolutions | ≈3000 |
| **I** | coordinate-convention-corrected re-run of the pixel-regression VLM baseline, and the adaptive stage's no-op verification on well-conditioned tasks | `pipeline/campaign_i.py` | 4 tasks × 30 seeds × arms | ≈400 |
| **J** | the deployable head-to-head: MOKA re-implemented *faithfully* (mark-based visual prompting) against this method, both driven by the **same real VLM**, same scenes and executor | `pipeline/campaign_j.py` | 5 tasks × seeds × 5 arms | ≈1200 |
| **K** | Campaign J follow-ups: adaptive registration on J's scenes, and MOKA given *our* candidate set | `pipeline/campaign_k.py` | reuses J's cached VLM answers | ≈600 |
| **L** | ReKep re-implemented as in the released method (DINOv2 keypoint proposal + real-VLM constraint authoring + sandboxed code execution) | `pipeline/campaign_l.py` | 5 tasks × seeds × arms | ≈900 |
| **M** | generality across VLM families: the J/L protocol re-run against a second, lineage-disjoint model | `pipeline/campaign_j.py` / `campaign_l.py` with `ALK_VLM_MODEL` pointed at the second server | subset of J/L | ≈400 |

Multi-phase drivers (`campaign_c`, `d`, `e`, `i`, `j`, `k`, `l`) take a phase
as the first positional argument. `run`/`execute` produces rollouts;
`report` only reads what is already there. `freeze` phases select prompts and
hyper-parameters **on non-evaluation scenes** (easy tier, seeds 1–5, and the
demonstration scenes) and write a `freeze.json`; nothing after a freeze is
tuned. Run them in the documented order — each driver's module docstring
lists its phases.

### Concrete examples

```bash
# Campaign A, one task (run the five tasks as five concurrent processes)
python -m pipeline.campaign --task pour --tier medium \
    --seed-start 1000 --seed-end 1059
# is the placement window feasible for the scripted expert at all?
python -m pipeline.campaign --task pour --report-feasibility \
    --seed-start 1000 --seed-end 1009

# Campaign F: which ALK points enter the closed-form sum
python -m pipeline.campaign --task pour --methods alk4 alk3_c134 \
    alk3_c123 alk2_c12 --out-root results/campaign_f

# Campaign C: discrete-error injection
python -m pipeline.campaign_c run --tasks nut_loosen cap_twist \
    --seed-start 1000 --seed-end 1039
python -m pipeline.campaign_c report

# Campaign E part 1: symmetry re-scoring, no simulator needed
python -m pipeline.campaign_e symmetry
```

### Tables from records

```bash
# generic: tidy CSV + success/error tables for any record tree
python -m stats.aggregate <record-root> --csv rollouts.csv
python -m stats.make_tables <record-root> --prefix campaign_a \
    --ours ours_full --out-dir results/tables
# campaign-specific table scripts (noise grid, ALK subsets, resolution sweep)
python -m stats.campaign_b_tables --root <record-root>
python -m stats.campaign_f_tables --root <record-root>
python -m stats.campaign_h_tables --root <record-root>
# the design's power calculation (the N=60 justification)
python -m stats.power
```

The `stats.campaign_*_tables` scripts emit both `.md` and `.tex`; the LaTeX
files are the ones in the paper.

## Demo videos

The simulation clips (omitted from the anonymized copy) were rendered with
`python demo/record_rollout.py --task <task> --seed <seed>` (nut_loosen /
rim_grasp / cap_twist / box_open at seed 1000, pour at seed 1001; box_open
and pour with `--camera sideview`): the same oracle-answer rollout as
`demo/quickstart.py`, plus an offscreen agentview frame captured every 2
control steps and written as an mp4. Rendering is opt-in; nothing else in
the pipeline renders during rollouts.

## Statistics protocol (frozen before the runs)

* **N = 60** seeds per task per method, seed set `{1000..1059}` shared by all
  methods — a **paired** design. `stats/power.py` gives N ≥ 56 to detect
  0.68 vs 0.40 at 80 % power, two-sided α = 0.05, under the conservative
  independent assumption; N = 60 leaves margin (the hardware study's N = 10
  per task would have 8.3 % power for the same contrast).
* **Primary metric**: success rate with Wilson 95 % CI. Method-vs-method
  comparisons use the **exact paired McNemar** test (`stats/tests.py`), with
  Fisher's exact test only where pairing genuinely does not hold. p-values are
  reported as numbers, not stars; families are corrected with
  Holm–Bonferroni.
* **Secondary metrics**: `T_map` rotation and translation error against ground
  truth, reported separately from success so pose quality and task outcome are
  not conflated.
* **Discrete answers**: the large-N campaigns use the simulator oracle (zero
  API cost, and methodologically cleaner — it isolates the continuous stage).
  The real-VLM campaigns (D, I, J, K, L, M) measure the discrete error rate
  and the end-to-end success under real answers on smaller n. Where a
  VLM-driven arm is compared against an oracle-driven one, the matched
  reference is `ours_nocorr` — the grasp-point correction consumes the oracle
  `φ4/φ5` region and would otherwise partially rescue wrong VLM answers.

## Verifying the code rather than the numbers

```bash
python -m pytest tests -q        # 117 passed, 2 skipped on a fresh checkout
python -m simtasks.run_smoke     # captures, back-projection, all 5 scripted demos
python demo/quickstart.py        # ~20 s from clean, one full rollout + a figure
python demo/mini_benchmark.py    # 6 min 55 s measured, the paired protocol at demo scale
```

Two tests skip by default, both on purpose:

* `tests/test_baseline_moka.py` — a live-VLM test; set `ALK_VLM_TEST=1` with a
  server running (see [VLM_SETUP.md](VLM_SETUP.md)).
* `tests/test_pour_criterion.py` — needs the canonical `pour` demonstration
  recording, which does not exist in a fresh checkout because no scene data is
  shipped. Record it once and the test runs:

  ```bash
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
      python -c "from pipeline import retarget_runner as rr; rr.ensure_demo('pour')"
  ```

  after which the suite is 118 passed, 1 skipped.
