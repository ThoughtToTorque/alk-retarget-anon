# alk-retarget

One-shot contact-rich manipulation via keypoint-anchored retargeting: the
method, a 5-task simulation benchmark for it, the baselines it is compared
against, and the statistics harness.

A single RGB-D demonstration plus three TCP keyframes is transferred to a new
scene in which the object has been re-randomised. The transfer is split in
two: a **small discrete** part — which end of the object is the head, which
side is which, where to grasp: five integer variables `Φ = (φ1..φ5)` — is
answered *semantically*, by a VLM when deployed and by the simulator oracle in
the large-N evaluation; a **continuous** part — the actual object pose — is
then solved in closed form by Procrustes on four anchored landmark keypoints
(ALK) built deterministically from `Φ` and the geometry, and refined by a
*bounded* Chamfer search that can polish the pose but cannot jump to a
different symmetry branch. The demo waypoints are mapped through the resulting
transform, corrected at the grasp point, and executed. Splitting it this way
is what lets a near-symmetric object (a nut, a can) be retargeted at all:
dense geometric registration has no way to prefer the semantically right
alignment among several equally good geometric ones.

## Demos

One successful rollout per task. Top row: the simulation benchmark
(oracle discrete answers, the exact pipeline `demo/quickstart.py` runs);
bottom row: the same method on the hardware setup (UR3 with an OnRobot RG2 gripper, a RealSense L515
RGB-D camera and a wrist F/T sensor, single RGB-D demonstration). Simulation clips play at 2.5x real time;
hardware clips are sped up as stated per clip.

| nut_loosen | rim_grasp | pour | box_open | cap_twist |
|---|---|---|---|---|

`python demo/quickstart.py` runs the same rollout the top row shows, end to
end, on your machine. The mp4 originals of the simulation clips are in
(demo media omitted from the anonymized copy); they were rendered with `demo/record_rollout.py`.


## Install

```bash
git clone https://github.com/ThoughtToTorque/alk-retarget-anon && cd alk-retarget
pip install -e .            # numpy, scipy, robosuite==1.4.1, mujoco==3.2.3, imageio, matplotlib
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl    # headless rendering
```

Python ≥ 3.8, Linux. No GPU, no dataset download, no API key: **no scene data
is shipped, because scene pairs are a deterministic function of a seed** and
are regenerated when you run something. Add `pip install -e '.[vlm]'` only for
the optional real-VLM path ([docs/VLM_SETUP.md](docs/VLM_SETUP.md)) and
`'.[dev]'` for pytest.

**Headless rendering.** MuJoCo picks its GL backend from `MUJOCO_GL` at import
time. `egl` is right on any machine with an NVIDIA driver (the driver ships
`libEGL`); on a machine with no GPU install osmesa
(`apt-get install libosmesa6-dev`) and use `MUJOCO_GL=osmesa
PYOPENGL_PLATFORM=osmesa`; on a desktop with a display, `MUJOCO_GL=glfw`
works. Both demos default to `egl` on their own, so `python demo/quickstart.py`
needs no env prefix; the campaign drivers do not, so export the variables
before those.

## Quickstart

```bash
python demo/quickstart.py
```

Generates the demo→target scene pair for one task, records the scripted
demonstration, runs one full rollout with oracle discrete answers, and writes
one figure. **≈20 s on a fresh checkout**, ≈8 s afterwards (the demonstration
is cached). Real output:

```
quickstart: task=nut_loosen seed=1000
  scene pairs are generated from the seed; cache dir: .../alk-retarget/data
  discrete answers: simulator oracle (no VLM / GPU / network)
  MUJOCO_GL=egl
  ... generating the scene pair, recording the scripted demo and running one rollout

========================================================================
one-shot retargeting -- task=nut_loosen  pair seed=1000
========================================================================

perception (ground-truth instance segmentation stands in for
GroundingDINO+SAM; k-means candidates, k=8, camera=agentview)
  demo   mask=  1070 px  cloud=  1070 pts  centroid err=  8.2 mm  sanity=True
  target mask=  1179 px  cloud=  1179 pts  centroid err= 16.5 mm  sanity=True

discrete variables Phi (oracle answers; a VLM would answer these)
  demo   {'phi1': 1, 'phi2': 4, 'phi3': 0, 'phi4': None, 'phi5': None}
  target {'phi1': 0, 'phi2': 4, 'phi3': 0, 'phi4': 1, 'phi5': 3}

ALK quadruple (world frame, metres)
        demo C_i                 target C_i
  C1    [-0.0647 -0.1382  0.8358]  [-0.1336 -0.1094  0.8395]
  C2    [-0.1730 -0.1441  0.8413]  [-0.0938 -0.1846  0.8406]
  C3    [-0.1125 -0.1621  0.8384]  [-0.0799 -0.1426  0.8366]
  C4    [-0.1097 -0.1123  0.8375]  [-0.1311 -0.1628  0.8391]

closed-form pose T0 = Procrustes(ALK_demo, ALK_target)
    [-0.41610 -0.90722  0.06178 | -0.3355 ]
    [ 0.90879 -0.41721 -0.00570 | -0.0987 ]
    [ 0.03095  0.05378  0.99807 |  0.0134 ]
  rotation error      12.19 deg
  translation error    9.47 mm

registered pose T_map = bounded Chamfer refinement of T0
    [-0.16671 -0.98429  0.05820 | -0.3152 ]
    [ 0.98552 -0.16819 -0.02149 | -0.0333 ]
    [ 0.03095  0.05378  0.99807 |  0.0134 ]
  rotation error       4.88 deg
  translation error    4.72 mm

Chamfer distance (demo cloud mapped into the target frame)
  before registration    5.23 mm
  after  registration    4.28 mm

grasp-point translation correction  2.02 mm

execution in the target scene (retargeted waypoints, 306 control steps)
  grasped              True
  waypoints converged  [True, False, True]
  TASK SUCCESS         True

wall clock: 17.0 s
rollout record: .../data/nut_loosen/1000/rollout_full.json
figure:         .../demo_out/quickstart_nut_loosen_1000.png
total wall clock: 17.8 s
```

The figure shows the demo object cloud mapped into the target frame before
(closed-form `T0`) and after (`T_map`) bounded registration, in two
orthogonal projections, with both ALK quadruples marked.

`--task {nut_loosen,rim_grasp,pour,box_open,cap_twist}` and `--seed N` pick a
different scene pair.

## The comparison, at demo scale

```bash
python demo/mini_benchmark.py     # all 5 tasks x 4 paired seeds x 3 methods
```

Runs the same driver the paper's campaigns use (`pipeline/campaign.py`) and
the same statistics code. **6 min 55 s measured from a clean checkout** (60
rollouts plus recording the five scripted demonstrations); resumable, so an
interrupted run continues. Real output, tail:

```
task         method          k/n    rate Wilson 95% CI       McNemar p
----------------------------------------------------------------------
nut_loosen   ours_full      4/4    1.00  [0.51, 1.00]        (ref)
nut_loosen   ours_noreg     4/4    1.00  [0.51, 1.00]           1.000
nut_loosen   icp            1/4    0.25  [0.05, 0.70]           0.250
----------------------------------------------------------------------
rim_grasp    ours_full      3/4    0.75  [0.30, 0.95]        (ref)
rim_grasp    ours_noreg     4/4    1.00  [0.51, 1.00]           1.000
rim_grasp    icp            4/4    1.00  [0.51, 1.00]           1.000
----------------------------------------------------------------------
pour         ours_full      2/4    0.50  [0.15, 0.85]        (ref)
pour         ours_noreg     2/4    0.50  [0.15, 0.85]           1.000
pour         icp            1/4    0.25  [0.05, 0.70]           1.000
----------------------------------------------------------------------
box_open     ours_full      4/4    1.00  [0.51, 1.00]        (ref)
box_open     ours_noreg     3/4    0.75  [0.30, 0.95]           1.000
box_open     icp            3/4    0.75  [0.30, 0.95]           1.000
----------------------------------------------------------------------
cap_twist    ours_full      4/4    1.00  [0.51, 1.00]        (ref)
cap_twist    ours_noreg     4/4    1.00  [0.51, 1.00]           1.000
cap_twist    icp            2/4    0.50  [0.15, 0.85]           0.500
----------------------------------------------------------------------

pooled over nut_loosen, rim_grasp, pour, box_open, cap_twist
(all)        ours_full     17/20   0.85  [0.64, 0.95]        (ref)
(all)        ours_noreg    17/20   0.85  [0.64, 0.95]           1.000  (b=1, c=1 discordant)
(all)        icp           11/20   0.55  [0.34, 0.74]           0.070  (b=7, c=1 discordant)
```

`--tasks`, `--seeds` and `--methods` scale it up. The demo prints, and this
bears repeating: **the paper's numbers come from 60 paired seeds per task over
5 tasks and 11 methods** — at 4 seeds these intervals are far too wide to
support any claim, and `ours_noreg` matching `ours_full` here is exactly the
kind of thing 4 seeds cannot resolve. It illustrates the protocol; it does not
reproduce the results.

## Code map

| package | one sentence |
|---|---|
| `alkbench/` | the method: k-means candidates, ALK construction, Procrustes, bounded and conditioning-adaptive registration, waypoint retargeting, the discrete-variable solver interface (oracle and VLM). Pure numpy + scipy, and deliberately free of any simulator import so the same code can drive hardware |
| `simtasks/` | the 5 robosuite task environments, object-centric success checkers, RGB-D + segmentation capture, the scripted expert that records the single demonstration, and the seed → scene-pair generator |
| `baselines/` | the comparison methods behind one registry — ICP, MOKA-style pixel marking (oracle-noise and real-VLM), ReKep-style constraint optimisation, and the keypoint-form ablations — all evaluated through the same executor and success checker as the method itself |
| `pipeline/` | the perception stage, the oracle, the single-rollout closed loop (`retarget_runner.py`), and one driver per campaign |
| `stats/` | Wilson intervals, exact paired McNemar / Fisher, Holm–Bonferroni, the design's power calculation, and the paper's table generators |

Deeper: [docs/CODE_MAP.md](docs/CODE_MAP.md) walks the data flow file by file
and function by function, and documents how to add a task and how to add a
baseline. `alkbench/NOTES.md` lists every deliberate deviation from the
paper's text; `simtasks/TASKS.md` documents the task↔environment mapping and
the success semantics.

## Paper, records and data

* **Paper**: *One-shot contact-rich manipulation via keypoint-anchored
  Paper identifier withheld for anonymized review.
* **Per-rollout records and generated tables** (≈340 MB unpacked, 69 MB
  compressed, on the order of 26,000 rollouts) are attached to the
  [`records-v1` release](https://github.com/ThoughtToTorque/alk-retarget-anon)
  of this repository as `alk-retarget-records-v1.tar.gz`; extracting it at
  the repository root creates `results/` (sha256 in the release notes).
  The repository also contains everything needed to *regenerate* them; [docs/REPRODUCE.md](docs/REPRODUCE.md) maps each campaign
  to its command and states the real cost (≈46 h of single-process wall
  clock for the full set).
* **Scene data**: none is shipped, and none is needed. Scene pairs are
  regenerated from their seeds.
* **Real-robot experiments** are documented in the paper; this repository
  contains the simulation study.

## Citing

See [CITATION.cff](CITATION.cff). The arXiv identifier is the one remaining
  Paper identifier withheld for anonymized review.

## License

MIT, see [LICENSE](LICENSE). robosuite is MIT and MuJoCo is Apache-2.0, both
compatible.
