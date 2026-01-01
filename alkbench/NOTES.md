# alkbench — implementation notes

Pure numpy + scipy package implementing the geometry pipeline of
"One-Shot Contact-Rich Manipulation via Keypoint-Anchored Retargeting"
(Sections 3.3–3.8 of the manuscript). Verified under Python 3.8
(numpy 1.22.4 / scipy 1.10.1) and system Python 3.9.
Test suite: `tests/` (run `python -m pytest tests/` from the repository root).

## Implementation choices (all of them are what the reported results use)

1. **k-means is hand-rolled** (plain Lloyd's, fixed `RandomState` seed,
   empty-cluster re-seeding at the farthest point); no sklearn dependency.
   Deterministic under a fixed seed, but centroids will not bit-match
   sklearn's k-means++.
2. **Clustering is restricted to masked pixels with *valid depth*** (the same
   set that gets backprojected), not all mask pixels, so every cluster has a
   3D centroid. Order is as in Section 3.3: cluster in 2D pixel space,
   average in 3D.
3. **Halfplane line anchors**: the line through the two axial endpoints is
   anchored at the clusters' 2D k-means centres (`cands.centers2d`), which is
   what every campaign in this repository passes (Section 3.4). Passing
   `intrinsics=` to `alk_from_candidates` switches to the pinhole projections
   of the 3D centroids; no reported result uses that path. The
   depth-consistency prior averages the z components and is only meaningful
   in the camera frame (see the docstring); the manuscript evaluates it and
   recommends against it (Section 5.3).
4. **Chamfer definition** (Eq. 8): the *average* of the two directed mean
   nearest-neighbour distances,
   d_ch = ½(mean_a min_b ‖a−b‖ + mean_b min_a ‖b−a‖). Every Chamfer value in
   the manuscript uses this form.
5. **Bounded registration search** (Section 3.6): the ±15°/5° × ±20 mm/10 mm
   grids define 7³·5³ = 42 875 poses, which the solver never enumerates. The
   default is first-improvement **coordinate descent** over the six
   perturbation parameters in a fixed order, two passes: 30 evaluations per
   pass plus the initial one, 61 analytic Chamfer evaluations (measured median
   63) on 2000-point subsamples, well under 1 s. An exact `method="grid"`
   option exists for coarser steps (warns above 10 000 poses). The descent
   never leaves the per-axis box and never increases the objective above the
   initialization's value; global optimality over the box is not claimed.
6. **Perturbation parameterization**: the SE(3) neighbourhood rotates about
   the centroid of the transformed source subsample (Euler xyz, degrees) and
   then translates, with the delta left-multiplied onto T_init. Rotating about
   the centroid keeps the ±20 mm translation bound meaningful for off-origin
   objects.
7. **Conditioning-adaptive bounds** (Section 3.7): `adaptive_bounds` reads the
   conditioning ratio (σ₂+σ₃)/σ₁ of the centred demonstration ALK and, below
   τ_σ = 0.40, widens only the rotation bound about the ALK principal axis to
   ±90° with a 15° coarse sweep followed by the base refinement (median 76
   evaluations per call).
8. **VLM prompts**: the prompts in `alkbench/discrete.py` and
   `pipeline/campaign_d.py` are the ones the reported real-VLM campaigns
   used (structured-JSON answer over 1-based marker indices, syntactic
   validation, one retry with error feedback). Markup rendering and PNG
   encoding are pure numpy + stdlib (zlib); `openai` is imported lazily
   inside `solve()`.
9. **φ4/φ5 (grasp region variables)** are carried by `DiscreteChoice` and
   the solvers; the grasp-region subdivision itself lives in the task-suite
   code (`pipeline/`).
10. **Test tolerances**: the trace/arccos rotation-angle formula has a numeric
    floor of ~1e-6 deg near identity, so `rotation_angle_deg` uses scipy's
    quaternion magnitude; noiseless Procrustes recovery then passes < 1e-6
    deg. For the slender synthetic object, rotation about the long axis is
    weakly observable from Chamfer, so the noisy-registration test asserts
    Chamfer reduction (the Eq. 8 objective), not pose-error reduction; the
    end-to-end test uses 0.5 mm keypoint noise (lateral centroids are only
    ~1.5 cm apart, so keypoint noise maps almost 1:1 into axial rotation
    error) and TCP waypoints with realistic few-cm lever arms.
