"""Campaign C: controlled corruption of the oracle's TARGET-side discrete
answers (VLM error injection -- the direct test of the conditional-
decoupling argument of paper Section 4.1).

Semantics (joint injection, frozen in docs/REPRODUCE.md):
with probability rho per rollout (seeded per (task, seed, rho)) the
target-side discrete choice is corrupted AFTER the oracle solved it:

  * (phi1, phi2)  -> a random WRONG ordered candidate pair, uniform over
                     all ordered pairs (i, j), i != j, excluding the
                     oracle's pair;
  * phi3          -> flipped with probability 0.5 (a uniformly random
                     wrong-or-right bit, i.e. the corruption a VLM that
                     guesses the swap bit would produce);
  * (phi4, phi5)  -> a random WRONG (coarse, fine) grasp-region combo,
                     uniform over all non-empty combos excluding the
                     oracle's, with the region points / centroid rebuilt
                     exactly the way pipeline.oracle.grasp_region_choice
                     builds them (same k-means, same seed).

The DEMO side is never touched: a real VLM error manifests as a wrong
*target* correspondence for a fixed (verified-once) demo annotation.

Everything downstream (target ALK reconstruction, Procrustes, bounded
registration, grasp projection, execution) runs unchanged on the corrupted
choice; if the corrupted endpoint pair yields a degenerate halfplane split
(alkbench raises ValueError), that IS the pipeline's behavior on such an
answer, so the injector reports it and the rollout is recorded as an
``alk_degenerate`` failure instead of being silently redrawn.

The injector is passed to pipeline.retarget_runner.run_pair via its
optional ``inject=`` hook (default None = existing behavior, untouched).
No shared module behavior is modified by importing this file.
"""
import zlib

import numpy as np

from alkbench import alk_from_candidates
from alkbench.candidates import kmeans

# must match pipeline.oracle.grasp_region_choice defaults
K_COARSE = 5
K_FINE = 5

RHOS = (0.1, 0.2, 0.5, 1.0)  # rho=0 cell reuses campaign_a ours_full


def rng_for(task, seed, rho):
    """Deterministic per-(task, seed, rho) generator (frozen draw order:
    1 injection coin, 2 axial-pair index, 3 phi3 coin, 4 region index)."""
    return np.random.default_rng(np.random.SeedSequence(
        [zlib.crc32(task.encode("utf8")), int(seed), int(round(rho * 1000))]))


def wrong_axial_pair(k, correct, rng):
    """Uniform draw over ordered pairs (i, j), i != j, excluding `correct`."""
    pairs = [(i, j) for i in range(k) for j in range(k)
             if i != j and (i, j) != tuple(correct)]
    if not pairs:
        raise ValueError("no wrong axial pair exists (k=%d)" % k)
    return pairs[int(rng.integers(len(pairs)))]


def grasp_region_combos(t_cands, k_coarse=K_COARSE, k_fine=K_FINE, seed=0):
    """Enumerate every realizable (phi4, phi5) grasp-region combo with its
    point set, reproducing pipeline.oracle.grasp_region_choice's clustering
    (same pixel-space k-means, same seed, same fine-region fallback)."""
    pix = np.asarray(t_cands.pixels, dtype=np.float64)
    pts = np.asarray(t_cands.points3d, dtype=np.float64)
    k_c = min(k_coarse, pts.shape[0])
    _, labels = kmeans(pix, k=k_c, seed=seed)
    combos = []  # (phi4, phi5, region_pts)
    for c in range(k_c):
        m = labels == c
        sub_pts, sub_pix = pts[m], pix[m]
        if sub_pts.shape[0] == 0:
            continue
        k_f = min(k_fine, sub_pts.shape[0])
        if k_f >= 2:
            _, sl = kmeans(sub_pix, k=k_f, seed=seed)
            for f in range(k_f):
                rm = sl == f
                if rm.any():
                    combos.append((c, f, sub_pts[rm]))
        else:
            combos.append((c, 0, sub_pts))
    return combos


def wrong_grasp_region(t_cands, correct, rng, seed=0):
    """Uniform draw over non-empty (phi4, phi5) combos != `correct`.

    Returns (phi4, phi5, centroid, region_pts, changed). If the object has
    only one realizable region, the correct one is kept (changed=False)."""
    combos = grasp_region_combos(t_cands, seed=seed)
    wrong = [x for x in combos if (x[0], x[1]) != tuple(correct)]
    if not wrong:
        keep = [x for x in combos if (x[0], x[1]) == tuple(correct)]
        c, f, region = keep[0]
        return int(c), int(f), region.mean(axis=0), region, False
    c, f, region = wrong[int(rng.integers(len(wrong)))]
    return int(c), int(f), region.mean(axis=0), region, True


def corrupt_target_choice(orc, t_cands, rng, kmeans_seed=0):
    """Corrupt orc's target-side discrete answers (joint injection).

    Returns (orc_new_or_None, record).  orc_new is None iff the corrupted
    endpoint pair produced a degenerate halfplane split (pipeline failure).
    The input orc is not mutated.
    """
    tc = dict(orc["target_choice"])
    rec = {"oracle_target_choice": dict(tc)}

    k = int(np.asarray(t_cands.candidates3d).shape[0])
    t1n, t2n = wrong_axial_pair(k, (tc["phi1"], tc["phi2"]), rng)
    phi3_flip = bool(rng.random() < 0.5)
    phi3n = int(tc["phi3"]) ^ int(phi3_flip)
    phi4n, phi5n, g_t, region_pts, phi45_changed = wrong_grasp_region(
        t_cands, (tc["phi4"], tc["phi5"]), rng, seed=kmeans_seed)

    new_tc = {"phi1": int(t1n), "phi2": int(t2n), "phi3": int(phi3n),
              "phi4": int(phi4n), "phi5": int(phi5n)}
    rec["injected_target_choice"] = new_tc
    rec["phi3_flipped"] = phi3_flip
    rec["phi45_changed"] = phi45_changed
    rec["corrupted_fields"] = (["phi1", "phi2"]
                               + (["phi3"] if phi3_flip else [])
                               + (["phi4", "phi5"] if phi45_changed else []))
    try:
        alk0 = alk_from_candidates(t_cands, t1n, t2n, phi3=False,
                                   depth_consistent=orc["depth_consistent"])
    except ValueError as e:  # corrupted pair -> degenerate split -> failure
        rec["alk_degenerate"] = True
        rec["error"] = str(e)
        return None, rec
    alk = np.asarray(alk0).copy()
    if phi3n:
        alk[[2, 3]] = alk[[3, 2]]

    out = dict(orc)
    out["target_choice"] = new_tc
    out["target_alk"] = alk
    out["target_grasp_centroid"] = np.asarray(g_t)
    out["target_grasp_region_points"] = np.asarray(region_pts)
    return out, rec


def make_injector(task, seed, rho, kmeans_seed=0):
    """Injector callable for retarget_runner.run_pair(inject=...).

    Signature expected by the hook: inject(orc, p_demo, p_tgt) ->
    (orc_or_None, record).  The injection coin is drawn HERE (once per
    injector) so the decision is reproducible per (task, seed, rho).
    """
    rng = rng_for(task, seed, rho)
    do_inject = bool(rng.random() < float(rho))

    def inject(orc, p_demo, p_tgt):
        rec = {"rho": float(rho), "injected": do_inject}
        if not do_inject:
            return orc, rec
        out, crec = corrupt_target_choice(orc, p_tgt["cands"], rng,
                                          kmeans_seed=kmeans_seed)
        rec.update(crec)
        return out, rec

    return inject
