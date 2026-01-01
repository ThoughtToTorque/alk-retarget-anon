"""ReKep keypoint proposal (Huang et al., 2024, Sec. III-A) -- DINOv2 worker.

Faithful re-implementation of ReKep's keypoint proposal on our captures:

  1. DINOv2 (ViT-S/14, the checkpoint ReKep uses) patch features of the RGB
     image, bilinearly upsampled to pixel resolution;
  2. k-means (k = num_candidates_per_mask, fixed seed) on the L2-normalised
     features of the masked object pixels;
  3. one candidate per cluster: the masked pixel whose feature is nearest the
     cluster centre;
  4. candidates projected to 3D through the depth image (world frame);
  5. merging: candidates closer than `min_dist_bt_keypoints` (3D metres) are
     collapsed, keeping the candidate of the larger cluster (ReKep's
     mean-shift-style deduplication).

Differences from ReKep's own keypoint_proposal.py, declared in the campaign
adaptation ledger (docs/REPRODUCE.md):
  * masks come from the benchmark's ground-truth instance segmentation (the
    stand-in for SAM used by EVERY method in this benchmark, campaign J A1);
    only the task's target object is masked, because the benchmark tasks
    manipulate a single object;
  * the input image may be integer-upscaled before DINOv2 (config `upscale`)
    because our renders are 256 px and the object occupies ~40 px (~3x3
    DINOv2 patches at native resolution) -- the analogue of campaign J's A7;
    swept and FROZEN on non-eval scenes;
  * `num_candidates_per_mask` / `min_dist_bt_keypoints` likewise swept on
    non-eval scenes (ReKep verbatim: 5 / 0.06 m).

This module needs torch >= 2 (DINOv2), so it runs as a SUBPROCESS under
vlm_server/.venv (torch 2.13, CPU is enough); the benchmark venv talks to it
through request .npz files:

    python baselines/rekep_keypoint_proposal.py --requests <dir> \
        [--num-candidates 5] [--min-dist 0.06] [--upscale 1]

Every <dir>/*.npz with keys rgb (H,W,3 u8), mask (H,W bool), depth (H,W),
K (3,3), T_world_cam (4,4) produces <same-stem>.json:
    {"pixels": [[u,v]..], "xyz": [[x,y,z]..], "cluster_sizes": [..],
     "config": {...}}
Existing .json outputs are skipped (resumable).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_MODEL = None


def _dino():
    global _MODEL
    if _MODEL is None:
        import torch
        _MODEL = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        _MODEL.eval()
    return _MODEL


def dense_features(rgb, upscale=1):
    """DINOv2 ViT-S/14 patch features, bilinearly upsampled to the ORIGINAL
    pixel grid.  Returns (H, W, 384) float32."""
    import torch
    img = np.asarray(rgb, dtype=np.float32) / 255.0
    if upscale > 1:
        img = np.repeat(np.repeat(img, upscale, 0), upscale, 1)
    h, w = img.shape[:2]
    h14, w14 = (h // 14) * 14, (w // 14) * 14
    x = torch.from_numpy(img).permute(2, 0, 1)[None]
    x = torch.nn.functional.interpolate(x, size=(h14, w14), mode="bilinear",
                                        align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        out = _dino().forward_features(x)["x_norm_patchtokens"]
    ph, pw = h14 // 14, w14 // 14
    f = out.reshape(1, ph, pw, -1).permute(0, 3, 1, 2)
    H, W = np.asarray(rgb).shape[:2]
    f = torch.nn.functional.interpolate(f, size=(H, W), mode="bilinear",
                                        align_corners=False)
    return f[0].permute(1, 2, 0).numpy()


def _kmeans(X, k, seed=0, n_iter=50):
    """Plain seeded k-means (numpy); returns (labels, centers)."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    k = min(k, n)
    # k-means++ style init
    idx = [int(rng.randint(n))]
    d2 = np.linalg.norm(X - X[idx[0]], axis=1) ** 2
    for _ in range(1, k):
        p = d2 / max(d2.sum(), 1e-12)
        idx.append(int(rng.choice(n, p=p)))
        d2 = np.minimum(d2, np.linalg.norm(X - X[idx[-1]], axis=1) ** 2)
    C = X[idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        D = np.linalg.norm(X[:, None, :] - C[None], axis=2)
        new = np.argmin(D, axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                C[j] = X[m].mean(axis=0)
    return labels, C


def backproject(depth, K, T_wc):
    """Per-pixel world coordinates (H, W, 3) and validity mask (H, W)."""
    d = np.asarray(depth, dtype=np.float64)
    H, W = d.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = np.isfinite(d) & (d > 0)
    z = np.where(valid, d, 1.0)
    cam = np.stack([(u - K[0, 2]) / K[0, 0] * z,
                    (v - K[1, 2]) / K[1, 1] * z, z], axis=-1)
    world = cam @ np.asarray(T_wc)[:3, :3].T + np.asarray(T_wc)[:3, 3]
    return world, valid


def propose(rgb, mask, depth, K, T_wc, num_candidates=5, min_dist=0.06,
            upscale=1, seed=0):
    """ReKep keypoint proposal on one masked object.  Returns dict."""
    feats = dense_features(rgb, upscale=upscale)
    world, valid = backproject(depth, K, T_wc)
    m = np.asarray(mask, dtype=bool) & valid
    vs, us = np.nonzero(m)
    if vs.size < num_candidates:
        vs, us = np.nonzero(np.asarray(mask, dtype=bool))
    F = feats[vs, us]
    F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-9)
    labels, C = _kmeans(F, num_candidates, seed=seed)
    k = C.shape[0]
    cand = []
    for j in range(k):
        in_c = np.nonzero(labels == j)[0]
        if in_c.size == 0:
            continue
        d = np.linalg.norm(F[in_c] - C[j], axis=1)
        i = in_c[int(np.argmin(d))]
        u_, v_ = int(us[i]), int(vs[i])
        if not valid[v_, u_]:            # nearest valid masked pixel instead
            dv = (vs - v_) ** 2 + (us - u_) ** 2
            ok = valid[vs, us]
            if not ok.any():
                continue
            i2 = np.nonzero(ok)[0][int(np.argmin(dv[ok]))]
            u_, v_ = int(us[i2]), int(vs[i2])
        cand.append({"uv": (u_, v_), "xyz": world[v_, u_],
                     "size": int(in_c.size)})
    # merge candidates closer than min_dist in 3D, larger cluster wins
    cand.sort(key=lambda c: -c["size"])
    kept = []
    for c in cand:
        if all(np.linalg.norm(np.asarray(c["xyz"]) - np.asarray(o["xyz"]))
               >= min_dist for o in kept):
            kept.append(c)
    # stable presentation order: lexsort by (v, u) like ReKep's display
    kept.sort(key=lambda c: (c["uv"][1], c["uv"][0]))
    return {"pixels": [[int(c["uv"][0]), int(c["uv"][1])] for c in kept],
            "xyz": [[float(x) for x in c["xyz"]] for c in kept],
            "cluster_sizes": [c["size"] for c in kept],
            "config": {"num_candidates": int(num_candidates),
                       "min_dist": float(min_dist),
                       "upscale": int(upscale), "seed": int(seed),
                       "model": "dinov2_vits14"}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--num-candidates", type=int, default=5)
    ap.add_argument("--min-dist", type=float, default=0.06)
    ap.add_argument("--upscale", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    reqs = sorted(glob.glob(os.path.join(a.requests, "*.npz")))
    n_done = 0
    for rp in reqs:
        out = rp[:-4] + ".json"
        if os.path.exists(out):
            continue
        z = np.load(rp)
        rec = propose(z["rgb"], z["mask"], z["depth"], z["K"],
                      z["T_world_cam"], num_candidates=a.num_candidates,
                      min_dist=a.min_dist, upscale=a.upscale, seed=a.seed)
        with open(out + ".tmp", "w") as f:
            json.dump(rec, f)
        os.replace(out + ".tmp", out)
        n_done += 1
        print("[kp] %s -> %d keypoints" % (os.path.basename(rp),
                                           len(rec["pixels"])), flush=True)
    print("done: %d new / %d requests" % (n_done, len(reqs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
