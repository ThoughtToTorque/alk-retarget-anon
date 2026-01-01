"""Campaign I -- (1) coordinate-convention-corrected `coord_vlm` re-run,
                  (2) adaptive-stage no-op verification on well-conditioned
                      tasks (see docs/REPRODUCE.md).

This module is NEW.  `baselines/moka_style.py`, `alkbench/*` and every earlier
campaign runner are left untouched; the corrected variants are registered as
extra entries in `baselines.runner_hooks.REGISTRY` at import time and executed
through the SAME `pipeline.campaign.run_task` path as Campaign D's
`moka_vlm`, so the scene pairs, perception, anchoring, waypoint retarget,
execution and success checker are bit-for-bit the same code.

Job 1 background
----------------
Campaign D reports `coord_vlm` (registry name `moka_vlm`, real-VLM pixel
regression) at 0/120 and diagnoses the failure partly as the model answering
"in its internal resized-canvas coordinate space".  Step (a) of this campaign
measures the convention EMPIRICALLY instead of assuming it
(results/campaign_i/probe/):

  * Qwen3-VL's preprocessor uses patch 16 x merge 2 -> factor 32, min_pixels
    65536, max_pixels 16777216.  A 256x256 render is 65536 px and a multiple
    of 32, so smart_resize is the IDENTITY on our images -- the
    "resized-canvas" explanation cannot be right at 256 px.
  * Presenting the SAME image content at 256 / 512 / 768 px, the replies are
    invariant to image size (median ratio 1.14 / 1.16 where the
    absolute-pixel convention predicts 2.0 / 3.0).  Regressed against the
    0-1000 normalized grid: slope 1.008 / 1.018, r = 0.999 / 1.000.

  => the model answers on a **size-independent 0-1000 normalized grid**
     (Qwen relative coordinates), not in pixels and not in a resized canvas.

The corrected path therefore (i) asks on that grid, (ii) bounds-checks on that
grid, (iii) retries once with the image size and the grid restated, and
(iv) maps back to true pixels:  u_px = u/1000 * (W-1),  v_px = v/1000 * (H-1).

Two corrected arms are run (both frozen on NON-EVAL data before the first
eval query -- see freeze phase):

  moka_vlm_fix     minimal fix: only the convention/bounds/retry change;
                   the demo markup is the untouched
                   `alkbench.discrete.draw_candidate_markup(..., zoom=True)`
                   crop that Campaign D sent.
  moka_vlm_fixctx  steelman: the same convention fix PLUS a demo image that
                   shows the whole scene with the grasp point ringed.  (The
                   zoom=True markup crops a 48x48 window around the point and
                   upscales it 7x, so Campaign D's "demonstration" image did
                   not actually show the object in context -- a second
                   confound a reviewer could raise.  The steelman removes it.)

Registry names start with "moka" on purpose: `pipeline.campaign` keys the
marked-anchor retarget path off that prefix, so the corrected arms keep
Campaign D's anchoring semantics exactly.

Phases
------
  probe    -- the convention experiments (results/campaign_i/probe/*.py)
  freeze   -- prompt/decoding selection on NON-EVAL scenes (easy tier
              data/<task>/{1..5}); writes results/campaign_i/freeze.json.
              Nothing after this point is tuned.
  execute  -- the eval arms on the Campaign D grid (4 tasks x seeds
              1000..1029, medium tier, data_medium/).  Needs MUJOCO_GL=egl.
  job2     -- adaptive-stage no-op check (selective widening without the
              slender depth prior) on the four well-conditioned tasks.
  report   -- tables + summary.
"""
import argparse
import base64
import json
import os
import sys
import time

import numpy as np

SIMBENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SIMBENCH not in sys.path:
    sys.path.insert(0, SIMBENCH)

from alkbench import transform_points                       # noqa: E402
from alkbench.discrete import draw_candidate_markup, encode_png  # noqa: E402
from baselines import common as bc                          # noqa: E402
from baselines import moka_style as ms                      # noqa: E402
from baselines import runner_hooks as hooks                 # noqa: E402
from pipeline import oracle                                 # noqa: E402

DATA_ROOT = os.path.join(SIMBENCH, "data_medium")
EASY_DATA_ROOT = os.path.join(SIMBENCH, "data")
OUT_ROOT = os.path.join(SIMBENCH, "results", "campaign_i")
CAMPAIGN_D_ROOT = os.path.join(SIMBENCH, "results", "campaign_d")
TABLES_DIR = os.path.join(SIMBENCH, "results", "tables")

TASKS = ("nut_loosen", "cap_twist", "rim_grasp", "box_open")
SEEDS = tuple(range(1000, 1030))
DEV_SEEDS = (1, 2, 3, 4, 5)          # easy tier, disjoint from the eval seeds

GRID = 1000.0                        # Qwen relative-coordinate grid


# ---------------------------------------------------------------------------
# prompts (see freeze phase; FROZEN before any eval query)
# ---------------------------------------------------------------------------

# Campaign D's prompt, verbatim from baselines.moka_style._PIXEL_PROMPT.
PROMPT_ORIG = ms._PIXEL_PROMPT

# Convention-explicit prompt: asks on the grid the model actually answers on,
# and still states the true image size (the reviewer's requested change).
PROMPT_NORM = """You see two images of the same type of robot manipulation \
scene.
IMAGE 1 (demonstration): the red circle marks the point where the robot \
grasped the object.
IMAGE 2 (new scene): the same object is at a different position/orientation.

Task: {task}

Output the coordinates of the SAME grasp point on the object in IMAGE 2. \
IMAGE 2 is {w} pixels wide and {h} pixels tall. Give the coordinates as \
RELATIVE coordinates normalised to a 0-1000 grid over IMAGE 2: x = 0 is the \
left edge, x = 1000 the right edge, y = 0 the top edge, y = 1000 the bottom \
edge.

Answer with ONLY a JSON object, no other text, e.g.: {{"u": 123, "v": 45}}"""

RETRY_ORIG = ('Invalid reply (%s). IMAGE 2 is %d pixels wide (u: 0..%d) and '
              '%d pixels tall (v: 0..%d). Answer with ONLY the JSON object '
              '{"u": ..., "v": ...} with u and v inside those ranges.')
RETRY_NORM = ('Invalid reply (%s). IMAGE 2 is %d pixels wide and %d pixels '
              'tall; answer in RELATIVE coordinates on the 0-1000 grid '
              '(0 = left/top edge, 1000 = right/bottom edge). Answer with '
              'ONLY the JSON object {"u": ..., "v": ...} with both values '
              'between 0 and 1000.')


def decode_uv(u, v, w, h, decode):
    """Map a reply onto true pixel coordinates."""
    if decode == "abs":                       # Campaign D behaviour
        return np.array([u, v], dtype=np.float64)
    if decode == "norm1000":
        return np.array([u / GRID * (w - 1), v / GRID * (h - 1)],
                        dtype=np.float64)
    raise ValueError(decode)


def _parse_json_uv(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
    s, e = t.find("{"), t.rfind("}")
    if s < 0 or e < 0:
        raise ValueError("no JSON object in reply")
    obj = json.loads(t[s:e + 1])
    return float(obj["u"]), float(obj["v"])


def query_pixel(demo_img, target_rgb, task, prompt_kind="norm",
                decode="norm1000", model=None, base_url=None, api_key=None,
                max_retries=1, temperature=0.0):
    """One coordinate query with a bounds check on the DECLARED grid and one
    retry that restates the image size (and the grid, when the grid is what
    was asked for).  Returns (uv_px, info)."""
    import openai  # lazy, same convention as alkbench.discrete.VLMSolver

    model = model or os.environ.get("ALK_VLM_MODEL", ms.VLM_DEFAULTS["model"])
    base_url = base_url or os.environ.get("ALK_VLM_BASE_URL",
                                          ms.VLM_DEFAULTS["base_url"])
    api_key = api_key or os.environ.get("ALK_VLM_API_KEY",
                                        ms.VLM_DEFAULTS["api_key"])
    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    h, w = np.asarray(target_rgb).shape[:2]
    if prompt_kind == "orig":
        prompt = PROMPT_ORIG.format(task=ms.TASK_DESCRIPTIONS.get(task, task),
                                    w=w, h=h, w1=w - 1, h1=h - 1)
        hi_u, hi_v = w - 1.0, h - 1.0
    elif prompt_kind == "norm":
        prompt = PROMPT_NORM.format(task=ms.TASK_DESCRIPTIONS.get(task, task),
                                    w=w, h=h)
        hi_u = hi_v = GRID
    else:
        raise ValueError(prompt_kind)

    def part(arr):
        b64 = base64.b64encode(encode_png(arr)).decode("ascii")
        return {"type": "image_url",
                "image_url": {"url": "data:image/png;base64," + b64}}

    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        part(np.asarray(demo_img, dtype=np.uint8)),
        part(np.asarray(target_rgb, dtype=np.uint8)),
    ]}]
    replies = []
    last_err = None
    for attempt in range(1 + max_retries):
        resp = client.chat.completions.create(model=model, messages=messages,
                                              temperature=temperature)
        text = resp.choices[0].message.content
        replies.append(text)
        try:
            u, v = _parse_json_uv(text)
            if not (0 <= u <= hi_u and 0 <= v <= hi_v):
                raise ValueError("(u, v) outside the declared range")
            return decode_uv(u, v, w, h, decode), {
                "replies": replies, "attempts": attempt + 1,
                "u_raw": u, "v_raw": v, "prompt_kind": prompt_kind,
                "decode": decode, "img_w": w, "img_h": h}
        except (ValueError, KeyError, TypeError) as e:
            last_err = e
            messages.append({"role": "assistant", "content": text})
            retry = (RETRY_ORIG % (e, w, w - 1, h, h - 1)
                     if prompt_kind == "orig" else RETRY_NORM % (e, w, h))
            messages.append({"role": "user", "content": retry})
    raise RuntimeError("query_pixel: invalid reply after retries: %s (%r)"
                       % (last_err, replies[-1]))


# ---------------------------------------------------------------------------
# demo image variants
# ---------------------------------------------------------------------------

def demo_image_crop(demo_rgb, demo_uv):
    """Campaign D's demo image, verbatim (zoom=True 48x48 crop, 7x upscale)."""
    return draw_candidate_markup(demo_rgb, np.asarray(demo_uv)[None],
                                 radius=6)


def demo_image_context(demo_rgb, demo_uv):
    """Steelman: the full demo scene with the grasp point ringed in red."""
    img = np.array(demo_rgb, dtype=np.uint8, copy=True)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = float(demo_uv[0]), float(demo_uv[1])
    d2 = (xx - u) ** 2 + (yy - v) ** 2
    img[(d2 <= 81) & (d2 > 49)] = (255, 255, 255)
    img[(d2 <= 49) & (d2 > 16)] = (255, 0, 0)
    return img


DEMO_IMAGE = {"crop": demo_image_crop, "context": demo_image_context}


# ---------------------------------------------------------------------------
# the corrected coord_vlm method (registry entry)
# ---------------------------------------------------------------------------

def _run_coord_vlm(demo_capture, target_capture, ctx, variant, prompt_kind,
                   decode, demo_img_kind):
    """Same body as baselines.moka_style.run_moka_vlm, with the convention
    handling swapped in."""
    g_d, proj_dist, p_d = ms.demo_grasp_world(demo_capture, ctx)
    p_t = ctx.percept(target_capture, "target")
    d_cd = demo_capture["cameras"][p_d["camera"]]
    demo_uv = ms.world_to_pixel(g_d, d_cd["K"], d_cd["T_world_cam"])
    demo_rgb = bc.load_rgb(demo_capture, p_d["camera"])
    target_rgb = bc.load_rgb(target_capture, p_t["camera"])
    demo_img = DEMO_IMAGE[demo_img_kind](demo_rgb, demo_uv)
    uv_marked, info = query_pixel(
        demo_img, target_rgb, ctx.task, prompt_kind=prompt_kind,
        decode=decode, model=ctx.vlm_model, base_url=ctx.vlm_base_url,
        api_key=ctx.vlm_api_key)
    uv_gt = None
    if ctx.T_gt is not None:
        g_t_gt = transform_points(ctx.T_gt, g_d[None])[0]
        t_cd = target_capture["cameras"][p_t["camera"]]
        uv_gt = ms.world_to_pixel(g_t_gt, t_cd["K"], t_cd["T_world_cam"])
    return ms._finish(variant, ctx, target_capture, g_d, uv_gt, uv_marked,
                      {"vlm_raw_reply": info["replies"][-1],
                       "vlm_info": info,
                       "demo_uv": [float(x) for x in demo_uv],
                       "demo_grasp_projection_dist_m": proj_dist})


ARMS = {
    # frozen configuration of each eval arm (see freeze phase)
    "moka_vlm_fix": dict(prompt_kind="norm", decode="norm1000",
                         demo_img_kind="crop"),
    "moka_vlm_fixctx": dict(prompt_kind="norm", decode="norm1000",
                            demo_img_kind="context"),
}

for _name, _kw in ARMS.items():
    if _name not in hooks.REGISTRY:
        hooks.register(_name)(
            (lambda n, k: lambda d, t, c: _run_coord_vlm(d, t, c, n, **k))(
                _name, _kw))


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def _json_safe(x):
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json_safe(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(_json_safe(obj), f, indent=1)
    os.replace(path + ".tmp", path)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# phase: freeze  (prompt/decoding selection on NON-EVAL scenes)
# ---------------------------------------------------------------------------

DEV_VARIANTS = [
    # name,            prompt_kind, decode,      demo image
    ("D_orig_abs",     "orig", "abs",      "crop"),      # Campaign D control
    ("orig_norm1000",  "orig", "norm1000", "crop"),
    ("norm_norm1000",  "norm", "norm1000", "crop"),
    ("norm_ctx",       "norm", "norm1000", "context"),
]


def _dev_pair(task, seed):
    from pipeline import retarget_runner as rr
    pdir = os.path.join(EASY_DATA_ROOT, task, str(seed))
    pair = _read_json(os.path.join(pdir, "pair.json"))
    demo_cap = bc.load_capture(os.path.join(
        rr.demo_dir_for(task, EASY_DATA_ROOT), "scene"))
    target_cap = bc.load_capture(os.path.join(pdir, "target", "scene"))
    demo = rr.load_demo(task, EASY_DATA_ROOT)
    inst = pair["target_instance"]
    T_gt = oracle.gt_relative_transform(pair["demo_object_poses"][inst],
                                        pair["target_object_poses"][inst])
    ctx = bc.MethodContext(task=task, T_gt=T_gt, keyframes=demo["keyframes"],
                           waypoints=demo["waypoints"], sigma_px=0.0,
                           noise_seed=seed, random4_seed=seed,
                           sensor_noise=None)
    return demo_cap, target_cap, ctx


def freeze(tasks=TASKS, seeds=DEV_SEEDS, out_root=OUT_ROOT):
    """Evaluate the candidate prompt/decoding variants on NON-EVAL scenes.

    Metric: pixel error of the marked point against the projected
    ground-truth grasp point, and the 3D anchor error after depth lifting.
    NO eval scene (medium tier, seeds 1000..1029) is touched here."""
    rows = []
    for task in tasks:
        for seed in seeds:
            try:
                demo_cap, target_cap, ctx = _dev_pair(task, seed)
            except Exception as e:
                print("skip %s/%d: %r" % (task, seed, e))
                continue
            for name, pk, dec, dik in DEV_VARIANTS:
                t0 = time.time()
                rec = {"task": task, "seed": seed, "variant": name}
                try:
                    r = _run_coord_vlm(demo_cap, target_cap, ctx, name, pk,
                                       dec, dik)
                    rec.update({
                        "ok": True,
                        "pixel_err_px": r.get("pixel_err_px"),
                        "anchor_err_m": r.get("anchor_err_m"),
                        "uv_marked": r.get("uv_marked"),
                        "uv_gt": r.get("uv_gt"),
                        "u_raw": r["vlm_info"]["u_raw"],
                        "v_raw": r["vlm_info"]["v_raw"],
                        "attempts": r["vlm_info"]["attempts"]})
                except Exception as e:
                    rec.update({"ok": False, "error": repr(e)})
                rec["t"] = round(time.time() - t0, 2)
                rows.append(rec)
                print("[dev %-11s %d %-14s] ok=%s pix=%s anchor=%s (%.1fs)"
                      % (task, seed, name, rec["ok"],
                         "%.1f" % rec["pixel_err_px"]
                         if rec.get("pixel_err_px") is not None else "-",
                         "%.3f" % rec["anchor_err_m"]
                         if rec.get("anchor_err_m") is not None else "-",
                         rec["t"]), flush=True)
    summary = {}
    for name, _, _, _ in DEV_VARIANTS:
        rs = [r for r in rows if r["variant"] == name]
        ok = [r for r in rs if r["ok"]]
        pix = [r["pixel_err_px"] for r in ok if r["pixel_err_px"] is not None]
        anc = [r["anchor_err_m"] for r in ok if r["anchor_err_m"] is not None]
        summary[name] = {
            "n": len(rs), "n_ok": len(ok),
            "n_query_failed": len(rs) - len(ok),
            "median_pixel_err_px": float(np.median(pix)) if pix else None,
            "median_anchor_err_mm": 1e3 * float(np.median(anc)) if anc else None,
            "frac_anchor_within_20mm": (float(np.mean(np.asarray(anc) < 0.02))
                                        if anc else None),
        }
        print("VARIANT %-15s ok %d/%d  median pix %s px  median anchor %s mm"
              % (name, len(ok), len(rs),
                 "%.1f" % summary[name]["median_pixel_err_px"]
                 if pix else "-",
                 "%.0f" % summary[name]["median_anchor_err_mm"]
                 if anc else "-"))
    _write_json(os.path.join(out_root, "freeze.json"),
                {"dev_data_root": EASY_DATA_ROOT, "dev_seeds": list(seeds),
                 "variants": [list(v) for v in DEV_VARIANTS],
                 "rows": rows, "summary": summary,
                 "prompt_orig": PROMPT_ORIG, "prompt_norm": PROMPT_NORM,
                 "arms_frozen": ARMS})
    return summary


# ---------------------------------------------------------------------------
# phase: execute
# ---------------------------------------------------------------------------

# `baselines.difficulty` was edited on 2026-08-16 (Campaign A2): box_open's
# medium-tier yaw window was narrowed from [-112.92, -81.41] to
# [-112.92, -103.0] deg.  The execution path re-samples the target placement
# from the CURRENT sampler (`envs.reset_with_seed`), so with the new window
# the executed box_open scene no longer matches the capture stored in
# `data_medium/box_open/<seed>/` (verified: identical positions, yaw off by
# 3-9 deg) and T_gt read from pair.json would be wrong.  Campaign I job 1
# must reproduce Campaign D's scenes exactly, so the SUPERSEDED window is
# restored in-process for the duration of the run.  Nothing is written to
# baselines/difficulty.py and no A2 file is touched.
BOX_OPEN_YAW_CAMPAIGN_D = (-np.pi / 2.0 - 0.40, -np.pi / 2.0 + 0.15)


def _restore_campaign_d_box_open_window():
    from baselines import difficulty
    cur = difficulty.MEDIUM_SAMPLER["box_open"]["rotation"]
    difficulty.MEDIUM_SAMPLER["box_open"]["rotation"] = BOX_OPEN_YAW_CAMPAIGN_D
    print("[campaign_i] box_open medium yaw window restored to the "
          "Campaign-D value %s (was %s)"
          % (np.round(np.rad2deg(BOX_OPEN_YAW_CAMPAIGN_D), 2),
             np.round(np.rad2deg(cur), 2)), flush=True)


def verify_scene_reproduction(task, seeds, data_root=DATA_ROOT, tol=1e-6):
    """Guard: the placement `envs.reset_with_seed` reproduces must equal the
    pose recorded in pair.json, otherwise T_gt does not describe the scene
    that is actually executed."""
    from pipeline import campaign as cg
    from simtasks import capture, envs
    env = cg.make_tier_env(task, "medium")
    bad = []
    try:
        for seed in seeds:
            pair = _read_json(os.path.join(data_root, task, str(seed),
                                           "pair.json"))
            inst = pair["target_instance"]
            envs.reset_with_seed(env, pair["target_seed"])
            now = capture.get_object_poses(env)[inst]
            ref = pair["target_object_poses"][inst]
            d = max(np.abs(np.asarray(now["pos"]) - ref["pos"]).max(),
                    np.abs(np.asarray(now["quat_xyzw"])
                           - ref["quat_xyzw"]).max())
            if d > tol:
                bad.append((seed, float(d)))
    finally:
        env.close()
    return bad


def execute(tasks=TASKS, seeds=SEEDS, methods=("moka_vlm_fix",),
            out_root=OUT_ROOT, data_root=DATA_ROOT, verify=True):
    from pipeline import campaign as cg
    if "box_open" in tasks:
        _restore_campaign_d_box_open_window()
    if verify:
        for task in tasks:
            bad = verify_scene_reproduction(task, seeds, data_root)
            print("[campaign_i] scene-reproduction check %-11s: %s"
                  % (task, "OK (%d/%d)" % (len(seeds), len(seeds)) if not bad
                     else "MISMATCH %r" % bad[:5]), flush=True)
            if bad:
                raise RuntimeError(
                    "scene reproduction mismatch for %s: %r" % (task, bad[:5]))
    n = 0
    for task in tasks:
        n += cg.run_task(task, list(seeds), list(methods), "medium",
                         data_root, os.path.join(out_root, "coord_vlm"))
    return n


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--phase", required=True,
                   choices=["freeze", "execute"])
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--methods", nargs="*", default=["moka_vlm_fix"])
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--seed-end", type=int, default=1029)
    a = p.parse_args(argv)
    if a.phase == "freeze":
        freeze(a.tasks)
    else:
        execute(a.tasks, tuple(range(a.seed_start, a.seed_end + 1)),
                tuple(a.methods))


if __name__ == "__main__":
    main()
