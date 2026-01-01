"""B2 -- MOKA-style baseline: VLM marks a pixel, depth lifts it to 3D
(docs/CODE_MAP.md B2).

Reproduces MOKA's core coupling flaw: the (oracle or real) VLM outputs the
grasp/function point directly as PIXEL coordinates on the target image; the
pixel is lifted to 3D by depth backprojection; there is NO registration
refinement (faithful to MOKA's design).  The transferred motion is the demo
relative-pose template anchored at the marked point, i.e.

    T_map = [I | g_lifted - g_demo]   (pure translation, R = I)

so demo waypoint orientations and relative offsets are preserved verbatim and
only the anchor moves.  Any pixel-marking error and any object-yaw change
therefore propagates STRAIGHT into the grasp -- the coupling the paper
observed qualitatively ("a few pixels off and MOKA fails") is made
quantitative here via the controllable pixel noise sigma_px in {0,5,10,20}.

Oracle-mark variant (large-N):
  GT pixel = projection of the T_gt-mapped demo grasp point into the target
  camera, + isotropic Gaussian pixel noise sigma_px (seeded).

Real-VLM variant (small-n):
  pixel-coordinate prompting against an OpenAI-compatible endpoint, same
  client conventions as alkbench.discrete.VLMSolver (env ALK_VLM_BASE_URL /
  ALK_VLM_API_KEY / ALK_VLM_MODEL; defaults target the local Qwen server at
  http://127.0.0.1:8199/v1, model Qwen/Qwen3-VL-32B-Instruct-FP8, key
  "local").  The demo image is sent with the demo grasp point marked, and the
  model must answer with the corresponding (u, v) on the target image.

Adaptation notes (single-object one-shot setting, documented for the paper):
the marked anchor is the GRASP point; a separate "function point" degenerates
to the same anchor for our single-object tasks (the post-action target moves
rigidly with the object), so exactly one point is marked.
"""
import json
import os

import numpy as np

from alkbench import transform_points
from alkbench.candidates import project
from alkbench.discrete import draw_candidate_markup, encode_png
from pipeline import oracle  # pure numpy/scipy (no robosuite)

from baselines import common

VLM_DEFAULTS = {
    "base_url": "http://127.0.0.1:8199/v1",
    "model": "Qwen/Qwen3-VL-32B-Instruct-FP8",
    "api_key": "local",
}


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def world_to_pixel(p_world, K, T_world_cam):
    """World point -> (u, v) pixel in a capture camera (OpenCV axes)."""
    T_cw = np.linalg.inv(np.asarray(T_world_cam, dtype=np.float64))
    p_cam = transform_points(T_cw, np.asarray(p_world,
                                              dtype=np.float64)[None])
    uv = project(p_cam, K[0, 0], K[1, 1], K[0, 2], K[1, 2])[0]
    return uv


def lift_pixel(uv, depth, K, T_world_cam, max_search_px=25):
    """Depth-backproject a marked pixel to a WORLD point.

    The pixel is rounded and clamped to the image; if its depth is invalid
    (<=0 / non-finite), the nearest valid-depth pixel within max_search_px is
    used instead.  Deliberately NOT restricted to the object mask: noise that
    pushes the mark off the object lifts a background/table point -- that is
    precisely MOKA's failure mode and must be preserved.

    Returns (point_world (3,), used_uv (2,), fell_back bool).
    """
    depth = np.asarray(depth, dtype=np.float64)
    h, w = depth.shape
    u = int(np.clip(round(float(uv[0])), 0, w - 1))
    v = int(np.clip(round(float(uv[1])), 0, h - 1))
    fell_back = False
    if not (np.isfinite(depth[v, u]) and depth[v, u] > 0):
        valid = np.isfinite(depth) & (depth > 0)
        vv, uu = np.nonzero(valid)
        if vv.size == 0:
            raise ValueError("no valid depth in image")
        d2 = (uu - u) ** 2 + (vv - v) ** 2
        j = int(np.argmin(d2))
        if d2[j] > max_search_px ** 2:
            raise ValueError("no valid depth within %d px of the mark"
                             % max_search_px)
        u, v = int(uu[j]), int(vv[j])
        fell_back = True
    K = np.asarray(K, dtype=np.float64)
    z = depth[v, u]
    p_cam = np.array([(u - K[0, 2]) / K[0, 0] * z,
                      (v - K[1, 2]) / K[1, 1] * z, z])
    T_wc = np.asarray(T_world_cam, dtype=np.float64)
    p_world = T_wc[:3, :3] @ p_cam + T_wc[:3, 3]
    return p_world, np.array([u, v], dtype=np.float64), fell_back


def demo_grasp_world(demo_capture, ctx):
    """Demo grasp anchor: pre_grasp TCP projected onto the nearest demo
    object point (same definition as pipeline.oracle.demo_grasp_point)."""
    p_d = ctx.percept(demo_capture, "demo")
    g_d, proj_dist = oracle.demo_grasp_point(p_d["cands"],
                                             ctx.keyframe_tcp("pre_grasp"))
    return g_d, proj_dist, p_d


def translation_map(g_demo, g_target):
    T = np.eye(4)
    T[:3, 3] = np.asarray(g_target, dtype=np.float64) - np.asarray(
        g_demo, dtype=np.float64)
    return T


def _finish(variant, ctx, target_capture, g_d, uv_gt, uv_marked, aux):
    """Shared tail: lift the marked pixel, build the translation-only T_map."""
    p_t = ctx.percept(target_capture, "target")
    cd = target_capture["cameras"][p_t["camera"]]
    g_lift, uv_used, fell_back = lift_pixel(uv_marked, cd["depth"],
                                            cd["K"], cd["T_world_cam"])
    T_map = translation_map(g_d, g_lift)
    result = common.base_result(
        variant, T_map,
        demo_grasp=np.asarray(g_d).tolist(),
        target_grasp=g_lift.tolist(),
        uv_marked=[float(x) for x in np.asarray(uv_marked)],
        uv_used=uv_used.tolist(),
        lift_fell_back=fell_back,
        camera=p_t["camera"],
    )
    if uv_gt is not None:
        g_gt = transform_points(ctx.T_gt, np.asarray(g_d)[None])[0]
        result["uv_gt"] = [float(x) for x in uv_gt]
        result["pixel_err_px"] = float(
            np.linalg.norm(np.asarray(uv_marked) - np.asarray(uv_gt)))
        result["anchor_err_m"] = float(np.linalg.norm(g_lift - g_gt))
    result.update(aux)
    return result


# ---------------------------------------------------------------------------
# oracle-mark variant
# ---------------------------------------------------------------------------

def run_moka_oracle(demo_capture, target_capture, ctx, sigma_px=None):
    """Oracle-marked MOKA-style transfer with controllable pixel noise.

    sigma_px defaults to ctx.sigma_px; the noise RNG is seeded with
    ctx.noise_seed so large-N runs are reproducible.
    """
    T_gt = ctx.require_T_gt("moka_oracle")
    sigma = ctx.sigma_px if sigma_px is None else float(sigma_px)
    g_d, proj_dist, _ = demo_grasp_world(demo_capture, ctx)
    p_t = ctx.percept(target_capture, "target")
    g_t_gt = transform_points(T_gt, g_d[None])[0]
    cd = target_capture["cameras"][p_t["camera"]]
    uv_gt = world_to_pixel(g_t_gt, cd["K"], cd["T_world_cam"])
    rng = np.random.RandomState(ctx.noise_seed)
    uv_marked = uv_gt + rng.normal(0.0, sigma, size=2)
    return _finish("moka_oracle", ctx, target_capture, g_d, uv_gt, uv_marked,
                   {"sigma_px": sigma,
                    "demo_grasp_projection_dist_m": proj_dist})


# ---------------------------------------------------------------------------
# real-VLM variant (pixel-coordinate prompting)
# ---------------------------------------------------------------------------

_PIXEL_PROMPT = """You see two images of the same type of robot manipulation \
scene.
IMAGE 1 (demonstration): the red circle marks the point where the robot \
grasped the object.
IMAGE 2 (new scene): the same object is at a different position/orientation.

Task: {task}

Output the pixel coordinates of the SAME grasp point on the object in \
IMAGE 2. The image is {w} pixels wide (u: 0..{w1}, left to right) and {h} \
pixels tall (v: 0..{h1}, top to bottom).

Answer with ONLY a JSON object, no other text, e.g.: {{"u": 123, "v": 45}}"""

TASK_DESCRIPTIONS = {
    "nut_loosen": "grasp the round nut by its handle bar and lift it",
    "rim_grasp": "grasp the can near its top rim and lift it",
    "pour": "grasp the elongated block and pour (large wrist rotation)",
    "box_open": "grasp the door handle and swing the door open",
    "cap_twist": "grasp the square nut by its handle and twist it in place",
}


def _parse_uv(text, w, h):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in reply")
    obj = json.loads(text[start:end + 1])
    u, v = float(obj["u"]), float(obj["v"])
    if not (0 <= u < w and 0 <= v < h):
        raise ValueError("(u, v) out of image bounds")
    return np.array([u, v])


def query_vlm_pixel(demo_rgb, demo_uv, target_rgb, task, model=None,
                    base_url=None, api_key=None, max_retries=1,
                    temperature=0.0):
    """Ask an OpenAI-compatible VLM for the grasp pixel on the target image.
    Same client conventions as alkbench.discrete.VLMSolver (lazy openai
    import, env-var config, JSON-only answer with one validation retry)."""
    import base64
    import openai  # lazy

    model = model or os.environ.get("ALK_VLM_MODEL", VLM_DEFAULTS["model"])
    base_url = base_url or os.environ.get("ALK_VLM_BASE_URL",
                                          VLM_DEFAULTS["base_url"])
    api_key = api_key or os.environ.get("ALK_VLM_API_KEY",
                                        VLM_DEFAULTS["api_key"])
    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    marked = draw_candidate_markup(demo_rgb, np.asarray(demo_uv)[None],
                                   radius=6)
    h, w = np.asarray(target_rgb).shape[:2]
    prompt = _PIXEL_PROMPT.format(task=TASK_DESCRIPTIONS.get(task, task),
                                  w=w, h=h, w1=w - 1, h1=h - 1)

    def img_part(arr):
        b64 = base64.b64encode(encode_png(arr)).decode("ascii")
        return {"type": "image_url",
                "image_url": {"url": "data:image/png;base64," + b64}}

    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        img_part(marked),
        img_part(np.asarray(target_rgb, dtype=np.uint8)),
    ]}]
    last_err = None
    for _ in range(1 + max_retries):
        resp = client.chat.completions.create(model=model, messages=messages,
                                              temperature=temperature)
        text = resp.choices[0].message.content
        try:
            return _parse_uv(text, w, h), text
        except (ValueError, KeyError, TypeError) as e:
            last_err = e
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content":
                             "Invalid reply (%s). Answer with ONLY the JSON "
                             "object {\"u\": ..., \"v\": ...}." % e})
    raise RuntimeError("query_vlm_pixel: invalid reply after retries: %s"
                       % last_err)


def run_moka_vlm(demo_capture, target_capture, ctx):
    """Real-VLM MOKA-style transfer (pixel-coordinate prompting)."""
    g_d, proj_dist, p_d = demo_grasp_world(demo_capture, ctx)
    p_t = ctx.percept(target_capture, "target")
    d_cd = demo_capture["cameras"][p_d["camera"]]
    demo_uv = world_to_pixel(g_d, d_cd["K"], d_cd["T_world_cam"])
    demo_rgb = common.load_rgb(demo_capture, p_d["camera"])
    target_rgb = common.load_rgb(target_capture, p_t["camera"])
    uv_marked, raw_reply = query_vlm_pixel(
        demo_rgb, demo_uv, target_rgb, ctx.task,
        model=ctx.vlm_model, base_url=ctx.vlm_base_url,
        api_key=ctx.vlm_api_key)
    uv_gt = None
    if ctx.T_gt is not None:  # scoring info when GT is available
        g_t_gt = transform_points(ctx.T_gt, g_d[None])[0]
        t_cd = target_capture["cameras"][p_t["camera"]]
        uv_gt = world_to_pixel(g_t_gt, t_cd["K"], t_cd["T_world_cam"])
    return _finish("moka_vlm", ctx, target_capture, g_d, uv_gt, uv_marked,
                   {"vlm_raw_reply": raw_reply,
                    "demo_grasp_projection_dist_m": proj_dist})
