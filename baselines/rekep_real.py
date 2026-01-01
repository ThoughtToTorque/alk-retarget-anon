"""ReKep baseline with a REAL VLM authoring relational keypoint constraints
(campaign L; independent implementation of the published method of Huang et
al., "ReKep: Spatio-Temporal Reasoning of
Relational Keypoint Constraints for Robotic Manipulation", arXiv 2409.01652).

What is faithful to ReKep here
------------------------------
* keypoint proposal: DINOv2 patch features + per-mask k-means + 3D merging
  (baselines/rekep_keypoint_proposal.py, run under the torch>=2 venv);
* the VLM WRITES PYTHON CONSTRAINT FUNCTIONS over keypoints; each returns a
  scalar cost, satisfied when <= 0 (ReKep's convention);
* the generated code is executed in a restricted namespace (numpy only, no
  imports/dunders/file I/O -- ReKep's exec_safe, hardened) with timeouts;
* the pose is found by MINIMIZING the 200x hinge-penalised constraint costs
  (ReKep's subgoal objective) with scipy dual_annealing + SLSQP (ReKep's own
  optimiser pair), grasped/demo keypoints moving rigidly with the
  end-effector (ReKep's "movable keypoints").

What is adapted for the one-shot retargeting protocol (the campaign
docs/REPRODUCE.md prints the full ledger with who-it-favours labels)
--------------------------------------------------------------
* ONE SE(3) solve instead of per-stage subgoal+path problems re-solved
  closed-loop: the benchmark's paired design replays ONE retargeted
  demo-waypoint template through the shared executor (costs ReKep);
* the task is one-shot: the VLM sees the DEMONSTRATION scene (annotated with
  its own proposed keypoints and the demonstrated grasp point) next to the
  TARGET scene, and its constraints tie the transformed demo keypoints /
  end-effector to the target keypoints -- ReKep's movable-keypoint mechanics
  applied across the demo->target pair (favours ReKep: it gets the same
  demonstration every other method gets);
* no collision / reachability / IK terms in the solve: the shared executor
  imposes identical kinematics on every method (neutral).
"""
import json
import os
import re
import signal
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import dual_annealing, minimize
from scipy.spatial.transform import Rotation

from alkbench import transform_points
from alkbench.candidates import project
from alkbench.discrete import _MARKER_COLORS, crop_zoom, encode_png
from pipeline import oracle

from baselines import common
from baselines import moka_marks as mm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIMBENCH = REPO_ROOT   # kept for backwards compatibility of the module API
# The DINOv2 keypoint proposal runs in a SEPARATE interpreter so that torch
# stays an optional dependency of this repository (see docs/VLM_SETUP.md).
# Point ALK_TORCH_PYTHON at an interpreter that has torch installed; the
# default is the current interpreter, which works if you installed the
# [vlm] extra.
TORCH_PY = os.environ.get("ALK_TORCH_PYTHON", sys.executable)
WORKER = os.path.join(REPO_ROOT, "baselines", "rekep_keypoint_proposal.py")

# ReKep-verbatim proposal configuration (sweep + freeze may override)
REKEP_KP_VERBATIM = {"num_candidates": 5, "min_dist": 0.06, "upscale": 1}

UPSCALE = 3          # A7 analogue: 256 -> 768 px markup for VLM legibility


# ---------------------------------------------------------------------------
# keypoint proposal cache (worker runs under the torch venv)
# ---------------------------------------------------------------------------

def kp_cfg_tag(cfg):
    return "c%d_d%03d_u%d" % (cfg["num_candidates"],
                              round(1e3 * cfg["min_dist"]), cfg["upscale"])


def kp_request_stem(task, root_tag, seed, role):
    return "%s__%s__%s__%s" % (task, root_tag, seed, role)


def write_kp_request(cache_dir, stem, capture, ctx):
    """Write one .npz proposal request (rgb, mask, depth, K, T_world_cam).
    The mask is the SAME one every method in this benchmark consumes,
    including MOKA's A13 handle sub-mask on box_open."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, stem + ".npz")
    if os.path.exists(path) or os.path.exists(path[:-4] + ".json"):
        return path
    role = "demo" if stem.endswith("__demo") else "target"
    p = ctx.percept(capture, role)
    mask, cd = mm._object_mask(capture, p, task=ctx.task, subobject=True)
    rgb = common.load_rgb(capture, p["camera"])
    np.savez_compressed(path + ".tmp.npz", rgb=rgb.astype(np.uint8),
                        mask=mask.astype(bool),
                        depth=np.asarray(cd["depth"], dtype=np.float32),
                        K=np.asarray(cd["K"], dtype=np.float64),
                        T_world_cam=np.asarray(cd["T_world_cam"],
                                               dtype=np.float64))
    os.replace(path + ".tmp.npz", path)
    return path


def run_kp_worker(cache_dir, cfg):
    """Run the DINOv2 worker (torch venv, CPU) over all pending requests."""
    cmd = [TORCH_PY, WORKER, "--requests", cache_dir,
           "--num-candidates", str(cfg["num_candidates"]),
           "--min-dist", str(cfg["min_dist"]),
           "--upscale", str(cfg["upscale"])]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""      # GPU is busy serving the VLM
    r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                       timeout=7200)
    if r.returncode != 0:
        raise RuntimeError("kp worker failed:\n%s\n%s"
                           % (r.stdout[-2000:], r.stderr[-2000:]))
    return r.stdout


def load_kp(cache_dir, stem):
    with open(os.path.join(cache_dir, stem + ".json")) as f:
        rec = json.load(f)
    return (np.asarray(rec["pixels"], dtype=np.float64),
            np.asarray(rec["xyz"], dtype=np.float64), rec)


# ---------------------------------------------------------------------------
# markup (same rendering conventions as the other interfaces in this bench)
# ---------------------------------------------------------------------------

def draw_rekep_markup(image, marks_uv, upscale=UPSCALE, radius=9,
                      text_scale=3, grasp_uv=None):
    """Numbered keypoint marks (labels '0'..'k-1', ReKep's integer style)
    on the integer-upscaled image; optional red ring at the demonstrated
    grasp pixel (demo side only)."""
    img = np.array(image, dtype=np.uint8, copy=True)
    s = int(max(1, upscale))
    img = np.ascontiguousarray(np.repeat(np.repeat(img, s, 0), s, 1))
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    if grasp_uv is not None:
        u, v = np.asarray(grasp_uv, dtype=np.float64) * s + s / 2.0
        d2 = (xx - u) ** 2 + (yy - v) ** 2
        rr = radius + 7
        img[(d2 <= (rr + 3) ** 2) & (d2 > rr ** 2)] = (255, 255, 255)
        img[(d2 <= rr ** 2) & (d2 > (rr - 4) ** 2)] = (230, 25, 75)
    uv = np.asarray(marks_uv, dtype=np.float64) * s + s / 2.0
    for i, (u, v) in enumerate(uv):
        color = _MARKER_COLORS[i % len(_MARKER_COLORS)]
        d2 = (xx - u) ** 2 + (yy - v) ** 2
        img[(d2 <= (radius + 2) ** 2) & (d2 > radius ** 2)] = (255, 255, 255)
        img[(d2 <= radius ** 2) & (d2 > (radius - 3) ** 2)] = color
        label = "%d" % i
        tw = 4 * text_scale * len(label)
        tx, ty = int(u) + radius + 4, int(v) - int(2.5 * text_scale)
        if tx + tw + 2 >= w:
            tx = int(u) - radius - tw - 6
        mm._draw_glyphs(img, label, ty + 1, tx + 1, (0, 0, 0),
                        scale=text_scale)
        mm._draw_glyphs(img, label, ty, tx, (255, 255, 255),
                        scale=text_scale)
    return img


def draw_rekep_zoom(image, marks_uv, radius=9, text_scale=3,
                    target_extent=520, grasp_uv=None):
    """Crop-and-upscale companion image (A7b analogue), marks re-drawn."""
    img = np.array(image, dtype=np.uint8, copy=True)
    uv = np.asarray(marks_uv, dtype=np.float64)
    extra = None
    if grasp_uv is not None:
        uv = np.vstack([uv, np.asarray(grasp_uv, dtype=np.float64)[None]])
    img, uv, _ = crop_zoom(img, uv, target_extent=target_extent)
    if grasp_uv is not None:
        extra = uv[-1]
        uv = uv[:-1]
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    if extra is not None:
        d2 = (xx - extra[0]) ** 2 + (yy - extra[1]) ** 2
        rr = radius + 7
        img[(d2 <= (rr + 3) ** 2) & (d2 > rr ** 2)] = (255, 255, 255)
        img[(d2 <= rr ** 2) & (d2 > (rr - 4) ** 2)] = (230, 25, 75)
    for i, (u, v) in enumerate(uv):
        color = _MARKER_COLORS[i % len(_MARKER_COLORS)]
        d2 = (xx - u) ** 2 + (yy - v) ** 2
        img[(d2 <= (radius + 2) ** 2) & (d2 > radius ** 2)] = (255, 255, 255)
        img[(d2 <= radius ** 2) & (d2 > (radius - 3) ** 2)] = color
        label = "%d" % i
        tw = 4 * text_scale * len(label)
        tx, ty = int(u) + radius + 4, int(v) - int(2.5 * text_scale)
        if tx + tw + 2 >= w:
            tx = int(u) - radius - tw - 6
        mm._draw_glyphs(img, label, ty + 1, tx + 1, (0, 0, 0),
                        scale=text_scale)
        mm._draw_glyphs(img, label, ty, tx, (255, 255, 255),
                        scale=text_scale)
    return img


# ---------------------------------------------------------------------------
# prompt (frozen on non-eval data; see results/campaign_l/freeze.json)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
## Instructions
You command a robot arm that must repeat a DEMONSTRATED manipulation in a \
NEW scene.  A rigid-transform solver will replay the demonstrated \
trajectory after moving it by one rigid transform T; your job is to write \
Python constraint functions whose minimisation determines T, in the style \
of relational keypoint constraints.

You are shown {n_images} images:
- Image 1 (DEMONSTRATION scene): the object with its keypoints marked \
0..{kd_max}.  The demonstrated grasp point is circled by a LARGE RED RING.
- Image 2 (TARGET scene): the SAME object at a different position and \
orientation, with ITS OWN keypoints marked 0..{kt_max}.  The two mark sets \
are computed independently, so index i in one image does NOT automatically \
correspond to index i in the other; use what the marks look like and where \
they sit on the object to decide which pairs are the same physical part.\
{zoom_note}

The solver calls your functions with three numpy arrays:
- `end_effector`: shape (3,), the gripper position of the demonstrated \
grasp AFTER applying T.
- `demo_keypoints`: shape ({kd}, 3), the demonstration keypoints AFTER \
applying T (they move rigidly together with the end-effector, like a \
grasped object).
- `target_keypoints`: shape ({kt}, 3), the keypoints of the target scene \
(fixed).
Each function returns one scalar cost; the constraint is satisfied when \
the cost is <= 0 (for "a coincides with b" return \
`np.linalg.norm(a - b) - 0.01`).

Write:
1. one grasp constraint placing `end_effector` at the correct grasp \
location, expressed through `target_keypoints`;
2. alignment constraints tying transformed `demo_keypoints` to their \
corresponding `target_keypoints` -- ONLY for pairs you are confident mark \
the same physical part of the object.  Use at least two well-separated \
pairs so the object's orientation is pinned down, if you can.
Rules: numpy only (as `np`); no imports; no if/while statements; each \
function must be deterministic and finite.

## Task
{instruction}

## Output format (nothing else after the code block):
```python
# reasoning as short comments
grasp_keypoint = <int, index into target_keypoints nearest the correct \
grasp>
num_constraints = <int>
def constraint1(end_effector, demo_keypoints, target_keypoints):
    return ...
def constraint2(end_effector, demo_keypoints, target_keypoints):
    return ...
```"""

ZOOM_NOTE = ("\n- Images 3 and 4: zoomed-in crops of the demonstration and "
             "target objects with the SAME marks, for legibility.")


def build_messages(instruction, demo_img, target_img, kd, kt,
                   demo_zoom=None, target_zoom=None):
    n_img = 2 + (demo_zoom is not None) + (target_zoom is not None)
    prompt = PROMPT_TEMPLATE.format(
        n_images=n_img, kd=kd, kt=kt, kd_max=kd - 1, kt_max=kt - 1,
        zoom_note=(ZOOM_NOTE if demo_zoom is not None else ""),
        instruction=instruction)
    content = [{"type": "text", "text": prompt},
               mm._img_part(demo_img), mm._img_part(target_img)]
    if demo_zoom is not None:
        content.append(mm._img_part(demo_zoom))
    if target_zoom is not None:
        content.append(mm._img_part(target_zoom))
    return [{"role": "user", "content": content}], prompt


def query_constraints(instruction, demo_img, target_img, kd, kt,
                      demo_zoom=None, target_zoom=None, model=None,
                      base_url=None, api_key=None, max_retries=1,
                      temperature=0.0):
    """One constraint-authoring query.  Returns (code_str, info)."""
    cli = mm._client(base_url, api_key)
    messages, prompt = build_messages(instruction, demo_img, target_img,
                                      kd, kt, demo_zoom, target_zoom)
    replies, last = [], None
    for attempt in range(1 + max_retries):
        resp = cli.chat.completions.create(
            model=mm._model(model), messages=messages,
            temperature=temperature if attempt == 0 else 0.3,
            max_tokens=2048)
        text = resp.choices[0].message.content or ""
        replies.append(text)
        try:
            code = extract_code(text)
            return code, {"attempts": attempt + 1, "replies": replies,
                          "prompt": prompt}
        except ValueError as e:
            last = e
    raise ValueError("no usable code after %d attempts (%s)"
                     % (len(replies), last))


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    for blk in reversed(blocks):
        if "def constraint" in blk:
            return blk.strip()
    if "def constraint" in text:      # code emitted without fences
        i = text.find("grasp_keypoint")
        if i < 0:
            i = text.find("def constraint")
        return text[i:].strip()
    raise ValueError("reply contains no constraint code block")


# ---------------------------------------------------------------------------
# sandboxed execution of VLM-authored code (untrusted)
# ---------------------------------------------------------------------------

# word-boundary token bans (checked on comment-stripped code, so prose in
# comments like "evaluate" cannot false-positive an authoring failure)
BANNED_TOKENS = (
    "import", "exec", "eval", "compile", "open", "globals", "locals",
    "getattr", "setattr", "delattr", "vars", "dir", "input", "breakpoint",
    "while", "yield", "class", "subprocess", "socket", "load", "save",
    "savez", "savetxt", "loadtxt", "fromfile", "tofile", "memmap",
    "frombuffer", "genfromtxt", "DataSource")


def _strip_comments(code):
    return "\n".join(line.split("#", 1)[0] for line in code.splitlines())


class _Timeout(Exception):
    pass


class time_limit(object):
    def __init__(self, seconds):
        self.seconds = int(max(1, seconds))

    def __enter__(self):
        def handler(signum, frame):
            raise _Timeout()
        self._old = signal.signal(signal.SIGALRM, handler)
        signal.alarm(self.seconds)

    def __exit__(self, *a):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._old)
        return False


def exec_safe(code):
    """Execute VLM-authored constraint code in a restricted namespace.

    ReKep's own exec_safe bans 'import' and dunders; ours also bans file
    I/O escapes through numpy, while/yield/class, and attribute reflection.
    Only numpy (as np) and a small builtin whitelist are visible.  Returns
    (fns list ordered by name, namespace dict).
    """
    stripped = _strip_comments(code)
    if "__" in stripped:
        raise ValueError("banned token '__' in generated code")
    for bad in BANNED_TOKENS:
        if re.search(r"\b%s\b" % re.escape(bad), stripped):
            raise ValueError("banned token %r in generated code" % bad)
    if len(code) > 20000:
        raise ValueError("generated code too long")
    gvars = {"np": np, "__builtins__": {},
             "abs": abs, "min": min, "max": max, "len": len, "sum": sum,
             "float": float, "int": int, "bool": bool, "round": round,
             "range": range, "enumerate": enumerate, "zip": zip,
             "print": lambda *a, **k: None}
    # single namespace: module-level names (e.g. `grasp_keypoint`) must be
    # visible from inside the generated functions (dev-scene defect fix,
    # freeze.json "exec_namespace_fix" -- a harness bug, not a VLM error)
    with time_limit(5):
        exec(compile(code, "<vlm_constraints>", "exec"), gvars)
    fns = []
    for name in sorted(gvars):
        if re.match(r"^constraint\d+$", name) and callable(gvars[name]):
            fns.append((int(name[10:]), gvars[name]))
    fns.sort()
    gk = gvars.get("grasp_keypoint")
    gk = int(gk) if isinstance(gk, (int, np.integer)) else None
    declared = gvars.get("num_constraints")
    if not isinstance(declared, (int, np.integer)):
        declared = None
    return [f for _, f in fns], {"grasp_keypoint": gk,
                                 "n_constraints": len(fns),
                                 "declared": declared}


def validate_constraints(fns, g_d, demo_kps, target_kps):
    """One test call per function; raises on anything non-finite."""
    with time_limit(5):
        for i, fn in enumerate(fns):
            v = fn(np.asarray(g_d, dtype=np.float64),
                   np.asarray(demo_kps, dtype=np.float64),
                   np.asarray(target_kps, dtype=np.float64))
            v = float(v)
            if not np.isfinite(v):
                raise ValueError("constraint%d returned non-finite" % (i + 1))


# ---------------------------------------------------------------------------
# solver (ReKep subgoal solve collapsed to one SE(3) transform)
# ---------------------------------------------------------------------------

PENALTY = 200.0            # ReKep's constraint penalty weight
SOLVER_MAXFUN = 5000       # ReKep's from-scratch sampling budget
TRANS_BOUND = 0.5          # m, around the centroid-aligning translation


def hinge_cost(fns, T, g_d, demo_kps, target_kps):
    """200 * sum(max(0, cost_i)) at transform T (ReKep's penalty)."""
    R, t = np.asarray(T)[:3, :3], np.asarray(T)[:3, 3]
    ee = R @ np.asarray(g_d) + t
    dk = np.asarray(demo_kps) @ R.T + t
    total = 0.0
    per = []
    for fn in fns:
        try:
            c = float(fn(ee, dk, np.asarray(target_kps)))
        except Exception:
            c = 1e4
        if not np.isfinite(c):
            c = 1e4
        per.append(c)
        total += PENALTY * max(0.0, c)
    return total, per


def solve_constraints(fns, g_d, demo_kps, target_kps, seed=0,
                      maxfun=SOLVER_MAXFUN, time_budget_s=90):
    """Global (dual_annealing) + local (SLSQP) minimisation of the hinge
    objective over SE(3); rotation about the demo-keypoint centroid, so the
    centroid-aligning translation is the origin of the search box."""
    P = np.asarray(demo_kps, dtype=np.float64)
    Q = np.asarray(target_kps, dtype=np.float64)
    g = np.asarray(g_d, dtype=np.float64)
    c_d = P.mean(axis=0)
    t0 = Q.mean(axis=0) - c_d

    def make_T(x):
        R = Rotation.from_euler("xyz", x[:3]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = c_d + t0 + x[3:] - R @ c_d
        return T

    def objective(x):
        total, _ = hinge_cost(fns, make_T(x), g, P, Q)
        return total

    bounds = [(-np.pi, np.pi)] * 3 + [(-TRANS_BOUND, TRANS_BOUND)] * 3
    x_init = np.zeros(6)
    info = {"timeout": False}
    try:
        with time_limit(time_budget_s):
            res = dual_annealing(
                objective, bounds, maxfun=maxfun, seed=seed, x0=x_init,
                minimizer_kwargs={"method": "SLSQP",
                                  "options": {"maxiter": 100}})
            x_best = res.x
            pol = minimize(objective, x_best, method="SLSQP",
                           options={"maxiter": 200, "ftol": 1e-10})
            if np.isfinite(pol.fun) and pol.fun <= res.fun:
                x_best = pol.x
    except _Timeout:
        info["timeout"] = True
        x_best = x_init
    T = make_T(x_best)
    total, per = hinge_cost(fns, T, g, P, Q)
    info.update({"cost_final": float(total), "per_constraint": per,
                 "x": [float(v) for v in x_best]})
    return T, info


# ---------------------------------------------------------------------------
# oracle-constraint arm (upper bound isolating the authoring channel)
# ---------------------------------------------------------------------------

def oracle_constraint_fns(demo_kps, target_kps, g_d, T_gt, tol=0.005):
    """The same constraint STRUCTURE the prompt requests, authored from the
    ground truth: one grasp constraint (end-effector at the target keypoint
    nearest the GT-mapped demo grasp) + one alignment constraint per demo
    keypoint to its GT-NN-matched target keypoint (injective)."""
    P = np.asarray(demo_kps, dtype=np.float64)
    Q = np.asarray(target_kps, dtype=np.float64)
    g_gt = transform_points(np.asarray(T_gt), np.asarray(g_d)[None])[0]
    j_star = int(np.argmin(np.linalg.norm(Q - g_gt, axis=1)))
    # injective greedy matching over the GT-mapped demo keypoints; unlike
    # common.gt_nn_match it tolerates n_demo != n_target (DINOv2 proposals
    # merge to different counts per scene) and simply matches min(n, m) pairs
    M = transform_points(np.asarray(T_gt), P)
    D = np.linalg.norm(M[:, None, :] - Q[None, :, :], axis=2)
    order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
    used_i = np.zeros(P.shape[0], dtype=bool)
    used_j = np.zeros(Q.shape[0], dtype=bool)
    pairs = []
    for i, j in order:
        if used_i[i] or used_j[j]:
            continue
        used_i[i] = used_j[j] = True
        pairs.append((int(i), int(j), float(D[i, j])))
        if len(pairs) == min(P.shape[0], Q.shape[0]):
            break
    fns = [lambda ee, dk, tk, j=j_star: np.linalg.norm(ee - tk[j]) - tol]
    for i, j, _ in pairs:
        fns.append(lambda ee, dk, tk, i=i, j=j:
                   np.linalg.norm(dk[i] - tk[j]) - tol)
    meta = {"grasp_keypoint": j_star,
            "match": [(i, j) for i, j, _ in pairs],
            "match_dist_m": [d for _, _, d in pairs],
            "n_constraints": len(fns)}
    return fns, meta


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def code_diagnostics(code):
    return {
        "n_chars": len(code),
        "uses_demo_keypoints": bool(
            re.search(r"demo_keypoints\s*\[", code)),
        "n_target_indices": len(set(
            re.findall(r"target_keypoints\s*\[\s*(\d+)", code))),
        "n_demo_indices": len(set(
            re.findall(r"demo_keypoints\s*\[\s*(\d+)", code))),
    }


# ---------------------------------------------------------------------------
# correspondence diagnostic (the decisive mechanism measurement)
# ---------------------------------------------------------------------------

PAIR_RE = re.compile(
    r"demo_keypoints\s*\[\s*(\d+)\s*\]\s*-\s*target_keypoints\s*\[\s*(\d+)\s*\]"
    r"|target_keypoints\s*\[\s*(\d+)\s*\]\s*-\s*demo_keypoints\s*\[\s*(\d+)\s*\]")


def declared_pairs(code):
    """(demo_index, target_index) pairs the generated code actually ties
    together in a difference expression.  Comments are stripped first."""
    body = _strip_comments(code)
    out = []
    for m in PAIR_RE.finditer(body):
        if m.group(1) is not None:
            out.append((int(m.group(1)), int(m.group(2))))
        else:
            out.append((int(m.group(4)), int(m.group(3))))
    return sorted(set(out))


def correspondence_accuracy(code, demo_kps, target_kps, T_gt, tol=0.03):
    """How many of the code's declared demo->target pairs are geometrically
    right: the GT-mapped demo keypoint must be within `tol` of the target
    keypoint it is tied to, AND be that target keypoint's nearest GT-mapped
    demo keypoint."""
    P = np.asarray(demo_kps, dtype=np.float64)
    Q = np.asarray(target_kps, dtype=np.float64)
    M = transform_points(np.asarray(T_gt), P)
    pairs = declared_pairs(code)
    n_ok = 0
    errs = []
    for i, j in pairs:
        if not (0 <= i < P.shape[0] and 0 <= j < Q.shape[0]):
            errs.append(None)
            continue
        d = float(np.linalg.norm(M[i] - Q[j]))
        errs.append(d)
        nearest = int(np.argmin(np.linalg.norm(M - Q[j], axis=1)))
        if d <= tol and nearest == i:
            n_ok += 1
    return {"n_pairs": len(pairs), "n_correct": n_ok, "pairs": pairs,
            "pair_err_m": errs}
