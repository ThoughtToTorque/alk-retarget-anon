"""Phase-4 Campaign D runner: real-VLM discrete error rate (local Qwen).

Measurements (docs/REPRODUCE.md, Campaign D; 4 stable tasks x 30 medium-tier
target scenes, seeds 1000..1029; the pour task is excluded while it is being
regenerated):

  1. Index-classification accuracy of alkbench.discrete.VLMSolver (crop-zoom
     markup, ONE fixed prompt per task) on phi1/phi2/phi3, scored against the
     recomputed oracle answers (pipeline.oracle) -- both RAW agreement and
     SYMMETRY-CORRECTED agreement (a choice is correct if the ALK transform it
     induces equals the oracle's up to a stabilizer of the demo object:
     trivial group for nut/cap/door, SO(2) about the can axis for rim_grasp).
  2. End-to-end success feeding the VLM DiscreteChoice through the full
     pipeline (retarget_runner path: ALK -> Procrustes -> bounded
     registration -> waypoint retarget -> execution).  Registration ON,
     grasp-point correction OFF: the correction consumes the oracle phi4/phi5
     grasp region, which would let ground truth partially rescue wrong VLM
     answers; the matched oracle-discrete reference is therefore
     campaign_a's ours_nocorr on the same seeds.
  3. moka_vlm contrast (pixel-regression prompting, baselines.moka_style) on
     the same scenes: pixel error / anchor 3D error / end-to-end success.
  4. Decoupling estimate: P(success | VLM discrete correct) vs
     P(success | incorrect), Wilson CIs (stats.tests).

Phases (run in order; every phase is resumable, existing JSONs are skipped):

  smoke    -- prompt sanity check on NON-EVAL scenes (easy-tier data/, seeds
              1..3) + the demo images; output to stdout only.  Prompts are
              frozen after this phase; they are never tuned per eval scene.
  classify -- all VLM discrete queries (demo once per task + 30 targets) +
              scoring; no simulator needed.
  execute  -- end-to-end rollouts ours_vlm (cached choices, no new VLM
              queries) and moka_vlm (queries inside).  Needs MUJOCO_GL=egl.
  report   -- aggregate tables (results/tables/campaign_d.{md,tex}) + summary.

Usage:
  .venv/bin/python -m pipeline.campaign_d --phase classify
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m pipeline.campaign_d \
      --phase execute --tasks nut_loosen
  .venv/bin/python -m pipeline.campaign_d --phase report
"""
import argparse
import json
import os
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from alkbench import (alk_from_candidates, procrustes, bounded_registration,
                      transform_points, rotation_angle_deg, VLMSolver)
from baselines import common as bc
from pipeline import oracle
from stats import tests as st

SIMBENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(SIMBENCH, "data_medium")
EASY_DATA_ROOT = os.path.join(SIMBENCH, "data")
OUT_ROOT = os.path.join(SIMBENCH, "results", "campaign_d")
CAMPAIGN_A_ROOT = os.path.join(SIMBENCH, "results", "campaign_a")
TABLES_DIR = os.path.join(SIMBENCH, "results", "tables")

TASKS_D = ("nut_loosen", "cap_twist", "rim_grasp", "box_open")
SEEDS_D = tuple(range(1000, 1030))
DEMO_PAIR_SEED = 0
K = 8
KMEANS_SEED = 0

# symmetry-equivalence tolerances = the pipeline's own "mapping ok" heuristic
ROT_TOL_DEG = 20.0
TRANS_TOL_M = 0.05

# continuous rotational symmetry axis in the OBJECT frame (only the can);
# nut/cap/door are scored as asymmetric (trivial stabilizer)
SYM_AXIS_LOCAL = {"rim_grasp": np.array([0.0, 0.0, 1.0])}

VLM_DEFAULTS = {
    "base_url": "http://127.0.0.1:8199/v1",
    "model": "Qwen/Qwen3-VL-32B-Instruct-FP8",
    "api_key": "local",
}

# ONE fixed task_description per task (fills alkbench.discrete
# _PROMPT_TEMPLATE's {task}).  Frozen after the smoke phase on non-eval
# scenes; identical for the demo query and all 30 target queries.
# phi3 note: from a single image the lateral-swap bit cannot be judged
# against the (unseen) demonstration; the camera is fixed and never mirrors
# the view, so the prompt pins phi3 = 0 and the classification burden sits
# entirely on phi1/phi2 (see FINDINGS for the honest discussion).
TASK_PROMPTS = {
    "nut_loosen": (
        "grasp the round nut by its handle. The object is a ring with a "
        "small handle tab sticking out. phi1 = the marker nearest the tip "
        "of the handle tab. phi2 = the marker farthest from the handle, "
        "across the hole. Always answer phi3 = 0."),
    "cap_twist": (
        "grasp the square nut by its handle. The object is a square ring "
        "(a plate with a rectangular hole) with a small handle tab sticking "
        "out. phi1 = the marker nearest the tip of the handle tab. phi2 = "
        "the marker farthest from the handle, across the hole. Always "
        "answer phi3 = 0."),
    "rim_grasp": (
        "grasp the upright can near its top rim. The object is an upright "
        "can on a table. phi1 = the marker nearest the BOTTOM of the can "
        "(where it meets the table). phi2 = the marker nearest the very "
        "TOP of the can. Always answer phi3 = 0."),
    "box_open": (
        "pull the door handle to open the door. The object is a "
        "rectangular door panel with a wooden handle near one vertical "
        "edge. phi1 = the marker nearest the door's TOP corner on the "
        "handle side. phi2 = the marker nearest the door's BOTTOM corner "
        "on the other side (diagonally opposite phi1). Always answer "
        "phi3 = 0."),
}


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
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_json_safe(obj), f, indent=2)
    os.rename(tmp, path)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def draw_legible_markup(image, candidates_uv, radius=9, text_scale=3,
                        target_extent=448):
    """Crop-zoom markup with LARGER digit labels than the alkbench default.

    Dev-scene probing (non-eval seeds) showed Qwen3-VL perceives the objects
    and the marker COLORS correctly but misreads the default 6x10-px bitmap
    digits (e.g. answering "6" for the purple marker 5), sometimes falling
    back to a hallucinated ordered layout.  This variant keeps the exact
    ring/palette/crop design of alkbench.discrete.draw_candidate_markup and
    only raises the digit scale (3x) and the crop target extent (448 px).
    One fixed setting for all tasks, frozen before the eval seeds were run.
    """
    from alkbench.discrete import crop_zoom, _draw_text, _MARKER_COLORS
    img = np.array(image, dtype=np.uint8, copy=True)
    uv = np.asarray(candidates_uv, dtype=np.float64)
    img, uv, _ = crop_zoom(img, uv, target_extent=target_extent)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    for i, (u, v) in enumerate(uv):
        color = _MARKER_COLORS[i % len(_MARKER_COLORS)]
        d2 = (xx - u) ** 2 + (yy - v) ** 2
        img[(d2 <= (radius + 2) ** 2) & (d2 > radius ** 2)] = (255, 255, 255)
        img[(d2 <= radius ** 2) & (d2 > (radius - 3) ** 2)] = color
        tw = 4 * text_scale
        tx, ty = int(u) + radius + 4, int(v) - int(2.5 * text_scale)
        if tx + tw + 2 >= w:
            tx = int(u) - radius - tw - 6
        _draw_text(img, str(i + 1), ty + 1, tx + 1, (0, 0, 0),
                   scale=text_scale)
        _draw_text(img, str(i + 1), ty, tx, (255, 255, 255),
                   scale=text_scale)
    return img


class CampaignVLMSolver(VLMSolver):
    """VLMSolver with the legible markup variant; prompt template, JSON
    parsing, validation-retry loop and client conventions are inherited
    unchanged (the query loop below mirrors VLMSolver.solve verbatim except
    for the draw_legible_markup call)."""

    def solve(self, image=None, candidates_uv=None, task_description=""):
        import base64
        from alkbench.discrete import encode_png, _PROMPT_TEMPLATE
        if image is None or candidates_uv is None:
            raise ValueError("solve needs image and candidates_uv")
        k = len(candidates_uv)
        markup = draw_legible_markup(image, candidates_uv)
        b64 = base64.b64encode(encode_png(markup)).decode("ascii")
        prompt = _PROMPT_TEMPLATE.format(k=k, task=task_description, extra="")
        client = self._get_client()
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64," + b64}},
        ]}]
        last_err = None
        for _ in range(1 + self.max_retries):
            resp = client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=self.temperature)
            text = resp.choices[0].message.content
            try:
                return self._parse(text, k)
            except (ValueError, KeyError, TypeError) as e:
                last_err = e
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content":
                                 "Invalid reply (%s). Answer with ONLY the "
                                 "JSON object in the required format." % e})
        raise RuntimeError("CampaignVLMSolver: invalid reply after retries: %s"
                           % last_err)


def make_recording_solver():
    """VLMSolver against the local endpoint, with every raw completion (text,
    latency) captured through a thin client proxy.  Returns (solver, log)."""
    model = os.environ.get("ALK_VLM_MODEL", VLM_DEFAULTS["model"])
    base_url = os.environ.get("ALK_VLM_BASE_URL", VLM_DEFAULTS["base_url"])
    api_key = os.environ.get("ALK_VLM_API_KEY", VLM_DEFAULTS["api_key"])
    solver = CampaignVLMSolver(model=model, base_url=base_url,
                               api_key=api_key, max_retries=1,
                               temperature=0.0)
    real = solver._get_client()
    log = []

    class _Completions(object):
        @staticmethod
        def create(**kw):
            t0 = time.time()
            resp = real.chat.completions.create(**kw)
            log.append({"text": resp.choices[0].message.content,
                        "dt": time.time() - t0})
            return resp

    class _Chat(object):
        completions = _Completions()

    class _Client(object):
        chat = _Chat()

    solver._client = _Client()
    return solver, log


# ---------------------------------------------------------------------------
# perception / oracle plumbing (shared by all phases; identical parameters to
# pipeline.retarget_runner: same camera priority, k=8, kmeans seed 0, no
# sensor noise at medium tier)
# ---------------------------------------------------------------------------

def demo_percept(task, data_root=DATA_ROOT):
    demo_dir = os.path.join(data_root, task, str(DEMO_PAIR_SEED), "demo")
    cap = bc.load_capture(os.path.join(demo_dir, "scene"))
    cam = bc.pick_camera(cap, task)
    p = bc.perceive(cap, task, k=K, seed=KMEANS_SEED, camera=cam)
    with open(os.path.join(demo_dir, "keyframes.json")) as f:
        keyframes = json.load(f)
    pre_grasp = [kf for kf in keyframes if kf["name"] == "pre_grasp"][0]
    return {"cap": cap, "cam": cam, "percept": p,
            "pre_grasp_tcp": np.asarray(pre_grasp["tcp_pos"])}


def target_percept(task, seed, cam, data_root=DATA_ROOT):
    pair = _read_json(os.path.join(data_root, task, str(seed), "pair.json"))
    cap = bc.load_capture(os.path.join(data_root, task, str(seed),
                                       "target", "scene"))
    p = bc.perceive(cap, task, k=K, seed=KMEANS_SEED, camera=cam)
    return pair, cap, p


# ---------------------------------------------------------------------------
# symmetry-corrected scoring
# ---------------------------------------------------------------------------

def _twist_swing_deg(R, axis):
    """Split rotation matrix R into twist about `axis` + residual swing.
    Returns (twist_deg, swing_deg)."""
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    p = float(np.dot(q[:3], a))
    tw = np.array([a[0] * p, a[1] * p, a[2] * p, q[3]])
    n = np.linalg.norm(tw)
    if n < 1e-12:  # pure 180-deg swing
        return 0.0, float(rotation_angle_deg(R))
    tw /= n
    R_tw = Rotation.from_quat(tw).as_matrix()
    swing = R @ R_tw.T
    twist_deg = float(np.degrees(
        2.0 * np.arccos(np.clip(abs(tw[3]), -1.0, 1.0))))
    return twist_deg, float(rotation_angle_deg(swing))


def stabilizer_residual(task, T_ref, T_est, demo_pose):
    """How far S = T_ref^-1 @ T_est is from a symmetry of the DEMO object.

    Returns dict(swing_deg, twist_deg, center_disp_m, full_rot_deg).
    Asymmetric tasks: swing = the full rotation angle of S (twist = 0).
    rim_grasp: twist about the demo can axis is quotiented out.
    """
    T_ref = np.asarray(T_ref, dtype=np.float64)
    T_est = np.asarray(T_est, dtype=np.float64)
    S = np.linalg.inv(T_ref) @ T_est
    R_S = S[:3, :3]
    c = np.asarray(demo_pose["pos"], dtype=np.float64)
    center_disp = float(np.linalg.norm(transform_points(S, c[None])[0] - c))
    full = float(rotation_angle_deg(R_S))
    axis_local = SYM_AXIS_LOCAL.get(task)
    if axis_local is None:
        return {"swing_deg": full, "twist_deg": 0.0,
                "center_disp_m": center_disp, "full_rot_deg": full}
    R_demo = Rotation.from_quat(demo_pose["quat_xyzw"]).as_matrix()
    axis_world = R_demo @ axis_local
    twist, swing = _twist_swing_deg(R_S, axis_world)
    return {"swing_deg": swing, "twist_deg": twist,
            "center_disp_m": center_disp, "full_rot_deg": full}


def sym_correct(resid):
    return bool(resid["swing_deg"] <= ROT_TOL_DEG
                and resid["center_disp_m"] <= TRANS_TOL_M)


def map_errors(T_est, T_gt, demo_pos, target_pos):
    T_est = np.asarray(T_est, dtype=np.float64)
    T_gt = np.asarray(T_gt, dtype=np.float64)
    rot = float(rotation_angle_deg(T_est[:3, :3] @ T_gt[:3, :3].T))
    mapped = transform_points(T_est, np.asarray(demo_pos, np.float64)[None])[0]
    trans = float(np.linalg.norm(mapped - np.asarray(target_pos, np.float64)))
    return rot, trans


# ---------------------------------------------------------------------------
# VLM query helpers
# ---------------------------------------------------------------------------

def query_choice(task, rgb, cands, solver, log):
    """One VLMSolver query.  Returns a dict with the parsed choice (or
    parse_failed) + raw replies and latency."""
    n0 = len(log)
    t0 = time.time()
    out = {"parse_failed": False, "error": None}
    try:
        ch = solver.solve(image=rgb, candidates_uv=cands.centers2d,
                          task_description=TASK_PROMPTS[task])
        out["choice"] = {"phi1": ch.phi1, "phi2": ch.phi2,
                         "phi3": int(ch.phi3)}
    except Exception as e:  # invalid after retries / transport error
        out["parse_failed"] = True
        out["error"] = repr(e)
        out["choice"] = None
    out["latency_s"] = round(time.time() - t0, 2)
    out["n_api_calls"] = len(log) - n0
    out["raw_replies"] = [r["text"] for r in log[n0:]]
    return out


def ensure_demo_query(task, dp, solver, log, out_root=OUT_ROOT):
    """Query the demo side once per task (cached on disk)."""
    path = os.path.join(out_root, task, "vlm_demo_query.json")
    if os.path.exists(path):
        return _read_json(path)
    d1o, d2o = oracle.demo_axial_choice(dp["percept"]["cands"])
    rgb = bc.load_rgb(dp["cap"], dp["cam"])
    q = query_choice(task, rgb, dp["percept"]["cands"], solver, log)
    rec = {
        "task": task, "side": "demo", "camera": dp["cam"], "k": K,
        "model": solver.model,
        "task_prompt": TASK_PROMPTS[task],
        "oracle_choice": {"phi1": d1o, "phi2": d2o, "phi3": 0},
        "vlm": q,
    }
    if q["choice"] is not None:
        v1, v2 = q["choice"]["phi1"], q["choice"]["phi2"]
        rec["agree"] = {
            "phi1": v1 == d1o, "phi2": v2 == d2o,
            "pair_set": {v1, v2} == {d1o, d2o},
            "pair_swapped": (v1, v2) == (d2o, d1o),
        }
    _write_json(path, rec)
    return _read_json(path)


# ---------------------------------------------------------------------------
# phase: classify
# ---------------------------------------------------------------------------

def classify_task(task, seeds, out_root=OUT_ROOT, data_root=DATA_ROOT):
    solver, log = make_recording_solver()
    dp = demo_percept(task, data_root)
    demo_q = ensure_demo_query(task, dp, solver, log, out_root)
    d_cands = dp["percept"]["cands"]

    # demo-side ALKs (oracle convention phi3=0 on the demo by definition)
    demo_alk_vlm = None
    if demo_q["vlm"]["choice"] is not None:
        vc = demo_q["vlm"]["choice"]
        try:
            demo_alk_vlm = alk_from_candidates(d_cands, vc["phi1"],
                                               vc["phi2"], phi3=False)
        except ValueError:
            pass

    n_new = 0
    for seed in seeds:
        path = os.path.join(out_root, task, str(seed), "vlm_query.json")
        if os.path.exists(path):
            continue
        t0 = time.time()
        pair, tcap, pt = target_percept(task, seed, dp["cam"], data_root)
        inst = pair["target_instance"]
        demo_pose = pair["demo_object_poses"][inst]
        target_pose = pair["target_object_poses"][inst]
        T_gt = oracle.gt_relative_transform(demo_pose, target_pose)
        orc = oracle.solve(dp["percept"], pt, T_gt, dp["pre_grasp_tcp"],
                           seed=KMEANS_SEED)
        T0_orc = procrustes(orc["demo_alk"], orc["target_alk"])
        rot_orc, trans_orc = map_errors(T0_orc, T_gt, demo_pose["pos"],
                                        target_pose["pos"])

        # cross-check against the recorded campaign_a oracle answers
        rec_a = os.path.join(CAMPAIGN_A_ROOT, task, str(seed),
                             "rollout_ours_full.json")
        matches_a = None
        if os.path.exists(rec_a):
            a_orc = _read_json(rec_a).get("oracle")
            if a_orc:
                tc = a_orc["target_choice"]
                oc = orc["target_choice"]
                matches_a = all(int(tc[k]) == int(oc[k])
                                for k in ("phi1", "phi2", "phi3"))

        rgb = bc.load_rgb(tcap, pt["camera"])
        q = query_choice(task, rgb, pt["cands"], solver, log)

        rec = {
            "task": task, "seed": seed, "side": "target",
            "camera": pt["camera"], "k": K, "model": solver.model,
            "tier": "medium",
            "oracle_choice": {k: orc["target_choice"][k]
                              for k in ("phi1", "phi2", "phi3")},
            "oracle_matches_campaign_a": matches_a,
            "demo_vlm_choice": demo_q["vlm"]["choice"],
            "vlm": q,
            "T_gt": T_gt.tolist(),
            "errs": {"rot_T0_oracle_vs_gt_deg": rot_orc,
                     "trans_T0_oracle_vs_gt_m": trans_orc},
        }

        if q["choice"] is not None:
            oc = rec["oracle_choice"]
            v = q["choice"]
            rec["agree"] = {
                "phi1": v["phi1"] == oc["phi1"],
                "phi2": v["phi2"] == oc["phi2"],
                "phi3": v["phi3"] == oc["phi3"],
                "pair_set": {v["phi1"], v["phi2"]} == {oc["phi1"],
                                                       oc["phi2"]},
                "joint_raw": (v["phi1"] == oc["phi1"]
                              and v["phi2"] == oc["phi2"]
                              and v["phi3"] == oc["phi3"]),
            }
            # 3D distance between chosen and oracle-chosen endpoints
            C = pt["cands"].candidates3d
            rec["endpoint_dist_m"] = [
                float(np.linalg.norm(C[v["phi1"]] - C[oc["phi1"]])),
                float(np.linalg.norm(C[v["phi2"]] - C[oc["phi2"]]))]

        # symmetry-corrected joint correctness via the induced ALK transform
        sym = {"correct": False, "reason": None}
        if q["choice"] is None:
            sym["reason"] = "vlm_parse_failed"
        elif demo_alk_vlm is None:
            sym["reason"] = "demo_vlm_alk_unavailable"
        else:
            try:
                target_alk_vlm = alk_from_candidates(
                    pt["cands"], q["choice"]["phi1"], q["choice"]["phi2"],
                    phi3=bool(q["choice"]["phi3"]))
                T0_vlm = procrustes(demo_alk_vlm, target_alk_vlm)
                resid = stabilizer_residual(task, T0_orc, T0_vlm, demo_pose)
                sym.update(resid)
                sym["correct"] = sym_correct(resid)
                rot_v, trans_v = map_errors(T0_vlm, T_gt, demo_pose["pos"],
                                            target_pose["pos"])
                rec["errs"]["rot_T0_vlm_vs_gt_deg"] = rot_v
                rec["errs"]["trans_T0_vlm_vs_gt_m"] = trans_v
                gt_resid = stabilizer_residual(task, T_gt, T0_vlm, demo_pose)
                rec["errs"]["sym_rot_T0_vlm_vs_gt_deg"] = gt_resid["swing_deg"]
            except ValueError as e:  # degenerate halfplane split
                sym["reason"] = "alk_degenerate: %s" % e
        rec["sym"] = sym
        rec["time_s"] = round(time.time() - t0, 1)
        _write_json(path, rec)
        n_new += 1
        ag = rec.get("agree", {})
        print("[%s %d classify] raw=%s sym=%s swing=%s (%.1fs)"
              % (task, seed, ag.get("joint_raw"), sym["correct"],
                 ("%.1f" % sym["swing_deg"]) if "swing_deg" in sym else "-",
                 rec["time_s"]), flush=True)
    return n_new


# ---------------------------------------------------------------------------
# phase: smoke (prompt sanity check on NON-EVAL scenes; stdout only)
# ---------------------------------------------------------------------------

def smoke(tasks, seeds=(1, 2, 3)):
    solver, log = make_recording_solver()
    for task in tasks:
        dp = demo_percept(task, EASY_DATA_ROOT)
        d1o, d2o = oracle.demo_axial_choice(dp["percept"]["cands"])
        rgb = bc.load_rgb(dp["cap"], dp["cam"])
        q = query_choice(task, rgb, dp["percept"]["cands"], solver, log)
        print("[%s demo ] oracle=(%d,%d) vlm=%s  raw=%r" % (
            task, d1o + 1, d2o + 1,
            None if q["choice"] is None else
            (q["choice"]["phi1"] + 1, q["choice"]["phi2"] + 1),
            q["raw_replies"][-1][:120]), flush=True)
        for seed in seeds:
            try:
                pair, tcap, pt = target_percept(task, seed, dp["cam"],
                                                EASY_DATA_ROOT)
            except FileNotFoundError:
                continue
            inst = pair["target_instance"]
            T_gt = oracle.gt_relative_transform(
                pair["demo_object_poses"][inst],
                pair["target_object_poses"][inst])
            orc = oracle.solve(dp["percept"], pt, T_gt, dp["pre_grasp_tcp"],
                               seed=KMEANS_SEED)
            oc = orc["target_choice"]
            rgb_t = bc.load_rgb(tcap, pt["camera"])
            q = query_choice(task, rgb_t, pt["cands"], solver, log)
            print("[%s %5d] oracle=(%d,%d,phi3=%d) vlm=%s" % (
                task, seed, oc["phi1"] + 1, oc["phi2"] + 1, oc["phi3"],
                None if q["choice"] is None else
                (q["choice"]["phi1"] + 1, q["choice"]["phi2"] + 1,
                 q["choice"]["phi3"])), flush=True)


# ---------------------------------------------------------------------------
# phase: execute (ours_vlm end-to-end + moka_vlm registry rollouts)
# ---------------------------------------------------------------------------

def execute_ours_vlm(task, seeds, out_root=OUT_ROOT, data_root=DATA_ROOT):
    """Feed the cached VLM discrete choices through the full pipeline and
    execute (registration ON, correction OFF -- see module docstring)."""
    from pipeline import campaign as cg
    from pipeline import retarget_runner as rr
    from simtasks import envs, motion

    demo_q = _read_json(os.path.join(out_root, task, "vlm_demo_query.json"))
    dp = demo_percept(task, data_root)
    d_cands = dp["percept"]["cands"]
    demo_alk_vlm = None
    demo_alk_err = None
    if demo_q["vlm"]["choice"] is not None:
        vc = demo_q["vlm"]["choice"]
        try:
            demo_alk_vlm = alk_from_candidates(d_cands, vc["phi1"],
                                               vc["phi2"], phi3=False)
        except ValueError as e:
            demo_alk_err = str(e)

    demo = rr.load_demo(task, data_root)
    env = cg.make_tier_env(task, "medium")
    n_new = 0
    try:
        for seed in seeds:
            out_path = os.path.join(out_root, task, str(seed),
                                    "rollout_ours_vlm.json")
            if os.path.exists(out_path):
                continue
            t0 = time.time()
            q = _read_json(os.path.join(out_root, task, str(seed),
                                        "vlm_query.json"))
            pair, tcap, pt = target_percept(task, seed, dp["cam"], data_root)
            inst = pair["target_instance"]
            demo_pose = pair["demo_object_poses"][inst]
            target_pose = pair["target_object_poses"][inst]
            T_gt = oracle.gt_relative_transform(demo_pose, target_pose)

            result = {
                "task": task, "seed": seed, "method": "ours_vlm",
                "variant": "ours_vlm", "tier": "medium",
                "registration": True, "correction": False,
                "success": False, "failure_stage": None,
                "camera": pt["camera"], "T_gt": T_gt.tolist(),
                "discrete_correct": bool(q["sym"]["correct"]),
                "vlm": {
                    "model": q["model"],
                    "demo_choice": demo_q["vlm"]["choice"],
                    "target_choice": q["vlm"]["choice"],
                    "oracle_choice": q["oracle_choice"],
                    "agree": q.get("agree"),
                    "sym": q["sym"],
                },
                "perception": {
                    "demo": {"sanity_ok": dp["percept"]["sanity_ok"],
                             "centroid_err_m": dp["percept"]["centroid_err_m"]},
                    "target": {"sanity_ok": pt["sanity_ok"],
                               "centroid_err_m": pt["centroid_err_m"]},
                },
            }

            fail = None
            if q["vlm"]["choice"] is None:
                fail = "vlm_discrete"
            elif demo_alk_vlm is None:
                fail = ("alk_degenerate_demo"
                        if demo_alk_err else "vlm_discrete_demo")
            if fail is None:
                vc_t = q["vlm"]["choice"]
                try:
                    target_alk_vlm = alk_from_candidates(
                        pt["cands"], vc_t["phi1"], vc_t["phi2"],
                        phi3=bool(vc_t["phi3"]))
                except ValueError as e:
                    fail = "alk_degenerate"
                    result["error"] = str(e)
            if fail is not None:
                result["failure_stage"] = fail
                result["time_s"] = round(time.time() - t0, 1)
                _write_json(out_path, result)
                n_new += 1
                print("[%s %d ours_vlm] success=False stage=%s (no exec)"
                      % (task, seed, fail), flush=True)
                continue

            T0 = procrustes(demo_alk_vlm, target_alk_vlm)
            reg = bounded_registration(d_cands.points3d, pt["cands"].points3d,
                                       T0)
            T_map = reg["T"]
            rot0, trans0 = map_errors(T0, T_gt, demo_pose["pos"],
                                      target_pose["pos"])
            rot_err, trans_err = map_errors(T_map, T_gt, demo_pose["pos"],
                                            target_pose["pos"])
            cond = bc.conditioning(np.asarray(demo_alk_vlm))
            result.update({
                "T_init": np.asarray(T0).tolist(),
                "T_map": np.asarray(T_map).tolist(),
                "chamfer_before_m": float(reg["chamfer_init"]),
                "chamfer_after_m": float(reg["chamfer_final"]),
                "rot_err_init_deg": rot0, "trans_err_init_m": trans0,
                "rot_err_deg": rot_err, "trans_err_m": trans_err,
                "conditioning": cond["sigma23"],
                "conditioning_detail": cond,
            })

            wps = rr.retarget_waypoint_dicts(demo["waypoints"], T_map)
            envs.reset_with_seed(env, pair["target_seed"])
            exec_res = motion.execute_waypoints(env, wps,
                                                grasp_check=rr._grasp_check(task))
            success = bool(envs.success_checker(task)(env))
            result.update({
                "grasped": exec_res["grasped"],
                "waypoints_converged": exec_res["converged"],
                "n_exec_steps": exec_res["n_steps"],
                "success": success,
            })
            if not success:
                if not (dp["percept"]["sanity_ok"] and pt["sanity_ok"]):
                    stage = "perception"
                elif trans_err > 0.05 or rot_err > 20.0:
                    stage = "mapping"
                elif exec_res["grasped"] is False:
                    stage = "grasp"
                else:
                    stage = "post_action"
                result["failure_stage"] = stage
            result["time_s"] = round(time.time() - t0, 1)
            _write_json(out_path, result)
            n_new += 1
            print("[%s %d ours_vlm] success=%s stage=%s rot=%.1fdeg "
                  "trans=%.1fmm disc_ok=%s (%.1fs)"
                  % (task, seed, success, result["failure_stage"], rot_err,
                     1e3 * trans_err, result["discrete_correct"],
                     result["time_s"]), flush=True)
    finally:
        env.close()
    return n_new


def execute_moka_vlm(task, seeds, out_root=OUT_ROOT, data_root=DATA_ROOT):
    """moka_vlm through the shared campaign runner (registry method)."""
    from pipeline import campaign as cg
    return cg.run_task(task, list(seeds), ["moka_vlm"], "medium", data_root,
                       out_root)


# ---------------------------------------------------------------------------
# phase: report
# ---------------------------------------------------------------------------

def _wilson_str(k, n):
    if n == 0:
        return "-"
    lo, hi = st.wilson_ci(k, n)
    return "%d/%d = %.2f [%.2f, %.2f]" % (k, n, k / n, lo, hi)


def _load_campaign_a_success(task, seeds, method):
    out = {}
    for seed in seeds:
        p = os.path.join(CAMPAIGN_A_ROOT, task, str(seed),
                         "rollout_%s.json" % method)
        if os.path.exists(p):
            out[seed] = int(bool(_read_json(p).get("success")))
    return out


def collect(tasks=TASKS_D, seeds=SEEDS_D, out_root=OUT_ROOT):
    data = {}
    for task in tasks:
        rows = []
        for seed in seeds:
            row = {"seed": seed}
            qp = os.path.join(out_root, task, str(seed), "vlm_query.json")
            if os.path.exists(qp):
                row["q"] = _read_json(qp)
            op = os.path.join(out_root, task, str(seed),
                              "rollout_ours_vlm.json")
            if os.path.exists(op):
                row["ours"] = _read_json(op)
            mp = os.path.join(out_root, task, str(seed),
                              "rollout_moka_vlm.json")
            if os.path.exists(mp):
                row["moka"] = _read_json(mp)
            rows.append(row)
        dq = os.path.join(out_root, task, "vlm_demo_query.json")
        data[task] = {
            "rows": rows,
            "demo_q": _read_json(dq) if os.path.exists(dq) else None,
            "ref_nocorr": _load_campaign_a_success(task, seeds,
                                                   "ours_nocorr"),
            "ref_moka": _load_campaign_a_success(task, seeds, "moka_oracle"),
        }
    return data


def report(tasks=TASKS_D, seeds=SEEDS_D, out_root=OUT_ROOT):
    data = collect(tasks, seeds, out_root)
    md = ["# Campaign D -- real-VLM discrete error rate (medium tier, "
          "seeds %d..%d, N=%d/task)\n" % (seeds[0], seeds[-1], len(seeds)),
          "Model: %s. Raw per-query records under results/campaign_d/.\n"
          % VLM_DEFAULTS["model"]]
    tex = []
    summary = {}

    # ---- table 1: classification accuracy ---------------------------------
    md.append("## 1. Index-classification accuracy (VLMSolver, "
              "crop-zoom markup)\n")
    md.append("| task | phi1 raw | phi2 raw | pair-set | phi3 raw | "
              "joint raw | sym-corrected | demo query |")
    md.append("|---|---|---|---|---|---|---|---|")
    t1 = []
    for task in tasks:
        d = data[task]
        qs = [r["q"] for r in d["rows"] if "q" in r]
        n = len(qs)
        ag = lambda key: sum(1 for q in qs if q.get("agree", {}).get(key))
        symk = sum(1 for q in qs if q["sym"]["correct"])
        demo_ok = "-"
        if d["demo_q"] is not None and "agree" in d["demo_q"]:
            demo_ok = ("pair-set %s" %
                       ("OK" if d["demo_q"]["agree"]["pair_set"] else "WRONG"))
        row = (task, ag("phi1"), ag("phi2"), ag("pair_set"), ag("phi3"),
               ag("joint_raw"), symk, n)
        t1.append(row)
        md.append("| %s | %d/%d | %d/%d | %d/%d | %d/%d | %d/%d | **%s** | %s |"
                  % (task, row[1], n, row[2], n, row[3], n, row[4], n,
                     row[5], n, _wilson_str(symk, n), demo_ok))
        summary.setdefault("classification", {})[task] = {
            "n": n, "phi1": row[1], "phi2": row[2], "pair_set": row[3],
            "phi3": row[4], "joint_raw": row[5], "sym_corrected": symk}
    pooled_sym = sum(r[6] for r in t1)
    pooled_n = sum(r[7] for r in t1)
    md.append("| **pooled** |  |  |  |  |  | **%s** |  |\n"
              % _wilson_str(pooled_sym, pooled_n))

    tex.append("% Campaign D table 1: VLM index-classification accuracy")
    tex.append("\\begin{tabular}{lcccccc}")
    tex.append("\\toprule")
    tex.append("task & $\\varphi_1$ & $\\varphi_2$ & pair-set & "
               "$\\varphi_3$ & joint & sym.-corr. \\\\")
    tex.append("\\midrule")
    for row in t1:
        task, p1, p2, ps, p3, jr, sy, n = row
        tex.append("%s & %d/%d & %d/%d & %d/%d & %d/%d & %d/%d & "
                   "\\textbf{%d/%d} \\\\"
                   % (task.replace("_", "\\_"), p1, n, p2, n, ps, n, p3, n,
                      jr, n, sy, n))
    tex.append("\\midrule")
    tex.append("pooled &  &  &  &  &  & \\textbf{%d/%d} \\\\"
               % (pooled_sym, pooled_n))
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}\n")

    # ---- table 2: end-to-end + moka contrast -------------------------------
    md.append("## 2/3. End-to-end success + classification-vs-regression "
              "contrast\n")
    md.append("ours_vlm = full pipeline (registration ON, correction OFF) "
              "with real-VLM phi1/phi2/phi3; reference = campaign_a "
              "ours_nocorr (identical stack, oracle discrete). moka_vlm = "
              "real-VLM pixel regression; reference = campaign_a moka_oracle "
              "(sigma_px = 0).\n")
    md.append("| task | ours_vlm | ours_nocorr (oracle ref) | McNemar p | "
              "moka_vlm | moka_oracle (ref) | moka pix err px (med) | "
              "moka anchor err mm (med) |")
    md.append("|---|---|---|---|---|---|---|---|")
    t2 = []
    for task in tasks:
        d = data[task]
        ours = {r["seed"]: int(bool(r["ours"]["success"]))
                for r in d["rows"] if "ours" in r}
        moka = {r["seed"]: int(bool(r["moka"]["success"]))
                for r in d["rows"] if "moka" in r}
        shared_o = sorted(set(ours) & set(d["ref_nocorr"]))
        p_mc = None
        if shared_o:
            res = st.mcnemar_from_vectors(
                [ours[s] for s in shared_o],
                [d["ref_nocorr"][s] for s in shared_o])
            p_mc = res.pvalue
        pix = [r["moka"].get("aux", {}).get("pixel_err_px")
               for r in d["rows"] if "moka" in r]
        pix = [x for x in pix if x is not None]
        anc = [r["moka"].get("aux", {}).get("anchor_err_m")
               for r in d["rows"] if "moka" in r]
        anc = [x for x in anc if x is not None]
        ko, no = sum(ours.values()), len(ours)
        kr = sum(d["ref_nocorr"].get(s, 0) for s in shared_o)
        km, nm = sum(moka.values()), len(moka)
        kmr = sum(d["ref_moka"].values())
        nmr = len(d["ref_moka"])
        md.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            task, _wilson_str(ko, no),
            _wilson_str(kr, len(shared_o)) if shared_o else "-",
            ("%.3g" % p_mc) if p_mc is not None else "-",
            _wilson_str(km, nm), _wilson_str(kmr, nmr),
            ("%.1f" % float(np.median(pix))) if pix else "-",
            ("%.0f" % (1e3 * float(np.median(anc)))) if anc else "-"))
        t2.append((task, ko, no, kr, len(shared_o), p_mc, km, nm, kmr, nmr,
                   float(np.median(pix)) if pix else None,
                   float(np.median(anc)) if anc else None))
        summary.setdefault("end_to_end", {})[task] = {
            "ours_vlm": [ko, no], "ours_nocorr_ref": [kr, len(shared_o)],
            "mcnemar_p": p_mc, "moka_vlm": [km, nm],
            "moka_oracle_ref": [kmr, nmr],
            "moka_pixel_err_px_median": (float(np.median(pix))
                                         if pix else None),
            "moka_anchor_err_mm_median": (1e3 * float(np.median(anc))
                                          if anc else None)}
    ko_p = sum(r[1] for r in t2); no_p = sum(r[2] for r in t2)
    kr_p = sum(r[3] for r in t2); nr_p = sum(r[4] for r in t2)
    km_p = sum(r[6] for r in t2); nm_p = sum(r[7] for r in t2)
    md.append("| **pooled** | **%s** | %s |  | **%s** |  |  |  |\n" % (
        _wilson_str(ko_p, no_p), _wilson_str(kr_p, nr_p),
        _wilson_str(km_p, nm_p)))

    tex.append("% Campaign D table 2: end-to-end + moka contrast")
    tex.append("\\begin{tabular}{lcccccc}")
    tex.append("\\toprule")
    tex.append("task & ours\\_vlm & ours (oracle $\\varphi$) & $p$ & "
               "moka\\_vlm & moka (oracle px) & anchor err [mm] \\\\")
    tex.append("\\midrule")
    for (task, ko, no, kr, nr, p_mc, km, nm, kmr, nmr, pix, anc) in t2:
        tex.append("%s & %d/%d & %d/%d & %s & %d/%d & %d/%d & %s \\\\" % (
            task.replace("_", "\\_"), ko, no, kr, nr,
            ("%.3g" % p_mc) if p_mc is not None else "--",
            km, nm, kmr, nmr,
            ("%.0f" % (1e3 * anc)) if anc is not None else "--"))
    tex.append("\\midrule")
    tex.append("pooled & %d/%d & %d/%d &  & %d/%d &  &  \\\\"
               % (ko_p, no_p, kr_p, nr_p, km_p, nm_p))
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}\n")

    # ---- decoupling --------------------------------------------------------
    md.append("## 4. Decoupling: P(success | VLM discrete correct / "
              "incorrect)\n")
    md.append("Success from ours_vlm rollouts; 'correct' = symmetry-"
              "corrected joint classification.\n")
    md.append("| scope | P(success given correct) | P(success given incorrect) |")
    md.append("|---|---|---|")
    pool_c = [0, 0]
    pool_i = [0, 0]
    for task in tasks:
        d = data[task]
        kc = nc = ki = ni = 0
        for r in d["rows"]:
            if "ours" not in r:
                continue
            s = int(bool(r["ours"]["success"]))
            if r["ours"].get("discrete_correct"):
                kc += s; nc += 1
            else:
                ki += s; ni += 1
        pool_c[0] += kc; pool_c[1] += nc
        pool_i[0] += ki; pool_i[1] += ni
        md.append("| %s | %s | %s |" % (task, _wilson_str(kc, nc),
                                        _wilson_str(ki, ni)))
        summary.setdefault("decoupling", {})[task] = {
            "given_correct": [kc, nc], "given_incorrect": [ki, ni]}
    md.append("| **pooled** | **%s** | **%s** |\n"
              % (_wilson_str(*pool_c), _wilson_str(*pool_i)))
    summary["decoupling"]["pooled"] = {"given_correct": pool_c,
                                       "given_incorrect": pool_i}

    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(os.path.join(TABLES_DIR, "campaign_d.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(TABLES_DIR, "campaign_d.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    _write_json(os.path.join(out_root, "summary.json"), summary)
    print("\n".join(md))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True,
                   choices=["smoke", "classify", "execute", "report"])
    p.add_argument("--tasks", nargs="*", default=list(TASKS_D))
    p.add_argument("--methods", nargs="*", default=["ours_vlm", "moka_vlm"])
    p.add_argument("--seed-start", type=int, default=SEEDS_D[0])
    p.add_argument("--seed-end", type=int, default=SEEDS_D[-1])
    args = p.parse_args(argv)
    for t in args.tasks:
        if t not in TASKS_D:
            raise SystemExit("task %r not in campaign D set %s"
                             % (t, TASKS_D))
    seeds = tuple(range(args.seed_start, args.seed_end + 1))
    if args.phase == "smoke":
        smoke(args.tasks)
    elif args.phase == "classify":
        for t in args.tasks:
            classify_task(t, seeds)
    elif args.phase == "execute":
        for t in args.tasks:
            if "ours_vlm" in args.methods:
                execute_ours_vlm(t, seeds)
            if "moka_vlm" in args.methods:
                execute_moka_vlm(t, seeds)
    elif args.phase == "report":
        report(tuple(args.tasks), seeds)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
