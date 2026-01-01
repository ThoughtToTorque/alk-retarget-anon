"""Phase-4 Campaign E: symmetry quantification + cross-category stress test.

Part 1 (``symmetry``, NO new rollouts): re-scores the existing Campaign A
rollouts (incl. the preserved out-of-scope cube pour) with symmetry-corrected
rotation errors.  For each task the object's symmetry group G (body frame)
is fixed a priori:

    nut_loosen / cap_twist / box_open : trivial  (handle / handle / door
                                        break all symmetries)
    rim_grasp (can)                   : SO(2) about the body z axis
    pour (elongated 8.0x2.2x4.4 box)  : C2 = {I, Rz(180)} primary
                                        (full D2 also evaluated; the two
                                        extra flips turn the box upside
                                        down and are reported if taken)
    pour_cube (out-of-scope near-cube): full octahedral rotation group
                                        (24 proper rotations)

A method's estimate T_est is symmetry-equivalent to T_gt iff
T_est = T_gt . P_d g P_d^{-1} for some g in G (P_d = demo object pose), so

    corrected_rot_err = min_{g in G} angle(M g^T),
    M = R_d^T R_gt^T R_est R_d          (relative rotation in body frame)

(for SO(2): the swing angle of M about the symmetry axis; the twist about
the axis is the unidentifiable stabilizer component).  The translation error
at the object center is invariant under g (g fixes the body origin), so only
rotation is re-scored.  Also extracts the Prop-2 identifiability margins
delta_ax = |c1-c2| and delta_lat = min_j dist(c_j, line(c1,c2)) from the
ALK quadruples already stored in the ours_full rollout JSONs.

Part 2 (``cross`` / ``cross_report``, new rollouts): cross-CATEGORY stress
test reproducing the paper's cross-object experiment (drink-bottle demo ->
bolt scene).  The demo of one nut task is retargeted onto the OTHER nut
category's target scenes; the task type follows the DEMO, the object follows
the TARGET scene:

    lift_r2s : nut_loosen demo (round nut, grasp handle + lift) executed in
               cap_twist target scenes (square nut).  Target task = "loosen
               (lift off) the SQUARE nut"; success = lift functional
               (>= 5 cm above settled height) on SquareNut_main.
    twist_s2r: cap_twist demo (square nut, grasp handle + in-place 45 deg
               twist) executed in nut_loosen target scenes (round nut).
               Target task = "twist the ROUND nut in place"; success =
               twist functional (>= 30 deg yaw, center within 3 cm) on
               RoundNut_main.

The scene-native checker of the scene's ORIGINAL task (twist for cap_twist
scenes, lift for nut_loosen scenes) is structurally unsatisfiable by the
cross-type demo motion (a lift demo never yaws the nut 30 deg; the twist
demo never lifts 5 cm) and is recorded per rollout as ``success_native``
purely as evidence of that action-type mismatch.  Both nuts carry their
handle at body +x with identical handle-bar geometry, so the cross-object
T_gt = pose_target(other nut) o pose_demo(demo nut)^{-1} is the semantically
correct handle->handle, ring->ring correspondence map.

Usage:
  python -m pipeline.campaign_e symmetry
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m pipeline.campaign_e cross \
      --direction lift_r2s --seed-start 1000 --seed-end 1029
  python -m pipeline.campaign_e cross_report
"""
import argparse
import csv
import itertools
import json
import os
import time
import traceback

import numpy as np

SIMBENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_MEDIUM = os.path.join(SIMBENCH, "data_medium")
CAMPAIGN_A = os.path.join(SIMBENCH, "results", "campaign_a")
OUT_ROOT = os.path.join(SIMBENCH, "results", "campaign_e")

SEEDS_A = tuple(range(1000, 1060))          # campaign A paired seeds
SEEDS_CROSS = tuple(range(1000, 1030))      # cross-category seeds (N=30)
PART1_METHODS = ("ours_full", "icp_centroid")

# (pseudo-task key) -> (rollout dir under campaign_a, pair dir under
# data_medium, target_instance)
PART1_TASKS = {
    "nut_loosen": ("nut_loosen", "nut_loosen", "RoundNut"),
    "cap_twist": ("cap_twist", "cap_twist", "SquareNut"),
    "rim_grasp": ("rim_grasp", "rim_grasp", "Can"),
    "box_open": ("box_open", "box_open", "Door"),
    "pour": ("pour", "pour", "cube"),
    "pour_cube": ("pour_cube_outofscope", "pour_cube_outofscope", "cube"),
}


# ---------------------------------------------------------------------------
# symmetry groups (body frame)
# ---------------------------------------------------------------------------

def _rot(axis, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _octahedral_group():
    """The 24 proper rotations of the cube: signed permutation matrices with
    det = +1."""
    mats, names = [], []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                M[i, p] = s
            if np.linalg.det(M) > 0.5:
                mats.append(M)
                ang = rotation_angle_deg_np(M)
                names.append("O%02d_%ddeg" % (len(mats) - 1, int(round(ang))))
    return list(zip(names, mats))


def rotation_angle_deg_np(R):
    tr = float(np.trace(np.asarray(R, dtype=np.float64)))
    return float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))))


# task -> ("discrete", [(name, R), ...]) or ("so2", axis)
SYM_GROUPS = {
    "nut_loosen": ("discrete", [("I", np.eye(3))]),
    "cap_twist": ("discrete", [("I", np.eye(3))]),
    "box_open": ("discrete", [("I", np.eye(3))]),
    "rim_grasp": ("so2", np.array([0.0, 0.0, 1.0])),
    # full D2 of a box with three distinct side lengths; Rz180 (flip about
    # the vertical axis, reversing the long axis) is the C2 subgroup the
    # campaign-A "flip-adjusted" column used
    "pour": ("discrete", [("I", np.eye(3)), ("Rz180", _rot("z", 180)),
                          ("Rx180", _rot("x", 180)),
                          ("Ry180", _rot("y", 180))]),
    "pour_cube": ("discrete", None),  # filled lazily (octahedral, 24)
}


def _twist_swing_deg(M, axis):
    """Split rotation M into twist about `axis` + residual swing (deg)."""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(M).as_quat()  # xyzw
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    p = float(np.dot(q[:3], a))
    tw = np.array([a[0] * p, a[1] * p, a[2] * p, q[3]])
    n = np.linalg.norm(tw)
    if n < 1e-12:  # pure 180-deg swing
        return 0.0, rotation_angle_deg_np(M)
    tw /= n
    R_tw = Rotation.from_quat(tw).as_matrix()
    swing = M @ R_tw.T
    twist_deg = float(np.degrees(
        2.0 * np.arccos(np.clip(abs(tw[3]), -1.0, 1.0))))
    return twist_deg, rotation_angle_deg_np(swing)


def sym_corrected_rot(task, T_est, T_gt, demo_pose):
    """Raw + mod-symmetry rotation error (deg) of T_est vs T_gt.

    Returns dict(raw, corrected, element, twist_deg, corrected_c2) where
    `element` is the argmin group element name ('twist' for SO(2)),
    corrected_c2 is the {I, Rz180}-only value (pour box; None elsewhere).
    """
    from scipy.spatial.transform import Rotation
    T_est = np.asarray(T_est, dtype=np.float64)
    T_gt = np.asarray(T_gt, dtype=np.float64)
    R_d = Rotation.from_quat(demo_pose["quat_xyzw"]).as_matrix()
    M = R_d.T @ T_gt[:3, :3].T @ T_est[:3, :3] @ R_d
    raw = rotation_angle_deg_np(M)
    kind, spec = SYM_GROUPS[task]
    if kind == "so2":
        twist, swing = _twist_swing_deg(M, spec)
        return {"raw": raw, "corrected": swing, "element": "twist",
                "twist_deg": twist, "corrected_c2": None}
    if spec is None:  # octahedral, lazily built
        spec = SYM_GROUPS[task] = ("discrete", _octahedral_group())[1]
        SYM_GROUPS[task] = ("discrete", spec)
    errs = [(rotation_angle_deg_np(M @ g.T), name) for name, g in spec]
    corrected, element = min(errs)
    c2 = None
    if task == "pour":
        c2 = min(e for e, name in errs if name in ("I", "Rz180"))
    return {"raw": raw, "corrected": corrected, "element": element,
            "twist_deg": None, "corrected_c2": c2}


# ---------------------------------------------------------------------------
# Prop-2 margins from an ALK quadruple
# ---------------------------------------------------------------------------

def alk_margins(alk):
    """delta_ax = |c1-c2|; delta_lat = min over lateral rows of the distance
    to the infinite line through (c1, c2).  Units: m."""
    A = np.asarray(alk, dtype=np.float64)
    c1, c2 = A[0], A[1]
    d_ax = float(np.linalg.norm(c1 - c2))
    if d_ax < 1e-9:
        return {"delta_ax_m": d_ax, "delta_lat_m": 0.0}
    u = (c2 - c1) / d_ax
    lats = []
    for j in (2, 3):
        v = A[j] - c1
        lats.append(float(np.linalg.norm(v - u * float(v @ u))))
    return {"delta_ax_m": d_ax, "delta_lat_m": float(min(lats))}


# ---------------------------------------------------------------------------
# Part 1 driver
# ---------------------------------------------------------------------------

def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _quantiles(xs, qs=(0.5, 0.9)):
    xs = np.asarray(xs, dtype=np.float64)
    return [float(np.quantile(xs, q)) for q in qs] if len(xs) else \
        [float("nan")] * len(qs)


def cmd_symmetry(args):
    os.makedirs(OUT_ROOT, exist_ok=True)
    rows = []
    margin_rows = []
    n_checked = 0
    for task, (roll_dir, pair_dir, inst) in PART1_TASKS.items():
        demo_pose = None
        for seed in SEEDS_A:
            pj = os.path.join(DATA_MEDIUM, pair_dir, str(seed), "pair.json")
            if not os.path.exists(pj):
                continue
            pair = _load_json(pj)
            if demo_pose is None:
                demo_pose = pair["demo_object_poses"][inst]
            for method in PART1_METHODS:
                rp = os.path.join(CAMPAIGN_A, roll_dir, str(seed),
                                  "rollout_%s.json" % method)
                if not os.path.exists(rp):
                    continue
                r = _load_json(rp)
                if r.get("T_map") is None:
                    continue
                res = sym_corrected_rot(task, r["T_map"], r["T_gt"],
                                        demo_pose)
                # sanity: recomputed raw == stored rot_err_deg
                stored = r.get("rot_err_deg")
                if stored is not None:
                    assert abs(res["raw"] - stored) < 1e-6, \
                        (task, seed, method, res["raw"], stored)
                    n_checked += 1
                rows.append({
                    "task": task, "method": method, "seed": seed,
                    "success": int(bool(r.get("success"))),
                    "rot_raw_deg": res["raw"],
                    "rot_sym_deg": res["corrected"],
                    "rot_sym_c2_deg": res["corrected_c2"],
                    "sym_element": res["element"],
                    "twist_deg": res["twist_deg"],
                    "trans_err_m": r.get("trans_err_m"),
                })
                if method == "ours_full" and r.get("oracle"):
                    md = alk_margins(r["oracle"]["demo_alk"])
                    mt = alk_margins(r["oracle"]["target_alk"])
                    margin_rows.append({
                        "task": task, "seed": seed,
                        "delta_ax_demo_m": md["delta_ax_m"],
                        "delta_lat_demo_m": md["delta_lat_m"],
                        "delta_ax_target_m": mt["delta_ax_m"],
                        "delta_lat_target_m": mt["delta_lat_m"],
                    })
    print("part 1: %d rollouts scored, %d raw-error sanity checks passed"
          % (len(rows), n_checked))

    # tidy CSVs
    for name, rws in (("symmetry_rollouts.csv", rows),
                      ("margins_rollouts.csv", margin_rows)):
        path = os.path.join(OUT_ROOT, name)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rws[0].keys()))
            w.writeheader()
            w.writerows(rws)
        print("wrote", path)

    _write_symmetry_tables(rows, margin_rows)
    return 0


GROUP_LABEL = {
    "nut_loosen": "trivial", "cap_twist": "trivial", "box_open": "trivial",
    "rim_grasp": "SO(2) axis", "pour": "C2/D2 flips",
    "pour_cube": "octahedral (24)",
}
TASK_ORDER = ("nut_loosen", "cap_twist", "box_open", "rim_grasp", "pour",
              "pour_cube")


def _agg(rows, task, method):
    sel = [r for r in rows if r["task"] == task and r["method"] == method]
    if not sel:
        return None
    raw = [r["rot_raw_deg"] for r in sel]
    cor = [r["rot_sym_deg"] for r in sel]
    succ = [r for r in sel if r["success"]]
    nontriv = [r for r in sel if r["sym_element"] not in ("I", "twist")
               and (r["rot_raw_deg"] - r["rot_sym_deg"]) > 1.0]
    out = {
        "n": len(sel), "k_succ": len(succ),
        "raw_med": _quantiles(raw)[0], "raw_p90": _quantiles(raw)[1],
        "cor_med": _quantiles(cor)[0], "cor_p90": _quantiles(cor)[1],
        "n_nontrivial": len(nontriv),
        "succ_raw_gt20": sum(1 for r in succ if r["rot_raw_deg"] > 20.0),
        "succ_cor_gt20": sum(1 for r in succ if r["rot_sym_deg"] > 20.0),
        "elements": {},
        "twist_med_succ": None,
    }
    for r in sel:
        out["elements"][r["sym_element"]] = \
            out["elements"].get(r["sym_element"], 0) + 1
    tw = [r["twist_deg"] for r in succ if r["twist_deg"] is not None]
    if tw:
        out["twist_med_succ"] = _quantiles(tw)[0]
    return out


def _write_symmetry_tables(rows, margin_rows):
    md = ["# Campaign E — symmetry-corrected rotation error "
          "(raw vs mod-G, Campaign A rollouts)", "",
          "corrected = min over the object's symmetry group G (body frame); "
          "'non-triv' = rollouts whose argmin is a non-identity element "
          "(improvement > 1 deg).  'succ raw>20 / sym>20' = SUCCESSFUL "
          "rollouts past the pipeline's own 20-deg mapping-failure "
          "heuristic before/after correction.", "",
          "| task | G | method | n | succ | raw med/p90 (deg) | "
          "mod-G med/p90 (deg) | non-triv | succ raw>20 | succ sym>20 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    tex = ["% Campaign E symmetry-corrected rotation errors",
           "\\begin{tabular}{llrrrrrrr}", "\\toprule",
           "task & $G$ & method & succ. & raw med/p90 & mod-$G$ med/p90 & "
           "non-triv. & \\shortstack{succ.\\\\raw$>$20} & "
           "\\shortstack{succ.\\\\sym$>$20} \\\\", "\\midrule"]
    summary = {}
    for task in TASK_ORDER:
        for method in PART1_METHODS:
            a = _agg(rows, task, method)
            if a is None:
                continue
            summary["%s/%s" % (task, method)] = a
            md.append(
                "| %s | %s | %s | %d | %d | %.1f / %.1f | %.1f / %.1f | "
                "%d | %d | %d |"
                % (task, GROUP_LABEL[task], method, a["n"], a["k_succ"],
                   a["raw_med"], a["raw_p90"], a["cor_med"], a["cor_p90"],
                   a["n_nontrivial"], a["succ_raw_gt20"], a["succ_cor_gt20"]))
            tex.append(
                "%s & %s & %s & %d/%d & %.1f / %.1f & %.1f / %.1f & %d & "
                "%d & %d \\\\"
                % (task.replace("_", "\\_"), GROUP_LABEL[task],
                   method.replace("_", "\\_"), a["k_succ"], a["n"],
                   a["raw_med"], a["raw_p90"], a["cor_med"], a["cor_p90"],
                   a["n_nontrivial"], a["succ_raw_gt20"], a["succ_cor_gt20"]))
        tex.append("\\midrule" if task != TASK_ORDER[-1] else "\\bottomrule")
    tex.append("\\end{tabular}")

    # margins table
    md += ["", "## Prop-2 identifiability margins (measured from ALK, "
           "ours_full rollouts)", "",
           "| task | delta_ax demo (cm) | delta_lat demo (cm) | "
           "delta_ax target med [p10, p90] (cm) | "
           "delta_lat target med [p10, p90] (cm) |",
           "|---|---|---|---|---|"]
    tex += ["", "% Prop-2 margins", "\\begin{tabular}{lrrrr}", "\\toprule",
            "task & $\\delta_{ax}$ demo & $\\delta_{lat}$ demo & "
            "$\\delta_{ax}$ target med [p10,p90] & "
            "$\\delta_{lat}$ target med [p10,p90] \\\\", "\\midrule"]
    for task in TASK_ORDER:
        sel = [m for m in margin_rows if m["task"] == task]
        if not sel:
            continue
        axd = sel[0]["delta_ax_demo_m"] * 100
        latd = sel[0]["delta_lat_demo_m"] * 100
        axt = np.asarray([m["delta_ax_target_m"] for m in sel]) * 100
        latt = np.asarray([m["delta_lat_target_m"] for m in sel]) * 100
        fmt = lambda v: "%.2f [%.2f, %.2f]" % (
            np.quantile(v, 0.5), np.quantile(v, 0.1), np.quantile(v, 0.9))
        md.append("| %s | %.2f | %.2f | %s | %s |"
                  % (task, axd, latd, fmt(axt), fmt(latt)))
        tex.append("%s & %.2f & %.2f & %s & %s \\\\"
                   % (task.replace("_", "\\_"), axd, latd, fmt(axt),
                      fmt(latt)))
        summary.setdefault("margins", {})[task] = {
            "delta_ax_demo_cm": axd, "delta_lat_demo_cm": latd,
            "delta_ax_target_cm_q10_50_90":
                [float(np.quantile(axt, q)) for q in (0.1, 0.5, 0.9)],
            "delta_lat_target_cm_q10_50_90":
                [float(np.quantile(latt, q)) for q in (0.1, 0.5, 0.9)],
        }
    tex += ["\\bottomrule", "\\end{tabular}"]

    with open(os.path.join(OUT_ROOT, "symmetry_table.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(OUT_ROOT, "symmetry_table.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    with open(os.path.join(OUT_ROOT, "symmetry_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", os.path.join(OUT_ROOT, "symmetry_table.{md,tex}"),
          "and symmetry_summary.json")


# ---------------------------------------------------------------------------
# Part 2: cross-category runner
# ---------------------------------------------------------------------------

# direction -> (demo_task, target_task)
DIRECTIONS = {
    "lift_r2s": ("nut_loosen", "cap_twist"),
    "twist_s2r": ("cap_twist", "nut_loosen"),
}
CROSS_METHODS = ("ours_full", "ours_full_adaptive")


def _skill_success(env, demo_task, target_task):
    """The DEMO task's success functional evaluated on the TARGET object
    (= the target task of the cross-category experiment)."""
    import robosuite.utils.transform_utils as T
    from simtasks import envs
    body = envs.TASKS[target_task].target_body
    init = env._simtasks_init_obj_pose
    pos, quat = envs._body_pose(env, body)
    if demo_task == "nut_loosen":     # lift-off functional
        return bool(pos[2] - init["pos"][2] >= 0.05)
    yaw = float(T.mat2euler(T.quat2mat(quat))[2])   # twist functional
    dyaw = abs(envs._wrap_angle(yaw - init["yaw"]))
    dxy = float(np.linalg.norm(np.asarray(pos[:2])
                               - np.asarray(init["pos"])[:2]))
    return bool(dyaw >= np.deg2rad(30.0) and dxy <= 0.03)


def run_cross_pair(direction, seed, env, adaptive=False):
    """One cross-category rollout: demo of DIRECTIONS[direction][0] retargeted
    into the target scene of DIRECTIONS[direction][1] (same seed as
    campaign A / data_medium).  Mirrors pipeline.retarget_runner.run_pair with
    the demo and target sides drawn from different tasks."""
    from alkbench import (procrustes, bounded_registration,
                          adaptive_registration, transform_points)
    from simtasks import capture, envs, motion
    from pipeline import oracle, perception
    from pipeline import retarget_runner as rr

    demo_task, target_task = DIRECTIONS[direction]
    t0 = time.time()
    method = "ours_full_adaptive" if adaptive else "ours_full"
    result = {"direction": direction, "demo_task": demo_task,
              "target_task": target_task, "seed": seed, "method": method,
              "variant": method, "tier": "medium", "adaptive": bool(adaptive),
              "success": False, "success_native": False,
              "failure_stage": None}

    # demo side (fixed demo scene of the demo task)
    demo = rr.load_demo(demo_task, DATA_MEDIUM)
    # demo_object_poses is identical in every seed's pair.json (fixed demo
    # scene); seed 1000 always exists in the campaign-A medium data root
    demo_pair = _load_json(os.path.join(DATA_MEDIUM, demo_task, "1000",
                                        "pair.json"))
    demo_inst = envs.TASKS[demo_task].target_instance
    demo_pose = demo_pair["demo_object_poses"][demo_inst]

    # target side (campaign-A medium-tier scene of the OTHER task)
    pair = _load_json(os.path.join(DATA_MEDIUM, target_task, str(seed),
                                   "pair.json"))
    target_inst = pair["target_instance"]
    target_pose = pair["target_object_poses"][target_inst]
    target_scene = capture.load_capture(
        os.path.join(DATA_MEDIUM, target_task, str(seed), "target", "scene"))

    # perception: each side masks its OWN task's instance (round vs square)
    cam = perception.pick_camera(demo["scene"], demo_task)
    p_demo = perception.perceive(demo["scene"], demo_task, k=8, seed=0,
                                 camera=cam)
    p_tgt = perception.perceive(target_scene, target_task, k=8, seed=0,
                                camera=cam)
    result["camera"] = cam
    result["perception"] = {
        "demo": {k: p_demo[k] for k in
                 ("mask_pixels", "n_points", "centroid_err_m", "sanity_ok")},
        "target": {k: p_tgt[k] for k in
                   ("mask_pixels", "n_points", "centroid_err_m",
                    "sanity_ok")},
    }

    # cross-object semantic GT: both nuts carry the handle at body +x, so the
    # body-frame relative transform maps handle->handle, ring->ring
    T_gt = oracle.gt_relative_transform(demo_pose, target_pose)
    result["T_gt"] = T_gt.tolist()

    pre_grasp = [kf for kf in demo["keyframes"]
                 if kf["name"] == "pre_grasp"][0]
    try:
        orc = oracle.solve(p_demo, p_tgt, T_gt,
                           np.asarray(pre_grasp["tcp_pos"]), seed=0,
                           depth_consistent="auto" if adaptive else False)
    except ValueError as e:
        result["failure_stage"] = "alk_degenerate"
        result["error"] = str(e)
        result["time_s"] = round(time.time() - t0, 1)
        return result
    result["oracle"] = rr._json_safe({
        "demo_choice": orc["demo_choice"],
        "target_choice": orc["target_choice"],
        "demo_alk": orc["demo_alk"], "target_alk": orc["target_alk"],
        "demo_grasp": orc["demo_grasp"],
        "target_grasp_centroid": orc["target_grasp_centroid"],
        "diagnostics": orc["diagnostics"],
        "depth_consistent": orc["depth_consistent"],
        "slender": orc["slender"],
    })

    T0 = procrustes(orc["demo_alk"], orc["target_alk"])
    src = p_demo["cands"].points3d
    tgt = p_tgt["cands"].points3d
    if adaptive:
        reg = adaptive_registration(src, tgt, T0, orc["demo_alk"])
        result["adaptive_triggered"] = bool(
            reg["adaptive"]["adaptive_triggered"])
    else:
        reg = bounded_registration(src, tgt, T0)
    T_map = reg["T"]
    rot0, trans0 = rr.map_errors(T0, T_gt, demo_pose["pos"],
                                 target_pose["pos"])
    rot_err, trans_err = rr.map_errors(T_map, T_gt, demo_pose["pos"],
                                       target_pose["pos"])
    result.update({
        "T_init": np.asarray(T0).tolist(),
        "T_map": np.asarray(T_map).tolist(),
        "chamfer_before_m": float(reg["chamfer_init"]),
        "chamfer_after_m": float(reg["chamfer_final"]),
        "rot_err_init_deg": rot0, "trans_err_init_m": trans0,
        "rot_err_deg": rot_err, "trans_err_m": trans_err,
    })

    # grasp-point correction (identical to run_pair; correction always ON)
    region = np.asarray(orc["target_grasp_region_points"])
    mapped_gd = transform_points(T_map, orc["demo_grasp"][None])[0]
    target_grasp = region[np.argmin(
        np.linalg.norm(region - mapped_gd, axis=1))]
    result["target_grasp_used"] = target_grasp.tolist()
    result["grasp_correction_norm_m"] = float(
        np.linalg.norm(target_grasp - mapped_gd))
    mapped_wps = rr.retarget_waypoint_dicts(
        demo["waypoints"], T_map, demo_grasp=orc["demo_grasp"],
        target_grasp=target_grasp)

    # execute in the target task's scene; grasp check on the TARGET object
    envs.reset_with_seed(env, pair["target_seed"])
    exec_res = motion.execute_waypoints(env, mapped_wps,
                                        grasp_check=rr._grasp_check(
                                            target_task))
    succ_skill = _skill_success(env, demo_task, target_task)
    succ_native = bool(envs.success_checker(target_task)(env))
    result.update({
        "grasped": exec_res["grasped"],
        "waypoints_converged": exec_res["converged"],
        "n_exec_steps": exec_res["n_steps"],
        "success": succ_skill,          # the cross-category target task
        "success_skill": succ_skill,
        "success_native": succ_native,  # scene's ORIGINAL task (mismatched)
    })
    if not succ_skill:
        if not (p_demo["sanity_ok"] and p_tgt["sanity_ok"]):
            stage = "perception"
        elif trans_err > 0.05 or rot_err > 20.0:
            stage = "mapping"
        elif exec_res["grasped"] is False:
            stage = "grasp"
        else:
            stage = "post_action"
        result["failure_stage"] = stage
    result["time_s"] = round(time.time() - t0, 1)
    return result


def cmd_cross(args):
    from simtasks import envs
    from pipeline import campaign as camp
    from pipeline import retarget_runner as rr
    direction = args.direction
    demo_task, target_task = DIRECTIONS[direction]
    out_dir = os.path.join(OUT_ROOT, "cross", direction)
    os.makedirs(out_dir, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_end + 1))
    env = camp.make_tier_env(target_task, "medium")
    n_done = 0
    try:
        for seed in seeds:
            for method, adaptive in (("ours_full", False),
                                     ("ours_full_adaptive", True)):
                path = os.path.join(out_dir, str(seed),
                                    "rollout_%s.json" % method)
                if os.path.exists(path):
                    continue
                t0 = time.time()
                try:
                    r = run_cross_pair(direction, seed, env,
                                       adaptive=adaptive)
                except Exception as e:
                    traceback.print_exc()
                    r = {"direction": direction, "demo_task": demo_task,
                         "target_task": target_task, "seed": seed,
                         "method": method, "success": False,
                         "success_native": False,
                         "failure_stage": "exception", "error": repr(e),
                         "time_s": round(time.time() - t0, 1)}
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(rr._json_safe(r), f, indent=2)
                os.rename(tmp, path)
                n_done += 1
                print("[%s %d %-18s] skill=%s native=%s stage=%-11s rot=%s "
                      "trans=%s (%.1fs)"
                      % (direction, seed, method, r.get("success"),
                         r.get("success_native"),
                         str(r.get("failure_stage")),
                         ("%.1fdeg" % r["rot_err_deg"])
                         if r.get("rot_err_deg") is not None else "-",
                         ("%.1fmm" % (1e3 * r["trans_err_m"]))
                         if r.get("trans_err_m") is not None else "-",
                         time.time() - t0), flush=True)
    finally:
        env.close()
    print("done: %s, %d new rollouts" % (direction, n_done), flush=True)
    return 0


def cmd_cross_diag(args):
    """Re-execute the stored ours_full cross rollouts (same T_map / grasp
    correction, deterministic reset) and record the FINAL object state
    (dz, dyaw, dxy vs the settled initial pose) that the success functionals
    threshold on -- attribution data for the post_action failures.  Also a
    determinism check: the reproduced success bit must match the stored one.
    Writes results/campaign_e/cross/<direction>/diag.json."""
    import robosuite.utils.transform_utils as T
    from simtasks import envs, motion
    from pipeline import campaign as camp
    from pipeline import retarget_runner as rr
    for direction, (demo_task, target_task) in DIRECTIONS.items():
        out_dir = os.path.join(OUT_ROOT, "cross", direction)
        demo = rr.load_demo(demo_task, DATA_MEDIUM)
        env = camp.make_tier_env(target_task, "medium")
        body = envs.TASKS[target_task].target_body
        diags = []
        try:
            for seed in SEEDS_CROSS:
                p = os.path.join(out_dir, str(seed),
                                 "rollout_ours_full.json")
                if not os.path.exists(p):
                    continue
                r = _load_json(p)
                if r.get("T_map") is None:
                    continue
                pair = _load_json(os.path.join(DATA_MEDIUM, target_task,
                                               str(seed), "pair.json"))
                wps = rr.retarget_waypoint_dicts(
                    demo["waypoints"], np.asarray(r["T_map"]),
                    demo_grasp=np.asarray(r["oracle"]["demo_grasp"]),
                    target_grasp=np.asarray(r["target_grasp_used"]))
                envs.reset_with_seed(env, pair["target_seed"])
                init = env._simtasks_init_obj_pose
                motion.execute_waypoints(env, wps,
                                         grasp_check=rr._grasp_check(
                                             target_task))
                pos, quat = envs._body_pose(env, body)
                yaw = float(T.mat2euler(T.quat2mat(quat))[2])
                d = {
                    "seed": seed,
                    "dz_m": float(pos[2] - init["pos"][2]),
                    "dyaw_deg": float(np.degrees(
                        abs(envs._wrap_angle(yaw - init["yaw"])))),
                    "dxy_m": float(np.linalg.norm(
                        np.asarray(pos[:2])
                        - np.asarray(init["pos"])[:2])),
                    "stored_success": bool(r["success"]),
                    "reproduced_success": _skill_success(env, demo_task,
                                                         target_task),
                }
                d["deterministic"] = (d["stored_success"]
                                      == d["reproduced_success"])
                diags.append(d)
                print("[diag %s %d] dz=%.3f dyaw=%.1f dxy=%.3f stored=%s "
                      "repro=%s" % (direction, seed, d["dz_m"],
                                    d["dyaw_deg"], d["dxy_m"],
                                    d["stored_success"],
                                    d["reproduced_success"]), flush=True)
        finally:
            env.close()
        with open(os.path.join(out_dir, "diag.json"), "w") as f:
            json.dump(diags, f, indent=2)
        n_det = sum(1 for d in diags if d["deterministic"])
        print("%s: %d diags, %d/%d deterministic reproductions"
              % (direction, len(diags), n_det, len(diags)), flush=True)
    return 0


# ---------------------------------------------------------------------------
# Part 2 report
# ---------------------------------------------------------------------------

def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def cmd_cross_report(args):
    rows = []
    for direction in DIRECTIONS:
        for seed in SEEDS_CROSS:
            for method in CROSS_METHODS:
                p = os.path.join(OUT_ROOT, "cross", direction, str(seed),
                                 "rollout_%s.json" % method)
                if os.path.exists(p):
                    r = _load_json(p)
                    r["_direction"] = direction
                    rows.append(r)
    # within-category references on the SAME seeds from campaign A
    refs = {}
    for task in ("nut_loosen", "cap_twist"):
        ks = []
        for seed in SEEDS_CROSS:
            p = os.path.join(CAMPAIGN_A, task, str(seed),
                             "rollout_ours_full.json")
            if os.path.exists(p):
                ks.append(int(bool(_load_json(p).get("success"))))
        refs[task] = (sum(ks), len(ks))

    md = ["# Campaign E — cross-category stress test (nut demo <-> nut "
          "target, N=%d seeds %d..%d)" % (len(SEEDS_CROSS), SEEDS_CROSS[0],
                                          SEEDS_CROSS[-1]), "",
          "success = the cross-category TARGET task (demo task's functional "
          "on the target object); native = the scene's original same-object "
          "checker (structurally unsatisfiable by the cross-type demo, "
          "recorded as evidence of the action-type mismatch).", "",
          "| direction | method | success | grasped | native | rot med "
          "(deg) | trans med (mm) | within-cat ref (same seeds) |",
          "|---|---|---|---|---|---|---|---|"]
    tex = ["% Campaign E cross-category", "\\begin{tabular}{llrrrrr}",
           "\\toprule",
           "direction & method & success & grasped & rot med & trans med & "
           "within-cat.\\ ref \\\\", "\\midrule"]
    summary = {"refs": {k: {"k": v[0], "n": v[1]} for k, v in refs.items()}}
    for direction in DIRECTIONS:
        demo_task, target_task = DIRECTIONS[direction]
        ref_k, ref_n = refs[demo_task]
        for method in CROSS_METHODS:
            sel = [r for r in rows if r["_direction"] == direction
                   and r["method"] == method]
            if not sel:
                continue
            n = len(sel)
            k = sum(1 for r in sel if r.get("success"))
            kn = sum(1 for r in sel if r.get("success_native"))
            kg = sum(1 for r in sel if r.get("grasped"))
            rot = [r["rot_err_deg"] for r in sel
                   if r.get("rot_err_deg") is not None]
            tr = [1e3 * r["trans_err_m"] for r in sel
                  if r.get("trans_err_m") is not None]
            p, lo, hi = _wilson(k, n)
            md.append("| %s (%s demo -> %s scene) | %s | %d/%d = %.2f "
                      "[%.2f, %.2f] | %d/%d | %d/%d | %.1f | %.1f | "
                      "%d/%d = %.2f |"
                      % (direction, demo_task, target_task, method, k, n, p,
                         lo, hi, kg, n, kn, n, np.median(rot), np.median(tr),
                         ref_k, ref_n, ref_k / max(ref_n, 1)))
            tex.append("%s & %s & %d/%d = %.2f & %d/%d & %.1f & %.1f & "
                       "%d/%d = %.2f \\\\"
                       % (direction.replace("_", "\\_"),
                          method.replace("_", "\\_"), k, n, p, kg, n,
                          np.median(rot), np.median(tr), ref_k, ref_n,
                          ref_k / max(ref_n, 1)))
            stages = {}
            for r in sel:
                if not r.get("success"):
                    stages[str(r.get("failure_stage"))] = \
                        stages.get(str(r.get("failure_stage")), 0) + 1
            summary["%s/%s" % (direction, method)] = {
                "k": k, "n": n, "wilson": [p, lo, hi], "grasped": kg,
                "native": kn, "rot_med_deg": float(np.median(rot)),
                "rot_p90_deg": float(np.quantile(rot, 0.9)),
                "trans_med_mm": float(np.median(tr)),
                "failure_stages": stages,
                "ref_same_seeds": {"task": demo_task, "k": ref_k, "n": ref_n},
            }
        tex.append("\\midrule" if direction != list(DIRECTIONS)[-1]
                   else "\\bottomrule")
    tex.append("\\end{tabular}")
    with open(os.path.join(OUT_ROOT, "cross_table.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(OUT_ROOT, "cross_table.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    with open(os.path.join(OUT_ROOT, "cross_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n".join(md))
    print("wrote cross_table.{md,tex} + cross_summary.json in", OUT_ROOT)
    return 0


# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("symmetry")
    pc = sub.add_parser("cross")
    pc.add_argument("--direction", required=True,
                    choices=sorted(DIRECTIONS.keys()))
    pc.add_argument("--seed-start", type=int, default=SEEDS_CROSS[0])
    pc.add_argument("--seed-end", type=int, default=SEEDS_CROSS[-1])
    sub.add_parser("cross_report")
    sub.add_parser("cross_diag")
    args = p.parse_args(argv)
    return {"symmetry": cmd_symmetry, "cross": cmd_cross,
            "cross_report": cmd_cross_report,
            "cross_diag": cmd_cross_diag}[args.cmd](args)


if __name__ == "__main__":
    import sys
    sys.exit(main())
