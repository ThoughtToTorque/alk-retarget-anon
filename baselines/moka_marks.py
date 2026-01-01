"""Campaign J -- MOKA re-implemented with its ACTUAL interface: mark-based
visual prompting (Liu, Fang, Abbeel, Levine, "MOKA: Open-World Robotic
Manipulation through Mark-Based Visual Prompting", RSS 2024 / arXiv 2403.03174).

Why this module exists
----------------------
`baselines/moka_style.py` (registry `moka_oracle` / `moka_vlm`) asks the model
for FREE-FORM PIXEL COORDINATES.  MOKA never does that: its whole thesis is
that "VLMs are better at multiple-choice problems than directly producing
continuous-valued locations", so it annotates candidate marks on the image and
has the VLM SELECT among them.  That is the same interface class our pipeline
uses, so a coordinate-regression MOKA is not the published method and not a
fair baseline.  This module implements the published method.  It is an independent
implementation written for this benchmark, not the authors' code; every
adaptation is stated in the paper where the baseline is introduced.

What MOKA actually does (paper Sec. IV-B..IV-D + Appendix B)
------------------------------------------------------------
1. HIGH-LEVEL REASONING (one query per task, Table IV): task string + initial
   observation -> a JSON list of subtasks, each with
   `instruction` / `object_grasped` / `object_unattached` / `motion_direction`.
2. POINT-BASED AFFORDANCE PROPOSAL (Appendix B-B): GroundedSAM segments the
   named objects; `K` boundary keypoints are drawn by FARTHEST POINT SAMPLING
   ON THE OBJECT CONTOUR, plus the mask's geometric center -> `K+1` indexed,
   colour-coded candidate marks per object ("P[i]" on object_grasped, "Q[j]"
   on object_unattached).  Free-space waypoints get a 5x5 grid of tiles named
   in chess notation (columns a..e left->right, rows 1..5 bottom->top).
3. LOW-LEVEL REASONING (one query per subtask, Tables V-VII): the annotated
   image + the subtask fields -> a JSON dict selecting
   `grasp_keypoint`, `function_keypoint`, `target_keypoint`,
   `pre_contact_tile`, `post_contact_tile`, `pre_contact_height`,
   `post_contact_height`, `target_angle`.
4. MOTION GENERATION (Sec. IV-D + Appendix B-D): object keypoints are lifted
   with the registered depth image; free-space waypoints are sampled uniformly
   inside the selected tile at the VLM-declared height ("same"/"above");
   the grasp is NOT taken from the VLM point directly -- MOKA samples
   *30 antipodal 4-DoF grasp proposals* (DexNet-2.0 sampler) on the object's
   depth crop and executes "the position and orientation of the grasp
   candidate that is closest to x_grasp"; the object orientation during the
   manipulation phase is the x_grasp -> x_function axis with the VLM's
   `target_angle`.

Adaptations to the single-object, one-shot benchmark (EVERY one of them, for
the paper's adaptation-notes list -- see docs/REPRODUCE.md)
------------------------------------------------------------------------
A1. SEGMENTATION.  MOKA runs GroundedSAM from object names.  We hand MOKA the
    simulator's ground-truth instance mask -- the SAME mask our pipeline's
    perception consumes.  Apples-to-apples, and strictly generous to MOKA
    (perfect open-vocabulary detection).
A2. NUMBER OF MARKS.  MOKA's figures show K+1 ~ 5-6 marks.  We use K = 7
    contour FPS points + the mask centroid = 8 marks, i.e. exactly the number
    of candidates our own discrete interface offers, so neither method gets a
    marking-density advantage.
A3. CANDIDATE GENERATION.  MOKA's, not ours: farthest point sampling on the
    2D mask CONTOUR plus the geometric center.  Our pipeline instead uses
    2D k-means cluster centres over the whole mask.  Deliberately different --
    this is the one place the task instructions require MOKA's own procedure.
A4. object_unattached IS ALWAYS EMPTY.  All five benchmark tasks manipulate a
    single object with no second, unattached object.  MOKA's own spec says
    `target_keypoint` is empty exactly when object_unattached is empty, so
    only the P[i] marks are drawn and `target_keypoint` is not requested.
    The tile channel and `function_keypoint` are still requested and parsed.
A5. SUBTASK COUNT.  Each benchmark task is one subtask (one grasp + one
    prescribed post-grasp motion), so MOKA's loop runs once: 1 high-level
    query per task (cached, exactly like our cached demo-side query) and 1
    low-level query per rollout.  Query counts are therefore IDENTICAL to
    ours per rollout.
A6. TOP-DOWN CAMERA.  MOKA assumes a top-down camera.  We keep the benchmark's
    per-task camera (agentview / sideview for box_open) so both methods see
    the same pixels.  The grid still divides the image with MOKA's chess
    convention.
A7. IMAGE UPSCALING.  MOKA's marks are drawn on the FULL image (the grid needs
    it), but our renders are 256x256 and Campaign D established that Qwen3-VL
    misreads small bitmap labels.  The full image is integer-upscaled 3x
    (-> 768 px) before the marks and the grid are drawn.  This is the exact
    analogue of the crop-zoom our own markup uses, and it changes no geometry.
A8. TILE -> WAYPOINT.  MOKA samples the free-space waypoint UNIFORMLY inside
    the chosen tile; we take the tile CENTRE (deterministic, and the minimum
    expected error inside the tile -- generous to MOKA).
A9. GRASP-PROPOSAL BUDGET.  MOKA samples 30 antipodal proposals; we sample up
    to 200 so the nearest-proposal step is not quantisation-limited.  Again
    generous to MOKA.  `n_proposals` is a parameter, so the 30-proposal
    setting is reproducible.
A10. 4-DoF GRASP -> RIGID TRANSFORM.  MOKA executes its 4-DoF grasp and then a
    motion of its own construction; our shared executor needs ONE world-frame
    T_map applied to the demo waypoint template (that is what makes every
    method in the campaign comparable).  We therefore build
        T_map = Translate(g_t) . Rz(dpsi) . Translate(-g_d)
    where g_d, g_t are the anchor POSITIONS (the antipodal grasp proposals
    closest to the demo grasp point and to MOKA's selected grasp keypoint --
    MOKA's own "use the candidate closest to x_grasp" rule; which side is
    snapped is the `snap` argument, frozen to "both" on non-eval data), and
    dpsi = wrap_pi(psi_t - psi_d) is the change in the azimuth of the
    grasp -> function AXIS, which is MOKA's own object-orientation
    representation (Sec. IV-D: "we use the vector from x_grasp to x_function
    to specify the orientation of the object").  The (grasp, function) index
    pair is selected by the VLM on both the demo and the target image.
    This gives MOKA a genuine ROTATION channel -- strictly more than the
    translation-only `moka_oracle` of Campaign A.
    NOTE: the antipodal proposal's OWN yaw was measured on non-eval scenes and
    is near-useless here (a 4-DoF antipodal grasp on a nut ring is ambiguous
    between "across the handle bar" and "along it"), so it is recorded as a
    diagnostic but not used to build T_map; the axis channel is used instead.
    Both choices are MOKA's; the better one was frozen on dev data.
A11. DEMO-SIDE ANCHOR.  MOKA is zero-shot and has no demonstration.  Its demo
    anchor here is the demonstration's own recorded pre_grasp TCP projected
    onto the nearest demo object point (`pipeline.oracle.demo_grasp_point`) --
    identical to Campaign A's `moka_oracle`, and available to every method.
A12. NO REGISTRATION.  Faithful to MOKA: the pose comes entirely from the
    keypoint + grasp proposal.  No Chamfer refinement, no ALK.
"""
import json
import os

import numpy as np

from alkbench import transform_points
from alkbench.candidates import project
from alkbench.discrete import _MARKER_COLORS, encode_png
from pipeline import oracle

from baselines import common

VLM_DEFAULTS = {
    "base_url": "http://127.0.0.1:8199/v1",
    "model": "Qwen/Qwen3-VL-32B-Instruct-FP8",
    "api_key": "local",
}

K_BOUNDARY = 7          # A2: 7 contour FPS points + 1 centroid = 8 marks
GRID_N = 5              # MOKA: 5 x 5 tiles
UPSCALE = 3             # A7: 256 -> 768 px
N_GRASP_PROPOSALS = 200  # A9 (MOKA's own number is 30)
GRIPPER_MAX_WIDTH = 0.085   # Robotiq-85
FRICTION_ANGLE_DEG = 30.0   # DexNet-2.0 antipodality cone (mu ~ 0.58)
ABOVE_HEIGHT_M = 0.10       # A8: "above" = contact height + 10 cm
HANDLE_RADIUS_M = 0.08      # A13: GroundedSAM("door handle") stand-in radius


# ---------------------------------------------------------------------------
# bitmap font (digits live in alkbench.discrete; add the letters MOKA needs)
# ---------------------------------------------------------------------------

_GLYPHS = {
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "010", "010", "010"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
    "A": ["111", "101", "111", "101", "101"],
    "B": ["110", "101", "110", "101", "110"],
    "C": ["111", "100", "100", "100", "111"],
    "D": ["110", "101", "101", "101", "110"],
    "E": ["111", "100", "111", "100", "111"],
    "P": ["111", "101", "111", "100", "100"],
}

COLUMNS = "ABCDE"[:GRID_N]


def _draw_glyphs(img, text, top, left, color, scale=3):
    h, w = img.shape[:2]
    x = left
    for ch in text:
        pat = _GLYPHS.get(ch)
        if pat is None:
            x += 4 * scale
            continue
        for r in range(5):
            for c in range(3):
                if pat[r][c] == "1":
                    r0, c0 = top + r * scale, x + c * scale
                    img[max(0, r0):min(h, r0 + scale),
                        max(0, c0):min(w, c0 + scale)] = color
        x += 4 * scale
    return x


# ---------------------------------------------------------------------------
# A3 -- MOKA candidate generation: contour FPS + geometric centre
# ---------------------------------------------------------------------------

def mask_contour_pixels(mask):
    """(u, v) pixels on the 4-connected boundary of a binary mask."""
    m = np.asarray(mask, dtype=bool)
    inner = np.zeros_like(m)
    inner[1:-1, 1:-1] = (m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1]
                         & m[1:-1, :-2] & m[1:-1, 2:])
    border = m & ~inner
    v, u = np.nonzero(border)
    return np.stack([u, v], axis=1).astype(np.float64)


def moka_candidates(mask, depth, K, T_world_cam, k_boundary=K_BOUNDARY):
    """MOKA's keypoint proposal (Appendix B-B), our depth lift.

    Returns (marks_uv (k_boundary+1, 2), marks_xyz (k_boundary+1, 3),
             info dict).  Mark 0..k_boundary-1 are contour FPS points, the
    LAST mark is the mask's geometric centre.  All marks are lifted with the
    registered depth image; a mark whose own pixel has no valid depth takes
    the nearest valid-depth pixel inside the mask.
    """
    m = np.asarray(mask, dtype=bool)
    contour = mask_contour_pixels(m)
    if contour.shape[0] < k_boundary:
        raise ValueError("mask contour has only %d pixels" % contour.shape[0])
    sel = common.farthest_point_indices(contour, k_boundary)
    uv = contour[sel]
    v_all, u_all = np.nonzero(m)
    centre = np.array([u_all.mean(), v_all.mean()], dtype=np.float64)
    uv = np.concatenate([uv, centre[None]], axis=0)

    # lift with the registered depth (nearest in-mask valid pixel fallback)
    d = np.asarray(depth, dtype=np.float64)
    valid = m & np.isfinite(d) & (d > 0)
    vv, uu = np.nonzero(valid)
    K = np.asarray(K, dtype=np.float64)
    T_wc = np.asarray(T_world_cam, dtype=np.float64)
    pts, used = [], []
    for (u, v) in uv:
        ui = int(np.clip(round(u), 0, d.shape[1] - 1))
        vi = int(np.clip(round(v), 0, d.shape[0] - 1))
        if not valid[vi, ui]:
            j = int(np.argmin((uu - ui) ** 2 + (vv - vi) ** 2))
            ui, vi = int(uu[j]), int(vv[j])
        z = d[vi, ui]
        p_cam = np.array([(ui - K[0, 2]) / K[0, 0] * z,
                          (vi - K[1, 2]) / K[1, 1] * z, z])
        pts.append(T_wc[:3, :3] @ p_cam + T_wc[:3, 3])
        used.append([ui, vi])
    return (uv, np.asarray(pts, dtype=np.float64),
            {"n_contour_px": int(contour.shape[0]),
             "used_uv": np.asarray(used, dtype=np.float64).tolist()})


# ---------------------------------------------------------------------------
# MOKA markup: P[i] dots + the 5x5 chess-notation grid (A7 upscaling)
# ---------------------------------------------------------------------------

def draw_moka_markup(image, marks_uv, upscale=UPSCALE, grid_n=GRID_N,
                     radius=9, text_scale=3):
    """Full-image MOKA markup: numbered candidate dots + the labelled grid.

    Marker rings and the colour palette are IDENTICAL to
    `alkbench.discrete.draw_candidate_markup` / `campaign_d.draw_legible_markup`
    so the two interfaces differ only in what is asked, not in how the marks
    are rendered.  Labels read "P1".."Pk"; tiles read "A1".."E5" with rows
    numbered from the BOTTOM of the image (MOKA's chess convention).
    """
    img = np.array(image, dtype=np.uint8, copy=True)
    s = int(max(1, upscale))
    img = np.repeat(np.repeat(img, s, axis=0), s, axis=1)
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]

    # --- grid lines + tile labels ------------------------------------------
    cw, ch = w / float(grid_n), h / float(grid_n)
    for i in range(1, grid_n):
        x = int(round(i * cw))
        img[:, max(0, x - 1):x + 1] = (255, 255, 255)
        y = int(round(i * ch))
        img[max(0, y - 1):y + 1, :] = (255, 255, 255)
    for ci in range(grid_n):
        for ri in range(grid_n):          # ri = 0 -> row 1 -> BOTTOM
            name = "%s%d" % (COLUMNS[ci], ri + 1)
            x0 = int(round(ci * cw)) + 3
            y0 = int(round((grid_n - 1 - ri) * ch)) + 3
            _draw_glyphs(img, name, y0 + 1, x0 + 1, (0, 0, 0),
                         scale=text_scale - 1)
            _draw_glyphs(img, name, y0, x0, (255, 255, 255),
                         scale=text_scale - 1)

    # --- candidate marks ----------------------------------------------------
    uv = np.asarray(marks_uv, dtype=np.float64) * s + s / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    for i, (u, v) in enumerate(uv):
        color = _MARKER_COLORS[i % len(_MARKER_COLORS)]
        d2 = (xx - u) ** 2 + (yy - v) ** 2
        img[(d2 <= (radius + 2) ** 2) & (d2 > radius ** 2)] = (255, 255, 255)
        img[(d2 <= radius ** 2) & (d2 > (radius - 3) ** 2)] = color
        label = "P%d" % (i + 1)
        tw = 4 * text_scale * len(label)
        tx, ty = int(u) + radius + 4, int(v) - int(2.5 * text_scale)
        if tx + tw + 2 >= w:
            tx = int(u) - radius - tw - 6
        _draw_glyphs(img, label, ty + 1, tx + 1, (0, 0, 0), scale=text_scale)
        _draw_glyphs(img, label, ty, tx, (255, 255, 255), scale=text_scale)
    return img


def draw_zoom_markup(image, marks_uv, radius=9, text_scale=3,
                     target_extent=520):
    """Crop-and-upscale markup around the candidate marks, WITHOUT the grid.

    MOKA's marks live on the full image because the 5x5 waypoint grid needs
    it, but at our 256 px render an 8-mark object occupies ~40 px and the
    P-labels overlap into an unreadable blob (see
    results/campaign_j/markups/).  This companion image is the exact analogue
    of the crop-zoom our own discrete interface sends
    (`pipeline.campaign_d.draw_legible_markup`), so the two interfaces are
    equally legible; it is sent IN ADDITION to the gridded full image, never
    instead of it.  Frozen on non-eval scenes (see freeze.json).
    """
    from alkbench.discrete import crop_zoom
    img = np.array(image, dtype=np.uint8, copy=True)
    uv = np.asarray(marks_uv, dtype=np.float64)
    img, uv, _ = crop_zoom(img, uv, target_extent=target_extent)
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    for i, (u, v) in enumerate(uv):
        color = _MARKER_COLORS[i % len(_MARKER_COLORS)]
        d2 = (xx - u) ** 2 + (yy - v) ** 2
        img[(d2 <= (radius + 2) ** 2) & (d2 > radius ** 2)] = (255, 255, 255)
        img[(d2 <= radius ** 2) & (d2 > (radius - 3) ** 2)] = color
        label = "P%d" % (i + 1)
        tw = 4 * text_scale * len(label)
        tx, ty = int(u) + radius + 4, int(v) - int(2.5 * text_scale)
        if tx + tw + 2 >= w:
            tx = int(u) - radius - tw - 6
        _draw_glyphs(img, label, ty + 1, tx + 1, (0, 0, 0), scale=text_scale)
        _draw_glyphs(img, label, ty, tx, (255, 255, 255), scale=text_scale)
    return img


def draw_demo_reference(image, demo_uv, radius=8):
    """Demo scene with the demonstrated grasp point ringed in red (the extra
    context MOKA is GIVEN here so it sees the same demonstration our pipeline
    does; see FINDINGS 'fair advantages')."""
    img = np.array(image, dtype=np.uint8, copy=True)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = (xx - float(demo_uv[0])) ** 2 + (yy - float(demo_uv[1])) ** 2
    img[(d2 <= (radius + 2) ** 2) & (d2 > radius ** 2)] = (255, 255, 255)
    img[(d2 <= radius ** 2) & (d2 > (radius - 3) ** 2)] = (230, 25, 75)
    s = int(UPSCALE)
    return np.ascontiguousarray(np.repeat(np.repeat(img, s, 0), s, 1))


def tile_centre_uv(tile, w, h, grid_n=GRID_N):
    """Chess-notation tile name -> its centre pixel in ORIGINAL image
    coordinates (A8: centre instead of a uniform sample inside the tile)."""
    t = str(tile).strip().upper()
    if len(t) < 2 or t[0] not in COLUMNS:
        raise ValueError("bad tile %r" % tile)
    ci = COLUMNS.index(t[0])
    ri = int(t[1:])
    if not (1 <= ri <= grid_n):
        raise ValueError("bad tile row %r" % tile)
    cw, ch = w / float(grid_n), h / float(grid_n)
    u = (ci + 0.5) * cw
    v = (grid_n - ri + 0.5) * ch      # rows numbered from the bottom
    return np.array([u, v], dtype=np.float64)


# ---------------------------------------------------------------------------
# MOKA grasp synthesis: antipodal 4-DoF proposals (DexNet-2.0 style)
# ---------------------------------------------------------------------------

def _footprint(points3d, cell=0.002):
    """Top-down occupancy footprint of a world-frame object cloud.

    A 4-DoF (x, y, z, yaw) grasp closes in the horizontal plane, so the
    antipodal contacts must be found on the object's TOP-DOWN silhouette, not
    on the camera-image silhouette (which an oblique camera skews).  Returns
    (occupancy (H,W) bool, top_z (H,W), origin (2,), cell).
    """
    P = np.asarray(points3d, dtype=np.float64)
    lo = P[:, :2].min(axis=0) - 2 * cell
    hi = P[:, :2].max(axis=0) + 2 * cell
    n = np.maximum(np.ceil((hi - lo) / cell).astype(int) + 1, 3)
    gi = np.clip(((P[:, 0] - lo[0]) / cell).astype(int), 0, n[0] - 1)
    gj = np.clip(((P[:, 1] - lo[1]) / cell).astype(int), 0, n[1] - 1)
    occ = np.zeros((n[1], n[0]), dtype=bool)
    top = np.full((n[1], n[0]), -np.inf)
    occ[gj, gi] = True
    np.maximum.at(top, (gj, gi), P[:, 2])
    return occ, top, lo, cell


def _contour_normals(occ):
    """Outward normals at the boundary cells of a 2D occupancy grid.
    Returns (contour_ij (n,2) as (col, row), normals (n,2))."""
    m = np.asarray(occ, dtype=np.float64)
    sm = m.copy()
    for _ in range(2):                       # cheap 3x3 box blur
        pad = np.pad(sm, 1, mode="edge")
        sm = sum(pad[i:i + m.shape[0], j:j + m.shape[1]]
                 for i in range(3) for j in range(3)) / 9.0
    gy, gx = np.gradient(sm)
    ij = mask_contour_pixels(occ)            # (col, row) = (x-idx, y-idx)
    idx = (ij[:, 1].astype(int), ij[:, 0].astype(int))
    n = np.stack([-gx[idx], -gy[idx]], axis=1)   # outward = -grad(occupancy)
    nn = np.linalg.norm(n, axis=1)
    keep = nn > 1e-9
    return ij[keep], n[keep] / nn[keep, None]


def sample_antipodal_grasps(points3d, n_proposals=N_GRASP_PROPOSALS, seed=0,
                            max_width=GRIPPER_MAX_WIDTH,
                            friction_deg=FRICTION_ANGLE_DEG,
                            cell=0.002, max_seeds=800, perp_tol_cells=1.5):
    """MOKA's grasping phase: 30 (here `n_proposals`) antipodal 4-DoF grasp
    proposals from the observed object point cloud (DexNet-2.0 sampler,
    Mahler et al. 2017), as MOKA specifies.

    Following DexNet 2.0, a contact is sampled on the object's silhouette and
    the opposing contact is found by walking INTO the object along the inward
    surface normal; the pair is kept when the segment lies inside the friction
    cone of BOTH outward normals.  The silhouette is the TOP-DOWN footprint of
    the observed cloud (a 4-DoF grasp closes horizontally).  Each accepted
    pair yields
      centre = midpoint of the contacts, at the object's local top height
      yaw    = azimuth of the closing direction in the world xy-plane
      width  = contact separation (<= gripper max width)

    Returns a list sorted by antipodality residual, truncated to
    `n_proposals`.
    """
    P = np.asarray(points3d, dtype=np.float64)
    if P.shape[0] < 8:
        return []
    occ, top, lo, cell = _footprint(P, cell=cell)
    ij, nrm = _contour_normals(occ)
    n = ij.shape[0]
    if n < 2:
        return []
    rng = np.random.RandomState(int(seed))
    seeds = (np.arange(n) if n <= max_seeds
             else rng.choice(n, size=max_seeds, replace=False))
    cos_a = np.cos(np.deg2rad(friction_deg))
    max_span = max_width / cell
    out = []
    for i in seeds:
        p_i, n_i = ij[i], nrm[i]
        d = ij - p_i
        t = d @ (-n_i)
        perp = np.linalg.norm(d + t[:, None] * n_i[None], axis=1)
        cand = (t > 0.5) & (t <= max_span) & (perp <= perp_tol_cells)
        if not cand.any():
            continue
        js = np.nonzero(cand)[0]
        L = np.linalg.norm(ij[js] - p_i, axis=1)
        dh = (ij[js] - p_i) / L[:, None]
        c1 = -(dh @ n_i)
        c2 = np.sum(dh * nrm[js], axis=1)
        ok = (c1 > cos_a) & (c2 > cos_a)
        if not ok.any():
            continue
        js, c1, c2, L = js[ok], c1[ok], c2[ok], L[ok]
        b = int(np.argmin(2.0 - (c1 + c2)))
        j = int(js[b])
        w = float(L[b] * cell)
        if not (5e-4 < w <= max_width):
            continue
        a_xy = lo + (p_i + 0.5) * cell
        b_xy = lo + (ij[j] + 0.5) * cell
        mid = 0.5 * (a_xy + b_xy)
        za = top[int(p_i[1]), int(p_i[0])]
        zb = top[int(ij[j][1]), int(ij[j][0])]
        # local top height along the closing segment (the 4-DoF grasp height)
        z = float(max(za, zb))
        close = b_xy - a_xy
        out.append({"centre": [float(mid[0]), float(mid[1]), z],
                    "yaw": float(np.mod(np.arctan2(close[1], close[0]),
                                        np.pi)),
                    "width_m": w,
                    "antipodal_resid": float(2.0 - (c1[b] + c2[b]))})
    if not out:
        return []
    out.sort(key=lambda g: g["antipodal_resid"])
    return out[:int(n_proposals)]


def nearest_proposal(proposals, x_point):
    """MOKA: 'use the position and orientation of the grasp candidate that is
    closest to x_grasp'."""
    if not proposals:
        return None, None
    C = np.asarray([g["centre"] for g in proposals], dtype=np.float64)
    k = int(np.argmin(np.linalg.norm(C - np.asarray(x_point,
                                                    dtype=np.float64), axis=1)))
    return proposals[k], float(np.linalg.norm(C[k] - np.asarray(x_point)))


def wrap_pi_half(a):
    """Wrap an angle into (-pi/2, pi/2] (parallel-jaw grasps are pi-symmetric;
    used only for the diagnostic grasp-proposal yaw)."""
    return float(np.mod(a + np.pi / 2.0, np.pi) - np.pi / 2.0)


def wrap_pi(a):
    """Wrap an angle into (-pi, pi]."""
    return float(np.mod(a + np.pi, 2.0 * np.pi) - np.pi)


def axis_azimuth(p_from, p_to):
    """World xy azimuth of the grasp -> function axis (MOKA's object
    orientation representation, Sec. IV-D)."""
    v = np.asarray(p_to, dtype=np.float64) - np.asarray(p_from,
                                                        dtype=np.float64)
    if np.linalg.norm(v[:2]) < 1e-6:
        return None
    return float(np.arctan2(v[1], v[0]))


def yaw_transform(g_demo, psi_demo, g_target, psi_target):
    """T_map = Translate(g_t) . Rz(dpsi) . Translate(-g_d)  (adaptation A10)."""
    dpsi = wrap_pi(float(psi_target) - float(psi_demo))
    c, s = np.cos(dpsi), np.sin(dpsi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(g_target, dtype=np.float64) \
        - R @ np.asarray(g_demo, dtype=np.float64)
    return T, dpsi


# ---------------------------------------------------------------------------
# prompts (adapted from MOKA Tables IV-VII; FROZEN before any eval seed)
# ---------------------------------------------------------------------------

HIGH_LEVEL_PROMPT = """You are a robot manipulation planner. Decompose the \
task into subtasks.

The input request contains:
- A string describing the task.
- An image of the current table-top environment.

Task: {task}

This table-top scene contains a SINGLE manipulable object and the task is \
carried out in ONE subtask (one grasp followed by the prescribed motion), so \
return a list containing exactly one dictionary. "object_grasped" must name \
the object the gripper holds; "object_unattached" must be an empty string.

The output response is a list of dictionaries in JSON form. Each dictionary \
specifies one subtask, in execution order, with these fields:
- "instruction": a string describing the subtask in natural language.
- "object_grasped": the object the robot gripper will hold in hand while \
executing the subtask. Empty string if there is no such object.
- "object_unattached": the object the gripper will interact with directly or \
via another object WITHOUT holding it in hand. Empty string if there is no \
such object.
- "motion_direction": a string describing the direction of the gripper motion \
while performing the subtask (e.g. "upward", "from right to left").

Answer with ONLY the JSON list, no other text."""

LOW_LEVEL_PROMPT = """Please describe the robot gripper's motion to solve the \
task by selecting keypoints and waypoints.

The input request contains:
- The task information with these fields:
  - instruction: {instruction}
  - object_grasped: {object_grasped}
  - object_unattached: {object_unattached}
  - motion_direction: {motion_direction}
- An image of the current table-top environment, annotated with a set of \
visual marks:
  - candidate keypoints on object_grasped: coloured dots marked as P1..P{k} \
on the image.
  - grid for waypoints: grid lines that uniformly divide the image into \
tiles. The grid divides the image into columns marked A, B, C, D, E from left \
to right and rows marked 1, 2, 3, 4, 5 from bottom to top, so each tile is \
named like "A1" (bottom-left) or "E5" (top-right).

The motion consists of a grasping phase and a manipulation phase, specified \
by grasp keypoint, function keypoint, pre-contact waypoint and post-contact \
waypoint. The definitions of these points are:
- grasp keypoint: the point on object_grasped indicating the part where the \
robot gripper should hold.
- function keypoint: the point on object_grasped indicating the functional \
part of the object; the axis from the grasp keypoint to the function keypoint \
specifies the orientation of the object during the motion.
- pre contact waypoint: the waypoint in free space that the gripper moves to \
before making contact.
- post contact waypoint: the waypoint in free space that the gripper moves to \
after the motion.

There is no object_unattached in this scene, so target keypoint is empty.

The response should be a dictionary in JSON form, which contains:
- "grasp_keypoint": selected from the candidate keypoints marked P1..P{k}.
- "function_keypoint": selected from the candidate keypoints marked P1..P{k}.
- "pre_contact_tile": the tile the pre-contact waypoint should be in, \
selected from the tiles marked on the image.
- "post_contact_tile": the tile the post-contact waypoint should be in.
- "pre_contact_height": "same" or "above".
- "post_contact_height": "same" or "above".
- "target_angle": a string describing how the object should be oriented \
during this motion, in terms of the axis pointing from the grasp keypoint to \
the function keypoint.

{cot}Give the answer as a JSON dictionary, for example:
{{"grasp_keypoint": "P3", "function_keypoint": "P1", "pre_contact_tile": \
"C4", "post_contact_tile": "C5", "pre_contact_height": "above", \
"post_contact_height": "above", "target_angle": "horizontal"}}"""

# MOKA Table VII instructs the VLM to reason before answering.  Kept as a
# switch so the choice is made on NON-EVAL data (see freeze.json).
COT_CLAUSE = """Think about this problem step by step and explain the \
reasoning steps. First, choose grasp keypoint and function keypoint on the \
correct parts of the object. Next, describe which tile the object is located \
in. Then choose pre contact tile, post contact tile, pre contact height and \
post contact height such that the resultant motion from the pre-contact \
waypoint through the object to the post-contact waypoint in 3D follows the \
"motion_direction" input. Remember that the columns are marked A, B, C, D, E \
from left to right and the rows are marked 1, 2, 3, 4, 5 from bottom to top.
Keep the reasoning under 150 words, then output the final answer as a JSON \
dictionary on its own line.

"""
NO_COT_CLAUSE = "Answer with ONLY the JSON dictionary, no other text. "

# Optional extra image (see FINDINGS "fair advantages"): the demonstration
# scene with the demonstrated grasp point ringed in red.
DEMO_IMAGE_CLAUSE = """
The SECOND image shows the SAME task performed earlier in a different object \
placement; the red circle marks the point where the robot grasped the object \
in that demonstration. Pick the marked candidate keypoint in the FIRST image \
that lies on the SAME part of the object."""

ZOOM_IMAGE_CLAUSE = """
An ADDITIONAL image is provided: a zoomed-in crop of the object carrying the \
SAME candidate keypoint marks P1..P{k} (no grid). Use it to read the mark \
labels; use the gridded image to choose the tiles."""

# One fixed free-form task instruction per task, taken verbatim from the
# benchmark's own TaskSpec.description wording (simtasks/envs.py) so MOKA and
# our pipeline are told the same thing.
# FAIRNESS NOTE.  Our own discrete interface is driven by a hand-written
# per-task prompt (pipeline.campaign_d.TASK_PROMPTS) that describes the object
# and names the part to grasp.  MOKA's only text channel is the free-form task
# instruction, so it is given the SAME descriptive content through that
# channel -- otherwise our prompt would be tuned and MOKA's would not.  What
# is NOT transferred is the index-level rule ("phi1 = the marker nearest ...")
# because MOKA's interface has no phi1/phi2; MOKA is told which PART to grasp,
# exactly as we are, and must still pick the mark itself.
TASK_INSTRUCTIONS = {
    "nut_loosen": "Grasp the round nut by its handle and lift it clear off "
                  "the table. The object is a metal ring with a small handle "
                  "tab sticking out on one side; the gripper must close on "
                  "that handle tab, not on the ring.",
    "rim_grasp": "Grasp the can cylinder near its upper rim and lift it "
                 "straight up. The object is an upright cylindrical can "
                 "standing on the table; the gripper must close near the top "
                 "rim of the can.",
    "pour": "Grasp the elongated block across its long axis, lift it and "
            "rotate the wrist by a large angle (pouring). The object is a "
            "single rectangular block lying on the table; the gripper must "
            "close on the MIDDLE of the block, across its short axis.",
    "box_open": "Grasp the door handle and swing the door open. The object "
                "is the wooden handle bar mounted on the door panel; the "
                "gripper must close on the middle of that handle bar.",
    "cap_twist": "Grasp the square nut by its handle and twist it in place "
                 "about the vertical axis. The object is a square ring (a "
                 "flat plate with a rectangular hole) with a small handle "
                 "tab sticking out on one side; the gripper must close on "
                 "that handle tab, not on the plate.",
}


# ---------------------------------------------------------------------------
# VLM plumbing (same client conventions as alkbench.discrete.VLMSolver)
# ---------------------------------------------------------------------------

def _client(base_url=None, api_key=None):
    import openai  # lazy
    return openai.OpenAI(
        base_url=base_url or os.environ.get("ALK_VLM_BASE_URL",
                                            VLM_DEFAULTS["base_url"]),
        api_key=api_key or os.environ.get("ALK_VLM_API_KEY",
                                          VLM_DEFAULTS["api_key"]))


def _model(model=None):
    return model or os.environ.get("ALK_VLM_MODEL", VLM_DEFAULTS["model"])


def _img_part(arr):
    import base64
    b64 = base64.b64encode(encode_png(np.asarray(arr,
                                                 dtype=np.uint8))).decode()
    return {"type": "image_url",
            "image_url": {"url": "data:image/png;base64," + b64}}


def _strip_json(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
    return t


def _iter_json_objects(text):
    """Yield every balanced {...} substring, last one first.

    MOKA's low-level prompt asks the VLM to REASON step by step before
    emitting the dictionary (Table VII), so the reply legitimately contains
    prose -- and often braces -- around the answer.  A naive
    first-'{' .. last-'}' slice breaks on that."""
    t = _strip_json(text)
    starts, spans = [], []
    for i, ch in enumerate(t):
        if ch == "{":
            starts.append(i)
        elif ch == "}" and starts:
            j = starts.pop()
            if not starts:
                spans.append((j, i))
    for a, b in reversed(spans):
        yield t[a:b + 1]


def _parse_json_obj(text, require=None):
    last = None
    for blob in _iter_json_objects(text):
        try:
            obj = json.loads(blob)
        except ValueError as e:
            last = e
            continue
        if not isinstance(obj, dict):
            continue
        if require is None or require in obj:
            return obj
        last = ValueError("JSON object without %r" % require)
    raise ValueError("no usable JSON object in reply (%s)" % last)


def _mark_index(val, k, field):
    """'P3' / 'p3' / 3 / '3' -> 0-based index; validated against k."""
    if val is None:
        raise ValueError("%s missing" % field)
    s = str(val).strip().upper()
    if s.startswith("P"):
        s = s[1:]
    if not s or not s.lstrip("-").isdigit():
        raise ValueError("%s=%r is not a P index" % (field, val))
    i = int(s)
    if not (1 <= i <= k):
        raise ValueError("%s=%r outside P1..P%d" % (field, val, k))
    return i - 1


def query_high_level(rgb, task_instruction, model=None, base_url=None,
                     api_key=None, max_retries=1, temperature=0.0):
    """MOKA step 1 (Table IV).  Returns (subtask dict, info)."""
    cli = _client(base_url, api_key)
    prompt = HIGH_LEVEL_PROMPT.format(task=task_instruction)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt},
                                             _img_part(rgb)]}]
    replies, last = [], None
    for attempt in range(1 + max_retries):
        r = cli.chat.completions.create(model=_model(model), messages=messages,
                                        temperature=temperature)
        text = r.choices[0].message.content
        replies.append(text)
        try:
            t = _strip_json(text)
            s, e = t.find("["), t.rfind("]")
            obj = json.loads(t[s:e + 1]) if s >= 0 and e >= 0 \
                else [_parse_json_obj(text)]
            if not isinstance(obj, list) or not obj:
                raise ValueError("expected a non-empty JSON list")
            # A5: our tasks are single-subtask; if the VLM still decomposes
            # into approach/grasp steps, take the first subtask that actually
            # names an object to grasp (MOKA's grasping phase).
            sub = next((x for x in obj if isinstance(x, dict)
                        and str(x.get("object_grasped") or "").strip()),
                       obj[0])
            out = {kk: str(sub.get(kk, "") or "")
                   for kk in ("instruction", "object_grasped",
                              "object_unattached", "motion_direction")}
            if not out["instruction"]:
                out["instruction"] = task_instruction
            return out, {"replies": replies, "attempts": attempt + 1,
                         "subtasks": obj}
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as ex:
            last = ex
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content":
                             "Invalid reply (%s). Answer with ONLY the JSON "
                             "list of subtask dictionaries." % ex})
    raise RuntimeError("query_high_level: invalid reply after retries: %s"
                       % last)


def query_low_level(markup, subtask, k, demo_image=None, zoom_image=None,
                    cot=True, model=None, base_url=None, api_key=None,
                    max_retries=1, temperature=0.0, max_tokens=1400):
    """MOKA step 3 (Tables V-VII).  Returns (selection dict, info).

    Images are sent in a fixed order: the gridded full-scene markup (MOKA's
    own interface), then the optional zoom crop, then the optional demo
    reference."""
    cli = _client(base_url, api_key)
    prompt = LOW_LEVEL_PROMPT.format(
        k=k, cot=COT_CLAUSE if cot else NO_COT_CLAUSE, **subtask)
    if zoom_image is not None:
        prompt = prompt + ZOOM_IMAGE_CLAUSE.format(k=k)
    if demo_image is not None:
        prompt = prompt + DEMO_IMAGE_CLAUSE
    content = [{"type": "text", "text": prompt}, _img_part(markup)]
    if zoom_image is not None:
        content.append(_img_part(zoom_image))
    if demo_image is not None:
        content.append(_img_part(demo_image))
    messages = [{"role": "user", "content": content}]
    replies, last = [], None
    for attempt in range(1 + max_retries):
        r = cli.chat.completions.create(model=_model(model), messages=messages,
                                        temperature=temperature,
                                        max_tokens=max_tokens)
        text = r.choices[0].message.content
        replies.append(text)
        try:
            obj = _parse_json_obj(text, require="grasp_keypoint")
            sel = {
                "grasp_keypoint": _mark_index(obj.get("grasp_keypoint"), k,
                                              "grasp_keypoint"),
                "function_keypoint": None,
                "pre_contact_tile": obj.get("pre_contact_tile") or None,
                "post_contact_tile": obj.get("post_contact_tile") or None,
                "pre_contact_height": str(obj.get("pre_contact_height")
                                          or "same").strip().lower(),
                "post_contact_height": str(obj.get("post_contact_height")
                                           or "same").strip().lower(),
                "target_angle": str(obj.get("target_angle") or ""),
            }
            fk = obj.get("function_keypoint")
            if fk not in (None, "", "null"):
                sel["function_keypoint"] = _mark_index(fk, k,
                                                       "function_keypoint")
            for kk in ("pre_contact_tile", "post_contact_tile"):
                if sel[kk] is not None:
                    tile_centre_uv(sel[kk], 100, 100)  # validate name
            return sel, {"replies": replies, "attempts": attempt + 1,
                         "raw": obj}
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as ex:
            last = ex
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content":
                             'Invalid reply (%s). Answer with ONLY the JSON '
                             'dictionary; "grasp_keypoint" and '
                             '"function_keypoint" must be one of P1..P%d and '
                             'the tiles must be named like "C4".' % (ex, k)})
    raise RuntimeError("query_low_level: invalid reply after retries: %s"
                       % last)


# ---------------------------------------------------------------------------
# the method itself
# ---------------------------------------------------------------------------

def _object_mask(capture, percept, task=None, subobject=True):
    """Segmentation mask MOKA's low-level stage is given (adaptation A1/A13).

    Normally the benchmark's ground-truth INSTANCE mask -- the same mask our
    pipeline's perception consumes.  For `box_open` the instance mask covers
    the whole door assembly (frame + panel + handle, 2.7 m across), whereas
    MOKA's GroundedSAM would be asked for the object its own high-level
    reasoning names ("door handle") and would return the handle only.  With
    `subobject=True` the mask is therefore restricted to pixels within
    HANDLE_RADIUS_M of the handle site recorded in the capture (A13); this is
    an explicit ORACLE-segmentation advantage given to MOKA and NOT to our
    pipeline.
    """
    cd = capture["cameras"][percept["camera"]]
    mask = cd["seg"] == percept["instance_id"]
    if subobject and task == "box_open":
        hp = (capture["meta"].get("extra_state") or {}).get("handle_pos")
        if hp is not None:
            d = np.asarray(cd["depth"], dtype=np.float64)
            K = np.asarray(cd["K"], dtype=np.float64)
            T_wc = np.asarray(cd["T_world_cam"], dtype=np.float64)
            v, u = np.nonzero(mask & np.isfinite(d) & (d > 0))
            z = d[v, u]
            cam = np.stack([(u - K[0, 2]) / K[0, 0] * z,
                            (v - K[1, 2]) / K[1, 1] * z, z], axis=1)
            w = cam @ T_wc[:3, :3].T + T_wc[:3, 3]
            near = np.linalg.norm(w - np.asarray(hp, dtype=np.float64),
                                  axis=1) <= HANDLE_RADIUS_M
            sub = np.zeros_like(mask)
            sub[v[near], u[near]] = True
            if sub.sum() >= 40:
                return sub, cd
    return mask, cd


def _side(capture, ctx, role, n_proposals=N_GRASP_PROPOSALS,
          k_boundary=K_BOUNDARY, subobject=True, candidates="contour_fps"):
    """MOKA's per-image stage: mask -> marks -> antipodal grasp proposals, on
    either the demo or the target scene.

    `candidates` selects the mark generator:
      "contour_fps"  MOKA's own (Appendix B-B): farthest point sampling on the
                     mask contour + the geometric centre  [default];
      "kmeans"       OUR candidate pool (2D k-means cluster centres over the
                     whole mask, alkbench.candidates).  Used only by the
                     `moka_real_ourcands` diagnostic arm, which separates
                     "MOKA's interface is weaker" from "MOKA's candidate
                     GENERATION is weaker" -- everything else is unchanged.
    """
    p = ctx.percept(capture, role)
    mask, cd = _object_mask(capture, p, task=ctx.task, subobject=subobject)
    if candidates == "kmeans":
        marks_uv = np.asarray(p["cands"].centers2d, dtype=np.float64)
        marks_xyz = np.asarray(p["cands"].candidates3d, dtype=np.float64)
        info = {"candidates": "kmeans"}
    else:
        marks_uv, marks_xyz, info = moka_candidates(mask, cd["depth"],
                                                    cd["K"],
                                                    cd["T_world_cam"],
                                                    k_boundary=k_boundary)
    cloud = _mask_cloud(mask, cd)
    props = sample_antipodal_grasps(
        cloud, n_proposals=n_proposals,
        seed=0 if role == "demo" else ctx.noise_seed)
    return {"percept": p, "mask": mask, "cd": cd, "marks_uv": marks_uv,
            "marks_xyz": marks_xyz, "proposals": props, "info": info,
            "cloud": cloud}


def _mask_cloud(mask, cd):
    d = np.asarray(cd["depth"], dtype=np.float64)
    K = np.asarray(cd["K"], dtype=np.float64)
    T_wc = np.asarray(cd["T_world_cam"], dtype=np.float64)
    v, u = np.nonzero(mask & np.isfinite(d) & (d > 0))
    z = d[v, u]
    cam = np.stack([(u - K[0, 2]) / K[0, 0] * z,
                    (v - K[1, 2]) / K[1, 1] * z, z], axis=1)
    return cam @ T_wc[:3, :3].T + T_wc[:3, 3]


def demo_side(demo_capture, ctx, n_proposals=N_GRASP_PROPOSALS,
              k_boundary=K_BOUNDARY, subobject=True,
              candidates="contour_fps"):
    """Demo scene: MOKA marks + the demonstration's own grasp anchor (A11)."""
    s = _side(demo_capture, ctx, "demo", n_proposals=n_proposals,
              k_boundary=k_boundary, subobject=subobject,
              candidates=candidates)
    g_d, proj_dist = oracle.demo_grasp_point(s["percept"]["cands"],
                                             ctx.keyframe_tcp("pre_grasp"))
    s["g_d"] = g_d
    s["proj_dist"] = proj_dist
    s["proposal"], s["proposal_dist_m"] = nearest_proposal(s["proposals"],
                                                           g_d)
    return s


def _finish(variant, ctx, dside, tside, dsel, tsel, aux, snap="target"):
    """MOKA's motion generation (Sec. IV-D) reduced to one rigid T_map (A10).

    position  : lifted target grasp keypoint, snapped to the closest antipodal
                4-DoF grasp proposal (MOKA verbatim);
    rotation  : change in the azimuth of the grasp -> function axis, which is
                MOKA's own object-orientation representation.
    """
    g_d = np.asarray(dside["g_d"], dtype=np.float64)
    x_grasp = np.asarray(tside["marks_xyz"])[tsel["grasp_keypoint"]]
    grasp_t, dist_t = nearest_proposal(tside["proposals"], x_grasp)
    snapped = snap in ("target", "both") and grasp_t is not None
    g_t = (np.asarray(grasp_t["centre"], dtype=np.float64) if snapped
           else np.asarray(x_grasp, dtype=np.float64))
    if snap == "both" and dside.get("proposal") is not None:
        g_d = np.asarray(dside["proposal"]["centre"], dtype=np.float64)

    psi_d = psi_t = None
    if dsel.get("function_keypoint") is not None \
            and dsel["function_keypoint"] != dsel["grasp_keypoint"]:
        psi_d = axis_azimuth(np.asarray(dside["marks_xyz"])[
            dsel["grasp_keypoint"]],
            np.asarray(dside["marks_xyz"])[dsel["function_keypoint"]])
    if tsel.get("function_keypoint") is not None \
            and tsel["function_keypoint"] != tsel["grasp_keypoint"]:
        psi_t = axis_azimuth(np.asarray(tside["marks_xyz"])[
            tsel["grasp_keypoint"]],
            np.asarray(tside["marks_xyz"])[tsel["function_keypoint"]])
    if psi_d is None or psi_t is None:   # no usable axis -> translation only
        T_map = np.eye(4)
        T_map[:3, 3] = g_t - g_d
        dpsi = 0.0
        axis_ok = False
    else:
        T_map, dpsi = yaw_transform(g_d, psi_d, g_t, psi_t)
        axis_ok = True

    res = common.base_result(
        variant, T_map,
        demo_grasp=g_d.tolist(),
        target_grasp=g_t.tolist(),
        demo_selection={k: v for k, v in dsel.items()},
        selection={k: v for k, v in tsel.items()},
        x_grasp_lifted=np.asarray(x_grasp).tolist(),
        marks_uv=np.asarray(tside["marks_uv"]).tolist(),
        n_marks=int(np.asarray(tside["marks_uv"]).shape[0]),
        grasp_proposal_target=grasp_t,
        grasp_proposal_dist_target_m=dist_t,
        grasp_proposal_snapped=snapped,
        n_proposals_demo=len(dside["proposals"]),
        n_proposals_target=len(tside["proposals"]),
        delta_yaw_deg=float(np.degrees(dpsi)),
        axis_channel_used=axis_ok,
        demo_axis_deg=None if psi_d is None else float(np.degrees(psi_d)),
        target_axis_deg=None if psi_t is None else float(np.degrees(psi_t)),
        camera=tside["percept"]["camera"],
        demo_grasp_projection_dist_m=dside["proj_dist"],
        mask_pixels_demo=int(dside["mask"].sum()),
        mask_pixels_target=int(tside["mask"].sum()),
    )
    if ctx.T_gt is not None:  # scoring info only, never used by the method
        g_gt = transform_points(ctx.T_gt, g_d[None])[0]
        dists = np.linalg.norm(np.asarray(tside["marks_xyz"]) - g_gt, axis=1)
        res["anchor_err_m"] = float(np.linalg.norm(g_t - g_gt))
        res["best_possible_mark_err_m"] = float(dists.min())
        res["oracle_mark"] = int(np.argmin(dists))
        res["selected_mark_err_m"] = float(np.linalg.norm(x_grasp - g_gt))
        res["mark_choice_correct"] = bool(
            tsel["grasp_keypoint"] == int(np.argmin(dists)))
        gt_yaw = float(np.degrees(wrap_pi(np.arctan2(ctx.T_gt[1, 0],
                                                     ctx.T_gt[0, 0]))))
        res["gt_yaw_deg"] = gt_yaw
        res["yaw_err_deg"] = float(np.degrees(
            wrap_pi(np.radians(res["delta_yaw_deg"] - gt_yaw))))
    res.update(aux)
    return res


def markup_for(side, capture):
    """The gridded full-scene image MOKA's low-level query sees."""
    rgb = common.load_rgb(capture, side["percept"]["camera"])
    return draw_moka_markup(rgb, side["marks_uv"])


def zoom_for(side, capture):
    """The companion zoom crop carrying the same marks (legibility aid)."""
    rgb = common.load_rgb(capture, side["percept"]["camera"])
    return draw_zoom_markup(rgb, side["marks_uv"])


def demo_reference_image(dside, demo_capture):
    d_cd = demo_capture["cameras"][dside["percept"]["camera"]]
    duv = _world_to_pixel(dside["g_d"], d_cd["K"], d_cd["T_world_cam"])
    return draw_demo_reference(
        common.load_rgb(demo_capture, dside["percept"]["camera"]), duv)


def oracle_demo_selection(dside):
    """Oracle demo-side marks: grasp = the mark nearest the demonstrated grasp
    point; function = the mark farthest from it (the object's functional
    extremity, MOKA's grasp->function axis)."""
    M = np.asarray(dside["marks_xyz"])
    ig = int(np.argmin(np.linalg.norm(M - dside["g_d"], axis=1)))
    i_f = int(np.argmax(np.linalg.norm(M - M[ig], axis=1)))
    return {"grasp_keypoint": ig, "function_keypoint": i_f, "oracle": True}


def run_moka_real(demo_capture, target_capture, ctx, variant="moka_real",
                  subtask=None, demo_selection=None, demo_image=True,
                  n_proposals=N_GRASP_PROPOSALS, k_boundary=K_BOUNDARY,
                  selection=None, subobject=True, snap="target",
                  zoom_image=True, cot=True, candidates="contour_fps"):
    """MOKA with its published mark-selection interface and a real VLM.

    `subtask` and `demo_selection` are the per-task cached high-level output
    and demo-side mark selection (one query each per task, mirroring our
    pipeline's cached demo-side query); when None they are queried here.
    `selection` short-circuits the target-side VLM query (replay / oracle).
    """
    dside = demo_side(demo_capture, ctx, n_proposals=n_proposals,
                      k_boundary=k_boundary, subobject=subobject,
                      candidates=candidates)
    tside = _side(target_capture, ctx, "target", n_proposals=n_proposals,
                  k_boundary=k_boundary, subobject=subobject,
                  candidates=candidates)
    k = int(np.asarray(tside["marks_uv"]).shape[0])
    aux = {"n_vlm_calls": 0}
    if subtask is None:
        subtask, hinfo = query_high_level(
            common.load_rgb(target_capture, tside["percept"]["camera"]),
            TASK_INSTRUCTIONS.get(ctx.task, ctx.task),
            model=ctx.vlm_model, base_url=ctx.vlm_base_url,
            api_key=ctx.vlm_api_key)
        aux["high_level_info"] = hinfo
        aux["n_vlm_calls"] += hinfo["attempts"]
    if demo_selection is None:
        demo_selection, dinfo = query_low_level(
            markup_for(dside, demo_capture), subtask, k, demo_image=None,
            zoom_image=zoom_for(dside, demo_capture) if zoom_image else None,
            cot=cot, model=ctx.vlm_model, base_url=ctx.vlm_base_url,
            api_key=ctx.vlm_api_key)
        aux["demo_query_info"] = {"attempts": dinfo["attempts"],
                                  "raw": dinfo["raw"]}
        aux["n_vlm_calls"] += dinfo["attempts"]
    if selection is None:
        dimg = demo_reference_image(dside, demo_capture) if demo_image else None
        selection, linfo = query_low_level(
            markup_for(tside, target_capture), subtask, k, demo_image=dimg,
            zoom_image=(zoom_for(tside, target_capture) if zoom_image
                        else None),
            cot=cot, model=ctx.vlm_model, base_url=ctx.vlm_base_url,
            api_key=ctx.vlm_api_key)
        aux["low_level_info"] = {"attempts": linfo["attempts"],
                                 "raw": linfo["raw"],
                                 "reply": linfo["replies"][-1][-1500:]}
        aux["n_vlm_calls"] += linfo["attempts"]
    aux["subtask"] = subtask
    return _finish(variant, ctx, dside, tside, demo_selection, selection, aux,
                   snap=snap)


def run_moka_marks_oracle(demo_capture, target_capture, ctx,
                          variant="moka_marks_oracle",
                          n_proposals=N_GRASP_PROPOSALS,
                          k_boundary=K_BOUNDARY, subobject=True,
                          snap="target", candidates="contour_fps"):
    """Same MOKA pipeline with ORACLE mark selection: the target marks nearest
    (in 3D) to the ground-truth-mapped demo marks.  Isolates MOKA's mark-
    SELECTION error from its candidate-generation and motion-mapping error."""
    T_gt = ctx.require_T_gt("moka_marks_oracle")
    dside = demo_side(demo_capture, ctx, n_proposals=n_proposals,
                      k_boundary=k_boundary, subobject=subobject,
                      candidates=candidates)
    tside = _side(target_capture, ctx, "target", n_proposals=n_proposals,
                  k_boundary=k_boundary, subobject=subobject,
                  candidates=candidates)
    dsel = oracle_demo_selection(dside)
    M = transform_points(T_gt, np.asarray(dside["marks_xyz"]))
    T = np.asarray(tside["marks_xyz"])
    tsel = {"grasp_keypoint": int(np.argmin(np.linalg.norm(
                T - M[dsel["grasp_keypoint"]], axis=1))),
            "function_keypoint": int(np.argmin(np.linalg.norm(
                T - M[dsel["function_keypoint"]], axis=1))),
            "oracle": True}
    return _finish(variant, ctx, dside, tside, dsel, tsel,
                   {"oracle_selection": True, "n_vlm_calls": 0}, snap=snap)


def _world_to_pixel(p_world, K, T_world_cam):
    T_cw = np.linalg.inv(np.asarray(T_world_cam, dtype=np.float64))
    p_cam = transform_points(T_cw, np.asarray(p_world,
                                              dtype=np.float64)[None])
    return project(p_cam, K[0, 0], K[1, 1], K[0, 2], K[1, 2])[0]
