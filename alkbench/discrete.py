"""Discrete variable interface: Phi = (phi1, phi2 axial endpoint ids,
phi3 lateral swap bit, phi4 coarse grasp region, phi5 fine sub-region).

Solvers: OracleSolver (caller-injected ground truth) and VLMSolver
(OpenAI-compatible endpoint, lazily imported so `openai` is optional).
"""

import json
import os
import struct
import zlib

import numpy as np


class DiscreteChoice(object):
    """Container for the discrete variables. Indices are 0-based."""

    def __init__(self, phi1, phi2, phi3, phi4=None, phi5=None):
        self.phi1 = int(phi1)
        self.phi2 = int(phi2)
        self.phi3 = bool(phi3)
        self.phi4 = None if phi4 is None else int(phi4)
        self.phi5 = None if phi5 is None else int(phi5)

    def __repr__(self):
        return ("DiscreteChoice(phi1=%r, phi2=%r, phi3=%r, phi4=%r, phi5=%r)"
                % (self.phi1, self.phi2, self.phi3, self.phi4, self.phi5))

    def __eq__(self, other):
        if not isinstance(other, DiscreteChoice):
            return NotImplemented
        return (self.phi1, self.phi2, self.phi3, self.phi4, self.phi5) == \
               (other.phi1, other.phi2, other.phi3, other.phi4, other.phi5)

    def as_dict(self):
        return {"phi1": self.phi1, "phi2": self.phi2, "phi3": int(self.phi3),
                "phi4": self.phi4, "phi5": self.phi5}


class DiscreteSolver(object):
    """Abstract solver mapping (image, candidates, task) -> DiscreteChoice."""

    def solve(self, image=None, candidates_uv=None, task_description=""):
        raise NotImplementedError


class OracleSolver(DiscreteSolver):
    """Returns a ground-truth DiscreteChoice injected by the caller."""

    def __init__(self, choice):
        self.choice = choice

    def solve(self, image=None, candidates_uv=None, task_description=""):
        return self.choice


# ---------------------------------------------------------------------------
# image markup helpers (pure numpy + stdlib PNG encoding)

_DIGITS = {  # 3x5 bitmap font
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
}


def _draw_text(img, text, top, left, color, scale=2):
    h, w = img.shape[:2]
    x = left
    for ch in text:
        pat = _DIGITS.get(ch)
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


# distinct, high-contrast marker colors (index 1..8)
_MARKER_COLORS = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (255, 225, 25),
]


def crop_zoom(image, candidates_uv, margin_frac=0.6, min_margin=24,
              target_extent=320):
    """Crop the image around the candidate bounding box and integer-upscale.

    Small objects in full-scene renders leave the numbered markers
    unreadable; VLM queries should see a zoomed crop. Geometry stays in
    original pixel coordinates -- only the markup/query path uses this.

    Returns (cropped_upscaled_image, transformed_uv, (scale, u0, v0)).
    """
    img = np.asarray(image)
    h, w = img.shape[:2]
    uv = np.asarray(candidates_uv, dtype=np.float64)
    u_min, v_min = uv.min(axis=0)
    u_max, v_max = uv.max(axis=0)
    extent = max(u_max - u_min, v_max - v_min, 1.0)
    m = max(min_margin, margin_frac * extent)
    u0 = int(max(0, np.floor(u_min - m)))
    v0 = int(max(0, np.floor(v_min - m)))
    u1 = int(min(w, np.ceil(u_max + m)))
    v1 = int(min(h, np.ceil(v_max + m)))
    crop = img[v0:v1, u0:u1]
    scale = max(1, int(round(target_extent / max(crop.shape[0], crop.shape[1]))))
    up = np.repeat(np.repeat(crop, scale, axis=0), scale, axis=1)
    uv_t = (uv - np.array([u0, v0])) * scale + scale / 2.0
    return np.ascontiguousarray(up), uv_t, (scale, u0, v0)


def draw_candidate_markup(image, candidates_uv, radius=8,
                          circle_color=None, text_color=(255, 255, 255),
                          zoom=True):
    """Overlay numbered markers (1-based) at candidate pixel locations.

    Parameters
    ----------
    image : (H, W, 3) uint8 RGB
    candidates_uv : (k, 2) pixel coordinates (u, v)
    circle_color : fixed color for all markers, or None for a distinct
        per-index palette (recommended -- avoids blending with the object)
    zoom : crop-and-upscale around the candidates before drawing
        (markers stay legible when the object is small in frame)

    Returns
    -------
    (H', W', 3) uint8 image with ring markers and index labels
    """
    img = np.array(image, dtype=np.uint8, copy=True)
    uv = np.asarray(candidates_uv, dtype=np.float64)
    if zoom:
        img, uv, _ = crop_zoom(img, uv)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    for i, (u, v) in enumerate(uv):
        color = circle_color if circle_color is not None else \
            _MARKER_COLORS[i % len(_MARKER_COLORS)]
        d2 = (xx - u) ** 2 + (yy - v) ** 2
        img[(d2 <= (radius + 2) ** 2) & (d2 > radius ** 2)] = (255, 255, 255)
        img[(d2 <= radius ** 2) & (d2 > (radius - 3) ** 2)] = color
        tx, ty = int(u) + radius + 4, int(v) - 6
        if tx + 10 >= w:
            tx = int(u) - radius - 12
        _draw_text(img, str(i + 1), ty + 1, tx + 1, (0, 0, 0))
        _draw_text(img, str(i + 1), ty, tx, text_color)
    return img


def encode_png(image):
    """Encode an (H, W, 3) uint8 RGB array as PNG bytes (stdlib only)."""
    img = np.ascontiguousarray(image, dtype=np.uint8)
    h, w = img.shape[:2]
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """You are selecting keypoints for robot manipulation.
The image shows a zoomed view of an object with {k} numbered candidate
points (colored rings with white numbers, labels 1..{k}). Task: {task}

Choose:
- "phi1": index of the FIRST axial endpoint of the object's main axis
- "phi2": index of the SECOND axial endpoint (must differ from phi1)
- "phi3": 0 or 1 -- set 1 if the lateral sides (left/right of the axis
  from phi1 to phi2) must be swapped to match the demonstration, else 0
{extra}
Answer with ONLY a JSON object, no other text, e.g.:
{{"phi1": 1, "phi2": 5, "phi3": 0}}
Indices are 1-based as printed in the image."""


class VLMSolver(DiscreteSolver):
    """Queries an OpenAI-compatible VLM endpoint for the discrete choice.

    base_url / api_key default to env vars ALK_VLM_BASE_URL / ALK_VLM_API_KEY.
    The `openai` package is imported lazily on first solve().
    """

    def __init__(self, model, base_url=None, api_key=None, max_retries=1,
                 temperature=0.0):
        self.model = model
        self.base_url = base_url if base_url is not None else os.environ.get("ALK_VLM_BASE_URL")
        self.api_key = api_key if api_key is not None else os.environ.get("ALK_VLM_API_KEY")
        self.max_retries = max_retries
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai  # lazy: package works without openai installed
            if not self.api_key:
                raise RuntimeError("VLMSolver: no API key (set ALK_VLM_API_KEY)")
            self._client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    @staticmethod
    def _parse(text, k):
        """Parse and validate a JSON reply. Returns DiscreteChoice (0-based)."""
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            raise ValueError("no JSON object in reply")
        obj = json.loads(text[start:end + 1])
        phi1, phi2 = int(obj["phi1"]), int(obj["phi2"])
        phi3 = int(obj.get("phi3", 0))
        if not (1 <= phi1 <= k and 1 <= phi2 <= k):
            raise ValueError("phi1/phi2 out of range 1..%d" % k)
        if phi1 == phi2:
            raise ValueError("phi1 == phi2")
        if phi3 not in (0, 1):
            raise ValueError("phi3 not in {0, 1}")
        phi4 = obj.get("phi4")
        phi5 = obj.get("phi5")
        return DiscreteChoice(phi1 - 1, phi2 - 1, bool(phi3),
                              None if phi4 is None else int(phi4) - 1,
                              None if phi5 is None else int(phi5) - 1)

    def solve(self, image=None, candidates_uv=None, task_description=""):
        import base64
        if image is None or candidates_uv is None:
            raise ValueError("VLMSolver.solve needs image and candidates_uv")
        k = len(candidates_uv)
        markup = draw_candidate_markup(image, candidates_uv)
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
                                 "Invalid reply (%s). Answer with ONLY the JSON "
                                 "object in the required format." % e})
        raise RuntimeError("VLMSolver: invalid reply after retries: %s" % last_err)
