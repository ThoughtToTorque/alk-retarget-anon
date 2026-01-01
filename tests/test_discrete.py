"""Tests for the discrete-variable interface (oracle + VLM stub plumbing)."""

import numpy as np
import pytest

from alkbench import DiscreteChoice, OracleSolver, VLMSolver
from alkbench.discrete import draw_candidate_markup, encode_png


def test_discrete_choice_equality_and_repr():
    a = DiscreteChoice(0, 5, True, phi4=2)
    b = DiscreteChoice(0, 5, 1, phi4=2)
    assert a == b
    assert a != DiscreteChoice(0, 5, False, phi4=2)
    assert "phi1=0" in repr(a)
    assert a.as_dict()["phi3"] == 1


def test_oracle_solver_returns_injected():
    gt = DiscreteChoice(1, 6, False)
    assert OracleSolver(gt).solve() is gt


def test_vlm_solver_constructs_without_openai():
    # lazy import: instantiation must not require the openai package
    s = VLMSolver(model="fake-model", base_url="http://x", api_key=None)
    assert s.model == "fake-model"


def test_vlm_parse_validation():
    ok = VLMSolver._parse('{"phi1": 2, "phi2": 7, "phi3": 1}', k=8)
    assert ok == DiscreteChoice(1, 6, True)
    ok2 = VLMSolver._parse('```json\n{"phi1": 1, "phi2": 8}\n```', k=8)
    assert ok2 == DiscreteChoice(0, 7, False)
    with pytest.raises(ValueError):
        VLMSolver._parse('{"phi1": 3, "phi2": 3, "phi3": 0}', k=8)
    with pytest.raises(ValueError):
        VLMSolver._parse('{"phi1": 0, "phi2": 9, "phi3": 0}', k=8)
    with pytest.raises(ValueError):
        VLMSolver._parse("no json here", k=8)
    with pytest.raises(ValueError):
        VLMSolver._parse('{"phi1": 1, "phi2": 2, "phi3": 5}', k=8)


def test_markup_and_png_encoding():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    uv = np.array([[16.0, 16.0], [48.0, 40.0]])
    marked = draw_candidate_markup(img, uv, zoom=False)
    assert marked.shape == img.shape
    assert (marked != 0).any()          # something was drawn
    assert (img == 0).all()             # input untouched
    zoomed = draw_candidate_markup(img, uv, zoom=True)
    assert zoomed.shape[0] >= img.shape[0]  # crop is upscaled for legibility
    assert (zoomed != 0).any()
    png = encode_png(marked)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert png.endswith(b"IEND" + png[-4:])  # IEND chunk present
