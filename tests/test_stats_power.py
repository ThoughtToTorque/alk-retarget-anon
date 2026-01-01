"""Unit tests for stats/power.py (deterministic seeds) and the report-side
statistics in stats/report.py that feed the paper tables."""

import math

import numpy as np
import pytest

from stats import power
from stats import report
from stats.tests import wilson_ci


# ---------------------------------------------------------------------------
# joint_probs model
# ---------------------------------------------------------------------------

def test_joint_probs_independent():
    c = power.joint_probs(0.4, 0.68, rho=0.0)
    assert c["p11"] == pytest.approx(0.68 * 0.4)
    assert c["p10"] == pytest.approx(0.68 * 0.6)
    assert c["p01"] == pytest.approx(0.32 * 0.4)
    assert sum(c.values()) == pytest.approx(1.0)


def test_joint_probs_correlated_and_infeasible():
    c = power.joint_probs(0.4, 0.68, rho=0.3)
    assert sum(c.values()) == pytest.approx(1.0)
    assert c["p11"] > 0.68 * 0.4  # positive rho raises agreement
    with pytest.raises(ValueError):
        # p11 = 0.09 + 0.9*0.09 = 0.171 > p0 = 0.1 -> p01 < 0: infeasible
        power.joint_probs(0.1, 0.9, rho=0.9)
    with pytest.raises(ValueError):
        power.joint_probs(-0.1, 0.5)


# ---------------------------------------------------------------------------
# power simulations: determinism, size, monotonicity
# ---------------------------------------------------------------------------

def test_power_deterministic():
    a = power.power_mcnemar(0.4, 0.68, 30, n_sim=1000, seed=5)
    b = power.power_mcnemar(0.4, 0.68, 30, n_sim=1000, seed=5)
    assert a == b
    a = power.power_fisher(0.4, 0.68, 30, n_sim=500, seed=5)
    b = power.power_fisher(0.4, 0.68, 30, n_sim=500, seed=5)
    assert a == b


def test_power_size_under_null():
    # under H0 (p0 == p1), rejection rate must be <= alpha (exact tests
    # are conservative)
    for fn in (power.power_mcnemar, power.power_fisher):
        size = fn(0.5, 0.5, 40, alpha=0.05, n_sim=2000, seed=11)
        assert size <= 0.07, fn.__name__


def test_power_increases_with_n_and_effect():
    p_small = power.power_mcnemar(0.4, 0.68, 15, n_sim=2000, seed=3)
    p_big = power.power_mcnemar(0.4, 0.68, 80, n_sim=2000, seed=3)
    assert p_big > p_small
    weak = power.power_fisher(0.4, 0.5, 50, n_sim=1000, seed=3)
    strong = power.power_fisher(0.4, 0.8, 50, n_sim=1000, seed=3)
    assert strong > weak


def test_positive_rho_increases_mcnemar_power():
    p0 = power.power_mcnemar(0.4, 0.68, 30, rho=0.0, n_sim=3000, seed=9)
    p3 = power.power_mcnemar(0.4, 0.68, 30, rho=0.3, n_sim=3000, seed=9)
    assert p3 >= p0  # rho=0 is the conservative choice quoted in the letter


# ---------------------------------------------------------------------------
# minimum-N searches
# ---------------------------------------------------------------------------

def test_min_n_headline_close_to_analytic():
    # analytic (normal-approx) n for McNemar, 0.68 vs 0.40, 80%: ~52
    n_mc = power.min_n_mcnemar(0.40, 0.68, power=0.8, n_sim=4000, seed=0)
    assert 35 <= n_mc <= 70
    # two-proportion approx: ~49/group; Fisher exact is a bit conservative
    n_fi = power.min_n_fisher(0.40, 0.68, power=0.8, n_sim=4000, seed=0)
    assert 40 <= n_fi <= 75
    # deterministic given the seed
    assert n_mc == power.min_n_mcnemar(0.40, 0.68, power=0.8, n_sim=4000, seed=0)


def test_min_n_huge_effect_is_small():
    n = power.min_n_mcnemar(0.05, 0.95, power=0.8, n_sim=2000, seed=0)
    assert n <= 12


def test_min_n_raises_when_unreachable():
    with pytest.raises(RuntimeError):
        power.min_n_mcnemar(0.50, 0.52, power=0.8, n_sim=200, seed=0, n_max=60)


# ---------------------------------------------------------------------------
# justification table
# ---------------------------------------------------------------------------

def test_justification_table_shape_and_content():
    rows = power.justification_table(0.40, 0.68, ns=(10, 50), n_sim=1500, seed=0)
    assert [r["n"] for r in rows] == [10, 50]
    assert rows[0]["power_mcnemar"] < rows[1]["power_mcnemar"]
    assert rows[0]["power_mcnemar"] < 0.5      # N=10 underpowered
    assert rows[1]["power_mcnemar"] > 0.7      # N=50 ~ adequately powered
    md = power.justification_markdown(0.40, 0.68, ns=(10, 50), n_sim=1500, seed=0)
    assert "| 10 |" in md and "| 50 |" in md and "Minimum N" in md


# ---------------------------------------------------------------------------
# report-side math (linearity fit, decoupling, holm wiring)
# ---------------------------------------------------------------------------

def test_linear_fit_exact_line():
    f = report.linear_fit([0, 0.5, 1.0], [0.7, 0.4, 0.1])
    assert f["slope"] == pytest.approx(-0.6)
    assert f["intercept"] == pytest.approx(0.7)
    assert f["r2"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        report.linear_fit([1.0, 1.0], [0.0, 1.0])


def test_linear_fit_matches_numpy_r2():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 12)
    y = 0.8 - 0.5 * x + rng.normal(0, 0.03, 12)
    f = report.linear_fit(x, y)
    r = np.corrcoef(x, y)[0, 1]
    assert f["r2"] == pytest.approx(r * r, abs=1e-10)


def _toy_rows():
    rows = []
    for e, ks in [(0.0, 14), (0.5, 8), (1.0, 2)]:
        for i in range(20):
            wrong = i < int(round(e * 20))
            rows.append({"task": "t", "seed": i, "method": "m",
                         "discrete_correct": 0 if wrong else 1,
                         "success": 1 if i < ks else 0,
                         "injected_error_rate": e})
    return rows


def test_decoupling_analysis_structure():
    res = report.decoupling_analysis(_toy_rows())
    assert res["n_labeled"] == 60
    for key in ("given_correct", "given_wrong"):
        c = res[key]
        assert c is not None and 0 <= c["rate"] <= 1
        lo, hi = c["ci"]
        assert lo <= c["rate"] <= hi
    assert res["fit"] is not None
    assert res["fit"]["slope"] < 0  # more injected error -> less success
    assert len(res["points"]) == 3
    # no flags -> graceful empty result
    empty = report.decoupling_analysis([{"task": "t", "seed": 0,
                                         "method": "m", "success": 1,
                                         "discrete_correct": None}])
    assert empty["n_labeled"] == 0 and empty["fit"] is None
    assert "no rollouts carry" in report.decoupling_markdown(empty)


def test_success_table_paired_mcnemar_and_holm():
    # two tasks, three methods on shared seeds -> McNemar + per-task Holm
    rng = np.random.default_rng(1)
    rows = []
    for task in ("a", "b"):
        for seed in range(30):
            base = rng.random()
            rows.append({"task": task, "seed": seed, "method": "full",
                         "success": int(base < 0.8)})
            rows.append({"task": task, "seed": seed, "method": "m1",
                         "success": int(base < 0.3)})
            rows.append({"task": task, "seed": seed, "method": "m2",
                         "success": int(rng.random() < 0.5)})
    t = report.make_success_table(rows, ours="full")
    assert t["methods"][0] == "full"
    cell = t["cells"][("a", "m1")]
    assert cell["test"] == "mcnemar" and cell["pvalue"] is not None
    assert cell["p_holm"] >= cell["pvalue"]
    ours_cell = t["cells"][("a", "full")]
    assert ours_cell["pvalue"] is None
    lo, hi = ours_cell["ci"]
    assert (lo, hi) == wilson_ci(ours_cell["k"], ours_cell["n"])
    pooled = t["cells"][(report.POOLED, "full")]
    assert pooled["n"] == 60
    # unpaired fallback: different seed sets -> fisher
    rows2 = [{"task": "a", "seed": s, "method": "full", "success": 1}
             for s in range(10)]
    rows2 += [{"task": "a", "seed": s + 100, "method": "m1", "success": 0}
              for s in range(10)]
    t2 = report.make_success_table(rows2, ours="full")
    assert t2["cells"][("a", "m1")]["test"] == "fisher"
    # renderers do not crash and contain the key markers
    md = report.success_table_markdown(t)
    tex = report.success_table_latex(t)
    assert "Wilson" in md and r"\toprule" in tex and r"\bottomrule" in tex
