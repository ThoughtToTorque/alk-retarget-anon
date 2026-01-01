"""Unit tests for stats/tests.py against scipy references and hand-computed
small cases.  scipy is available in the pinned environment; the module under test
itself must NOT need scipy (checked in test_no_scipy_import)."""

import math

import numpy as np
import pytest
import scipy.stats as ss

from stats.tests import (wilson_ci, mcnemar_exact, mcnemar_from_vectors,
                         paired_counts, fisher_exact, paired_bootstrap_diff,
                         holm_bonferroni, norm_ppf)


# ---------------------------------------------------------------------------
# norm_ppf
# ---------------------------------------------------------------------------

def test_norm_ppf_vs_scipy():
    for p in [1e-10, 1e-4, 0.01, 0.025, 0.2, 0.5, 0.7, 0.975, 0.999, 1 - 1e-10]:
        assert norm_ppf(p) == pytest.approx(ss.norm.ppf(p), abs=1e-9)


def test_norm_ppf_edges():
    assert norm_ppf(0.0) == -math.inf
    assert norm_ppf(1.0) == math.inf
    with pytest.raises(ValueError):
        norm_ppf(-0.1)


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------

def test_wilson_hand_computed():
    # classic textbook case: k=8, n=10, 95% -> (0.4902, 0.9433)
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.4902, abs=2e-3)
    assert hi == pytest.approx(0.9433, abs=2e-3)


def test_wilson_closed_form():
    # independent re-derivation of the closed form with scipy's z
    for k, n, alpha in [(0, 10, 0.05), (10, 10, 0.05), (3, 10, 0.05),
                        (34, 50, 0.05), (17, 50, 0.01), (1, 3, 0.1)]:
        z = ss.norm.ppf(1 - alpha / 2)
        ph = k / n
        denom = 1 + z**2 / n
        center = (ph + z**2 / (2 * n)) / denom
        half = z * math.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / denom
        lo, hi = wilson_ci(k, n, alpha)
        assert lo == pytest.approx(max(0.0, center - half), abs=1e-12)
        assert hi == pytest.approx(min(1.0, center + half), abs=1e-12)


def test_wilson_bounds_and_errors():
    lo, hi = wilson_ci(0, 5)
    assert lo == 0.0 and 0 < hi < 1
    lo, hi = wilson_ci(5, 5)
    assert hi == 1.0 and 0 < lo < 1
    with pytest.raises(ValueError):
        wilson_ci(6, 5)
    with pytest.raises(ValueError):
        wilson_ci(1, 0)


# ---------------------------------------------------------------------------
# exact McNemar
# ---------------------------------------------------------------------------

def test_mcnemar_hand_computed():
    # b=1, c=5: X~Bin(6, .5); P(X<=1) = (1+6)/64 = 7/64; p = 2*7/64 = 0.21875
    res = mcnemar_exact(1, 5)
    assert res.pvalue == pytest.approx(0.21875, abs=1e-12)
    # b=0, c=8: p = 2 * (1/256) = 0.0078125
    assert mcnemar_exact(0, 8).pvalue == pytest.approx(2 / 256, abs=1e-12)


def test_mcnemar_vs_scipy_binomtest():
    for b, c in [(0, 0), (0, 1), (1, 1), (2, 8), (5, 5), (3, 12), (0, 20),
                 (7, 19), (25, 40)]:
        p = mcnemar_exact(b, c).pvalue
        if b + c == 0:
            assert p == 1.0
        else:
            ref = ss.binomtest(min(b, c), b + c, 0.5).pvalue
            assert p == pytest.approx(ref, rel=1e-10)


def test_mcnemar_symmetry_and_degenerate():
    assert mcnemar_exact(3, 9).pvalue == mcnemar_exact(9, 3).pvalue
    assert mcnemar_exact(4, 4).pvalue == 1.0
    assert mcnemar_exact(0, 0).pvalue == 1.0
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_mcnemar_from_vectors():
    a = [1, 1, 0, 1, 0, 0, 1, 1]
    b = [1, 0, 0, 0, 1, 0, 1, 0]
    bb, cc = paired_counts(a, b)
    assert (bb, cc) == (3, 1)  # a-only successes: idx 1,3,7; b-only: idx 4
    res = mcnemar_from_vectors(a, b)
    assert (res.b, res.c) == (3, 1)
    assert res.pvalue == pytest.approx(ss.binomtest(1, 4, 0.5).pvalue, rel=1e-10)
    with pytest.raises(ValueError):
        paired_counts([1, 0], [1, 0, 1])


# ---------------------------------------------------------------------------
# Fisher exact
# ---------------------------------------------------------------------------

def test_fisher_hand_computed():
    # [[3, 1], [1, 3]]: classic; two-sided p = 0.485714... (=34/70)
    res = fisher_exact([[3, 1], [1, 3]])
    assert res.pvalue == pytest.approx(0.4857142857, abs=1e-9)
    assert res.odds_ratio == pytest.approx(9.0)


def test_fisher_vs_scipy():
    rng = np.random.default_rng(42)
    tables = [[[0, 10], [10, 0]], [[5, 5], [5, 5]], [[0, 0], [0, 5]],
              [[34, 16], [20, 30]], [[1, 49], [10, 40]]]
    for _ in range(30):
        tables.append(rng.integers(0, 25, size=(2, 2)).tolist())
    for t in tables:
        ours = fisher_exact(t)
        ref_or, ref_p = ss.fisher_exact(t)
        assert ours.pvalue == pytest.approx(ref_p, rel=1e-9, abs=1e-12), t
        if math.isnan(ref_or):
            assert math.isnan(ours.odds_ratio)
        else:
            assert ours.odds_ratio == pytest.approx(ref_or)


def test_fisher_input_validation():
    with pytest.raises(ValueError):
        fisher_exact([[1, -2], [3, 4]])
    with pytest.raises(ValueError):
        fisher_exact([[1.5, 2], [3, 4]])


# ---------------------------------------------------------------------------
# paired bootstrap
# ---------------------------------------------------------------------------

def test_bootstrap_deterministic_and_ordered():
    rng = np.random.default_rng(7)
    a = (rng.random(50) < 0.7).astype(int)
    b = (rng.random(50) < 0.4).astype(int)
    r1 = paired_bootstrap_diff(a, b, n_boot=2000, seed=123)
    r2 = paired_bootstrap_diff(a, b, n_boot=2000, seed=123)
    assert r1 == r2
    assert r1.lo <= r1.diff <= r1.hi
    assert r1.diff == pytest.approx(a.mean() - b.mean())


def test_bootstrap_identical_vectors_gives_zero():
    a = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    r = paired_bootstrap_diff(a, a, n_boot=500, seed=0)
    assert r.diff == 0.0 and r.lo == 0.0 and r.hi == 0.0


def test_bootstrap_coverage_sanity():
    # true diff 0.3 with n=200: 95% CI should comfortably exclude 0
    rng = np.random.default_rng(0)
    a = (rng.random(200) < 0.7).astype(int)
    b = (rng.random(200) < 0.4).astype(int)
    r = paired_bootstrap_diff(a, b, n_boot=5000, seed=1)
    assert r.lo > 0.0
    with pytest.raises(ValueError):
        paired_bootstrap_diff([], [], seed=0)


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------

def test_holm_hand_computed():
    # sorted p = (.01, .02, .03): adj = (3*.01, 2*.02, 1*.03) -> (.03, .04, .04)
    adj = holm_bonferroni([0.02, 0.03, 0.01])
    assert adj == pytest.approx([0.04, 0.04, 0.03])


def test_holm_monotone_clipped_and_order():
    adj = holm_bonferroni([0.6, 0.04, 0.5, 0.01])
    # sorted: .01*4=.04, .04*3=.12, .5*2=1.0, .6*1=.6 -> monotone: .04,.12,1.,1.
    assert adj == pytest.approx([1.0, 0.12, 1.0, 0.04])
    assert max(adj) <= 1.0


def test_holm_single_and_empty():
    assert holm_bonferroni([0.03]) == pytest.approx([0.03])
    assert holm_bonferroni([]) == []
    with pytest.raises(ValueError):
        holm_bonferroni([0.1, 1.2])


# ---------------------------------------------------------------------------
# module purity: stats.tests must not require scipy at import/run time
# ---------------------------------------------------------------------------

def test_no_scipy_import():
    import stats.tests as m
    src = open(m.__file__).read()
    for line in src.splitlines():
        s = line.strip()
        assert not s.startswith(("import scipy", "from scipy")), line
