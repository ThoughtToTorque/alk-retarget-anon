"""Statistical tests for benchmark success-rate comparisons.

Which test to use (frozen protocol, baselines/docs/CODE_MAP.md):

  wilson_ci            -- CI for a single success rate k/n.  Always report
                          this next to every k/n in the paper's tables.
  mcnemar_exact        -- PRIMARY method-vs-method test.  Use whenever both
                          methods were run on the SAME seed set (our paired
                          design: seeds shared across all methods).  Exact
                          binomial version; valid at any N, including small
                          discordant counts.
  fisher_exact         -- FALLBACK for unpaired comparisons (methods run on
                          different seeds / different N, e.g. numbers quoted
                          from another paper).  Less powerful than McNemar
                          on paired data -- do not use it when pairing exists.
  paired_bootstrap_diff-- effect-size CI for the success-rate DIFFERENCE
                          under the paired design (resamples seeds jointly).
                          Complements the McNemar p-value: report both.
  holm_bonferroni      -- multiple-comparison correction across the family
                          of pairwise-vs-ours tests.  Controls FWER at alpha
                          with no independence assumption; uniformly more
                          powerful than plain Bonferroni.

Everything is pure numpy + stdlib math (exact tests via log-gamma), so this
runs both in the pinned py3.8 environment and in a bare system python without
scipy.  scipy is used only as a cross-check in tests/test_stats_tests.py.
"""

import math
from collections import namedtuple
from typing import List, Sequence, Tuple

import numpy as np

__all__ = [
    "wilson_ci",
    "mcnemar_exact",
    "mcnemar_from_vectors",
    "paired_counts",
    "fisher_exact",
    "paired_bootstrap_diff",
    "holm_bonferroni",
    "norm_ppf",
]

McNemarResult = namedtuple("McNemarResult", ["b", "c", "pvalue"])
FisherResult = namedtuple("FisherResult", ["odds_ratio", "pvalue"])
BootstrapResult = namedtuple("BootstrapResult", ["diff", "lo", "hi"])


# ---------------------------------------------------------------------------
# normal quantile (no scipy): Acklam's rational approximation + one Halley
# refinement with math.erfc  ->  ~1e-15 accuracy over (0, 1).
# ---------------------------------------------------------------------------

_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF. Pure python, ~machine precision."""
    if not 0.0 < p < 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError("p must be in [0, 1], got %r" % (p,))
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = ((((( _C[0]*q + _C[1])*q + _C[2])*q + _C[3])*q + _C[4])*q + _C[5]) / \
            (((( _D[0]*q + _D[1])*q + _D[2])*q + _D[3])*q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = ((((( _A[0]*r + _A[1])*r + _A[2])*r + _A[3])*r + _A[4])*r + _A[5])*q / \
            ((((( _B[0]*r + _B[1])*r + _B[2])*r + _B[3])*r + _B[4])*r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -((((( _C[0]*q + _C[1])*q + _C[2])*q + _C[3])*q + _C[4])*q + _C[5]) / \
            (((( _D[0]*q + _D[1])*q + _D[2])*q + _D[3])*q + 1.0)
    # one Halley step against the true CDF (erfc is in stdlib math);
    # evaluate the residual in the tail nearest x to avoid cancellation
    if p <= 0.5:
        e = 0.5 * math.erfc(-x / math.sqrt(2.0)) - p
    else:
        e = (1.0 - p) - 0.5 * math.erfc(x / math.sqrt(2.0))
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Use for: the CI printed next to every success rate k/n in the paper's
    tables.  Preferred over the Wald interval because it behaves correctly
    at k=0, k=n and small n (never leaves [0,1], has near-nominal coverage
    down to n~5).

    Parameters: k successes out of n trials; two-sided level 1-alpha.
    Returns (lo, hi).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    z = norm_ppf(1.0 - alpha / 2.0)
    phat = k / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, center - half)
    hi = 1.0 if k == n else min(1.0, center + half)
    return lo, hi


# ---------------------------------------------------------------------------
# exact binomial helpers (log-space, safe for large n)
# ---------------------------------------------------------------------------

def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if p == 0.0:
        return 0.0 if k == 0 else -math.inf
    if p == 1.0:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log(1.0 - p))


def _binom_cdf_half(k: int, n: int) -> float:
    """P(X <= k) for X ~ Binomial(n, 1/2)."""
    return math.fsum(math.exp(_log_binom_pmf(i, n, 0.5)) for i in range(k + 1))


# ---------------------------------------------------------------------------
# exact McNemar (paired)
# ---------------------------------------------------------------------------

def mcnemar_exact(b: int, c: int) -> McNemarResult:
    """Exact (binomial) McNemar test for paired binary outcomes.

    Use for: comparing two methods evaluated on the SAME seeds (this benchmark's
    paired design).  Only the discordant pairs matter:
        b = #(method A succeeded, method B failed)
        c = #(method A failed,   method B succeeded)
    Under H0 (equal marginal success probabilities) the discordant outcomes
    are Binomial(b+c, 1/2); the two-sided exact p-value is
        p = min(1, 2 * P(X <= min(b,c))).
    This exact form is valid at any sample size (the chi-square McNemar
    approximation is not, below ~25 discordant pairs).

    Returns McNemarResult(b, c, pvalue).
    """
    if b < 0 or c < 0:
        raise ValueError("b, c must be non-negative")
    n = b + c
    if n == 0:
        return McNemarResult(b, c, 1.0)
    k = min(b, c)
    p = min(1.0, 2.0 * _binom_cdf_half(k, n))
    return McNemarResult(b, c, p)


def paired_counts(success_a: Sequence[int], success_b: Sequence[int]) -> Tuple[int, int]:
    """Discordant-pair counts (b, c) from two aligned 0/1 success vectors.

    b = #(a=1, b=0), c = #(a=0, b=1).  Vectors must be aligned by seed.
    """
    a = np.asarray(success_a, dtype=int)
    b_ = np.asarray(success_b, dtype=int)
    if a.shape != b_.shape:
        raise ValueError("success vectors must have equal length "
                         "(paired design: same seeds)")
    b = int(np.sum((a == 1) & (b_ == 0)))
    c = int(np.sum((a == 0) & (b_ == 1)))
    return b, c


def mcnemar_from_vectors(success_a: Sequence[int],
                         success_b: Sequence[int]) -> McNemarResult:
    """mcnemar_exact on two seed-aligned 0/1 success vectors."""
    b, c = paired_counts(success_a, success_b)
    return mcnemar_exact(b, c)


# ---------------------------------------------------------------------------
# Fisher exact (unpaired fallback)
# ---------------------------------------------------------------------------

def fisher_exact(table: Sequence[Sequence[int]]) -> FisherResult:
    """Two-sided Fisher exact test for a 2x2 contingency table.

    Use for: UNPAIRED comparisons only -- methods evaluated on different
    seed sets or with different N (e.g. success counts quoted from another
    paper).  When both methods share seeds, use mcnemar_exact instead
    (Fisher ignores the pairing and wastes power).

    table = [[a, b], [c, d]], e.g.
        [[k_ours, n_ours - k_ours], [k_base, n_base - k_base]].
    Two-sided p-value sums all tables (with the same margins) whose
    hypergeometric probability is <= that of the observed table, matching
    scipy.stats.fisher_exact.  Odds ratio is the conditional sample OR
    a*d / (b*c) (inf if b*c == 0 and a*d > 0, nan for 0/0).

    Returns FisherResult(odds_ratio, pvalue).
    """
    (a, b), (c, d) = table
    for v in (a, b, c, d):
        if v < 0 or int(v) != v:
            raise ValueError("table entries must be non-negative integers")
    a, b, c, d = int(a), int(b), int(c), int(d)
    n = a + b + c + d
    if n == 0:
        return FisherResult(math.nan, 1.0)
    row1, col1 = a + b, a + c

    def log_pmf(x: int) -> float:
        # hypergeometric: x successes in row1 draws, col1 marked, n total
        return (math.lgamma(row1 + 1) - math.lgamma(x + 1) - math.lgamma(row1 - x + 1)
                + math.lgamma(n - row1 + 1) - math.lgamma(col1 - x + 1)
                - math.lgamma(n - row1 - col1 + x + 1)
                - (math.lgamma(n + 1) - math.lgamma(col1 + 1) - math.lgamma(n - col1 + 1)))

    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    p_obs = math.exp(log_pmf(a))
    # relative tolerance identical in spirit to scipy's
    cutoff = p_obs * (1.0 + 1e-7)
    p = math.fsum(math.exp(log_pmf(x)) for x in range(lo, hi + 1)
                  if math.exp(log_pmf(x)) <= cutoff)
    p = min(1.0, p)
    if b * c == 0:
        oratio = math.nan if a * d == 0 else math.inf
    else:
        oratio = (a * d) / float(b * c)
    return FisherResult(oratio, p)


# ---------------------------------------------------------------------------
# paired bootstrap CI for a success-rate difference
# ---------------------------------------------------------------------------

def paired_bootstrap_diff(success_a: Sequence[int],
                          success_b: Sequence[int],
                          n_boot: int = 10000,
                          seed: int = 0,
                          alpha: float = 0.05) -> BootstrapResult:
    """Percentile bootstrap CI for rate(A) - rate(B) under the paired design.

    Use for: the effect-size interval accompanying the McNemar p-value.
    Seeds (= pairs) are resampled JOINTLY, so the between-method correlation
    induced by shared initial conditions is preserved; resampling the two
    vectors independently would be wrong and inflate the interval.

    Vectors must be aligned by seed.  Deterministic given ``seed``.
    Returns BootstrapResult(diff, lo, hi) with diff = observed difference.
    """
    a = np.asarray(success_a, dtype=float)
    b = np.asarray(success_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("need two 1-D success vectors of equal length")
    n = a.size
    if n == 0:
        raise ValueError("empty success vectors")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])
    return BootstrapResult(float(a.mean() - b.mean()), float(lo), float(hi))


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------

def holm_bonferroni(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values.

    Use for: correcting the family of pairwise-vs-ours comparisons (one
    family per table).  Controls the family-wise error rate at alpha under
    arbitrary dependence; reject H0_i iff adjusted p_i <= alpha.  Uniformly
    more powerful than plain Bonferroni.

    Returns adjusted p-values in the ORIGINAL input order, monotone and
    clipped to [0, 1].
    """
    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvals must be 1-D")
    if p.size == 0:
        return []
    if np.any((p < 0) | (p > 1) | np.isnan(p)):
        raise ValueError("p-values must be in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    adj_sorted = (m - np.arange(m)) * p[order]
    adj_sorted = np.maximum.accumulate(adj_sorted)   # enforce monotonicity
    adj_sorted = np.minimum(adj_sorted, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adj_sorted
    return out.tolist()
