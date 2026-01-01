"""Power analysis for the benchmark evaluation protocol.

Answers "how many rollouts does a claim of this size actually need?" -- the
original ten-rollout design had 8.3% power, which is why N was fixed at 60
before any campaign was run.

Simulation-based (seeded, deterministic) power for the two tests actually
used in the paper:

  - exact McNemar (paired design, seeds shared across methods) -- primary
  - Fisher exact (unpaired fallback)                            -- secondary

and the corresponding minimum-N searches at a target power (default 80%).

Model for the paired simulation: each seed yields a correlated Bernoulli
pair (ours ~ p1, baseline ~ p0) with success correlation ``rho``.  rho=0
(independent given the seed) is the CONSERVATIVE choice for McNemar power:
positive correlation (easy seeds are easy for everyone -- the realistic
case for a shared-seed design) shrinks the discordant mass symmetrically
and *increases* power for a fixed marginal difference, so the rho=0 minimum
N is an upper bound: the design is at least as well powered as this.

Pure numpy + stats.tests; no scipy required.  Python 3.8 compatible.
"""

import math
from functools import lru_cache
from typing import Dict, List, Sequence

import numpy as np

try:  # package import (python -m stats.power) and script import both work
    from stats.tests import mcnemar_exact, fisher_exact, norm_ppf
except ImportError:  # pragma: no cover
    from tests import mcnemar_exact, fisher_exact, norm_ppf  # type: ignore

__all__ = [
    "joint_probs",
    "power_mcnemar",
    "power_fisher",
    "min_n_mcnemar",
    "min_n_fisher",
    "justification_table",
    "justification_markdown",
]


# ---------------------------------------------------------------------------
# cached p-values (both tests depend only on integer counts)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _mcnemar_p(b: int, c: int) -> float:
    return mcnemar_exact(b, c).pvalue


@lru_cache(maxsize=None)
def _fisher_p(k1: int, n1: int, k2: int, n2: int) -> float:
    return fisher_exact([[k1, n1 - k1], [k2, n2 - k2]]).pvalue


# ---------------------------------------------------------------------------
# paired-outcome model
# ---------------------------------------------------------------------------

def joint_probs(p0: float, p1: float, rho: float = 0.0) -> Dict[str, float]:
    """Joint cell probabilities of a correlated Bernoulli pair.

    Returns {'p11','p10','p01','p00'} where the first index is OURS (rate
    p1) and the second the BASELINE (rate p0).  ``rho`` is the Pearson
    correlation of the two indicators; raises if the requested (p0,p1,rho)
    is infeasible (some cell < 0).
    """
    for p in (p0, p1):
        if not 0.0 <= p <= 1.0:
            raise ValueError("probabilities must be in [0, 1]")
    cov = rho * math.sqrt(p1 * (1 - p1) * p0 * (1 - p0))
    p11 = p1 * p0 + cov
    p10 = p1 - p11
    p01 = p0 - p11
    p00 = 1.0 - p11 - p10 - p01
    cells = {"p11": p11, "p10": p10, "p01": p01, "p00": p00}
    if any(v < -1e-12 for v in cells.values()):
        raise ValueError("infeasible (p0=%g, p1=%g, rho=%g): %r"
                         % (p0, p1, rho, cells))
    return {k: max(0.0, v) for k, v in cells.items()}


# ---------------------------------------------------------------------------
# simulated power
# ---------------------------------------------------------------------------

def power_mcnemar(p0: float, p1: float, n: int, alpha: float = 0.05,
                  rho: float = 0.0, n_sim: int = 4000, seed: int = 0) -> float:
    """Simulated power of the exact McNemar test at N paired seeds.

    n_sim experiments are drawn from the correlated-pair model
    (:func:`joint_probs`); power = fraction with two-sided p < alpha.
    Deterministic given ``seed``.
    """
    cells = joint_probs(p0, p1, rho)
    probs = [cells["p11"], cells["p10"], cells["p01"], cells["p00"]]
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, probs, size=int(n_sim))
    b_arr, c_arr = counts[:, 1], counts[:, 2]
    hits = sum(1 for b, c in zip(b_arr, c_arr)
               if _mcnemar_p(int(b), int(c)) < alpha)
    return hits / float(n_sim)


def power_fisher(p0: float, p1: float, n: int, alpha: float = 0.05,
                 n_sim: int = 4000, seed: int = 0) -> float:
    """Simulated power of the two-sided Fisher exact test with N per group.

    Two independent Binomial(n, .) samples (unpaired design).
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    k1 = rng.binomial(n, p1, size=int(n_sim))
    k2 = rng.binomial(n, p0, size=int(n_sim))
    hits = sum(1 for a, b in zip(k1, k2)
               if _fisher_p(int(a), n, int(b), n) < alpha)
    return hits / float(n_sim)


# ---------------------------------------------------------------------------
# analytic initial guesses (only used to bound the search window)
# ---------------------------------------------------------------------------

def _guess_n_mcnemar(p0: float, p1: float, alpha: float, power: float,
                     rho: float) -> int:
    cells = joint_probs(p0, p1, rho)
    pd = cells["p10"] + cells["p01"]
    delta = abs(p1 - p0)
    if delta == 0 or pd == 0:
        return 10
    za, zb = norm_ppf(1 - alpha / 2.0), norm_ppf(power)
    n = (za * math.sqrt(pd) + zb * math.sqrt(pd - delta * delta)) ** 2 / delta ** 2
    return max(4, int(math.ceil(n)))


def _guess_n_two_prop(p0: float, p1: float, alpha: float, power: float) -> int:
    delta = abs(p1 - p0)
    if delta == 0:
        return 10
    za, zb = norm_ppf(1 - alpha / 2.0), norm_ppf(power)
    pbar = (p0 + p1) / 2.0
    n = (za * math.sqrt(2 * pbar * (1 - pbar))
         + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2 / delta ** 2
    return max(4, int(math.ceil(n)))


def _search_min_n(power_fn, guess: int, target: float, n_max: int) -> int:
    """Smallest N with simulated power >= target, scanning step 1 from a
    window below the analytic guess (exact tests make power slightly
    non-monotone in N; step-1 scan from well below the guess is robust)."""
    start = max(4, int(guess * 0.5))
    for n in range(start, n_max + 1):
        if power_fn(n) >= target:
            return n
    raise RuntimeError("no N <= %d reaches the target power" % n_max)


def min_n_mcnemar(p0: float, p1: float, power: float = 0.8,
                  alpha: float = 0.05, rho: float = 0.0,
                  n_sim: int = 4000, seed: int = 0, n_max: int = 2000) -> int:
    """Minimum number of PAIRED seeds so the exact McNemar test detects
    p1 vs p0 with the given power (simulated, deterministic given seed).

    rho=0 is conservative for a shared-seed design (see module docstring).
    """
    guess = _guess_n_mcnemar(p0, p1, alpha, power, rho)
    return _search_min_n(
        lambda n: power_mcnemar(p0, p1, n, alpha, rho, n_sim, seed),
        guess, power, n_max)


def min_n_fisher(p0: float, p1: float, power: float = 0.8,
                 alpha: float = 0.05, n_sim: int = 4000, seed: int = 0,
                 n_max: int = 2000) -> int:
    """Minimum N PER GROUP so the two-sided Fisher exact test detects
    p1 vs p0 with the given power (simulated, deterministic given seed)."""
    guess = _guess_n_two_prop(p0, p1, alpha, power)
    return _search_min_n(
        lambda n: power_fisher(p0, p1, n, alpha, n_sim, seed),
        guess, power, n_max)


# ---------------------------------------------------------------------------
# the design-justification table quoted in the paper
# ---------------------------------------------------------------------------

def justification_table(p0: float, p1: float,
                        ns: Sequence[int] = (10, 30, 50, 100),
                        alpha: float = 0.05, rho: float = 0.0,
                        n_sim: int = 4000, seed: int = 0) -> List[Dict[str, float]]:
    """Power of both tests at candidate sample sizes.

    This is the table that answers the sample-size critique: it shows the
    power of the ORIGINAL N=10 protocol next to the chosen N, for the
    paper's headline effect (p1 = ours vs p0 = best baseline).

    Returns a list of dicts: {'n', 'power_mcnemar', 'power_fisher'}.
    """
    rows = []
    for n in ns:
        rows.append({
            "n": int(n),
            "power_mcnemar": power_mcnemar(p0, p1, n, alpha, rho, n_sim, seed),
            "power_fisher": power_fisher(p0, p1, n, alpha, n_sim, seed),
        })
    return rows


def justification_markdown(p0: float, p1: float,
                           ns: Sequence[int] = (10, 30, 50, 100),
                           alpha: float = 0.05, rho: float = 0.0,
                           n_sim: int = 4000, seed: int = 0,
                           target_power: float = 0.8) -> str:
    """Markdown table + minimum-N lines, ready to paste into the letter."""
    rows = justification_table(p0, p1, ns, alpha, rho, n_sim, seed)
    n_mc = min_n_mcnemar(p0, p1, target_power, alpha, rho, n_sim, seed)
    n_fi = min_n_fisher(p0, p1, target_power, alpha, n_sim, seed)
    lines = [
        "Detecting p1=%.2f (ours) vs p0=%.2f (baseline), two-sided alpha=%.2g, "
        "%d simulations, rho=%.2g:" % (p1, p0, alpha, n_sim, rho),
        "",
        "| N (seeds) | McNemar power (paired) | Fisher power (unpaired) |",
        "|---:|---:|---:|",
    ]
    for r in rows:
        lines.append("| %d | %.3f | %.3f |"
                     % (r["n"], r["power_mcnemar"], r["power_fisher"]))
    lines += [
        "",
        "Minimum N for %.0f%% power: **%d** (McNemar, paired) / "
        "**%d per group** (Fisher, unpaired)." % (100 * target_power, n_mc, n_fi),
    ]
    return "\n".join(lines)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="benchmark power analysis")
    ap.add_argument("--p0", type=float, default=0.40,
                    help="baseline success rate (default 0.40)")
    ap.add_argument("--p1", type=float, default=0.68,
                    help="our success rate (default 0.68)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--power", type=float, default=0.8)
    ap.add_argument("--n-sim", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    print(justification_markdown(args.p0, args.p1, alpha=args.alpha,
                                 rho=args.rho, n_sim=args.n_sim,
                                 seed=args.seed, target_power=args.power))


if __name__ == "__main__":
    main()
