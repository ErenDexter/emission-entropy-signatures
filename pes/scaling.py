"""How 1:N identification scales with the number of enrolled notes.

Verification re-ranks the top-k notes of the embedding search geometrically, so a note
is found only if its genuine embedding score beats all but k-1 of the other N-1 notes.
If one other note outranks the genuine score g with probability p = P(impostor > g),
the number that do is ~ Poisson((N-1) p). Recall@k at size N, and the k needed to keep
a target recall, follow from p alone.

p is estimated from measured impostor scores, standardized by their mean/std:
empirical below the 95th percentile, an exponential tail above it (generalized Pareto
with shape 0) so it can be extrapolated past the largest impostor score observed.
This is a projection from the measured scores, not a measurement at size N.
"""

from __future__ import annotations

import math

import numpy as np

SIZES = (1_000, 10_000, 100_000, 1_000_000, 10_000_000)


def standardize(genuine: np.ndarray, impostor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """z-scores relative to the impostor distribution (so scores of different models pool)."""
    mu, sd = float(np.mean(impostor)), float(np.std(impostor)) or 1.0
    return (np.asarray(genuine) - mu) / sd, (np.asarray(impostor) - mu) / sd


def exceed_prob(genuine: np.ndarray, impostor: np.ndarray, tail_q: float = 0.95) -> np.ndarray:
    """P(an impostor score beats each genuine score)."""
    g, imp = np.asarray(genuine, float), np.sort(np.asarray(impostor, float))
    u = float(np.quantile(imp, tail_q))
    exc = imp[imp > u] - u
    beta = float(exc.mean()) if len(exc) else 1e-9
    empirical = 1.0 - np.searchsorted(imp, g, side="right") / len(imp)
    tail = (1.0 - tail_q) * np.exp(-(g - u) / beta)
    return np.where(g <= u, empirical, tail)


def _poisson_cdf(k: int, lam: float) -> float:
    if lam == 0 or k > lam + 12 * math.sqrt(lam) + 60:
        return 1.0
    if lam > 2000:  # normal approximation with continuity correction
        return 0.5 * math.erfc(-(k + 0.5 - lam) / math.sqrt(2 * lam))
    return min(1.0, sum(math.exp(i * math.log(lam) - lam - math.lgamma(i + 1)) for i in range(k + 1)))


def recall_at(p: np.ndarray, n: int, k: int) -> float:
    """Expected fraction of genuine notes inside the top-k shortlist among n enrolled notes."""
    return float(np.mean([_poisson_cdf(k - 1, (n - 1) * pi) for pi in p]))


def shortlist_for_recall(p: np.ndarray, n: int, target: float = 0.99) -> int:
    """Smallest k with recall_at(p, n, k) >= target."""
    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        if recall_at(p, n, mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def decidability(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """d' = separation of the genuine and impostor means in pooled standard deviations."""
    g, i = np.asarray(genuine, float), np.asarray(impostor, float)
    return float(abs(g.mean() - i.mean()) / math.sqrt((g.var() + i.var()) / 2 + 1e-12))


def project(genuine: np.ndarray, impostor: np.ndarray, k: int = 20, target: float = 0.99,
            sizes: tuple[int, ...] = SIZES) -> dict:
    p = exceed_prob(genuine, impostor)
    return {
        "d_prime": decidability(genuine, impostor),
        "exceed_prob_median": float(np.median(p)),
        "shortlist_k": k,
        "recall_at_k": {n: recall_at(p, n, k) for n in sizes},
        "target_recall": target,
        "shortlist_needed": {n: shortlist_for_recall(p, n, target) for n in sizes},
    }
