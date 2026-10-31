"""Statistics used to decide with uncertainty instead of point estimates."""

from __future__ import annotations

import math
import random


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Args:
        successes: Number of successes; may be fractional when trials are averaged per case.
        n: Number of independent units, i.e. cases.
        z: Normal quantile; 1.96 gives a 95% interval.

    Returns:
        The lower and upper bounds; ``(0.0, 1.0)`` when ``n`` is zero.

    Raises:
        ValueError: If the counts are impossible.
    """
    if n < 0 or successes < 0 or successes > n:
        raise ValueError(f"invalid counts: {successes} successes out of {n}")
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def paired_bootstrap_diff(
    a: list[float], b: list[float], resamples: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Paired bootstrap of the mean difference ``a - b`` resampling cases.

    Args:
        a: Per-case scores of the candidate.
        b: Per-case scores of the baseline, aligned with ``a``.
        resamples: Number of bootstrap resamples.
        seed: Seed that makes the result reproducible.

    Returns:
        The mean difference and its 95% percentile interval.

    Raises:
        ValueError: If the inputs are empty or not aligned.
    """
    if len(a) != len(b):
        raise ValueError(f"paired samples differ in length: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("paired samples are empty")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(resamples))
    lo = means[int(0.025 * (resamples - 1))]
    hi = means[int(math.ceil(0.975 * (resamples - 1)))]
    return sum(diffs) / n, lo, hi


def pass_at_k(outcomes: list[bool]) -> bool:
    """Whether at least one of the k trials passed."""
    return any(outcomes)


def pass_hat_k(outcomes: list[bool]) -> bool:
    """Whether all k trials passed (pass^k, the reliability view)."""
    return bool(outcomes) and all(outcomes)
