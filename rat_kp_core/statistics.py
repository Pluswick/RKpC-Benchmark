"""Prespecified effect estimation and LOTO confirmatory inference."""

from __future__ import annotations

import itertools

import numpy as np


BOOTSTRAP_SEED = 20260619
BOOTSTRAP_REPLICATES = 10_000

# Exhaustive sign-flip enumeration costs 2**n. The real families are ten paired
# split seeds or eleven held-out tissues, so this cap cannot be reached by a
# legitimate run; it turns an accidental misuse into an error rather than an
# apparent hang.
MAX_SIGN_FLIP_DIFFERENCES = 20


def percentile_ci(values: np.ndarray, replicates: int = BOOTSTRAP_REPLICATES) -> tuple[float, float]:
    """Bootstrap split seeds for the Part 1 paired mean effect."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(sampled, [0.025, 0.975]).tolist())


def exact_sign_flip_pvalue(deltas: np.ndarray) -> float:
    """Two-sided exact sign-flip test on paired differences.

    Enumerates all 2**n sign assignments rather than sampling them, so the
    p-value is exact and identical on every run. It is valid because the null
    hypothesis is symmetry of the paired differences about zero, which needs no
    distributional assumption. Exhaustive enumeration is only tractable for the
    small n used here (10 split seeds, or 11 held-out tissues).
    """
    deltas = np.asarray(deltas, dtype=float)
    if len(deltas) > MAX_SIGN_FLIP_DIFFERENCES:
        raise ValueError(
            f"Exact sign-flip enumeration is capped at {MAX_SIGN_FLIP_DIFFERENCES} "
            f"paired differences; received {len(deltas)}"
        )
    observed = abs(deltas.mean())
    statistics = [
        abs(np.mean(deltas * np.asarray(signs, dtype=float)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(deltas))
    ]
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, returned in the input order.

    The running minimum over the reversed ranking enforces monotonicity, so an
    adjusted value can never fall below one from a smaller raw p-value.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def hierarchical_loto_ci(
    tissue_rows: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    """Resample tissues equally, then compounds/rows within each sampled tissue."""
    tissues = sorted(tissue_rows)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(replicates, dtype=float)
    for bootstrap_index in range(replicates):
        selected = rng.choice(tissues, size=len(tissues), replace=True)
        deltas = []
        for tissue in selected:
            y_true, structure, physiology = tissue_rows[str(tissue)]
            row_indices = rng.integers(0, len(y_true), size=len(y_true))
            truth = y_true[row_indices]
            structure_rmse = np.sqrt(np.mean((structure[row_indices] - truth) ** 2))
            physiology_rmse = np.sqrt(np.mean((physiology[row_indices] - truth) ** 2))
            deltas.append(structure_rmse - physiology_rmse)
        estimates[bootstrap_index] = np.mean(deltas)
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())

