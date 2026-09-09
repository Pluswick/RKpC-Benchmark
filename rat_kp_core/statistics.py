"""Prespecified effect estimation and LOTO confirmatory inference."""

from __future__ import annotations

import itertools

import numpy as np


BOOTSTRAP_SEED = 20260619
BOOTSTRAP_REPLICATES = 10_000


def percentile_ci(values: np.ndarray, replicates: int = BOOTSTRAP_REPLICATES) -> tuple[float, float]:
    """Bootstrap split seeds for the Part 1 paired mean effect."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(sampled, [0.025, 0.975]).tolist())


def exact_sign_flip_pvalue(deltas: np.ndarray) -> float:
    deltas = np.asarray(deltas, dtype=float)
    observed = abs(deltas.mean())
    statistics = [
        abs(np.mean(deltas * np.asarray(signs, dtype=float)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(deltas))
    ]
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
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

