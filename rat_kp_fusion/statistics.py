"""Prespecified inference utilities for the injection study."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


BOOTSTRAP_SEED = 20260803
BOOTSTRAP_REPLICATES = 10_000

# Exhaustive sign-flip enumeration costs 2**n. The real families are ten paired
# split seeds or eleven held-out tissues, so this cap cannot be reached by a
# legitimate run; it turns an accidental misuse into an error rather than an
# apparent hang.
MAX_SIGN_FLIP_DIFFERENCES = 20


def percentile_seed_ci(deltas, replicates: int = BOOTSTRAP_REPLICATES):
    """Percentile bootstrap interval for the mean of ten paired seed differences.

    The length is asserted: an observed-tissue contrast is defined over the ten
    frozen split seeds, so a shorter vector means runs are missing rather than
    that a smaller interval should be reported.
    """
    values = np.asarray(deltas, dtype=float)
    if values.ndim != 1 or len(values) != 10 or not np.isfinite(values).all():
        raise ValueError("Part 1 requires ten finite paired split-seed differences")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).tolist())


def exact_sign_flip_pvalue(deltas) -> float:
    """Two-sided exact sign-flip test on paired differences.

    Enumerates all 2**n sign assignments rather than sampling them, so the
    p-value is exact and identical on every run. It is valid because the null
    hypothesis is symmetry of the paired differences about zero, which needs no
    distributional assumption. Exhaustive enumeration is only tractable for the
    small n used here (10 split seeds, or 11 held-out tissues).
    """
    values = np.asarray(deltas, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Sign-flip input must be one-dimensional and finite")
    if len(values) > MAX_SIGN_FLIP_DIFFERENCES:
        raise ValueError(
            f"Exact sign-flip enumeration is capped at {MAX_SIGN_FLIP_DIFFERENCES} "
            f"paired differences; received {len(values)}"
        )
    observed = abs(values.mean())
    null = [
        abs(np.mean(values * np.asarray(signs, dtype=float)))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def benjamini_hochberg(pvalues) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, returned in the input order.

    The running minimum over the reversed ranking enforces monotonicity, so an
    adjusted value can never fall below one from a smaller raw p-value.
    """
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def tissue_macro_rmse(predictions: pd.DataFrame) -> float:
    """RMSE averaged over tissues with equal weight per tissue.

    The dataset is unevenly sampled across tissues, so a pooled RMSE is
    dominated by the best-represented tissues. Weighting tissues equally
    answers the deployment question of how well a model does per tissue.
    """
    required = {"Tissue", "y_true", "y_pred"}
    if required - set(predictions.columns):
        raise ValueError("Missing tissue-macro metric columns")
    values = []
    for _, frame in predictions.groupby("Tissue", sort=True):
        residual = frame["y_pred"].to_numpy(float) - frame["y_true"].to_numpy(float)
        values.append(float(np.sqrt(np.mean(residual ** 2))))
    return float(np.mean(values))


def loto_tissue_parent_cluster_ci(
    predictions: pd.DataFrame,
    comparison: str,
    reference: str,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    """Resample tissues equally, then parent groups within each selected tissue.

    Input predictions must already be averaged over the five training seeds and contain
    one row per condition and original test row.
    """
    required = {"Tissue", "parent_group_id", "row_index", "condition", "y_true", "y_pred"}
    if required - set(predictions.columns):
        raise ValueError("Missing LOTO cluster-bootstrap columns")
    selected = predictions[predictions["condition"].isin([comparison, reference])].copy()
    wide = selected.pivot(
        index=["Tissue", "parent_group_id", "row_index", "y_true"],
        columns="condition",
        values="y_pred",
    ).reset_index()
    if comparison not in wide or reference not in wide or wide[[comparison, reference]].isna().any().any():
        raise ValueError("Incomplete paired LOTO predictions")
    tissues = sorted(wide["Tissue"].unique())
    # The bootstrap unit is the parent group.  Pre-aggregate squared-error sums
    # and row counts once so each replicate only resamples small NumPy arrays.
    # This is mathematically identical to concatenating every sampled group's
    # original rows, including the multiplicity induced by repeated draws.
    tissue_group_summaries = {}
    for tissue in tissues:
        tissue_frame = wide[wide["Tissue"] == tissue].copy()
        tissue_frame["comparison_se"] = (
            tissue_frame[comparison].to_numpy(float) - tissue_frame["y_true"].to_numpy(float)
        ) ** 2
        tissue_frame["reference_se"] = (
            tissue_frame[reference].to_numpy(float) - tissue_frame["y_true"].to_numpy(float)
        ) ** 2
        group_summary = tissue_frame.groupby(
            "parent_group_id", sort=False, as_index=False
        ).agg(
            comparison_sse=("comparison_se", "sum"),
            reference_sse=("reference_se", "sum"),
            n_rows=("row_index", "size"),
        )
        tissue_group_summaries[tissue] = (
            group_summary["comparison_sse"].to_numpy(float),
            group_summary["reference_sse"].to_numpy(float),
            group_summary["n_rows"].to_numpy(float),
        )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled_tissues = rng.choice(tissues, size=len(tissues), replace=True)
        tissue_deltas = []
        for tissue in sampled_tissues:
            comparison_sse, reference_sse, row_counts = tissue_group_summaries[tissue]
            group_count = len(row_counts)
            sampled_groups = rng.choice(group_count, size=group_count, replace=True)
            sampled_n = row_counts[sampled_groups].sum()
            comparison_rmse = np.sqrt(comparison_sse[sampled_groups].sum() / sampled_n)
            reference_rmse = np.sqrt(reference_sse[sampled_groups].sum() / sampled_n)
            tissue_deltas.append(comparison_rmse - reference_rmse)
        estimates[replicate] = np.mean(tissue_deltas)
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())
