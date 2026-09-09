"""Optionally recalculate the manuscript's PT/RR and matched-panel metrics.

Record-level inputs are not distributed. Users must reconstruct the documented
local input tables from authorised access to the cited articles and errata. No
publisher-formatted material is extracted or read by this evaluator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BOOTSTRAP_SEED = 20260805
BOOTSTRAP_REPLICATES = 10_000
METRICS = ["rmse_log10", "mae_log10", "f2", "f3", "f4", "median_absolute_fold_error"]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = np.asarray(y_pred, float) - np.asarray(y_true, float)
    absolute = np.abs(residual)
    return {
        "rmse_log10": float(np.sqrt(np.mean(residual**2))),
        "mae_log10": float(np.mean(absolute)),
        "f2": float(np.mean(absolute <= np.log10(2.0) + 1e-15)),
        "f3": float(np.mean(absolute <= np.log10(3.0) + 1e-15)),
        "f4": float(np.mean(absolute <= np.log10(4.0) + 1e-15)),
        "median_absolute_fold_error": float(np.median(np.power(10.0, absolute))),
    }


def bootstrap_mean_ci(values: np.ndarray, offset: int) -> tuple[float, float]:
    values = np.asarray(values, float)
    rng = np.random.default_rng(BOOTSTRAP_SEED + offset)
    indexes = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    return tuple(np.quantile(values[indexes].mean(axis=1), [0.025, 0.975]).tolist())


def cluster_bootstrap(
    frame: pd.DataFrame, predictions: dict[str, str], offset: int
) -> tuple[dict[str, dict[str, float]], dict[str, float] | None]:
    clusters = frame["workbook_drug"].drop_duplicates().tolist()
    positions = {
        cluster: np.flatnonzero(frame["workbook_drug"].to_numpy() == cluster)
        for cluster in clusters
    }
    truth = frame["y_true_log10"].to_numpy(float)
    arrays = {name: frame[column].to_numpy(float) for name, column in predictions.items()}
    samples = {name: {metric: [] for metric in METRICS} for name in predictions}
    paired = [] if len(predictions) == 2 else None
    rng = np.random.default_rng(BOOTSTRAP_SEED + offset)
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        indexes = np.concatenate([positions[clusters[index]] for index in selected])
        replicate = {name: metrics(truth[indexes], values[indexes]) for name, values in arrays.items()}
        for name in predictions:
            for metric in METRICS:
                samples[name][metric].append(replicate[name][metric])
        if paired is not None:
            names = list(predictions)
            paired.append(replicate[names[0]]["rmse_log10"] - replicate[names[1]]["rmse_log10"])
    intervals = {}
    for name in predictions:
        intervals[name] = {}
        for metric in METRICS:
            low, high = np.quantile(samples[name][metric], [0.025, 0.975])
            intervals[name][f"ci95_{metric}_low"] = float(low)
            intervals[name][f"ci95_{metric}_high"] = float(high)
    paired_interval = None
    if paired is not None:
        low, high = np.quantile(paired, [0.025, 0.975])
        paired_interval = {"ci95_low": float(low), "ci95_high": float(high)}
    return intervals, paired_interval


def internal_summary(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_index, (key, frame) in enumerate(
        seed_metrics.groupby(["split_type", "model", "condition"], sort=True)
    ):
        row = {
            "evaluation_domain": "Kp_Data_internal_OOF",
            "method": "GNN",
            "split_type": key[0],
            "model": key[1],
            "condition": key[2],
            "n_split_seeds": int(frame["split_seed"].nunique()),
            "mean_n_rows": float(frame["n_rows"].mean()),
            "mean_n_parent_groups": float(frame["n_parent_groups"].mean()),
        }
        for metric_index, metric in enumerate(METRICS):
            values = frame[metric].to_numpy(float)
            low, high = bootstrap_mean_ci(values, 100 * group_index + metric_index)
            row[metric] = float(values.mean())
            row[f"ci95_{metric}_low"] = low
            row[f"ci95_{metric}_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def paper_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for method_index, (method, frame) in enumerate(frames.items()):
        intervals, _ = cluster_bootstrap(
            frame, {method: "baseline_pred_log10"}, 1000 + 100 * method_index
        )
        rows.append(
            {
                "evaluation_domain": f"{method}_source_paper_evaluation",
                "method": method,
                "split_type": "not_applicable",
                "model": "not_applicable",
                "condition": "not_applicable",
                "n_rows": int(len(frame)),
                "n_parent_groups": int(frame["workbook_drug"].nunique()),
                **metrics(frame["y_true_log10"], frame["baseline_pred_log10"]),
                **intervals[method],
            }
        )
    return pd.DataFrame(rows)


def direct_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows, paired_rows = [], []
    grouping = ["benchmark_method", "split_type", "model", "condition"]
    for group_index, (key, group) in enumerate(frame.groupby(grouping, sort=True)):
        method, split_type, model, condition = key
        intervals, paired = cluster_bootstrap(
            group,
            {"GNN": "gnn_pred_log10", method: "baseline_pred_log10"},
            2000 + 100 * group_index,
        )
        for evaluated, column in {"GNN": "gnn_pred_log10", method: "baseline_pred_log10"}.items():
            metric_rows.append(
                {
                    "evaluation_domain": f"{method}_paper_direct_panel",
                    "benchmark_method": method,
                    "split_type": split_type,
                    "model": model if evaluated == "GNN" else "not_applicable",
                    "condition": condition if evaluated == "GNN" else "not_applicable",
                    "method": evaluated,
                    "n_rows": int(len(group)),
                    "n_parent_groups": int(group["workbook_drug"].nunique()),
                    "min_oof_split_seeds": int(group["n_oof_split_seeds"].min()),
                    "median_oof_split_seeds": float(group["n_oof_split_seeds"].median()),
                    **metrics(group["y_true_log10"], group[column]),
                    **intervals[evaluated],
                }
            )
        gnn_rmse = metrics(group["y_true_log10"], group["gnn_pred_log10"])["rmse_log10"]
        baseline_rmse = metrics(group["y_true_log10"], group["baseline_pred_log10"])["rmse_log10"]
        paired_rows.append(
            {
                "evaluation_domain": f"{method}_paper_direct_panel",
                "benchmark_method": method,
                "split_type": split_type,
                "model": model,
                "condition": condition,
                "n_rows": int(len(group)),
                "n_parent_groups": int(group["workbook_drug"].nunique()),
                "gnn_minus_baseline_rmse_log10": gnn_rmse - baseline_rmse,
                **paired,
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(paired_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data/inputs/literature_benchmarks")
    )
    parser.add_argument("--output", type=Path, default=Path("results/manuscript_benchmarks"))
    args = parser.parse_args()
    source = args.input_dir
    required = [
        source / "internal_gnn_seed_metrics.csv",
        source / "pt_benchmark_processed.csv",
        source / "rr_benchmark_processed.csv",
        source / "direct_panel_record_predictions.csv",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        listed = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Record-level PT/RR inputs are intentionally not distributed. "
            "Reconstruct the documented local files from the cited original "
            f"articles and errata before running this evaluator. Missing:\n{listed}"
        )
    internal = pd.read_csv(source / "internal_gnn_seed_metrics.csv", encoding="utf-8-sig")
    papers = {
        "PT": pd.read_csv(source / "pt_benchmark_processed.csv", encoding="utf-8-sig"),
        "RR": pd.read_csv(source / "rr_benchmark_processed.csv", encoding="utf-8-sig"),
    }
    direct = pd.read_csv(source / "direct_panel_record_predictions.csv", encoding="utf-8-sig")
    direct_metrics, paired = direct_summary(direct)
    args.output.mkdir(parents=True, exist_ok=True)
    internal_summary(internal).to_csv(args.output / "internal_gnn_summary.csv", index=False)
    paper_summary(papers).to_csv(args.output / "paper_reproduction_summary.csv", index=False)
    direct_metrics.to_csv(args.output / "direct_panel_metrics.csv", index=False)
    paired.to_csv(args.output / "direct_panel_paired_rmse.csv", index=False)
    print(f"Wrote manuscript benchmark results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
