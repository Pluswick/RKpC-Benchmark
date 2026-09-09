"""Build PT and RR matched panels from authorised local record-level inputs.

This module contains matching logic only. It does not include or download any
record-level PT/RR values or Kp_Data records. All generated record-level files
remain local and are excluded from the public repository by ``.gitignore``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


REPRESENTATIVE_CONFIGS = {
    "parent_group": ("gine", "onehot_early"),
    "scaffold": ("d_mpnn", "onehot_late"),
}


def canonical_smiles(value: object) -> str | None:
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def prepare_paper_records(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    required = (
        "workbook_excel_row",
        "workbook_drug",
        "Tissue",
        "canonical_smiles",
        "paper_exp_kp",
        "baseline_pred_kp",
    )
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing {method} columns: {sorted(missing)}")
    result = frame[list(required)].copy()
    result["canonical_smiles"] = result["canonical_smiles"].map(canonical_smiles)
    for column in ("paper_exp_kp", "baseline_pred_kp"):
        result[column] = pd.to_numeric(result[column], errors="raise")
        if (result[column] <= 0).any():
            raise ValueError(f"{method} requires positive {column}")
    result["benchmark_method"] = method
    result["y_true_log10"] = np.log10(result["paper_exp_kp"])
    result["baseline_pred_log10"] = np.log10(result["baseline_pred_kp"])
    return result


def exact_paper_mapping(paper: pd.DataFrame, processed: pd.DataFrame) -> pd.DataFrame:
    required = {
        "row_index",
        "parent_group_id",
        "identity_unit_id",
        "Drug",
        "SMILES",
        "Tissue",
    }
    missing = required.difference(processed.columns)
    if missing:
        raise ValueError(f"Missing processed-data columns: {sorted(missing)}")
    processed = processed[list(required)].copy()
    processed["canonical_smiles"] = processed["SMILES"].map(canonical_smiles)
    mapping = paper.merge(
        processed,
        on=["canonical_smiles", "Tissue"],
        how="left",
        validate="one_to_many",
        suffixes=("_paper", "_training"),
    )
    mapping["exact_training_input_available"] = mapping["row_index"].notna()
    return mapping


def select_representative_oof(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "split_type",
        "split_seed",
        "model",
        "condition",
        "row_index",
        "Tissue",
        "y_pred",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing OOF-prediction columns: {sorted(missing)}")
    selected = []
    for split_type, (model, condition) in REPRESENTATIVE_CONFIGS.items():
        selected.append(
            predictions[
                predictions["split_type"].eq(split_type)
                & predictions["model"].eq(model)
                & predictions["condition"].eq(condition)
            ].copy()
        )
    result = pd.concat(selected, ignore_index=True)
    if result.empty:
        raise ValueError("No representative held-out predictions were selected")
    return result


def build_direct_panel(
    mappings: dict[str, pd.DataFrame], predictions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = []
    for method, mapping in mappings.items():
        available = mapping[mapping["exact_training_input_available"]].copy()
        available["row_index"] = available["row_index"].astype(int)
        linked = predictions.merge(
            available[
                [
                    "row_index",
                    "workbook_excel_row",
                    "workbook_drug",
                    "Tissue",
                    "canonical_smiles",
                    "paper_exp_kp",
                    "baseline_pred_kp",
                    "y_true_log10",
                    "baseline_pred_log10",
                ]
            ],
            on="row_index",
            how="inner",
            validate="many_to_many",
            suffixes=("_oof", "_paper"),
        )
        linked["benchmark_method"] = method
        if not linked["Tissue_oof"].eq(linked["Tissue_paper"]).all():
            raise RuntimeError("A row-index match produced inconsistent tissue labels")
        grouping = [
            "benchmark_method",
            "split_type",
            "split_seed",
            "model",
            "condition",
            "workbook_excel_row",
            "workbook_drug",
            "Tissue_paper",
            "canonical_smiles",
            "paper_exp_kp",
            "baseline_pred_kp",
            "y_true_log10",
            "baseline_pred_log10",
        ]
        seed_rows.append(
            linked.groupby(grouping, as_index=False, sort=True).agg(
                gnn_pred_log10=("y_pred", "mean"),
                n_matching_training_rows=("row_index", "nunique"),
                within_run_prediction_range=(
                    "y_pred",
                    lambda values: float(values.max() - values.min()),
                ),
            )
        )
    seed_level = pd.concat(seed_rows, ignore_index=True)
    pooled = seed_level.groupby(
        [
            "benchmark_method",
            "split_type",
            "model",
            "condition",
            "workbook_excel_row",
            "workbook_drug",
            "Tissue_paper",
            "canonical_smiles",
            "paper_exp_kp",
            "baseline_pred_kp",
            "y_true_log10",
            "baseline_pred_log10",
        ],
        as_index=False,
        sort=True,
    ).agg(
        gnn_pred_log10=("gnn_pred_log10", "mean"),
        n_oof_split_seeds=("split_seed", "nunique"),
        max_matching_training_rows=("n_matching_training_rows", "max"),
        max_within_run_prediction_range=("within_run_prediction_range", "max"),
    )
    pooled["gnn_pred_kp"] = np.power(10.0, pooled["gnn_pred_log10"])
    return seed_level, pooled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-data", type=Path, required=True)
    parser.add_argument("--oof-predictions", type=Path, required=True)
    parser.add_argument("--pt-records", type=Path, required=True)
    parser.add_argument("--rr-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/local_matched_panels"))
    args = parser.parse_args()

    processed = pd.read_csv(args.processed_data, encoding="utf-8-sig")
    predictions = select_representative_oof(
        pd.read_csv(args.oof_predictions, encoding="utf-8-sig")
    )
    papers = {
        "PT": prepare_paper_records(pd.read_csv(args.pt_records, encoding="utf-8-sig"), "PT"),
        "RR": prepare_paper_records(pd.read_csv(args.rr_records, encoding="utf-8-sig"), "RR"),
    }
    mappings = {method: exact_paper_mapping(frame, processed) for method, frame in papers.items()}
    seed_level, pooled = build_direct_panel(mappings, predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    seed_level.to_csv(args.output / "direct_panel_seed_predictions.csv", index=False)
    pooled.to_csv(args.output / "direct_panel_record_predictions.csv", index=False)


if __name__ == "__main__":
    main()
