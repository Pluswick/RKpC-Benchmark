"""Build the identity-resolved rat Kp datasets and leakage-safe splits.

The required source table is not distributed with the public code. When an
authorised local copy is supplied, all generated files remain inside this
repository's ignored data directories.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from rat_kp_core.data import TISSUES, _parse_kp, read_csv, sha256_file
from rat_kp_core.features import build_tabular_feature_cache


ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data" / "inputs" / "source_data_not_distributed.csv"
PROCESSED_DIR = ROOT / "data" / "inputs"
AUDIT_DIR = ROOT / "data" / "generated_audit"
SPLIT_DIR = ROOT / "data" / "inputs" / "splits"
LOTO_DIR = SPLIT_DIR / "loto"
SENSITIVITY_SPLIT_ROOT = ROOT / "data" / "inputs" / "splits_sensitivity"

AD_MW_THRESHOLD = 800.0
FLUNITRAZEPAM_SMILES = (
    "CN1C(=O)CN=C(c2ccccc2F)c2cc([N+](=O)[O-])ccc21"
)

CONDITION_PATTERNS = (
    (
        re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*ng/ml", re.IGNORECASE),
        "concentration",
        "ng/mL",
    ),
    (
        re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*mg/kg/day", re.IGNORECASE),
        "dose",
        "mg/kg/day",
    ),
)


def normalize_label(label: str) -> str:
    key = label.strip().lower()
    direct = {
        "phenobarbitone": "phenobarbital",
        "phenobarbital": "phenobarbital",
        "carvedilol r-enantiomer": "R-carvedilol",
        "r-carvedilol": "R-carvedilol",
        "carvedilol-r": "R-carvedilol",
        "carvedilol s-enantiomer": "S-carvedilol",
        "s-carvedilol": "S-carvedilol",
        "carvedilol-s": "S-carvedilol",
        "thiopentone": "thiopental",
        "disopyramide r": "R-disopyramide",
        "disopyramide r-enantiomer": "R-disopyramide",
        "disopyramide s": "S-disopyramide",
        "disopyramide s-enantiomer": "S-disopyramide",
    }
    if key in direct:
        return direct[key]
    if key.startswith("3,4-diaminopyridine "):
        return "3,4-diaminopyridine"
    if key == "cyclosporin" or key.startswith("cyclosporine "):
        return "cyclosporin A"
    return label.strip()


def parse_condition(label: str) -> tuple[str, float | None, str]:
    for pattern, kind, unit in CONDITION_PATTERNS:
        match = pattern.search(label)
        if match:
            return kind, float(match.group("value")), unit
    return "unspecified", None, ""


def molecule_fields(smiles: str) -> dict[str, object] | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    parent = max(
        fragments,
        key=lambda item: (
            item.GetNumHeavyAtoms(),
            item.GetNumAtoms(),
            Chem.MolToSmiles(item, canonical=True, isomericSmiles=True),
        ),
    )
    parent_smiles = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    parent_no_stereo = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=False)
    inchikey = Chem.MolToInchiKey(parent)
    neutral_parent = rdMolStandardize.Uncharger().uncharge(Chem.Mol(parent))
    scaffold_key = MurckoScaffold.MurckoScaffoldSmiles(
        mol=neutral_parent, includeChirality=False
    )
    return {
        "model_smiles": canonical,
        "parent_smiles": parent_smiles,
        "parent_no_stereo_smiles": parent_no_stereo,
        "parent_inchikey": inchikey,
        "parent_connectivity_key": inchikey.split("-")[0],
        "formula": rdMolDescriptors.CalcMolFormula(parent),
        "molecular_weight": float(Descriptors.MolWt(parent)),
        # Scaffold grouping is charge-neutralized only for partition assignment.
        # Model inputs retain the resolved formal charge and stereochemistry.
        "murcko_scaffold": scaffold_key,
    }


def stable_id(prefix: str, value: str, length: int = 12) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:length]}"


def classify_rat_row(row: pd.Series, parsed: dict[str, object] | None) -> tuple[str, str, str]:
    label = str(row["Drug"]).strip().lower()
    formula = "" if parsed is None else str(parsed["formula"])
    source_smiles = str(row["SMILES"])

    if label == "thiopental":
        return (
            "quarantine",
            "thiopental row carries thioridazine connectivity; correct thiopentone row exists",
            "https://pubchem.ncbi.nlm.nih.gov/compound/3000715",
        )
    if label == "ethoxybenzamide":
        return (
            "quarantine",
            "ambiguous 2-/4-ethoxybenzamide positional isomer; no source provenance",
            "https://webbook.nist.gov/cgi/cbook.cgi?ID=938-73-8",
        )
    if label == "digoxin" and formula == "C16H13ClN2O":
        return (
            "quarantine",
            "digoxin-labelled row carries diazepam connectivity",
            "https://pubchem.ncbi.nlm.nih.gov/compound/3062",
        )
    if label == "grepafloxacin" and formula == "C18H20FN3O3":
        return (
            "quarantine",
            "C18 structure is inconsistent with official C19 grepafloxacin",
            "https://pubchem.ncbi.nlm.nih.gov/compound/72474",
        )
    if label == "trihexyphenidyl" and formula in {"C19H30NO+", "C19H29NO"}:
        return (
            "quarantine",
            "trihexyphenidyl-labelled row carries procyclidine-like C19 connectivity",
            "https://pubchem.ncbi.nlm.nih.gov/compound/4919",
        )
    if label in {"glycyrrhetinic acid", "glycyrrhizin", "laniquidar"}:
        return (
            "quarantine",
            "invalid source structure requires unavailable experimental provenance",
            {
                "glycyrrhetinic acid": "https://pubchem.ncbi.nlm.nih.gov/compound/10114",
                "glycyrrhizin": "https://pubchem.ncbi.nlm.nih.gov/compound/14982",
                "laniquidar": "https://pubchem.ncbi.nlm.nih.gov/compound/6450806",
            }[label],
        )
    if label == "cyclosporin" and parsed is None:
        return (
            "quarantine",
            "invalid cyclosporin structure; valid cyclosporin A records are retained separately",
            "https://pubchem.ncbi.nlm.nih.gov/compound/5284373",
        )
    if label == "flunitrazepam" and parsed is None:
        return (
            "corrected_retained",
            "source SMILES syntax corrected to PubChem flunitrazepam connectivity",
            "https://pubchem.ncbi.nlm.nih.gov/compound/3380",
        )
    if parsed is None:
        return "quarantine", "unresolved invalid SMILES", ""
    return "retained", "valid source structure retained", ""


def build_resolution_manifest(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw.insert(0, "source_row_id", [f"RAW{i:04d}" for i in range(len(raw))])
    raw.insert(1, "source_csv_row", np.arange(len(raw), dtype=int) + 2)
    comparison_columns = [column for column in raw.columns if column not in {"source_row_id", "source_csv_row"}]
    raw["exact_duplicate"] = raw.duplicated(subset=comparison_columns, keep=False)
    first_by_key: dict[tuple[object, ...], str] = {}
    duplicate_of = []
    for row in raw.itertuples(index=False):
        key = tuple(
            "<NA>" if pd.isna(getattr(row, column)) else getattr(row, column)
            for column in comparison_columns
        )
        first = first_by_key.setdefault(key, row.source_row_id)
        duplicate_of.append("" if first == row.source_row_id else first)
    raw["duplicate_of_source_row_id"] = duplicate_of

    records: list[dict[str, object]] = []
    for row in raw.itertuples(index=False):
        item = row._asdict()
        species = str(item["Species"]).strip().lower()
        label = str(item["Drug"]).strip()
        condition_type, condition_value, condition_unit = parse_condition(label)
        source_smiles = str(item["SMILES"])
        corrected_smiles = FLUNITRAZEPAM_SMILES if label.lower() == "flunitrazepam" else source_smiles
        source_parsed = molecule_fields(source_smiles)
        corrected_parsed = molecule_fields(corrected_smiles)

        if species != "rat":
            status, reason, source_url = "out_of_scope_species", "non-rat row", ""
        else:
            status, reason, source_url = classify_rat_row(pd.Series(item), source_parsed)
            if status == "corrected_retained":
                corrected_parsed = molecule_fields(FLUNITRAZEPAM_SMILES)
            else:
                corrected_smiles = source_smiles
                corrected_parsed = source_parsed

            kp_values = [_parse_kp(item[tissue]) for tissue in TISSUES]
            has_positive_kp = any(np.isfinite(value) and value > 0 for value in kp_values)
            if status in {"retained", "corrected_retained"} and not has_positive_kp:
                status, reason = "no_target", "no positive/non-missing rat Kp values"

            if item["duplicate_of_source_row_id"] and status in {"retained", "corrected_retained"}:
                status, reason = "duplicate_removed", "all raw fields identical to earlier source row"

        parsed = corrected_parsed
        normalized = normalize_label(label)
        record = {
            **item,
            "species_normalized": species,
            "compound_name_normalized": normalized,
            "condition_type": condition_type,
            "condition_value": condition_value,
            "condition_unit": condition_unit,
            "source_smiles_valid": source_parsed is not None,
            "resolved_smiles": "" if parsed is None else parsed["model_smiles"],
            "resolution_status": status,
            "resolution_reason": reason,
            "resolution_source_url": source_url,
            "included_in_primary": status in {"retained", "corrected_retained"},
        }
        for key in (
            "parent_smiles",
            "parent_no_stereo_smiles",
            "parent_inchikey",
            "parent_connectivity_key",
            "formula",
            "molecular_weight",
            "murcko_scaffold",
        ):
            record[key] = "" if parsed is None else parsed[key]
        records.append(record)
    return pd.DataFrame(records)


def add_identity_ids(manifest: pd.DataFrame) -> pd.DataFrame:
    result = manifest.copy()
    included = result["included_in_primary"].astype(bool)
    parent_keys = sorted(result.loc[included, "parent_connectivity_key"].astype(str).unique())
    parent_ids = {key: f"PG{index:04d}" for index, key in enumerate(parent_keys, start=1)}
    result["parent_group_id"] = ""
    result.loc[included, "parent_group_id"] = result.loc[included, "parent_connectivity_key"].map(parent_ids)

    entity_keys = (
        result.loc[included, "compound_name_normalized"].astype(str).str.lower()
        + "||"
        + result.loc[included, "resolved_smiles"].astype(str)
    )
    unique_entities = sorted(entity_keys.unique())
    entity_ids = {key: f"IU{index:04d}" for index, key in enumerate(unique_entities, start=1)}
    result["identity_unit_id"] = ""
    result.loc[included, "identity_unit_id"] = entity_keys.map(entity_ids)
    return result


def condition_specific_long(manifest: pd.DataFrame) -> pd.DataFrame:
    retained = manifest[manifest["included_in_primary"].astype(bool)].copy()
    id_columns = [
        "source_row_id",
        "source_csv_row",
        "compound_name_normalized",
        "resolved_smiles",
        "parent_group_id",
        "identity_unit_id",
        "parent_connectivity_key",
        "murcko_scaffold",
        "molecular_weight",
        "condition_type",
        "condition_value",
        "condition_unit",
    ]
    long = retained.melt(
        id_vars=id_columns,
        value_vars=TISSUES,
        var_name="Tissue",
        value_name="Kp_raw",
    )
    long["Kp"] = long["Kp_raw"].map(_parse_kp)
    long = long[np.isfinite(long["Kp"]) & (long["Kp"] > 0)].copy()
    long["Species"] = "rat"
    long["Tissue"] = long["Tissue"].str.lower()
    long["log10Kp"] = np.log10(long["Kp"].astype(float))
    long["SMILES"] = long["resolved_smiles"]
    long["Drug"] = long["compound_name_normalized"]
    long["ad_high_mw"] = long["molecular_weight"].astype(float) > AD_MW_THRESHOLD
    result = long[
        [
            "source_row_id",
            "source_csv_row",
            "parent_group_id",
            "identity_unit_id",
            "Drug",
            "SMILES",
            "Species",
            "Tissue",
            "Kp",
            "log10Kp",
            "condition_type",
            "condition_value",
            "condition_unit",
            "molecular_weight",
            "ad_high_mw",
            "parent_connectivity_key",
            "murcko_scaffold",
        ]
    ].reset_index(drop=True)
    result.insert(0, "row_index", np.arange(len(result), dtype=int))
    return result


def aggregate_primary(condition_long: pd.DataFrame) -> pd.DataFrame:
    def join(values: pd.Series) -> str:
        return " | ".join(sorted({str(value) for value in values if pd.notna(value) and str(value) != ""}))

    grouped = (
        condition_long.groupby(
            [
                "parent_group_id",
                "identity_unit_id",
                "Drug",
                "SMILES",
                "Species",
                "Tissue",
                "parent_connectivity_key",
                "murcko_scaffold",
                "molecular_weight",
                "ad_high_mw",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            Kp=("Kp", "median"),
            n_measurements=("Kp", "size"),
            Kp_min=("Kp", "min"),
            Kp_max=("Kp", "max"),
            source_row_ids=("source_row_id", join),
            condition_types=("condition_type", join),
            condition_values=("condition_value", join),
            condition_units=("condition_unit", join),
        )
        .reset_index(drop=True)
    )
    grouped["log10Kp"] = np.log10(grouped["Kp"].astype(float))
    grouped.insert(0, "row_index", np.arange(len(grouped), dtype=int))
    return grouped[
        [
            "row_index",
            "parent_group_id",
            "identity_unit_id",
            "Drug",
            "SMILES",
            "Species",
            "Tissue",
            "Kp",
            "log10Kp",
            "n_measurements",
            "Kp_min",
            "Kp_max",
            "source_row_ids",
            "condition_types",
            "condition_values",
            "condition_units",
            "molecular_weight",
            "ad_high_mw",
            "parent_connectivity_key",
            "murcko_scaffold",
        ]
    ]


def split_parent_groups(groups: list[str], seed: int, fractions: tuple[float, float, float]) -> dict[str, str]:
    values = np.array(sorted(set(groups)), dtype=object)
    np.random.default_rng(seed).shuffle(values)
    n_train = round(len(values) * fractions[0])
    n_val = round(len(values) * fractions[1])
    return {
        str(group): "train" if index < n_train else "val" if index < n_train + n_val else "test"
        for index, group in enumerate(values)
    }


def scaffold_parent_split(long_df: pd.DataFrame, seed: int) -> dict[str, str]:
    parent = long_df[["parent_group_id", "murcko_scaffold"]].drop_duplicates()
    if parent.groupby("parent_group_id")["murcko_scaffold"].nunique().max() != 1:
        raise RuntimeError("A parent group has multiple Murcko scaffolds")
    scaffold_groups = [
        sorted(group["parent_group_id"].astype(str).tolist())
        for _, group in parent.groupby("murcko_scaffold", dropna=False, sort=True)
    ]
    np.random.default_rng(seed).shuffle(scaffold_groups)
    scaffold_groups.sort(key=len, reverse=True)
    total = parent["parent_group_id"].nunique()
    train_target, val_target = round(total * 0.70), round(total * 0.15)
    train_count = val_count = 0
    assignment: dict[str, str] = {}
    for group in scaffold_groups:
        if train_count < train_target:
            label = "train"
            train_count += len(group)
        elif val_count < val_target:
            label = "val"
            val_count += len(group)
        else:
            label = "test"
        assignment.update({parent_id: label for parent_id in group})
    return assignment


def write_splits(long_df: pd.DataFrame) -> dict[str, str]:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    LOTO_DIR.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    base_columns = ["row_index", "parent_group_id", "identity_unit_id", "SMILES", "Tissue"]
    for seed in range(10):
        assignments = {
            "random": split_parent_groups(long_df["parent_group_id"].tolist(), seed, (0.70, 0.15, 0.15)),
            "scaffold": scaffold_parent_split(long_df, seed),
        }
        for split_type, lookup in assignments.items():
            table = long_df[base_columns].copy()
            table["split"] = table["parent_group_id"].map(lookup)
            path = SPLIT_DIR / f"{split_type}_seed{seed}.csv"
            table.to_csv(path, index=False, encoding="utf-8-sig")
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)

    for heldout in sorted(long_df["Tissue"].unique()):
        nonheld = long_df[long_df["Tissue"] != heldout]
        lookup = split_parent_groups(nonheld["parent_group_id"].tolist(), 0, (0.85, 0.15, 0.0))
        table = long_df[base_columns].copy()
        table["split"] = [
            "test" if tissue == heldout else lookup[parent]
            for parent, tissue in zip(table["parent_group_id"], table["Tissue"])
        ]
        path = LOTO_DIR / f"heldout_{heldout}.csv"
        table.to_csv(path, index=False, encoding="utf-8-sig")
        hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    return hashes


def write_derived_splits(dataset: pd.DataFrame, analysis_set: str) -> dict[str, str]:
    """Project frozen primary parent-group assignments onto a sensitivity dataset."""
    root = SENSITIVITY_SPLIT_ROOT / analysis_set
    loto_root = root / "loto"
    root.mkdir(parents=True, exist_ok=True)
    loto_root.mkdir(parents=True, exist_ok=True)
    base_columns = ["row_index", "parent_group_id", "identity_unit_id", "SMILES", "Tissue"]
    hashes: dict[str, str] = {}
    for split_type in ("random", "scaffold"):
        for seed in range(10):
            primary_split = read_csv(SPLIT_DIR / f"{split_type}_seed{seed}.csv")
            group_assignment = primary_split.groupby("parent_group_id")["split"].first().to_dict()
            table = dataset[base_columns].copy()
            table["split"] = table["parent_group_id"].map(group_assignment)
            if table["split"].isna().any():
                raise RuntimeError(f"Sensitivity group missing from primary {split_type} split")
            path = root / f"{split_type}_seed{seed}.csv"
            table.to_csv(path, index=False, encoding="utf-8-sig")
            hashes[str(path.relative_to(ROOT))] = sha256_file(path)

    for heldout in sorted(dataset["Tissue"].unique()):
        primary_split = read_csv(LOTO_DIR / f"heldout_{heldout}.csv")
        non_test = primary_split[primary_split["split"] != "test"]
        group_assignment = non_test.groupby("parent_group_id")["split"].first().to_dict()
        table = dataset[base_columns].copy()
        table["split"] = [
            "test" if tissue == heldout else group_assignment[parent]
            for parent, tissue in zip(table["parent_group_id"], table["Tissue"])
        ]
        path = loto_root / f"heldout_{heldout}.csv"
        table.to_csv(path, index=False, encoding="utf-8-sig")
        hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    return hashes


def audit_splits(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(SPLIT_DIR.glob("*.csv")):
        split = read_csv(path)
        groups = split.groupby("parent_group_id")["split"].nunique()
        rows.append({
            "split_file": str(path.relative_to(ROOT)),
            "split_type": path.stem.split("_seed")[0],
            "heldout_tissue": "",
            "rows": len(split),
            "parent_groups": split["parent_group_id"].nunique(),
            "parent_group_leakage_count": int((groups > 1).sum()),
            "train_rows": int((split["split"] == "train").sum()),
            "val_rows": int((split["split"] == "val").sum()),
            "test_rows": int((split["split"] == "test").sum()),
        })
    for path in sorted(LOTO_DIR.glob("*.csv")):
        split = read_csv(path)
        heldout = path.stem.removeprefix("heldout_")
        non_test = split[split["split"] != "test"]
        leakage = non_test.groupby("parent_group_id")["split"].nunique()
        rows.append({
            "split_file": str(path.relative_to(ROOT)),
            "split_type": "loto",
            "heldout_tissue": heldout,
            "rows": len(split),
            "parent_groups": split["parent_group_id"].nunique(),
            "parent_group_leakage_count": int((leakage > 1).sum()),
            "train_rows": int((split["split"] == "train").sum()),
            "val_rows": int((split["split"] == "val").sum()),
            "test_rows": int((split["split"] == "test").sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    for directory in (PROCESSED_DIR, AUDIT_DIR, SPLIT_DIR, LOTO_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    raw = read_csv(RAW_PATH)
    manifest = add_identity_ids(build_resolution_manifest(raw))
    condition_long = condition_specific_long(manifest)
    primary = aggregate_primary(condition_long)
    ad_sensitivity = primary[~primary["ad_high_mw"].astype(bool)].copy().reset_index(drop=True)

    manifest.to_csv(AUDIT_DIR / "identity_resolution_manifest.csv", index=False, encoding="utf-8-sig")
    condition_long.to_csv(PROCESSED_DIR / "rat_kp_long_condition_specific.csv", index=False, encoding="utf-8-sig")
    primary.to_csv(PROCESSED_DIR / "rat_kp_long.csv", index=False, encoding="utf-8-sig")
    ad_sensitivity.to_csv(PROCESSED_DIR / "rat_kp_long_ad_sensitivity.csv", index=False, encoding="utf-8-sig")
    feature_cache = build_tabular_feature_cache(force=True)
    split_hashes = write_splits(primary)
    sensitivity_split_hashes = {}
    sensitivity_split_hashes.update(write_derived_splits(ad_sensitivity, "ad_excluded"))
    sensitivity_split_hashes.update(write_derived_splits(condition_long, "condition_specific"))
    split_audit = audit_splits(primary)
    split_audit.to_csv(AUDIT_DIR / "split_integrity_audit.csv", index=False, encoding="utf-8-sig")

    tissue_counts = (
        primary.groupby("Tissue")
        .agg(rows=("row_index", "size"), parent_groups=("parent_group_id", "nunique"), identity_units=("identity_unit_id", "nunique"))
        .reset_index()
    )
    tissue_counts.to_csv(AUDIT_DIR / "tissue_counts_v2.csv", index=False, encoding="utf-8-sig")

    if manifest.loc[manifest["included_in_primary"].astype(bool), "resolved_smiles"].eq("").any():
        raise RuntimeError("Included row lacks a resolved SMILES")
    if split_audit["parent_group_leakage_count"].ne(0).any():
        raise RuntimeError("Parent-group leakage remains in candidate splits")
    if tissue_counts["rows"].min() < 10:
        raise RuntimeError("A tissue has fewer than 10 primary rows after resolution")

    report = {
        "status": "candidate_frozen_before_v2_model_execution",
        "created_date": "2026-06-22",
        "raw_sha256": sha256_file(RAW_PATH),
        "raw_rows_all_species": int(len(raw)),
        "raw_rows_rat": int(raw["Species"].astype(str).str.lower().eq("rat").sum()),
        "exact_duplicate_groups_all_species": int(
            (raw.groupby(list(raw.columns), dropna=False).size() > 1).sum()
        ),
        "exact_duplicate_excess_rat": int(
            manifest[
                manifest["species_normalized"].eq("rat")
                & manifest["duplicate_of_source_row_id"].astype(str).ne("")
            ].shape[0]
        ),
        "resolution_status_counts_rat": (
            manifest[manifest["species_normalized"].eq("rat")]["resolution_status"].value_counts().sort_index().to_dict()
        ),
        "condition_specific_positive_kp_rows": int(len(condition_long)),
        "primary_rows": int(len(primary)),
        "primary_identity_units": int(primary["identity_unit_id"].nunique()),
        "primary_parent_groups": int(primary["parent_group_id"].nunique()),
        "primary_model_smiles": int(primary["SMILES"].nunique()),
        "primary_tissues": int(primary["Tissue"].nunique()),
        "ad_mw_threshold": AD_MW_THRESHOLD,
        "ad_flagged_identity_units": int(primary.loc[primary["ad_high_mw"], "identity_unit_id"].nunique()),
        "ad_flagged_drugs": sorted(primary.loc[primary["ad_high_mw"], "Drug"].unique().tolist()),
        "ad_sensitivity_rows": int(len(ad_sensitivity)),
        "tabular_feature_cache": feature_cache,
        "minimum_tissue_rows": int(tissue_counts["rows"].min()),
        "minimum_loto_test_rows": int(split_audit.loc[split_audit["split_type"].eq("loto"), "test_rows"].min()),
        "split_files": len(split_hashes),
        "split_hashes": split_hashes,
        "sensitivity_split_files": len(sensitivity_split_hashes),
        "sensitivity_split_hashes": sensitivity_split_hashes,
        "output_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (
                AUDIT_DIR / "identity_resolution_manifest.csv",
                AUDIT_DIR / "split_integrity_audit.csv",
                AUDIT_DIR / "tissue_counts_v2.csv",
                PROCESSED_DIR / "rat_kp_long_condition_specific.csv",
                PROCESSED_DIR / "rat_kp_long.csv",
                PROCESSED_DIR / "rat_kp_long_ad_sensitivity.csv",
            )
        },
    }
    (PROCESSED_DIR / "data_manifest_v2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
