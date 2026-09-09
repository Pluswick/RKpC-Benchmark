"""Independent raw-table parsing for the rat Kp tissue-context study."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


ID_COLUMNS = ["Drug", "SMILES", "Species"]
TISSUES = [
    "Adipose", "Bone", "Brain", "Gut", "Heart", "Kidneys",
    "Liver", "Lung", "Muscle", "Skin", "Spleen",
]
RANGE_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*-\s*"
    r"(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)


def sha256_file(path: str | Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path: str | Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode CSV: {path}")


def _parse_kp(value: object) -> float:
    if pd.isna(value) or not str(value).strip():
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        match = RANGE_RE.match(str(value))
        if not match:
            return float("nan")
        low, high = float(match.group(1)), float(match.group(2))
        if low <= 0 or high <= 0:
            return float("nan")
        return float(np.sqrt(low * high))


def build_rat_long(raw_path: str | Path) -> pd.DataFrame:
    """Create one positive Kp row per valid molecule/tissue using median log replicates."""
    wide = read_csv(raw_path)
    missing = set(ID_COLUMNS + TISSUES) - set(wide.columns)
    if missing:
        raise ValueError(f"Raw data missing columns: {sorted(missing)}")
    wide = wide[wide["Species"].astype(str).str.lower() == "rat"].copy()
    long_df = wide.melt(
        id_vars=ID_COLUMNS,
        value_vars=TISSUES,
        var_name="Tissue",
        value_name="Kp_raw",
    )
    long_df["Kp"] = long_df["Kp_raw"].map(_parse_kp)
    long_df = long_df[np.isfinite(long_df["Kp"]) & (long_df["Kp"] > 0)].copy()
    long_df["Species"] = "rat"
    long_df["Tissue"] = long_df["Tissue"].str.lower()

    RDLogger.DisableLog("rdApp.*")
    valid_smiles = {
        smiles
        for smiles in long_df["SMILES"].dropna().astype(str).unique()
        if Chem.MolFromSmiles(smiles) is not None
    }
    long_df = long_df[long_df["SMILES"].astype(str).isin(valid_smiles)].copy()
    long_df["_log10Kp"] = np.log10(long_df["Kp"].astype(float))

    def join_unique(values: pd.Series) -> str:
        return " | ".join(sorted({str(v).strip() for v in values if pd.notna(v)}))

    result = (
        long_df.groupby(["SMILES", "Species", "Tissue"], as_index=False)
        .agg(
            Drug=("Drug", join_unique),
            log10Kp=("_log10Kp", "median"),
            n_measurements=("Kp", "size"),
            Kp_min=("Kp", "min"),
            Kp_max=("Kp", "max"),
            Kp_raw_values=("Kp_raw", join_unique),
        )
        .reset_index(drop=True)
    )
    result["Kp"] = np.power(10.0, result["log10Kp"])
    return result[
        ["Drug", "SMILES", "Species", "Tissue", "Kp", "log10Kp",
         "n_measurements", "Kp_min", "Kp_max", "Kp_raw_values"]
    ]

