"""Independent compound-level split construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def random_split(smiles: list[str], seed: int, fractions=(0.70, 0.15, 0.15)) -> pd.DataFrame:
    unique = np.array(sorted(set(smiles)), dtype=object)
    np.random.default_rng(seed).shuffle(unique)
    n_train = round(len(unique) * fractions[0])
    n_val = round(len(unique) * fractions[1])
    labels = [
        "train" if i < n_train else "val" if i < n_train + n_val else "test"
        for i in range(len(unique))
    ]
    return pd.DataFrame({"SMILES": unique, "split": labels})


def scaffold_split(smiles: list[str], seed: int, fractions=(0.70, 0.15, 0.15)) -> pd.DataFrame:
    groups: dict[str, list[str]] = {}
    for smi in sorted(set(smiles)):
        mol = Chem.MolFromSmiles(smi)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        groups.setdefault(scaffold, []).append(smi)
    values = list(groups.values())
    np.random.default_rng(seed).shuffle(values)
    values.sort(key=len, reverse=True)
    train_target = round(len(set(smiles)) * fractions[0])
    val_target = round(len(set(smiles)) * fractions[1])
    train_count = val_count = 0
    rows = []
    for group in values:
        if train_count < train_target:
            label = "train"
            train_count += len(group)
        elif val_count < val_target:
            label = "val"
            val_count += len(group)
        else:
            label = "test"
        rows.extend({"SMILES": smi, "split": label} for smi in sorted(group))
    return pd.DataFrame(rows)


def attach_split(long_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    result = long_df.merge(split_df, on="SMILES", how="left", validate="many_to_one")
    if result["split"].isna().any() or result.groupby("SMILES")["split"].nunique().max() != 1:
        raise ValueError("Incomplete or leaking compound split")
    return result

