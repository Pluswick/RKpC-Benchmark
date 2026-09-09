"""Frozen tissue-context encoding contained entirely within the study folder."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import read_csv, sha256_file
from .paths import PHYSIOLOGY_PATH


PHYSIOLOGY_SHA256 = "63fc39f972af4008631c78d3132cd291ba1d0bbd22c7378f3055ac8cb4389cad"
PHYSIOLOGY_COLUMNS = [
    "neutral_lipid_fraction",
    "neutral_phospholipid_fraction",
    "extracellular_water_fraction",
    "intracellular_water_fraction",
    "acidic_phospholipid_mg_g",
]
CONTEXT_MODES = ("structure_only", "tissue_onehot", "physiology")


def load_physiology() -> pd.DataFrame:
    if sha256_file(PHYSIOLOGY_PATH) != PHYSIOLOGY_SHA256:
        raise ValueError("Independent physiological descriptor checksum mismatch")
    matrix = read_csv(PHYSIOLOGY_PATH)
    matrix["tissue"] = matrix["tissue"].astype(str).str.lower()
    if len(matrix) != 11 or matrix["tissue"].nunique() != 11:
        raise ValueError("Expected 11 unique tissues")
    values = matrix[PHYSIOLOGY_COLUMNS].apply(pd.to_numeric).to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid physiological values")
    return matrix


@dataclass
class ContextEncoder:
    mode: str
    matrix: pd.DataFrame
    scaling: str = "tissue_equal"
    tissue_order: list[str] = field(init=False)
    fitted_tissues: list[str] = field(default_factory=list, init=False)
    mean_: np.ndarray | None = field(default=None, init=False)
    scale_: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.mode not in CONTEXT_MODES:
            raise ValueError(f"Unknown context mode: {self.mode}")
        if self.scaling not in {"tissue_equal", "observation_weighted"}:
            raise ValueError(f"Unknown physiology scaling: {self.scaling}")
        self.tissue_order = self.matrix["tissue"].tolist()

    def fit(self, train: pd.DataFrame) -> "ContextEncoder":
        observed = set(train["Tissue"].astype(str).str.lower())
        if observed - set(self.tissue_order):
            raise ValueError("Training data contain unknown tissues")
        self.fitted_tissues = [t for t in self.tissue_order if t in observed]
        if self.mode == "physiology":
            lookup = self.matrix.set_index("tissue")[PHYSIOLOGY_COLUMNS]
            if self.scaling == "tissue_equal":
                values = lookup.loc[self.fitted_tissues].to_numpy(float)
            else:
                values = lookup.loc[train["Tissue"].astype(str).str.lower()].to_numpy(float)
            self.mean_ = values.mean(axis=0)
            self.scale_ = values.std(axis=0, ddof=0)
            if (self.scale_ == 0).any():
                raise ValueError("Zero-variance physiology column")
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted_tissues:
            raise RuntimeError("Context encoder is not fit")
        tissues = frame["Tissue"].astype(str).str.lower().tolist()
        if set(tissues) - set(self.tissue_order):
            raise ValueError("Unknown tissue during transform")
        if self.mode == "structure_only":
            return np.empty((len(frame), 0), dtype=np.float64)
        if self.mode == "tissue_onehot":
            index = {t: i for i, t in enumerate(self.tissue_order)}
            fitted = set(self.fitted_tissues)
            result = np.zeros((len(frame), 11), dtype=np.float64)
            for row, tissue in enumerate(tissues):
                if tissue in fitted:
                    result[row, index[tissue]] = 1.0
            return result
        lookup = self.matrix.set_index("tissue")[PHYSIOLOGY_COLUMNS]
        values = lookup.loc[tissues].to_numpy(float)
        return (values - self.mean_) / self.scale_

    def fit_transform(self, train: pd.DataFrame) -> np.ndarray:
        return self.fit(train).transform(train)

    @property
    def feature_names(self) -> list[str]:
        if self.mode == "tissue_onehot":
            return [f"tissue={t}" for t in self.tissue_order]
        if self.mode == "physiology":
            return [f"physiology={name}" for name in PHYSIOLOGY_COLUMNS]
        return []

