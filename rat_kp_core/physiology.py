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
    """Load the tissue physiology table, refusing any unrecorded edit.

    These five descriptors are the entire physiology context, so a changed
    value would silently redefine one of the study's conditions. The file is
    therefore pinned by checksum and the contents are re-checked for 11
    unique tissues and finite non-negative values.
    """
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
    """Encode tissue context as nothing, a one-hot vector, or physiology descriptors.

    ``structure_only`` produces a zero-width matrix, so the same code path
    serves the no-context condition without a special case.

    The encoder is fit on training rows only, and the distinction matters
    most under leave-one-tissue-out evaluation:

    * ``tissue_onehot`` keeps a fixed 11-column layout but emits an all-zero
      row for a tissue absent from training. An unseen tissue therefore
      carries no identity the model was ever trained to use, which is what
      makes one-hot context unable to extrapolate to a new tissue.
    * ``physiology`` standardizes on the training tissues, so an unseen
      tissue is expressed in training units and may land far outside the
      fitted range. That is the extrapolation the LOTO analysis measures.

    ``tissue_equal`` scaling gives each training tissue equal weight when
    computing the standardization, so unevenly sampled tissues do not pull
    the mean and scale toward whichever tissue has the most records;
    ``observation_weighted`` weights by record instead.
    """

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
        """Record the training tissues and fit physiology standardization to them."""
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
        """Encode context for arbitrary rows using the fitted training statistics.

        A tissue outside the known 11 raises; a known tissue that was absent
        from training encodes as all zeros under ``tissue_onehot``.
        """
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
        """Fit on the training rows and encode them in one call."""
        return self.fit(train).transform(train)

    @property
    def feature_names(self) -> list[str]:
        """Column names of the encoded context, empty for ``structure_only``."""
        if self.mode == "tissue_onehot":
            return [f"tissue={t}" for t in self.tissue_order]
        if self.mode == "physiology":
            return [f"physiology={name}" for name in PHYSIOLOGY_COLUMNS]
        return []

