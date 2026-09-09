"""Independent Morgan/RDKit2D and LinearSVR preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
import json

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler

from .physiology import ContextEncoder
from .data import read_csv, sha256_file
from .paths import (
    LONG_DATA_PATH,
    TABULAR_FEATURE_CACHE_PATH,
    TABULAR_FEATURE_INDEX_PATH,
)


def _molecules(smiles: pd.Series) -> list[Chem.Mol]:
    molecules = [Chem.MolFromSmiles(str(value)) for value in smiles]
    if any(molecule is None for molecule in molecules):
        raise ValueError("Invalid SMILES reached independent featurizer")
    return molecules


def _morgan(molecules: list[Chem.Mol]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    output = np.zeros((len(molecules), 2048), dtype=np.float32)
    for row, molecule in enumerate(molecules):
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), output[row])
    return output


def _descriptors(molecules: list[Chem.Mol]) -> np.ndarray:
    functions = [function for _, function in Descriptors.descList]
    output = np.empty((len(molecules), len(functions)), dtype=np.float64)
    for row, molecule in enumerate(molecules):
        values = []
        for function in functions:
            try:
                value = float(function(molecule))
            except Exception:
                value = np.nan
            values.append(value if np.isfinite(value) else np.nan)
        output[row] = values
    return output


def build_tabular_feature_cache(force: bool = False) -> dict:
    """Precompute molecular-only features once; context and scaling remain job-specific."""
    if TABULAR_FEATURE_CACHE_PATH.exists() and TABULAR_FEATURE_INDEX_PATH.exists() and not force:
        metadata = json.loads(TABULAR_FEATURE_INDEX_PATH.read_text(encoding="utf-8"))
        if metadata.get("long_data_sha256") == sha256_file(LONG_DATA_PATH):
            return metadata
    long_df = read_csv(LONG_DATA_PATH)
    smiles = sorted(long_df["SMILES"].astype(str).unique())
    molecules = _molecules(pd.Series(smiles))
    fingerprints = _morgan(molecules)
    descriptors = _descriptors(molecules)
    TABULAR_FEATURE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        TABULAR_FEATURE_CACHE_PATH,
        fingerprints=fingerprints,
        descriptors=descriptors,
    )
    metadata = {
        "scope": "molecular features only; no target, tissue context, split, or scaling",
        "long_data_sha256": sha256_file(LONG_DATA_PATH),
        "smiles": smiles,
        "morgan_shape": list(fingerprints.shape),
        "descriptor_shape": list(descriptors.shape),
    }
    TABULAR_FEATURE_INDEX_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _cached_features(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    metadata = build_tabular_feature_cache()
    if metadata["long_data_sha256"] != sha256_file(LONG_DATA_PATH):
        raise ValueError("Tabular feature cache data checksum mismatch")
    lookup = {smiles: index for index, smiles in enumerate(metadata["smiles"])}
    try:
        indices = np.asarray([lookup[str(smiles)] for smiles in frame["SMILES"]], dtype=int)
    except KeyError as exc:
        raise ValueError(f"SMILES absent from frozen feature cache: {exc}") from exc
    with np.load(TABULAR_FEATURE_CACHE_PATH) as cache:
        return cache["fingerprints"][indices], cache["descriptors"][indices]


@dataclass
class TabularPreprocessor:
    model_key: str
    context: ContextEncoder
    descriptor_medians_: np.ndarray | None = field(default=None, init=False)
    descriptor_mask_: np.ndarray | None = field(default=None, init=False)
    descriptor_min_: np.ndarray | None = field(default=None, init=False)
    descriptor_max_: np.ndarray | None = field(default=None, init=False)
    morgan_mask_: np.ndarray | None = field(default=None, init=False)
    descriptor_scaler_: StandardScaler | None = field(default=None, init=False)
    fitted_: bool = field(default=False, init=False)

    def fit(self, train: pd.DataFrame) -> "TabularPreprocessor":
        fingerprints, descriptors = _cached_features(train)
        with np.errstate(all="ignore"):
            self.descriptor_medians_ = np.nanmedian(descriptors, axis=0)
        self.descriptor_medians_[~np.isfinite(self.descriptor_medians_)] = 0.0
        descriptors = np.where(np.isfinite(descriptors), descriptors, self.descriptor_medians_)
        with np.errstate(over="ignore", invalid="ignore"):
            float32_descriptors = descriptors.astype(np.float32)
        self.descriptor_mask_ = np.isfinite(float32_descriptors).all(axis=0)
        if not self.descriptor_mask_.any():
            raise ValueError("No float32-safe RDKit2D descriptors in training data")
        descriptors = descriptors[:, self.descriptor_mask_]
        self.descriptor_min_ = descriptors.min(axis=0)
        self.descriptor_max_ = descriptors.max(axis=0)
        if self.model_key == "linear_svr":
            self.morgan_mask_ = np.ptp(fingerprints, axis=0) > 0
            self.descriptor_scaler_ = StandardScaler().fit(descriptors)
        elif self.model_key not in {"random_forest", "xgboost"}:
            raise ValueError(f"Unknown tabular model: {self.model_key}")
        self.context.fit(train)
        self.fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted_ or self.descriptor_medians_ is None or self.descriptor_mask_ is None:
            raise RuntimeError("Tabular preprocessor is not fit")
        fingerprints, descriptors = _cached_features(frame)
        descriptors = np.where(np.isfinite(descriptors), descriptors, self.descriptor_medians_)
        descriptors = descriptors[:, self.descriptor_mask_]
        descriptors = np.clip(descriptors, self.descriptor_min_, self.descriptor_max_)
        if self.model_key == "linear_svr":
            fingerprints = fingerprints[:, self.morgan_mask_]
            descriptors = self.descriptor_scaler_.transform(descriptors)
        result = np.concatenate([fingerprints, descriptors, self.context.transform(frame)], axis=1)
        if not np.isfinite(result).all():
            raise RuntimeError("Non-finite independent tabular features")
        return result

    def fit_transform(self, train: pd.DataFrame) -> np.ndarray:
        return self.fit(train).transform(train)
