"""Canonical repository-relative locations for every study artifact.

All paths are resolved from the repository root, so the pipelines behave
identically regardless of the working directory a script is launched from.
Request-only inputs live under ``data/inputs`` and are never distributed.
"""

from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = STUDY_ROOT / "data" / "inputs"
RAW_DATA_PATH = INPUT_DIR / "source_data_not_distributed.csv"
LONG_DATA_PATH = INPUT_DIR / "rat_kp_long.csv"
AD_SENSITIVITY_DATA_PATH = INPUT_DIR / "rat_kp_long_ad_sensitivity.csv"
CONDITION_SENSITIVITY_DATA_PATH = INPUT_DIR / "rat_kp_long_condition_specific.csv"
SPLIT_DIR = INPUT_DIR / "splits"
LOTO_SPLIT_DIR = SPLIT_DIR / "loto"
# Sensitivity splits project the frozen primary parent-group assignments onto the
# robustness datasets; ``prepare_data.write_derived_splits`` is their only writer.
SENSITIVITY_SPLIT_ROOT = INPUT_DIR / "splits_sensitivity"
AD_SENSITIVITY_SPLIT_DIR = SENSITIVITY_SPLIT_ROOT / "ad_excluded"
CONDITION_SENSITIVITY_SPLIT_DIR = SENSITIVITY_SPLIT_ROOT / "condition_specific"

# Filename stem for each split scheme, as ``<stem>_seed<N>.csv`` inside a split
# directory. The stem is NOT a description of the partitioning rule: the scheme
# the manuscript calls the parent-group split is stored under the stem "random"
# because both schemes assign whole parent groups (`prepare_data.split_parent_groups`
# and `prepare_data.scaffold_parent_split`), and no split ever divides a parent
# group across partitions. The historical stem is retained because those exact
# filenames are published in the RKpC Benchmark record and are covered by the
# frozen split hashes in the pre-experiment validation manifest, so renaming them
# would break the link to the deposited data. Always resolve stems through these
# mappings rather than writing the literal "random".
SPLIT_FILE_STEM = {"parent_group": "random", "scaffold": "scaffold"}
SPLIT_FILE_STEMS = tuple(SPLIT_FILE_STEM.values())
MANUSCRIPT_SPLIT_SCHEME = {stem: scheme for scheme, stem in SPLIT_FILE_STEM.items()}
MANIFEST_DIR = STUDY_ROOT / "manifests"
PRIMARY_JOB_MANIFEST_PATH = MANIFEST_DIR / "primary_jobs.csv"
AD_SENSITIVITY_JOB_MANIFEST_PATH = MANIFEST_DIR / "ad_excluded_jobs.csv"
CONDITION_SENSITIVITY_JOB_MANIFEST_PATH = MANIFEST_DIR / "condition_specific_jobs.csv"
EXPERIMENT_RESULT_DIR = STUDY_ROOT / "results" / "experiments"
ANALYSIS_RESULT_DIR = STUDY_ROOT / "results" / "analysis"
FEATURE_CACHE_DIR = STUDY_ROOT / "data" / "cache"
TABULAR_FEATURE_CACHE_PATH = FEATURE_CACHE_DIR / "tabular_molecular_features.npz"
TABULAR_FEATURE_INDEX_PATH = FEATURE_CACHE_DIR / "tabular_molecular_features.json"
PHYSIOLOGY_PATH = INPUT_DIR / "rat_tissue_physiology.csv"
CONFIG_PATH = STUDY_ROOT / "configs" / "FIXED_MODEL_CONFIG_V2.json"
SMOKE_RESULT_DIR = STUDY_ROOT / "results" / "smoke_test"
