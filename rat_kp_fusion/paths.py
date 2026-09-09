from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = STUDY_ROOT / "configs" / "INJECTION_STUDY_CONFIG_V1.json"
EXTENSION_ROOT = STUDY_ROOT / "injection_study"
MANIFEST_DIR = EXTENSION_ROOT / "manifests"
RESULT_DIR = EXTENSION_ROOT / "results"
SMOKE_RESULT_DIR = RESULT_DIR / "smoke"
GPU_SMOKE_RESULT_DIR = RESULT_DIR / "smoke_gpu"
EXPERIMENT_RESULT_DIR = RESULT_DIR / "experiments"
PLAN_MANIFEST_PATH = MANIFEST_DIR / "planned_jobs.csv"
NEW_JOB_MANIFEST_PATH = MANIFEST_DIR / "new_jobs.csv"
REUSE_MANIFEST_PATH = MANIFEST_DIR / "legacy_reuse_jobs.csv"
PREFLIGHT_REPORT_PATH = MANIFEST_DIR / "pre_experiment_validation.json"
