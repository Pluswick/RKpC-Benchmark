"""Repository-relative locations for the additive-tissue-intercept analyses.

The manuscript calls these the additive tissue-intercept control and the
preprocessing-robustness analyses; on disk they live under
``additional_context_analyses``. See ``docs/terminology.md``.
"""

from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = STUDY_ROOT / "configs" / "ADDITIONAL_CONTEXT_ANALYSES_CONFIG_V1.json"
ROOT = STUDY_ROOT / "additional_context_analyses"
MANIFEST_DIR = ROOT / "manifests"
RESULT_DIR = ROOT / "results"
EXPERIMENT_RESULT_DIR = RESULT_DIR / "experiments"
SMOKE_RESULT_DIR = RESULT_DIR / "smoke"
ANALYSIS_DIR = RESULT_DIR / "analysis"
PLAN_PATH = MANIFEST_DIR / "planned_jobs.csv"
PREFLIGHT_PATH = MANIFEST_DIR / "preflight.json"

