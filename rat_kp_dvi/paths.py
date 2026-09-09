from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[1]
ROOT = STUDY_ROOT / "vss_output_propagation"
CONFIG_PATH = STUDY_ROOT / "configs" / "VSS_OUTPUT_PROPAGATION_CONFIG_V1.json"
MANIFEST_DIR = ROOT / "manifests"
RESULT_DIR = ROOT / "results"
EXPERIMENT_DIR = RESULT_DIR / "experiments"
ANALYSIS_DIR = RESULT_DIR / "analysis"

