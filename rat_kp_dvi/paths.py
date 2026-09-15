"""Repository-relative locations for the Kp-to-distribution-volume analysis."""

from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[1]
ROOT = STUDY_ROOT / "vss_output_propagation"
CONFIG_PATH = STUDY_ROOT / "configs" / "VSS_OUTPUT_PROPAGATION_CONFIG_V1.json"
MANIFEST_DIR = ROOT / "manifests"
RESULT_DIR = ROOT / "results"
EXPERIMENT_DIR = RESULT_DIR / "experiments"
ANALYSIS_DIR = RESULT_DIR / "analysis"
# Written by prepare_matched_panels.py from authorised local PT/RR reconstructions.
# The record-level file stays local and is never distributed.
MATCHED_PANEL_DIR = STUDY_ROOT / "results" / "local_matched_panels"
MATCHED_PANEL_RECORDS_PATH = MATCHED_PANEL_DIR / "direct_panel_record_predictions.csv"
