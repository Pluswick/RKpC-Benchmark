"""Report completion and failure counts without reading performance values."""

import json

import pandas as pd

from rat_kp_core.paths import (
    AD_SENSITIVITY_JOB_MANIFEST_PATH,
    CONDITION_SENSITIVITY_JOB_MANIFEST_PATH,
    EXPERIMENT_RESULT_DIR,
    PRIMARY_JOB_MANIFEST_PATH,
)


def summarize(path, label):
    jobs = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    complete = failed = 0
    for row in jobs.itertuples(index=False):
        root = EXPERIMENT_RESULT_DIR / row.analysis_set / row.stage / row.job_id
        valid_complete = all((root / name).exists() for name in ("COMPLETE", "run.json", "predictions.csv"))
        complete += int(valid_complete)
        failed += int((root / "FAILED.json").exists() and not valid_complete)
    return {"manifest": label, "total": len(jobs), "complete": complete, "failed": failed, "pending": len(jobs)-complete-failed}


if __name__ == "__main__":
    print(json.dumps([
        summarize(PRIMARY_JOB_MANIFEST_PATH, "primary"),
        summarize(AD_SENSITIVITY_JOB_MANIFEST_PATH, "ad_excluded"),
        summarize(CONDITION_SENSITIVITY_JOB_MANIFEST_PATH, "condition_specific"),
    ], indent=2))
