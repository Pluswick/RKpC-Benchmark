"""Guarded full-study launcher. Dry-run is the default and safe pre-experiment action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rat_kp_core.data import sha256_file
from rat_kp_fusion.paths import (
    CONFIG_PATH,
    EXPERIMENT_RESULT_DIR,
    NEW_JOB_MANIFEST_PATH,
    PREFLIGHT_REPORT_PATH,
)
from rat_kp_fusion.training import execute_new_job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--confirm-full-run", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args()
    jobs = pd.read_csv(NEW_JOB_MANIFEST_PATH, encoding="utf-8-sig", keep_default_na=False)
    jobs["already_complete"] = jobs.job_id.map(
        lambda value: (
            EXPERIMENT_RESULT_DIR
            / jobs.loc[jobs.job_id == value, "analysis_set"].iloc[0]
            / jobs.loc[jobs.job_id == value, "stage"].iloc[0]
            / value
            / "COMPLETE"
        ).exists()
    )
    if args.max_jobs is not None:
        jobs = jobs.head(args.max_jobs)
    pending = jobs[~jobs.already_complete]
    summary = {
        "selected_new_jobs": len(jobs),
        "already_complete": int(jobs.already_complete.sum()),
        "pending": len(pending),
        "full_run_confirmed": args.confirm_full_run,
        "by_model": jobs.groupby("model").size().to_dict(),
        "by_stage": jobs.groupby("stage").size().to_dict(),
    }
    print(json.dumps(summary, indent=2))
    if not args.confirm_full_run:
        print("Dry-run only. Pass --confirm-full-run explicitly to start full experiments.")
        return
    if not PREFLIGHT_REPORT_PATH.exists():
        raise RuntimeError("Preflight report is missing")
    preflight = json.loads(PREFLIGHT_REPORT_PATH.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed" or preflight.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise RuntimeError("Preflight/config contract failed")
    for index, row in enumerate(pending.itertuples(index=False), start=1):
        print(f"[{index}/{len(pending)}] {row.job_id}", flush=True)
        try:
            execute_new_job(pd.Series(row._asdict()), device=args.device, smoke=False)
        except Exception as exc:
            output = EXPERIMENT_RESULT_DIR / row.analysis_set / row.stage / row.job_id
            output.mkdir(parents=True, exist_ok=True)
            (output / "FAILED.json").write_text(json.dumps({
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }, indent=2), encoding="utf-8")
            raise


if __name__ == "__main__":
    main()
