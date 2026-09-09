"""Resume-safe sequential orchestrator for primary and sensitivity jobs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import pandas as pd

from rat_kp_core.jobs import MODELS
from rat_kp_core.paths import (
    AD_SENSITIVITY_JOB_MANIFEST_PATH,
    CONDITION_SENSITIVITY_JOB_MANIFEST_PATH,
    EXPERIMENT_RESULT_DIR,
    PRIMARY_JOB_MANIFEST_PATH,
    STUDY_ROOT,
)
from rat_kp_core.physiology import CONTEXT_MODES


def _load(which: str) -> pd.DataFrame:
    frames = []
    paths = {
        "primary": PRIMARY_JOB_MANIFEST_PATH,
        "ad_excluded": AD_SENSITIVITY_JOB_MANIFEST_PATH,
        "condition_specific": CONDITION_SENSITIVITY_JOB_MANIFEST_PATH,
    }
    selected = paths if which == "all" else {which: paths[which]}
    for label, path in selected.items():
        frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
        frame["manifest"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _complete(row: pd.Series) -> bool:
    path = (
        EXPERIMENT_RESULT_DIR
        / str(row["analysis_set"])
        / str(row["stage"])
        / str(row["job_id"])
        / "COMPLETE"
    )
    return (
        path.exists()
        and (path.parent / "run.json").exists()
        and (path.parent / "predictions.csv").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-set",
        choices=["primary", "ad_excluded", "condition_specific", "all"],
        default="primary",
    )
    parser.add_argument("--stage", choices=["part1", "loto", "all"], default="all")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--contexts", nargs="+", choices=list(CONTEXT_MODES), default=list(CONTEXT_MODES))
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    jobs = _load(args.analysis_set)
    if args.stage != "all":
        jobs = jobs[jobs["stage"] == args.stage]
    jobs = jobs[jobs["model"].isin(args.models) & jobs["context"].isin(args.contexts)]
    jobs = jobs.copy()
    jobs["already_complete"] = jobs.apply(_complete, axis=1)
    pending = jobs[~jobs["already_complete"]]
    if args.max_jobs is not None:
        pending = pending.head(args.max_jobs)
    summary = {
        "selected_jobs": len(jobs),
        "already_complete": int(jobs["already_complete"].sum()),
        "pending_to_run": len(pending),
        "dry_run": args.dry_run,
        "by_stage": jobs.groupby("stage").size().to_dict(),
        "by_model": jobs.groupby("model").size().to_dict(),
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        if not pending.empty:
            print("First pending job IDs:")
            print("\n".join(pending["job_id"].head(10)))
        return

    failures = []
    for position, row in enumerate(pending.itertuples(index=False), start=1):
        print(f"[{position}/{len(pending)}] {row.job_id}", flush=True)
        command = [
            sys.executable,
            str(STUDY_ROOT / "run_job.py"),
            "--manifest", row.manifest,
            "--job-id", row.job_id,
            "--device", args.device,
        ]
        completed = subprocess.run(command, cwd=STUDY_ROOT, check=False)
        if completed.returncode != 0:
            failures.append(row.job_id)
            if not args.continue_on_error:
                raise SystemExit(completed.returncode)
    if failures:
        raise SystemExit(f"Failed jobs ({len(failures)}): {failures[:10]}")


if __name__ == "__main__":
    main()
