"""Guarded launcher for the 520 isolated additional-analysis jobs."""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--cuda-ordinal", type=int, default=0)
    parser.add_argument("--analysis", choices=[
        "all", "primary_context_decomposition", "loto_raw_fraction_robustness"
    ], default="all")
    parser.add_argument("--confirm-full-run", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args()
    if args.device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_ordinal)

    from rat_kp_core.data import sha256_file
    from rat_kp_controls.config import load_config
    from rat_kp_controls.paths import CONFIG_PATH, EXPERIMENT_RESULT_DIR, PLAN_PATH, PREFLIGHT_PATH
    from rat_kp_controls.training import execute_job
    if args.device == "gpu":
        import torch
        selected_name = torch.cuda.get_device_name(0)
        expected_name = load_config()["shared_training"]["expected_gpu"]
        if selected_name != expected_name:
            raise RuntimeError(
                f"CUDA ordinal {args.cuda_ordinal} resolved to {selected_name}, expected {expected_name}"
            )
    else:
        selected_name = None

    load_config(require_locked=args.confirm_full_run)
    jobs = pd.read_csv(PLAN_PATH, encoding="utf-8-sig", keep_default_na=False)
    if args.analysis != "all":
        jobs = jobs[jobs["analysis"] == args.analysis].copy()
    jobs["already_complete"] = jobs.apply(
        lambda row: (
            EXPERIMENT_RESULT_DIR / row.analysis / row.job_id / "COMPLETE"
        ).exists(),
        axis=1,
    )
    if args.max_jobs is not None:
        jobs = jobs.head(args.max_jobs)
    pending = jobs[~jobs["already_complete"]]
    summary = {
        "selected_jobs": len(jobs),
        "already_complete": int(jobs["already_complete"].sum()),
        "pending": len(pending),
        "full_run_confirmed": args.confirm_full_run,
        "cuda_ordinal": args.cuda_ordinal if args.device == "gpu" else None,
        "resolved_gpu_name": selected_name,
        "by_analysis": jobs.groupby("analysis").size().to_dict(),
        "by_model": jobs.groupby("model").size().to_dict(),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if not args.confirm_full_run:
        print("Dry-run only. Pass --confirm-full-run to start training.")
        return
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "passed"
        or preflight.get("additional_config_sha256") != sha256_file(CONFIG_PATH)
    ):
        raise RuntimeError("Additional-analysis preflight/config contract failed")
    for index, row in enumerate(pending.itertuples(index=False), start=1):
        print(f"[{index}/{len(pending)}] {row.job_id}", flush=True)
        try:
            execute_job(pd.Series(row._asdict()), device=args.device, smoke=False)
        except Exception as exc:
            output = EXPERIMENT_RESULT_DIR / row.analysis / row.job_id
            output.mkdir(parents=True, exist_ok=True)
            (output / "FAILED.json").write_text(json.dumps({
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }, indent=2), encoding="utf-8")
            raise


if __name__ == "__main__":
    main()
