"""Run the 20 frozen full-context jobs needed for complete 11-tissue inference."""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--cuda-ordinal", type=int, default=1)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if args.device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_ordinal)
    import pandas as pd
    import torch
    from rat_kp_dvi.jobs import build_full_context_jobs
    from rat_kp_dvi.paths import EXPERIMENT_DIR, MANIFEST_DIR
    from rat_kp_dvi.training import execute_job
    jobs = build_full_context_jobs()
    if args.max_jobs is not None:
        jobs = jobs.head(args.max_jobs)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(MANIFEST_DIR / "full_context_jobs.csv", index=False, encoding="utf-8-sig")
    gpu = torch.cuda.get_device_name(0) if args.device == "gpu" else None
    if args.device == "gpu" and gpu != "NVIDIA GeForce RTX 4090":
        raise RuntimeError(f"Expected RTX 4090, resolved {gpu}")
    pending = [row for row in jobs.itertuples(index=False) if not (EXPERIMENT_DIR / row.split_type / row.job_id / "COMPLETE").exists()]
    print(json.dumps({"jobs": len(jobs), "pending": len(pending), "resolved_gpu": gpu, "confirmed": args.confirm}, indent=2), flush=True)
    if not args.confirm:
        return
    for index, row in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {row.job_id}", flush=True)
        execute_job(pd.Series(row._asdict()), device=args.device)


if __name__ == "__main__":
    main()
