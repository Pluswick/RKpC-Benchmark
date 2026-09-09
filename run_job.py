"""Execute one immutable job-manifest row."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from rat_kp_core.paths import (
    AD_SENSITIVITY_JOB_MANIFEST_PATH,
    CONDITION_SENSITIVITY_JOB_MANIFEST_PATH,
    PRIMARY_JOB_MANIFEST_PATH,
)
from rat_kp_core.training import execute_job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        choices=["primary", "ad_excluded", "condition_specific"],
        required=True,
    )
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    args = parser.parse_args()
    path = {
        "primary": PRIMARY_JOB_MANIFEST_PATH,
        "ad_excluded": AD_SENSITIVITY_JOB_MANIFEST_PATH,
        "condition_specific": CONDITION_SENSITIVITY_JOB_MANIFEST_PATH,
    }[args.manifest]
    manifest = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    selected = manifest[manifest["job_id"] == args.job_id]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one job_id={args.job_id!r}, found {len(selected)}")
    output = execute_job(selected.iloc[0], device=args.device)
    print(json.dumps({"status": "complete", "job_id": args.job_id, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
