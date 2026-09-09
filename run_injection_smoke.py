"""Run one-batch smoke checks for every new model/condition implementation path."""

from __future__ import annotations

import argparse
import json

from rat_kp_fusion.jobs import build_plan
from rat_kp_fusion.training import execute_new_job


SMOKE_CASES = {
    "gcn": ["structure_only", "onehot_late", "onehot_early", "physiology_late", "physiology_early"],
    "gine": ["structure_only", "onehot_late", "onehot_early", "physiology_late", "physiology_early"],
    "d_mpnn": ["onehot_early", "physiology_early"],
    "attentive_fp": ["onehot_early", "physiology_early"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    plan = build_plan()
    selected = []
    for model, conditions in SMOKE_CASES.items():
        for condition in conditions:
            row = plan[
                (plan.analysis_set == "primary")
                & (plan.stage == "part1")
                & (plan.split_type == "parent_group")
                & (plan.split_seed == 0)
                & (plan.model == model)
                & (plan.condition == condition)
                & (plan.execution_mode == "new")
            ]
            if len(row) != 1:
                raise RuntimeError(f"Missing smoke case: {model}/{condition}")
            selected.append(row.iloc[0])
    outputs = [str(execute_new_job(row, device=args.device, smoke=True)) for row in selected]
    print(json.dumps({"status": "passed", "smoke_jobs": len(outputs), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
