"""Run four one-batch GPU checks without launching the full experiment matrix."""

from __future__ import annotations

import json

from rat_kp_fusion.jobs import build_plan
from rat_kp_fusion.paths import GPU_SMOKE_RESULT_DIR
from rat_kp_fusion.training import execute_new_job


CASES = (
    ("gcn", "physiology_early"),
    ("gine", "physiology_early"),
    ("d_mpnn", "onehot_early"),
    ("attentive_fp", "onehot_early"),
)


def main() -> None:
    plan = build_plan()
    outputs = []
    for model, condition in CASES:
        selected = plan[
            (plan.analysis_set == "primary")
            & (plan.stage == "part1")
            & (plan.split_type == "parent_group")
            & (plan.split_seed == 0)
            & (plan.model == model)
            & (plan.condition == condition)
            & (plan.execution_mode == "new")
        ]
        if len(selected) != 1:
            raise RuntimeError(f"Missing GPU preflight case: {model}/{condition}")
        outputs.append(str(execute_new_job(
            selected.iloc[0], device="gpu", smoke=True, output_root=GPU_SMOKE_RESULT_DIR
        )))
    print(json.dumps({"status": "passed", "gpu_smoke_jobs": len(outputs), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
