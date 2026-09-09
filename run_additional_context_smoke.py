"""Run one-batch checks for every new implementation path."""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--cuda-ordinal", type=int, default=0)
    args = parser.parse_args()
    if args.device == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_ordinal)

    from rat_kp_controls.jobs import build_plan
    from rat_kp_controls.training import execute_job
    if args.device == "gpu":
        import torch
        from rat_kp_controls.config import load_config
        selected_name = torch.cuda.get_device_name(0)
        expected_name = load_config()["shared_training"]["expected_gpu"]
        if selected_name != expected_name:
            raise RuntimeError(
                f"CUDA ordinal {args.cuda_ordinal} resolved to {selected_name}, expected {expected_name}"
            )
    else:
        selected_name = None

    plan = build_plan()
    selected = []
    for model in ("gcn", "gine", "d_mpnn", "attentive_fp"):
        row = plan[
            (plan["analysis"] == "primary_context_decomposition")
            & (plan["model"] == model)
            & (plan["split_type"] == "parent_group")
            & (plan["split_seed"] == 0)
        ]
        if len(row) != 1:
            raise RuntimeError(f"Missing additive smoke case for {model}")
        selected.append(row.iloc[0])
    for position in ("early", "late"):
        row = plan[
            (plan["analysis"] == "loto_raw_fraction_robustness")
            & (plan["model"] == "gine")
            & (plan["heldout_tissue"] == "adipose")
            & (plan["training_seed"] == 0)
            & (plan["injection_position"] == position)
        ]
        if len(row) != 1:
            raise RuntimeError(f"Missing raw-fraction smoke case for {position}")
        selected.append(row.iloc[0])
    outputs = [str(execute_job(row, device=args.device, smoke=True)) for row in selected]
    print(json.dumps({
        "status": "passed",
        "resolved_gpu_name": selected_name,
        "smoke_jobs": len(outputs),
        "outputs": outputs,
    }, indent=2))


if __name__ == "__main__":
    main()
