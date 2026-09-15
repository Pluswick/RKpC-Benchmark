"""Select the full-context jobs whose fitted states drive 11-tissue inference.

Rather than defining new jobs, this narrows the frozen context/fusion-position
plan to the representative model and condition named in the DVI configuration
for each split scheme. The count is asserted so that a plan or configuration
change cannot quietly alter which runs the propagation analysis rests on.
"""

from __future__ import annotations

import pandas as pd

from rat_kp_fusion.jobs import build_plan as build_injection_plan
from .config import load_config


def build_full_context_jobs() -> pd.DataFrame:
    """Select the frozen full-context jobs used for complete 11-tissue inference.

    Narrows the frozen plan to the representative model and condition named in
    the DVI configuration for each split scheme; the expected count is asserted
    so a plan or configuration change cannot quietly alter which runs the
    propagation analysis rests on.
    """
    cfg = load_config()
    plan = build_injection_plan()
    selected = []
    for split_type, spec in cfg["representative_models"].items():
        subset = plan[
            plan["analysis_set"].eq("primary")
            & plan["stage"].eq("part1")
            & plan["split_type"].eq(split_type)
            & plan["model"].eq(spec["model"])
            & plan["condition"].eq(spec["full_context"])
            & plan["split_seed"].isin(cfg["split_seeds"])
        ].copy()
        selected.append(subset)
    jobs = pd.concat(selected, ignore_index=True)
    if len(jobs) != 20:
        raise RuntimeError(f"Expected 20 full-context jobs, found {len(jobs)}")
    return jobs.sort_values(["split_type", "split_seed"]).reset_index(drop=True)

