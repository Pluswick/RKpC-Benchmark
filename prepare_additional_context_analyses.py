"""Write and validate the isolated additional-analysis job manifest."""

from __future__ import annotations

import json

from rat_kp_core.data import sha256_file
from rat_kp_fusion.config import load_config as load_base_config
from rat_kp_fusion.paths import CONFIG_PATH as BASE_CONFIG_PATH

from rat_kp_controls.config import load_config
from rat_kp_controls.jobs import write_plan
from rat_kp_controls.paths import CONFIG_PATH, PREFLIGHT_PATH, STUDY_ROOT


def main() -> None:
    load_base_config(require_frozen=True)
    config = load_config()
    plan = write_plan()
    paths = [STUDY_ROOT / value for value in plan["split_path"].unique()]
    paths.append(STUDY_ROOT / plan.iloc[0]["dataset_path"])
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing additional-analysis inputs: {missing}")
    report = {
        "status": "passed",
        "additional_config_status": config["status"],
        "additional_config_sha256": sha256_file(CONFIG_PATH),
        "base_config_sha256": sha256_file(BASE_CONFIG_PATH),
        "planned_jobs": len(plan),
        "by_analysis": plan.groupby("analysis").size().to_dict(),
        "by_model": plan.groupby("model").size().to_dict(),
        "input_sha256": {
            str(path.relative_to(STUDY_ROOT)): sha256_file(path)
            for path in paths
        },
    }
    PREFLIGHT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

