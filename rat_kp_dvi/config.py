from __future__ import annotations

import json
from .paths import CONFIG_PATH


def load_config() -> dict:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("schema_version") != "1.0" or cfg.get("status") != "frozen_before_vss_output_propagation":
        raise ValueError("Vss output-propagation config is not frozen")
    if len(cfg.get("tissue_volumes_ml", {})) != 11:
        raise ValueError("Exactly 11 tissue volumes are required")
    return cfg

