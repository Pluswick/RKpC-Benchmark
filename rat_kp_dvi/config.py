"""Load and contract-check the frozen DVI output-propagation configuration.

Verifies that the file is still marked frozen and that it carries a volume
for all 11 modelled tissues, because a missing volume would silently drop a
tissue from the volume-weighted sum rather than fail.
"""

from __future__ import annotations

import json
from .paths import CONFIG_PATH


def load_config() -> dict:
    """Load the frozen DVI output-propagation configuration and verify its contract.

    See the module docstring for what the integrity check guarantees.
    """
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("schema_version") != "1.0" or cfg.get("status") != "frozen_before_vss_output_propagation":
        raise ValueError("Vss output-propagation config is not frozen")
    if len(cfg.get("tissue_volumes_ml", {})) != 11:
        raise ValueError("Exactly 11 tissue volumes are required")
    return cfg

