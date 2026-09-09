from __future__ import annotations

import json

from rat_kp_core.data import sha256_file

from .paths import CONFIG_PATH


# Frozen after implementation, contract tests, and candidate smoke checks passed.
CONFIG_SHA256: str | None = "1828020c55e481c46205550c8eacd87968e0156d2eda26d0bf6c070e0b4d1362"


def load_config(*, require_frozen: bool = False) -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("schema_version") != "1.0" or config.get("target") != "log10(Kp)":
        raise ValueError("Injection-study config contract failed")
    if require_frozen:
        if CONFIG_SHA256 is None or config.get("status") != "approved_and_frozen_before_execution":
            raise ValueError("Injection-study config is not frozen")
        if sha256_file(CONFIG_PATH) != CONFIG_SHA256:
            raise ValueError("Injection-study config checksum mismatch")
    return config
