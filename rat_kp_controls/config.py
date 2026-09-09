from __future__ import annotations

import json

from rat_kp_core.data import sha256_file

from .paths import CONFIG_PATH


# Locked after contract tests and six GPU smoke checks, before any full job.
CONFIG_SHA256: str | None = "3f9797fefb7ffd98c284f11f45ccf931ec75b26c80995be8779f95765d4783f2"


def load_config(*, require_locked: bool = False) -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != "1.0"
        or config.get("target") != "log10(Kp)"
        or config.get("scope", {}).get("new_jobs") != 520
    ):
        raise ValueError("Additional-context analysis config contract failed")
    if require_locked:
        if CONFIG_SHA256 is None or config.get("status") != "locked_before_additional_execution":
            raise ValueError("Additional-context analysis config is not locked")
        if sha256_file(CONFIG_PATH) != CONFIG_SHA256:
            raise ValueError("Additional-context analysis config checksum mismatch")
    return config
