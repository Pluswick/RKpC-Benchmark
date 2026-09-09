from __future__ import annotations

import json
from rat_kp_dvi.analysis import run_analysis


if __name__ == "__main__":
    outputs = run_analysis()
    print(json.dumps({name: len(frame) for name, frame in outputs.items()}, indent=2))
