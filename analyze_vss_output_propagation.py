"""Run the DVI output-propagation analysis and report output row counts.

Entry point for ``rat_kp_dvi.analysis``; no analysis logic lives here.
"""

from __future__ import annotations

import json
from rat_kp_dvi.analysis import run_analysis


if __name__ == "__main__":
    outputs = run_analysis()
    print(json.dumps({name: len(frame) for name, frame in outputs.items()}, indent=2))
