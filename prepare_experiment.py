"""Create and audit the frozen v2 primary and robustness job manifests."""

import json

from rat_kp_core.jobs import write_job_manifests


if __name__ == "__main__":
    print(json.dumps(write_job_manifests(), indent=2))
