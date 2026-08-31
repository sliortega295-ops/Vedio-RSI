#!/usr/bin/env python3
"""Create the one persistent UUID lease after proving the H100 is idle."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

from gpu_guard import atomic_write_json, query_compute_apps, query_gpu


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--lease-file", required=True)
    parser.add_argument("--lock-path", required=True)
    parser.add_argument("--owner", default="sol-video-agent-reproducer")
    args = parser.parse_args()

    lease_file = Path(args.lease_file).expanduser().resolve()
    lock_path = Path(args.lock_path).expanduser().resolve()
    if not args.uuid.startswith("GPU-"):
        raise SystemExit("--uuid must be an NVIDIA GPU UUID")
    if lease_file.exists():
        raise SystemExit(f"refusing to overwrite existing lease: {lease_file}")

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        gpu = query_gpu(args.uuid)
        apps = query_compute_apps(args.uuid)
        if apps:
            raise SystemExit(f"GPU is not idle; refusing lease: {json.dumps(apps)}")
        payload: dict[str, object] = {
            "schema_version": 1,
            "status": "active",
            "gpu_uuid": args.uuid,
            "lock_path": str(lock_path),
            "lease_file": str(lease_file),
            "host": socket.gethostname(),
            "owner": args.owner,
            "leased_at_utc": datetime.now(timezone.utc).isoformat(),
            "lease_creator_pid": os.getpid(),
            "gpu_at_lease": gpu,
            "compute_apps_at_lease": apps,
        }
        atomic_write_json(lease_file, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
