"""Phase 3 site-only network deny for original CPU quality subprocesses.

Python imports this module from the fixed PYTHONPATH before the archived
collector, plan_eval, and LPIPS judge. It does not modify any quality metric.
"""

from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path


_AUDIT_ROOT = Path(
    "/share/project/shuchenweng/yongyan_liu/Experiments/SolAgent/20260913-a100-original-baseline"
) / "repo/output/experiments/sana2ba100-cache_ca-0002/worktree/runs"
_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_CONNECT_EX = socket.socket.connect_ex


def _record(kind: str) -> None:
    raw = os.environ.get("SOL_PHASE3_NETWORK_AUDIT", "")
    path = Path(raw).resolve() if raw else None
    if path is None or not path.is_relative_to(_AUDIT_ROOT) or path.name != "network_audit.jsonl":
        # Deny still applies even when the audit path is malformed/missing.
        return
    row = {"kind": kind, "pid": os.getpid(),
           "utc": datetime.now(timezone.utc).isoformat()}
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _connect(sock: socket.socket, address: object) -> object:
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        _record("blocked_socket_connect")
        raise OSError("NETWORK_DISABLED_PHASE3")
    return _ORIGINAL_CONNECT(sock, address)


def _connect_ex(sock: socket.socket, address: object) -> int:
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        _record("blocked_socket_connect_ex")
        raise OSError("NETWORK_DISABLED_PHASE3")
    return _ORIGINAL_CONNECT_EX(sock, address)


socket.socket.connect = _connect
socket.socket.connect_ex = _connect_ex
_record("guard_loaded")
