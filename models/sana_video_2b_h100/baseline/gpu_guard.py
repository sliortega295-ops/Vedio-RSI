"""UUID-scoped cooperative GPU lease and ownership checks.

The lock serializes every Sol-Agent run that cooperates with this experiment.
The live ``nvidia-smi`` check is still mandatory because foreign jobs do not
know about our lock.
"""

from __future__ import annotations

import csv
import fcntl
import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class GpuLease:
    gpu_uuid: str
    lock_path: Path
    lease_file: Path
    host: str
    owner: str
    leased_at_utc: str


def _nvidia_smi() -> str:
    return os.environ.get("SANA_NVIDIA_SMI", "nvidia-smi")


def _run_query(arguments: list[str]) -> list[list[str]]:
    proc = subprocess.run(
        [_nvidia_smi(), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi query failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return [
        [cell.strip() for cell in row]
        for row in csv.reader(proc.stdout.splitlines())
        if row and any(cell.strip() for cell in row)
    ]


def query_gpu(gpu_uuid: str) -> dict[str, object]:
    rows = _run_query(
        [
            "--query-gpu=uuid,index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    matches = [row for row in rows if row[0] == gpu_uuid]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one visible GPU with UUID {gpu_uuid}, found {len(matches)}"
        )
    row = matches[0]
    if len(row) != 6:
        raise RuntimeError(f"unexpected nvidia-smi GPU row: {row!r}")
    return {
        "uuid": row[0],
        "index": int(row[1]),
        "name": row[2],
        "memory_total_mib": int(row[3]),
        "memory_used_mib": int(row[4]),
        "utilization_gpu_percent": int(row[5]),
    }


def query_compute_apps(gpu_uuid: str) -> list[dict[str, object]]:
    rows = _run_query(
        [
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    apps: list[dict[str, object]] = []
    for row in rows:
        if row[0] != gpu_uuid:
            continue
        if len(row) != 4:
            raise RuntimeError(f"unexpected nvidia-smi compute-app row: {row!r}")
        apps.append(
            {
                "gpu_uuid": row[0],
                "pid": int(row[1]),
                "process_name": row[2],
                "used_gpu_memory_mib": int(row[3]),
            }
        )
    return apps


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_lease(path: str | Path) -> GpuLease:
    lease_file = Path(path).expanduser().resolve()
    try:
        payload = json.loads(lease_file.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read GPU lease {lease_file}: {exc}") from exc
    if payload.get("schema_version") != 1 or payload.get("status") != "active":
        raise RuntimeError(f"GPU lease is not active: {lease_file}")
    gpu_uuid = str(payload.get("gpu_uuid") or "")
    lock_path_raw = str(payload.get("lock_path") or "")
    if not gpu_uuid.startswith("GPU-") or not Path(lock_path_raw).is_absolute():
        raise RuntimeError(f"invalid GPU lease fields: {lease_file}")
    return GpuLease(
        gpu_uuid=gpu_uuid,
        lock_path=Path(lock_path_raw).resolve(),
        lease_file=lease_file,
        host=str(payload.get("host") or ""),
        owner=str(payload.get("owner") or ""),
        leased_at_utc=str(payload.get("leased_at_utc") or ""),
    )


@contextmanager
def locked_idle_lease(path: str | Path) -> Iterator[tuple[GpuLease, dict[str, object]]]:
    initial = load_lease(path)
    initial.lock_path.parent.mkdir(parents=True, exist_ok=True)
    with initial.lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        current = load_lease(path)
        if current.gpu_uuid != initial.gpu_uuid or current.lock_path != initial.lock_path:
            raise RuntimeError("GPU lease changed while waiting for its lock")
        gpu = query_gpu(current.gpu_uuid)
        apps = query_compute_apps(current.gpu_uuid)
        if apps:
            raise RuntimeError(
                f"refusing to disturb live compute apps on {current.gpu_uuid}: {apps}"
            )
        yield current, gpu
