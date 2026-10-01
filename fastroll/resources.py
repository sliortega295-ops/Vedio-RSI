"""Explicit time-bounded authority and cooperative physical-device locking."""
from __future__ import annotations

import fcntl
import os
import subprocess
import time
from pathlib import Path

from .contracts import ContractError, digest, read_json


class FileLock:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ContractError("lock path cannot be a symlink")
        self.fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(self.fd)
            raise

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def inventory() -> dict[str, dict]:
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,index,memory.free,memory.total",
        "--format=csv,noheader,nounits"], text=True, timeout=10)
    result = {}
    for line in raw.strip().splitlines():
        uid, index, free, total = [x.strip() for x in line.split(",")]
        result[uid] = {"uuid": uid, "index": int(index), "free_mib": int(free), "total_mib": int(total)}
    return result


def active_gpus() -> set[str]:
    raw = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        text=True, timeout=10)
    return set(filter(None, (x.strip() for x in raw.splitlines())))


class Authority:
    def __init__(self, path: Path, workspace: Path):
        self.path = path.resolve(strict=True)
        self.workspace = workspace.resolve()
        self.snapshot = read_json(self.path)
        self.check()

    def check(self):
        a = read_json(self.path)
        if a != self.snapshot:
            raise PermissionError("authorization changed/revoked; stop all owned work")
        if a.get("authorized") is not True or not a.get("issuer"):
            raise PermissionError("external resource-owner authorization required")
        if Path(a.get("workspace", "")).resolve() != self.workspace:
            raise PermissionError("authorization workspace mismatch")
        now = time.time()
        if not (type(a.get("not_before")) in (int, float) and type(a.get("expires_at")) in (int, float) and a["not_before"] <= now < a["expires_at"]):
            raise PermissionError("authorization not active or expired")
        ids = a.get("gpu_uuids")
        if not isinstance(ids, list) or len(set(ids)) != len(ids) or any(not isinstance(x, str) or not x.startswith("GPU-") for x in ids):
            raise PermissionError("authorization must list exact GPU UUIDs")
        if not isinstance(a.get("approved_sessions"), list):
            raise PermissionError("authorization must list approved session hashes")
        if not a["approved_sessions"] and not (a.get("allow_new_sessions_under_source_roots") is True and a.get("source_roots") and a.get("approved_backends")):
            raise PermissionError("empty sessions require explicit bounded autonomous-source approval")
        if not Path(a.get("lock_root", "")).is_absolute():
            raise PermissionError("authorization requires a shared absolute GPU lock root")
        return a

    def admit(self, session: dict):
        a = self.check()
        if digest(session) not in a["approved_sessions"]:
            if not (a.get("allow_new_sessions_under_source_roots") is True and session.get("backend") in a.get("approved_backends", []) and "source" in session):
                raise PermissionError("session was not approved by the resource owner")
        if "source" in session:
            root = Path(session["source"]["root"]).resolve()
            if not any(root.is_relative_to(Path(p).resolve()) for p in a.get("source_roots", [])):
                raise PermissionError("source root outside authorization")


class GPUReservation:
    def __init__(self, authority: Authority):
        self.locks = []
        self.devices = []
        try:
            a = authority.check()
            devices = inventory() if a["gpu_uuids"] else {}
            for uid in sorted(a["gpu_uuids"]):
                if uid not in devices:
                    raise PermissionError(f"authorized device is absent: {uid}")
                lock = FileLock(Path(a["lock_root"]) / (uid + ".lock"))
                self.locks.append(lock)
                if uid in active_gpus():
                    raise PermissionError(f"device already has a compute process: {uid}")
                self.devices.append(devices[uid])
        except BaseException:
            self.close()
            raise

    def close(self):
        for lock in self.locks:
            lock.close()
        self.locks = []
