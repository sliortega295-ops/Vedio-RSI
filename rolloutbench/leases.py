from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping


_GPU_UUID_RE = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9_.-]+\Z")


class LeaseContractError(RuntimeError):
    """Raised when a cooperative GPU lease record is missing or inconsistent."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise LeaseContractError(f"{label} must be a nonempty safe identifier")
    return value


def _request(
    *,
    authorization: str,
    owner: str,
    host: str,
    plan_id: str,
    run_id: str,
    gpu_uuid: str,
    lock_path: str,
) -> dict[str, str]:
    values = {
        "authorization": _identifier(authorization, "authorization"),
        "owner": _identifier(owner, "owner"),
        "host": _identifier(host, "host"),
        "plan_id": _identifier(plan_id, "plan_id"),
        "run_id": _identifier(run_id, "run_id"),
    }
    if not isinstance(gpu_uuid, str) or not _GPU_UUID_RE.fullmatch(gpu_uuid):
        raise LeaseContractError("gpu_uuid must be a canonical GPU UUID")
    if not isinstance(lock_path, str) or not lock_path:
        raise LeaseContractError("lock_path must be an absolute path")
    absolute_lock = Path(lock_path)
    if not absolute_lock.is_absolute() or ".." in absolute_lock.parts:
        raise LeaseContractError("lock_path must be an absolute normalized path")
    return {
        **values,
        "gpu_uuid": gpu_uuid,
        "lock_path": str(absolute_lock),
    }


def _lease_record(request: Mapping[str, str], lease_file: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "record_type": "cooperative_gpu_lease",
        # Keep the runtime-facing fields compatible with gpu_guard.load_lease().
        "status": "active",
        "authorization_scope": "cooperative_only_not_ownership",
        "ownership_claim": False,
        "lease_file": str(lease_file.resolve()),
        "leased_at_utc": "",
        **request,
    }


def _release_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.release.json")


def _load_record(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LeaseContractError(f"{label} is missing or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise LeaseContractError(f"{label} must be a JSON object")
    return value


def _assert_matches(record: Mapping[str, Any], request: Mapping[str, str]) -> None:
    if record.get("schema_version") != 1 or record.get("record_type") != "cooperative_gpu_lease":
        raise LeaseContractError("lease record schema is invalid")
    if record.get("status") != "active":
        raise LeaseContractError("lease record is not active")
    if record.get("ownership_claim") is not False:
        raise LeaseContractError("lease record must not claim GPU ownership")
    for key, expected in request.items():
        if record.get(key) != expected:
            raise LeaseContractError(f"lease {key} mismatch")


def _atomic_write_new_or_same(path: Path, content: bytes) -> bool:
    """Write content atomically; return false when an identical record already exists."""

    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise LeaseContractError(f"refusing overwrite of conflicting lease record: {path}")
        return False
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def acquire_cooperative_lease(
    lease_path: Path | str,
    *,
    authorization: str,
    owner: str,
    host: str,
    plan_id: str,
    run_id: str,
    gpu_uuid: str,
    lock_path: str,
) -> dict[str, Any]:
    """Create one explicit cooperative authorization record without probing a GPU."""

    request = _request(
        authorization=authorization,
        owner=owner,
        host=host,
        plan_id=plan_id,
        run_id=run_id,
        gpu_uuid=gpu_uuid,
        lock_path=lock_path,
    )
    target = Path(lease_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mutex = target.parent / f".{target.name}.contract.lock"
    with mutex.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            if _release_path(target).exists():
                raise LeaseContractError("lease was released and cannot be reactivated")
            record = _lease_record(request, target)
            _atomic_write_new_or_same(target, _canonical(record))
            return record
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_active_lease(
    lease_path: Path | str,
    *,
    authorization: str,
    owner: str,
    host: str,
    plan_id: str,
    run_id: str,
    gpu_uuid: str,
    lock_path: str,
) -> dict[str, Any]:
    """Read and validate an active cooperative lease without modifying its files."""

    request = _request(
        authorization=authorization,
        owner=owner,
        host=host,
        plan_id=plan_id,
        run_id=run_id,
        gpu_uuid=gpu_uuid,
        lock_path=lock_path,
    )
    target = Path(lease_path)
    mutex = target.parent / f".{target.name}.contract.lock"
    try:
        with mutex.open("rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                if _release_path(target).exists():
                    raise LeaseContractError("lease has been released")
                record = _load_record(target, "lease record")
                _assert_matches(record, request)
                return record
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError as exc:
        raise LeaseContractError("lease validation lock is missing or unreadable") from exc


def release_cooperative_lease(
    lease_path: Path | str,
    *,
    authorization: str,
    owner: str,
    host: str,
    plan_id: str,
    run_id: str,
    gpu_uuid: str,
    lock_path: str,
) -> dict[str, Any]:
    """Record an explicit release audit without deleting the immutable active record."""

    request = _request(
        authorization=authorization,
        owner=owner,
        host=host,
        plan_id=plan_id,
        run_id=run_id,
        gpu_uuid=gpu_uuid,
        lock_path=lock_path,
    )
    target = Path(lease_path)
    audit_path = _release_path(target)
    mutex = target.parent / f".{target.name}.contract.lock"
    with mutex.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            active = _load_record(target, "lease record")
            _assert_matches(active, request)
            audit = {
                "schema_version": 1,
                "record_type": "cooperative_gpu_lease_release",
                "status": "released",
                "active_lease_sha256": _sha256(_canonical(active)),
                "ownership_claim": False,
                **request,
            }
            _atomic_write_new_or_same(audit_path, _canonical(audit))
            return audit
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
