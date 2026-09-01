from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from .leases import LeaseContractError, validate_active_lease
from .pilot_runner import RunContext


class LaunchAuthorizationError(RuntimeError):
    """Raised when external GPU ownership evidence does not authorize a run."""


_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+\Z")
_GPU_UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
_AUTHORITY_KINDS = frozenset({"user", "cluster_admin", "scheduler"})
_SCOPE_PREFIX = "sol-rolloutbench-v0-formal"
_MAX_PREFLIGHT_AGE = timedelta(minutes=10)
_MAX_AUTHORIZATION_LIFETIME = timedelta(hours=24)


def _load_regular_json(path_value: Path | str, label: str) -> tuple[Path, dict[str, Any], bytes]:
    path = Path(path_value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise LaunchAuthorizationError(f"{label} must be an absolute regular file")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LaunchAuthorizationError(f"{label} is unreadable or invalid") from exc
    if not isinstance(value, dict):
        raise LaunchAuthorizationError(f"{label} must contain one JSON object")
    return path, value, raw


def _time(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise LaunchAuthorizationError(f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise LaunchAuthorizationError(f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LaunchAuthorizationError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise LaunchAuthorizationError(f"{label} must be a safe nonempty identifier")
    return value


def _absolute_map(value: Any, expected: set[str], label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise LaunchAuthorizationError(f"{label} must cover the exact run GPU UUID set")
    result: dict[str, str] = {}
    for gpu_uuid, path_value in value.items():
        path = Path(str(path_value))
        if not path.is_absolute() or ".." in path.parts:
            raise LaunchAuthorizationError(f"{label}.{gpu_uuid} must be an absolute normalized path")
        result[str(gpu_uuid)] = str(path)
    return result


def validate_launch_authorization(
    context: RunContext,
    authorization_path: Path | str,
    lease_files: Mapping[str, Path | str],
    *,
    preflight_spec: Mapping[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate a procedural external assertion plus exact active GPU leases.

    The JSON issuer fields are not cryptographically authenticated.  This gate
    therefore proves plan binding, freshness, and cooperative lease state, but
    the human or scheduler identity remains a procedural trust boundary.
    """

    path, authorization, raw = _load_regular_json(
        authorization_path, "launch authorization"
    )
    phase = context.run.get("scope")
    if phase not in {"pilot", "full"}:
        raise LaunchAuthorizationError("planned run scope is not pilot or full")
    required_header = {
        "schema_version": 1,
        "record_type": "gpu_launch_authorization",
        "status": "AUTHORIZED",
        "scope": f"{_SCOPE_PREFIX}-{phase}",
        "ownership_verified": True,
        "plan_id": context.plan_id,
        "plan_sha256": context.plan_sha256,
        "run_id": context.run.get("run_id"),
        "run_sha256": context.run_sha256,
    }
    mismatched = [
        key for key, expected in required_header.items()
        if authorization.get(key) != expected
    ]
    if mismatched:
        raise LaunchAuthorizationError(
            f"launch authorization header mismatch: {sorted(mismatched)}"
        )

    authorization_id = _safe_id(
        authorization.get("authorization_id"), "authorization_id"
    )
    owner = _safe_id(authorization.get("owner"), "owner")
    host = _safe_id(authorization.get("host"), "host")
    issued_by = _safe_id(authorization.get("issued_by"), "issued_by")
    authority_kind = authorization.get("authority_kind")
    if authority_kind not in _AUTHORITY_KINDS:
        raise LaunchAuthorizationError("authority_kind is not an allowed external authority")

    workers = context.run.get("workers")
    if not isinstance(workers, list) or not workers:
        raise LaunchAuthorizationError("planned run has no workers")
    planned_uuids = [str(worker.get("gpu_uuid", "")) for worker in workers]
    if (
        len(planned_uuids) != len(set(planned_uuids))
        or any(not _GPU_UUID.fullmatch(value) for value in planned_uuids)
    ):
        raise LaunchAuthorizationError("planned worker GPU UUIDs are invalid")
    declared_uuids = authorization.get("gpu_uuids")
    if not isinstance(declared_uuids, list) or declared_uuids != planned_uuids:
        raise LaunchAuthorizationError("authorization GPU UUID order does not match the run")
    uuid_set = set(planned_uuids)
    lock_paths = _absolute_map(
        authorization.get("lock_paths"), uuid_set, "lock_paths"
    )
    declared_leases = _absolute_map(
        authorization.get("lease_files"), uuid_set, "lease_files"
    )
    supplied_leases = {str(key): str(Path(value)) for key, value in lease_files.items()}
    if supplied_leases != declared_leases:
        raise LaunchAuthorizationError("supplied lease files disagree with authorization")

    observed_at = _time(authorization.get("preflight_observed_at_utc"), "preflight_observed_at_utc")
    issued_at = _time(authorization.get("issued_at_utc"), "issued_at_utc")
    expires_at = _time(authorization.get("expires_at_utc"), "expires_at_utc")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if (
        observed_at > issued_at
        or issued_at > current
        or current - observed_at > _MAX_PREFLIGHT_AGE
    ):
        raise LaunchAuthorizationError("authorization is not based on a fresh preflight")
    if (
        expires_at <= issued_at
        or expires_at <= current
        or expires_at - issued_at > _MAX_AUTHORIZATION_LIFETIME
    ):
        raise LaunchAuthorizationError("authorization is expired or exceeds 24 hours")

    preflight_path, preflight, preflight_raw = _load_regular_json(
        authorization.get("preflight_receipt_path", ""), "preflight receipt"
    )
    preflight_sha = hashlib.sha256(preflight_raw).hexdigest()
    if authorization.get("preflight_receipt_sha256") != preflight_sha:
        raise LaunchAuthorizationError("preflight receipt SHA-256 mismatch")
    if (
        preflight.get("schema_version") != 1
        or preflight.get("query_status") != "PASS"
        or preflight.get("technical_ready") is not True
        or preflight.get("runtime_ready") is not True
        or preflight.get("quality_ready") is not True
        or preflight.get("two_gpu_idle_point_in_time") is not True
        or preflight.get("launch_authorized") is not False
        or preflight.get("pilot_ready") is not False
        or preflight.get("gpu_idle_scope", {}).get("ownership_verified") is not False
        or preflight.get("gpu_idle_scope", {}).get("observed_at_utc")
        != authorization.get("preflight_observed_at_utc")
    ):
        raise LaunchAuthorizationError(
            "preflight must be technically ready while making no ownership claim"
        )
    observation = preflight.get("observation")
    observed_gpus = observation.get("gpus", []) if isinstance(observation, Mapping) else []
    observed_uuids = {
        str(row.get("uuid")) for row in observed_gpus if isinstance(row, Mapping)
    }
    if not uuid_set.issubset(observed_uuids):
        raise LaunchAuthorizationError("preflight does not contain every authorized GPU")
    spec_hashes = {
        key: preflight_spec.get(key)
        for key in (
            "profile_sha256",
            "model_profile_sha256",
            "suite_sha256",
            "quality_protocol_sha256",
            "artifacts_sha256",
        )
    }
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in spec_hashes.values()
    ):
        raise LaunchAuthorizationError("preflight spec hash closure is incomplete")
    if any(preflight.get(key) != value for key, value in spec_hashes.items()):
        raise LaunchAuthorizationError("preflight receipt does not match the formal profile")
    target_rows = preflight_spec.get("target_gpus")
    target_uuids = {
        str(row.get("uuid"))
        for row in target_rows
        if isinstance(row, Mapping)
    } if isinstance(target_rows, list) else set()
    if not uuid_set.issubset(target_uuids):
        raise LaunchAuthorizationError("planned GPUs are outside the formal preflight spec")

    lease_receipts: dict[str, Any] = {}
    for gpu_uuid in planned_uuids:
        try:
            lease_receipts[gpu_uuid] = validate_active_lease(
                declared_leases[gpu_uuid],
                authorization=authorization_id,
                owner=owner,
                host=host,
                plan_id=context.plan_id,
                run_id=str(context.run["run_id"]),
                gpu_uuid=gpu_uuid,
                lock_path=lock_paths[gpu_uuid],
            )
        except LeaseContractError as exc:
            raise LaunchAuthorizationError(
                f"active cooperative lease is invalid for {gpu_uuid}"
            ) from exc

    return {
        "schema_version": 1,
        "status": "VALIDATED",
        "authorization_path": str(path),
        "authorization_sha256": hashlib.sha256(raw).hexdigest(),
        "authorization_id": authorization_id,
        "owner": owner,
        "issued_by": issued_by,
        "authority_kind": authority_kind,
        "host": host,
        "plan_id": context.plan_id,
        "plan_sha256": context.plan_sha256,
        "run_id": context.run["run_id"],
        "run_sha256": context.run_sha256,
        "gpu_uuids": planned_uuids,
        "lock_paths": lock_paths,
        "lease_files": declared_leases,
        "lease_receipt_sha256": {
            gpu_uuid: hashlib.sha256(
                json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for gpu_uuid, receipt in lease_receipts.items()
        },
        "preflight_receipt_path": str(preflight_path),
        "preflight_receipt_sha256": preflight_sha,
        "ownership_assertion_accepted": True,
        "ownership_verified": False,
        "authority_verification": "procedural_assertion_not_cryptographically_verified",
        "cryptographic_issuer_verification": False,
        "issued_at_utc": issued_at.isoformat(),
        "expires_at_utc": expires_at.isoformat(),
        "performance_claim": False,
    }
