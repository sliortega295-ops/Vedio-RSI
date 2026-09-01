from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


EVENT_TYPES = frozenset(
    {
        "run_started",
        "episode_released",
        "stage_queued",
        "stage_started",
        "stage_completed",
        "stage_failed",
        "stage_interrupted",
        "lease_acquired",
        "lease_released",
        "cache_hit",
        "cache_miss",
        "decision_sealed",
        "frontier_updated",
        "worker_started",
        "worker_reset",
        "worker_stopped",
        "run_completed",
    }
)


class LedgerError(RuntimeError):
    """Base class for fail-closed event ledger errors."""


class CorruptLedgerError(LedgerError):
    """Raised when a complete JSONL record is corrupt or out of sequence."""


class IdempotencyConflictError(LedgerError):
    """Raised when an idempotency key is reused with different content."""


@dataclass(frozen=True)
class LedgerState:
    events: tuple[dict[str, Any], ...]
    decisions: dict[str, dict[str, Any]]
    stage_states: dict[tuple[str, str, int], str]
    interrupted_stages: set[tuple[str, str, int]]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


@functools.lru_cache(maxsize=1)
def _boot_id() -> str:
    path = Path("/proc/sys/kernel/random/boot_id")
    try:
        return path.read_text(encoding="ascii").strip()
    except OSError:
        # A stable process-local fallback keeps events comparable on non-Linux test hosts.
        return f"pid-{os.getpid()}"


class EventLedger:
    """A locked, append-only JSONL ledger with idempotent decision sealing."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_locked(self, handle: Any, *, repair_tail: bool) -> list[dict[str, Any]]:
        handle.seek(0)
        raw = handle.read()
        if raw and not raw.endswith(b"\n"):
            last_newline = raw.rfind(b"\n")
            complete_size = last_newline + 1 if last_newline >= 0 else 0
            if repair_tail:
                handle.seek(complete_size)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
            raw = raw[:complete_size]

        events: list[dict[str, Any]] = []
        event_ids: set[str] = set()
        for line_number, line in enumerate(raw.splitlines(), start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorruptLedgerError(f"invalid complete JSON record at line {line_number}") from exc
            if not isinstance(event, dict):
                raise CorruptLedgerError(f"event at line {line_number} is not an object")
            expected_sequence = len(events) + 1
            if event.get("sequence") != expected_sequence:
                raise CorruptLedgerError(
                    f"sequence mismatch at line {line_number}: expected {expected_sequence}"
                )
            if event.get("event_type") not in EVENT_TYPES:
                raise CorruptLedgerError(f"unknown event type at line {line_number}")
            event_id = event.get("event_id")
            if not isinstance(event_id, str) or not event_id or event_id in event_ids:
                raise CorruptLedgerError(f"invalid or duplicate event_id at line {line_number}")
            event_ids.add(event_id)
            events.append(event)
        return events

    def read(self, *, repair_tail: bool = True) -> list[dict[str, Any]]:
        with self.path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                return self._read_locked(handle, repair_tail=repair_tail)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def append(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unsupported event type: {event_type}")
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        if event_type == "decision_sealed":
            episode_id = payload.get("episode_id")
            if not isinstance(episode_id, str) or idempotency_key != f"decision:{episode_id}":
                raise ValueError("decision_sealed requires the canonical per-episode idempotency key")

        with self.path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                events = self._read_locked(handle, repair_tail=True)
                if idempotency_key is not None:
                    for event in events:
                        if event.get("idempotency_key") != idempotency_key:
                            continue
                        if event["event_type"] == event_type and _canonical(event["payload"]) == _canonical(payload):
                            return event
                        raise IdempotencyConflictError(
                            f"conflicting payload for idempotency key {idempotency_key}"
                        )

                event = {
                    "sequence": len(events) + 1,
                    "event_id": str(uuid.uuid4()),
                    "event_type": event_type,
                    "payload": payload,
                    "idempotency_key": idempotency_key,
                    "utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                    "monotonic_ns": time.monotonic_ns(),
                    "boot_id": _boot_id(),
                }
                handle.seek(0, os.SEEK_END)
                handle.write(_canonical(event) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
                return event
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def find_idempotent(self, idempotency_key: str) -> dict[str, Any] | None:
        match: dict[str, Any] | None = None
        for event in self.read():
            if event.get("idempotency_key") != idempotency_key:
                continue
            if match is not None:
                raise CorruptLedgerError(
                    f"duplicate physical records for idempotency key {idempotency_key}"
                )
            match = event
        return match

    def seal_decision(self, episode_id: str, decision: dict[str, Any]) -> dict[str, Any]:
        return self.append(
            "decision_sealed",
            {"episode_id": episode_id, "decision": decision},
            idempotency_key=f"decision:{episode_id}",
        )

    def reconstruct(self) -> LedgerState:
        events = self.read()
        decisions: dict[str, dict[str, Any]] = {}
        stage_states: dict[tuple[str, str, int], str] = {}
        for event in events:
            payload = event["payload"]
            if event["event_type"] == "decision_sealed":
                if payload["episode_id"] in decisions:
                    raise CorruptLedgerError(
                        f"duplicate physical decision record for {payload['episode_id']}"
                    )
                decisions[payload["episode_id"]] = payload["decision"]
            if event["event_type"].startswith("stage_"):
                key = (
                    str(payload["episode_id"]),
                    str(payload["stage"]),
                    int(payload.get("attempt", 1)),
                )
                stage_states[key] = event["event_type"].removeprefix("stage_")
        interrupted = {key for key, status in stage_states.items() if status == "started"}
        for key in interrupted:
            stage_states[key] = "interrupted"
        return LedgerState(tuple(events), decisions, stage_states, interrupted)


def atomic_write_stage_output(
    ledger: EventLedger,
    output_path: Path | str,
    data: bytes,
    *,
    episode_id: str,
    stage: str,
    attempt: int,
    fault_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Durably replace an output, then record its digest as stage completion."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()
    payload = {
        "episode_id": episode_id,
        "stage": stage,
        "attempt": attempt,
        "output_path": str(target),
        "output_sha256": digest,
        "output_size_bytes": len(data),
    }
    idempotency_key = f"stage:{episode_id}:{stage}:{attempt}:complete"
    lock_path = target.parent / f".{target.name}.lock"
    with lock_path.open("a+b") as output_lock:
        fcntl.flock(output_lock.fileno(), fcntl.LOCK_EX)
        try:
            existing = ledger.find_idempotent(idempotency_key)
            if existing is not None:
                if (
                    existing.get("event_type") != "stage_completed"
                    or _canonical(existing.get("payload")) != _canonical(payload)
                ):
                    raise IdempotencyConflictError(
                        f"conflicting payload for idempotency key {idempotency_key}"
                    )
                if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise CorruptLedgerError(
                        f"durable output disagrees with completed stage {idempotency_key}"
                    )
                return existing

            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=target.parent, prefix=f".{target.name}.", delete=False
                ) as handle:
                    temporary_path = Path(handle.name)
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                if fault_hook is not None:
                    fault_hook("before_replace")
                os.replace(temporary_path, target)
                temporary_path = None
                directory_fd = os.open(
                    target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                if fault_hook is not None:
                    fault_hook("after_replace")
                if fault_hook is not None:
                    fault_hook("before_completion_append")
                return ledger.append(
                    "stage_completed",
                    payload,
                    idempotency_key=idempotency_key,
                )
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        finally:
            fcntl.flock(output_lock.fileno(), fcntl.LOCK_UN)
