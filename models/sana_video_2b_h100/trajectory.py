#!/usr/bin/env python3
"""Append or validate the complete per-round Sol-Video-Agent trajectory."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path


CAPS = {"kernel": 40, "cache": 20}
REQUIRED_SECTIONS = ("session", "provenance", "build", "run", "validity", "decision")
DECISIONS = {"retain", "reject", "retry", "stop"}


def load_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        validate_record(record)
        records.append(record)
    validate_sequence(records)
    return records


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_record(record: dict[str, object]) -> None:
    if record.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    component = record.get("component")
    if component not in CAPS:
        raise ValueError(f"component must be one of {sorted(CAPS)}")
    round_number = record.get("round")
    if not isinstance(round_number, int) or isinstance(round_number, bool) or round_number < 1:
        raise ValueError("round must be a positive integer")
    if round_number > CAPS[str(component)]:
        raise ValueError(f"{component} round exceeds hard cap {CAPS[str(component)]}")
    if not _nonempty(record.get("hypothesis")):
        raise ValueError("hypothesis must be non-empty")
    for section in REQUIRED_SECTIONS:
        if not isinstance(record.get(section), dict):
            raise ValueError(f"{section} must be an object")
    session = record["session"]
    if not _nonempty(session.get("agent_id")) or not _nonempty(session.get("prompt_sha256")):
        raise ValueError("session requires agent_id and prompt_sha256")
    provenance = record["provenance"]
    for key in ("baseline_sha256", "parent_sha"):
        if not _nonempty(provenance.get(key)):
            raise ValueError(f"provenance requires {key}")
    build = record["build"]
    if build.get("status") not in {"passed", "failed", "not_run"}:
        raise ValueError("build.status must be passed, failed, or not_run")
    run = record["run"]
    if run.get("status") not in {"validated", "failed", "not_run"}:
        raise ValueError("run.status must be validated, failed, or not_run")
    if not _nonempty(run.get("command")):
        raise ValueError("run.command must preserve the exact attempted command")
    validity = record["validity"]
    if validity.get("status") not in {"validated", "failed", "not_run"}:
        raise ValueError("validity.status must be validated, failed, or not_run")
    decision = record["decision"]
    if decision.get("outcome") not in DECISIONS or not _nonempty(decision.get("reason")):
        raise ValueError(f"decision requires outcome in {sorted(DECISIONS)} and a reason")


def validate_sequence(records: list[dict[str, object]]) -> None:
    if not records:
        return
    components = {record["component"] for record in records}
    if len(components) != 1:
        raise ValueError("one trajectory ledger may contain only one component")
    rounds = [record["round"] for record in records]
    if rounds != list(range(1, len(records) + 1)):
        raise ValueError(f"rounds must be contiguous from 1, got {rounds}")
    stop_rounds = [record["round"] for record in records if record["decision"]["outcome"] == "stop"]
    if stop_rounds and stop_rounds != [rounds[-1]]:
        raise ValueError("stop may appear only on the final round")


def append_record(ledger: Path, record_path: Path) -> dict[str, object]:
    record = json.loads(record_path.read_text())
    validate_record(record)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing = []
        for line in handle.read().splitlines():
            if line.strip():
                existing.append(json.loads(line))
        for old in existing:
            validate_record(old)
        validate_sequence(existing)
        expected_round = len(existing) + 1
        if record["round"] != expected_round:
            raise ValueError(f"next round must be {expected_round}, got {record['round']}")
        if existing and record["component"] != existing[0]["component"]:
            raise ValueError("component does not match existing ledger")
        candidate = [*existing, record]
        validate_sequence(candidate)
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    append = subparsers.add_parser("append")
    append.add_argument("--ledger", type=Path, default=Path("TRAJECTORY.jsonl"))
    append.add_argument("--record", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--ledger", type=Path, default=Path("TRAJECTORY.jsonl"))
    args = parser.parse_args()
    if args.command == "append":
        record = append_record(args.ledger, args.record)
        print(json.dumps({"status": "appended", "component": record["component"], "round": record["round"]}))
        return 0
    records = load_records(args.ledger)
    print(json.dumps({"status": "valid", "rounds": len(records), "component": records[0]["component"] if records else None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
