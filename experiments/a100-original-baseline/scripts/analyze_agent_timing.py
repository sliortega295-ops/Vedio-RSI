#!/usr/bin/env python3
"""Summarize Codex stream event boundaries without inventing thinking time.

Timed JSONL is written by the site transport worker on the local host. Tool
start/completion pairs are observable stream intervals; gaps outside those
pairs can include API latency, model work, buffering, and agent reasoning and
must not be called a precise component time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def rows(path: Path) -> tuple[list[dict], int]:
    result = []
    malformed = 0
    for line in path.open():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            malformed += 1
    return result, malformed


def classify(item_type: str, command: str) -> str:
    if item_type == "file_change":
        return "file_edit_event"
    if "--help" in command and re.search(r"\bpython3?\s+\S+\.py\b", command):
        return "tool_help_preflight"
    for script, label in (
        ("orchestration/bin/spawn_executor.py", "spawn_executor_including_materialization"),
        ("orchestration/bin/poll_executor.py", "poll_executor"),
        ("orchestration/bin/verify_delivery.py", "verify_delivery"),
        ("orchestration/bin/resume_executor.py", "resume_executor"),
        ("scripts/launch_config.py", "launch_submit"),
        ("orchestration/site_remote.py", "remote_job_status_or_sync"),
        ("scripts/collect_run.py", "collect_and_quality"),
        ("search/plan_eval.py", "quality_assessment"),
    ):
        if re.search(r"\bpython3?\s+" + re.escape(script) + r"(?:\s|$)", command):
            return label
    if re.search(r"\bpython3?\s+-m\s+(?:py_compile|pytest)\b", command):
        return "cpu_preflight"
    return "other_tool_event"


def union_s(intervals: list[tuple[int, int]]) -> float:
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged) / 1e9


def summarize(name: str, path: Path, local_events: list[dict]) -> dict:
    timed, malformed = rows(path)
    attempts = {row.get("attempt") for row in timed if isinstance(row.get("attempt"), int)}
    if len(attempts) > 1:
        raise ValueError(f"timed stream mixes attempts: {path}")
    attempt = next(iter(attempts), None)
    starts = {}
    completed = []
    event_types = Counter()
    for row in timed:
        event_types[row.get("codex_type") or "unknown"] += 1
        try:
            inner = json.loads(row["line"])
        except (KeyError, json.JSONDecodeError):
            continue
        item = inner.get("item") or {}
        ident = item.get("id")
        if not ident:
            continue
        if inner.get("type") == "item.started":
            starts[ident] = (row, item)
        elif inner.get("type") == "item.completed" and ident in starts:
            begin, started_item = starts.pop(ident)
            item_type = item.get("type") or started_item.get("type") or "unknown"
            command = item.get("command") or started_item.get("command") or ""
            elapsed = (row["monotonic_ns"] - begin["monotonic_ns"]) / 1e9
            completed.append({
                "item_id": ident, "item_type": item_type,
                "category": classify(item_type, command),
                "utc_start": begin["utc"], "utc_end": row["utc"],
                "monotonic_start_ns": begin["monotonic_ns"],
                "monotonic_end_ns": row["monotonic_ns"], "elapsed_s": elapsed,
                "command_sha256": hashlib.sha256(command.encode()).hexdigest() if command else None,
                "exit_code": item.get("exit_code"),
            })
    intervals = [(item["monotonic_start_ns"], item["monotonic_end_ns"])
                 for item in completed]
    api_begins = [row for row in local_events if row.get("event") == "agent_api_begin"
                  and row.get("name") == name and row.get("attempt") == attempt]
    api_ends = [row for row in local_events if row.get("event") == "agent_api_end"
                and row.get("name") == name and row.get("attempt") == attempt]
    api_begin = min(api_begins, key=lambda row: row["monotonic_ns"]) if api_begins else None
    api_end = max(api_ends, key=lambda row: row["monotonic_ns"]) if api_ends else None
    later_begins = [row["monotonic_ns"] for row in local_events
                    if row.get("event") == "agent_api_begin" and row.get("name") == name
                    and isinstance(row.get("attempt"), int) and attempt is not None
                    and row["attempt"] > attempt]
    next_begin_ns = min(later_begins) if later_begins else None
    stops = [row for row in local_events if row.get("event") == "agent_session_stop"
             and row.get("name") == name
             and (api_begin is None or row["monotonic_ns"] >= api_begin["monotonic_ns"])
             and (next_begin_ns is None or row["monotonic_ns"] < next_begin_ns)]
    session_stop = max(stops, key=lambda row: row["monotonic_ns"]) if stops else None
    end_marker = api_end or session_stop
    observed_begin = api_begin["monotonic_ns"] if api_begin else (timed[0]["monotonic_ns"] if timed else None)
    observed_end = end_marker["monotonic_ns"] if end_marker else (timed[-1]["monotonic_ns"] if timed else None)
    observed_elapsed = ((observed_end - observed_begin) / 1e9
                        if observed_begin is not None and observed_end is not None else None)
    tool_union = union_s(intervals)
    by_category = defaultdict(lambda: {"count": 0, "summed_item_wall_s": 0.0})
    for item in completed:
        by_category[item["category"]]["count"] += 1
        by_category[item["category"]]["summed_item_wall_s"] += item["elapsed_s"]
    return {
        "session": name, "attempt": attempt, "timed_jsonl": str(path),
        "snapshot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "snapshot_last_event_utc": timed[-1].get("utc") if timed else None,
        "malformed_rows": malformed, "event_type_counts": dict(event_types),
        "api_begin_utc": api_begin["utc"] if api_begin else None,
        "api_end_utc": api_end["utc"] if api_end else None,
        "api_session_complete": api_end is not None,
        "observed_envelope_end_kind": (
            "agent_api_end" if api_end else "agent_session_stop" if session_stop else "last_stream_event"),
        "observed_envelope_s": observed_elapsed,
        "completed_tool_item_count": len(completed),
        "unfinished_tool_item_ids": sorted(starts),
        "completed_tool_union_wall_s": tool_union,
        "envelope_outside_completed_tool_intervals_s_unclassified": (
            max(0.0, observed_elapsed - tool_union) if observed_elapsed is not None else None),
        "category_sums_may_overlap": dict(by_category),
        "completed_items": sorted(completed, key=lambda item: item["monotonic_start_ns"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--session", action="append", required=True,
                    help="name=path/to/agent_attempt_XX.timed.jsonl")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    local_events, bad_events = rows(args.events)
    sessions = []
    for spec in args.session:
        name, separator, raw_path = spec.partition("=")
        if not separator or not name or not raw_path:
            ap.error("--session must be name=path")
        sessions.append(summarize(name, Path(raw_path), local_events))
    result = {
        "schema_version": 1, "local_event_rows": len(local_events),
        "local_event_malformed_rows": bad_events, "sessions": sessions,
        "method": "local_host_monotonic_Codex_stream_item_boundaries_plus_broker_API_events",
        "limitations": [
            "A non-tool stream interval is unclassified: it can contain API wait, model work, reasoning, event buffering or invisible operations; it is not measured thinking time.",
            "Concurrent tool item durations overlap. Category sums are not additive wall time; completed_tool_union_wall_s is their merged envelope.",
            "While a session is running, observed_envelope_s ends at the last stream event and omits ongoing work.",
            "A command_execution interval measures host-visible tool activity including subprocess wait, not necessarily CPU busy time.",
            "The clean Master and Executor attempts were briefly SIGSTOP-paused during site transport repairs; those pauses remain in observed envelopes and unclassified intervals. See failed_preflight_seq0002_transport.md. Do not count them as API/model reasoning time.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"sessions": len(sessions), "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
