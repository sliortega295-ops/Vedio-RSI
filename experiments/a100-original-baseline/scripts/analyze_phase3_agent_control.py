#!/usr/bin/env python3
"""Summarize measured local Codex/API and command windows without inferring thought time.

Both clocks are from the local host. Command windows are CLI stream events,
not CPU time; unclassified API time includes model wait, tool overhead, and
event-stream gaps, so it is deliberately not named "reasoning".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path("/home/lyy/Experiments/SolAgent/20260913-a100-original-baseline")
REPO = PROJECT / "repo"
SESSIONS = {
    "master": ("sana2ba100-master-0002",
               REPO / "output/orchestrated/sana2ba100-20260913-093521/master"),
    "cache": ("sana2ba100-cache_ca-0002",
              REPO / "output/experiments/sana2ba100-cache_ca-0002/worktree/goals/cache_ca"),
}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def category(command: str) -> str:
    try:
        argv = shlex.split(command)
    except ValueError:
        return "other_command"
    inner = argv[2] if len(argv) >= 3 and argv[:2] == ["/bin/bash", "-lc"] else command
    match = re.match(r"^(?:(?:[A-Z_][A-Z_0-9]*)=\S+\s+)*python(?:3)?\s+([^\s;]+)(?:\s|$)",
                     inner.strip())
    if not match:
        return "other_command"
    script = match.group(1).strip("'\"")
    if script.endswith("launch_config.py"):
        return "launch_config_submit_or_prepare"
    if script.endswith("verify_delivery.py"):
        return "native_verify_delivery"
    if script.endswith("resume_executor.py"):
        return "native_resume_executor"
    if script.endswith("poll_executor.py"):
        return "native_poll_executor"
    if script.endswith("spawn_executor.py"):
        return "native_spawn_executor"
    if script.endswith("site_remote.py") and " status " in inner:
        return "site_remote_status"
    if script.endswith("site_remote.py") and " sync " in inner:
        return "site_remote_sync"
    if script.endswith("plan_eval.py"):
        return "native_plan_eval"
    if script.endswith("collect_run.py"):
        return "native_collect_run"
    return "other_command"


def union_seconds(windows: list[tuple[int, int]]) -> float:
    ordered = sorted(windows)
    if not ordered:
        return 0.0
    total = 0
    start, end = ordered[0]
    for a, b in ordered[1:]:
        if a <= end:
            end = max(end, b)
        else:
            total += end - start
            start, end = a, b
    total += end - start
    return total / 1e9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", choices=SESSIONS, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    name, goal = SESSIONS[args.session]
    events_path = PROJECT / "telemetry/local_events.jsonl"
    stream_path = goal / f"agent_attempt_{args.attempt:02d}.timed.jsonl"
    matching = [row for row in jsonl(events_path)
                if row.get("name") == name and row.get("attempt") == args.attempt]
    begins = [row for row in matching if row.get("event") == "agent_api_begin"]
    ends = [row for row in matching if row.get("event") == "agent_api_end"]
    if len(begins) != 1 or len(ends) > 1:
        raise ValueError("agent API begin/end cardinality mismatch")
    begin = begins[0]
    end = ends[0] if ends else None
    asof = end["monotonic_ns"] if end else time.monotonic_ns()
    if asof < begin["monotonic_ns"]:
        raise ValueError("invalid local monotonic interval")
    starts: dict[str, tuple[int, dict]] = {}
    commands = []
    item_counts: dict[str, int] = defaultdict(int)
    non_json_stream_lines = 0
    for row in jsonl(stream_path):
        try:
            payload = json.loads(row["line"])
        except json.JSONDecodeError:
            non_json_stream_lines += 1
            continue
        if not isinstance(payload, dict):
            non_json_stream_lines += 1
            continue
        item = payload.get("item") or {}
        ident = item.get("id")
        item_type = item.get("type")
        kind = row.get("codex_type")
        if kind == "item.started" and ident:
            starts[ident] = (row["monotonic_ns"], item)
        elif kind == "item.completed" and ident:
            item_counts[item_type or "unknown"] += 1
            pair = starts.pop(ident, None)
            if item_type != "command_execution" or pair is None:
                continue
            begin_ns = pair[0]
            end_ns = row["monotonic_ns"]
            if end_ns < begin_ns:
                continue
            command = item.get("command") or pair[1].get("command") or ""
            commands.append({"item_id": ident, "category": category(command),
                             "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                             "start_monotonic_ns": begin_ns,
                             "end_monotonic_ns": end_ns,
                             "wall_s": (end_ns - begin_ns) / 1e9,
                             "exit_code": item.get("exit_code"),
                             "output_bytes": len((item.get("aggregated_output") or "").encode())})
    categories: dict[str, dict] = defaultdict(lambda: {"count": 0, "wall_s_sum": 0.0,
                                                 "output_bytes_sum": 0})
    clipped = []
    for command in commands:
        bucket = categories[command["category"]]
        bucket["count"] += 1
        bucket["wall_s_sum"] += command["wall_s"]
        bucket["output_bytes_sum"] += command["output_bytes"]
        a = max(begin["monotonic_ns"], command["start_monotonic_ns"])
        b = min(asof, command["end_monotonic_ns"])
        if b > a:
            clipped.append((a, b))
    api_wall = (asof - begin["monotonic_ns"]) / 1e9
    command_union = union_seconds(clipped)
    print(json.dumps({
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "session": name, "attempt": args.attempt,
        "source": {"events": str(events_path), "timed_stream": str(stream_path),
                   "events_sha256_at_read": hashlib.sha256(events_path.read_bytes()).hexdigest(),
                   "timed_stream_sha256_at_read": hashlib.sha256(stream_path.read_bytes()).hexdigest()},
        "api_begin_utc": begin["utc"], "api_end_utc": end["utc"] if end else None,
        "api_wall_s": api_wall, "api_status": "finished" if end else "open_as_of_snapshot",
        "transport_image_count": len(begin.get("transport_images") or []),
        "completed_item_type_counts": dict(item_counts),
        "non_json_stream_lines": non_json_stream_lines,
        "completed_command_count": len(commands),
        "completed_command_wall_s_sum_may_overlap": sum(c["wall_s"] for c in commands),
        "completed_command_wall_s_union": command_union,
        "api_wall_without_completed_command_events_s_not_thought_time": api_wall - command_union,
        "command_categories": dict(categories),
        "longest_commands": sorted(commands, key=lambda c: c["wall_s"], reverse=True)[:20],
        "unpaired_started_items": sorted(starts),
        "limitations": ["Codex stream item boundaries are local API/tool event timing, not CPU time.",
                        "Non-command API time cannot be uniquely attributed to reasoning or API wait.",
                        "Open attempts and in-progress commands are censored at snapshot time."]
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
