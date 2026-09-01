from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .executor import FakeExecutor, FakeExecution


SYSTEMS = frozenset({"serial1", "fifo2", "optroll1", "optroll2"})
_STAGE_DURATIONS = {
    "acquire_gpu": 0.1,
    "preflight": 0.4,
    "generate": 5.0,
    "collect": 0.5,
    "exact_validate": 1.0,
    "legacy_sanity": 1.0,
    "quality_v1": 4.0,
    "microbenchmark": 1.0,
    "decide": 0.2,
}


@dataclass(frozen=True)
class SimulationResult:
    system: str
    trace: tuple[dict[str, Any], ...]
    decisions: tuple[dict[str, Any], ...]
    frontier: dict[str, str]
    summary: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "trace": list(self.trace),
            "decisions": list(self.decisions),
            "frontier": self.frontier,
            "summary": self.summary,
        }


def load_public_episodes(source: Path | str | Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(source, (str, Path)):
        rows = [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines()]
    else:
        rows = list(source)
    public: list[dict[str, Any]] = []
    for row in rows:
        episode = copy.deepcopy(row)
        episode.pop("golden", None)
        public.append(episode)
    return public


def _duration(execution: FakeExecution, *, reuse_hit: bool) -> float:
    total = sum(_STAGE_DURATIONS[stage] for stage in execution.stages)
    return round(total - (0.3 if reuse_hit and "preflight" in execution.stages else 0.0), 6)


def _assign_fifo(
    ordered: list[dict[str, Any]], executions: dict[str, FakeExecution], workers: int
) -> dict[str, tuple[float, float, int]]:
    available = [0.0] * workers
    completion: dict[str, float] = {}
    assignments: dict[str, tuple[float, float, int]] = {}
    last_dispatch = 0.0
    for episode in ordered:
        episode_id = episode["episode_id"]
        dependency_ready = max((completion[item] for item in episode["depends_on"]), default=0.0)
        worker = min(range(workers), key=lambda index: (available[index], index))
        start = max(available[worker], dependency_ready, last_dispatch)
        end = round(start + _duration(executions[episode_id], reuse_hit=False), 6)
        assignments[episode_id] = (start, end, worker)
        completion[episode_id] = end
        available[worker] = end
        last_dispatch = start
    return assignments


def _assign_optroll(
    episodes: list[dict[str, Any]], executions: dict[str, FakeExecution], workers: int
) -> dict[str, tuple[float, float, int]]:
    streams = {
        component: sorted(
            (episode for episode in episodes if episode["component"] == component),
            key=lambda row: row["round"],
        )
        for component in ("kernel", "cache")
    }
    assignments: dict[str, tuple[float, float, int]] = {}
    if workers == 2:
        for worker, component in enumerate(("kernel", "cache")):
            now = 0.0
            for episode in streams[component]:
                episode_id = episode["episode_id"]
                reuse_hit = episode_id == "K02"
                end = round(now + _duration(executions[episode_id], reuse_hit=reuse_hit), 6)
                assignments[episode_id] = (now, end, worker)
                now = end
        return assignments

    positions = {"kernel": 0, "cache": 0}
    stream_ready = {"kernel": 0.0, "cache": 0.0}
    now = 0.0
    while any(positions[name] < len(streams[name]) for name in streams):
        candidates: list[tuple[float, float, int, str, dict[str, Any]]] = []
        for component, rows in streams.items():
            position = positions[component]
            if position >= len(rows):
                continue
            episode = rows[position]
            execution = executions[episode["episode_id"]]
            candidates.append(
                (
                    max(now, stream_ready[component]),
                    _duration(execution, reuse_hit=episode["episode_id"] == "K02"),
                    episode["global_fifo_index"],
                    component,
                    episode,
                )
            )
        start, duration, _, component, episode = min(candidates)
        end = round(start + duration, 6)
        assignments[episode["episode_id"]] = (start, end, 0)
        positions[component] += 1
        stream_ready[component] = end
        now = end
    return assignments


def _max_concurrency(assignments: dict[str, tuple[float, float, int]]) -> int:
    points: list[tuple[float, int]] = []
    for start, end, _ in assignments.values():
        points.extend(((start, 1), (end, -1)))
    active = maximum = 0
    for _, delta in sorted(points, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def simulate(
    system: str,
    episodes: Iterable[dict[str, Any]],
    executor: FakeExecutor | None = None,
) -> SimulationResult:
    if system not in SYSTEMS:
        raise ValueError(f"unknown system: {system}")
    public = load_public_episodes(episodes)
    if any("golden" in episode for episode in public):
        raise ValueError("scheduler input contains a golden oracle")
    if len(public) != 35 or len({episode["episode_id"] for episode in public}) != 35:
        raise ValueError("v0 simulation requires 35 unique episodes")

    fake = executor or FakeExecutor()
    executions = {episode["episode_id"]: fake.execute(episode) for episode in public}
    fifo = sorted(public, key=lambda row: row["global_fifo_index"])
    if system == "serial1":
        assignments = _assign_fifo(fifo, executions, 1)
    elif system == "fifo2":
        assignments = _assign_fifo(fifo, executions, 2)
    elif system == "optroll1":
        assignments = _assign_optroll(public, executions, 1)
    else:
        assignments = _assign_optroll(public, executions, 2)

    by_id = {episode["episode_id"]: episode for episode in public}
    trace: list[dict[str, Any]] = []
    for episode in fifo:
        trace.append({"event": "episode_released", "episode_id": episode["episode_id"], "time": 0.0})

    order = sorted(
        public,
        key=lambda episode: (
            assignments[episode["episode_id"]][0], episode["global_fifo_index"]
        ),
    )
    decisions: list[dict[str, Any]] = []
    frontier: dict[str, str] = {}
    for episode in order:
        episode_id = episode["episode_id"]
        start, end, worker = assignments[episode_id]
        execution = executions[episode_id]
        worker_type = (
            "one_shot"
            if system == "serial1"
            else "global_fifo"
            if system == "fifo2"
            else f"typed_{episode['component']}"
        )
        trace.append(
            {
                "event": "episode_started",
                "episode_id": episode_id,
                "component": episode["component"],
                "time": start,
                "worker": worker,
                "worker_type": worker_type,
            }
        )
        reuse_hit = system.startswith("optroll") and episode_id == "K02"
        if reuse_hit:
            declared = episode["reuse"]["inputs"][0]
            trace.append(
                {
                    "event": "cache_hit",
                    "episode_id": episode_id,
                    "source_episode_id": declared["episode_id"],
                    "artifact": declared["artifact"],
                    "time": start,
                }
            )
        stage_total = _duration(execution, reuse_hit=reuse_hit)
        cursor = start
        raw_total = sum(_STAGE_DURATIONS[stage] for stage in execution.stages)
        for stage in execution.stages:
            stage_duration = _STAGE_DURATIONS[stage]
            if reuse_hit and stage == "preflight":
                stage_duration -= 0.3
            cursor = round(cursor + stage_duration, 6)
            trace.append(
                {
                    "event": "stage_completed",
                    "episode_id": episode_id,
                    "stage": stage,
                    "time": cursor,
                    "worker": worker,
                }
            )
        if round(cursor - start, 6) != stage_total or raw_total <= 0:
            raise AssertionError("stage timing contract drift")
        decision = {"episode_id": episode_id, **execution.decision}
        decisions.append(decision)
        trace.append({"event": "decision_sealed", **decision, "time": end})
        if execution.frontier_candidate:
            frontier[episode["component"]] = episode_id
            trace.append(
                {
                    "event": "frontier_updated",
                    "episode_id": episode_id,
                    "component": episode["component"],
                    "time": end,
                }
            )
        trace.append({"event": "episode_completed", "episode_id": episode_id, "time": end})

    trace.sort(
        key=lambda row: (
            row["time"],
            {"episode_released": 0, "episode_started": 1}.get(row["event"], 2),
            by_id.get(row.get("episode_id"), {"global_fifo_index": -1})["global_fifo_index"],
        )
    )
    sealed = [row for row in trace if row["event"] == "decision_sealed"]
    recomputed_frontier: dict[str, str] = {}
    for row in trace:
        if row["event"] == "frontier_updated":
            recomputed_frontier[row["component"]] = row["episode_id"]
    summary = {
        "synthetic_contract_simulation": True,
        "performance_claim": False,
        "released_episodes": sum(row["event"] == "episode_released" for row in trace),
        "sealed_decisions": len(sealed),
        "max_gpu_concurrency": _max_concurrency(assignments),
        "makespan_units": max(end for _, end, _ in assignments.values()),
        "declared_reuse_hits": sum(row["event"] == "cache_hit" for row in trace),
        "decision_agreement": {
            row["episode_id"]: {
                "outcome": row["outcome"],
                "contract": row["contract"],
            }
            for row in sealed
        }
        == {row["episode_id"]: {"outcome": row["outcome"], "contract": row["contract"]} for row in decisions},
        "decision_agreement_scope": "fake_executor_vs_schedule_trace",
        "frontier_agreement": recomputed_frontier == frontier,
        "frontier_agreement_scope": "fake_executor_vs_schedule_trace",
        "frontier_semantics": "fake_contract_only_not_historical",
        "historical_oracle_checked": False,
        "frontier": frontier,
    }
    return SimulationResult(system, tuple(trace), tuple(decisions), frontier, summary)
