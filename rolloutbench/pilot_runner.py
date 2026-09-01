from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

from .events import EventLedger
from .quality_contract import DENSE_REFERENCE_ID
from .resume import ResumePlanError, plan_stage_resume
from .runtime_checkout import verify_runtime_receipt


class PilotRunnerError(RuntimeError):
    """Raised when a pilot run cannot advance from durable evidence."""


@dataclass(frozen=True)
class RunContext:
    plan_id: str
    plan_sha256: str
    run_sha256: str
    preparation_sha256: str
    run: dict[str, Any]
    preparation: dict[str, Any]
    plan_path: Path
    preparation_path: Path


@dataclass(frozen=True)
class Unit:
    unit_id: str
    episode_id: str
    component: str
    global_fifo_index: int
    depends_on: tuple[str, ...]
    worker_affinity: int | str
    unit_kind: str = "primary"
    historical_predecessors: tuple[str, ...] = ()
    quality_pair: dict[str, Any] | None = None
    preparation_episode_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DispatchSlot:
    start: float
    end: float
    worker_id: int
    worker_mode: str = "one_shot"


@dataclass(frozen=True)
class DispatchGrant:
    plan_id: str
    plan_sha256: str
    run_id: str
    run_sha256: str
    unit_id: str
    worker_id: int
    dispatch_policy: str
    dispatcher_id: str
    dispatch_index: int


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    wall_s: float
    stdout_path: Path
    stderr_path: Path
    stdout_sha256: str
    stdout_size_bytes: int
    stderr_sha256: str
    stderr_size_bytes: int


class StageExecutor(Protocol):
    def execute(self, invocation: Mapping[str, Any], *, log_dir: Path) -> ProcessResult: ...


_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+")
_QUALITY_KINDS = (
    "quality_dense_generate",
    "quality_candidate_generate",
    "quality_dense_vbench",
    "quality_candidate_vbench",
    "quality_lpips",
    "quality_compare",
)
_KIND_ORDER = {"primary": 0, **{kind: index + 1 for index, kind in enumerate(_QUALITY_KINDS)}}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _load_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotRunnerError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PilotRunnerError(f"{label} must be a JSON object")
    return value, raw


def _validate_receipts(preparation: Mapping[str, Any], episodes: list[Any]) -> None:
    experiment_root_value = preparation.get("experiment_root")
    derived_root_value = preparation.get("derived_root")
    if not isinstance(experiment_root_value, str) or not isinstance(derived_root_value, str):
        raise PilotRunnerError("preparation experiment/derived roots are missing")
    experiment_root = Path(experiment_root_value)
    derived_root = Path(derived_root_value)
    if not experiment_root.is_absolute() or not derived_root.is_absolute():
        raise PilotRunnerError("preparation experiment/derived roots must be absolute")
    try:
        derived_root.resolve(strict=False).relative_to(experiment_root.resolve(strict=False))
    except ValueError as exc:
        raise PilotRunnerError("preparation derived_root escapes experiment_root") from exc
    runtime_receipts = preparation.get("runtime_receipts")
    materialization_receipts = preparation.get("materialization_receipts")
    if not isinstance(runtime_receipts, Mapping) or not isinstance(materialization_receipts, Mapping):
        raise PilotRunnerError("preparation receipt mappings are missing")
    for value in episodes:
        if not isinstance(value, Mapping):
            raise PilotRunnerError("planned episode is malformed")
        episode_id = value.get("episode_id")
        runtime_contract = value.get("runtime_checkout")
        candidate = value.get("candidate")
        runtime = runtime_receipts.get(episode_id)
        material = materialization_receipts.get(episode_id)
        if not isinstance(runtime_contract, Mapping) or not isinstance(runtime, Mapping):
            raise PilotRunnerError(f"runtime receipt is missing for {episode_id}")
        expected_runtime = {
            "status": "READY",
            "runtime_ref": runtime_contract.get("git_ref"),
            "ref_role": runtime_contract.get("ref_role"),
            "runtime_tree_oid": runtime_contract.get("runtime_tree_oid"),
            "required_runtime_paths": runtime_contract.get("required_runtime_paths"),
        }
        if any(runtime.get(key) != expected for key, expected in expected_runtime.items()):
            raise PilotRunnerError(f"runtime receipt disagrees with episode {episode_id}")
        critical = runtime.get("critical_runtime_file_sha256")
        required_paths = runtime_contract.get("required_runtime_paths")
        if (
            not isinstance(runtime.get("worktree_path"), str)
            or not runtime["worktree_path"]
            or not isinstance(critical, Mapping)
            or not isinstance(required_paths, list)
            or set(critical) != set(required_paths)
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in critical.values())
        ):
            raise PilotRunnerError(f"runtime receipt is incomplete for {episode_id}")
        if not isinstance(candidate, Mapping) or not isinstance(material, Mapping):
            raise PilotRunnerError(f"materialization receipt is missing for {episode_id}")
        artifacts = material.get("artifacts")
        if (
            material.get("episode_id") != episode_id
            or material.get("authority_ref") != candidate.get("authority_ref")
            or not isinstance(artifacts, list)
            or not artifacts
        ):
            raise PilotRunnerError(f"materialization receipt disagrees with episode {episode_id}")
        expected_artifacts: list[dict[str, str]] = []
        config = candidate.get("config")
        probe = candidate.get("probe")
        if config is not None:
            if not isinstance(config, Mapping):
                raise PilotRunnerError(f"candidate config descriptor is malformed for {episode_id}")
            expected_artifacts.append(
                {
                    "kind": "config",
                    "source_path": str(config.get("path", "")),
                    "sha256": str(config.get("blob_sha256", "")),
                }
            )
        if probe is not None:
            source = probe.get("source") if isinstance(probe, Mapping) else None
            if not isinstance(source, Mapping):
                raise PilotRunnerError(f"candidate probe descriptor is malformed for {episode_id}")
            expected_artifacts.append(
                {
                    "kind": "probe_source",
                    "source_path": str(source.get("path", "")),
                    "sha256": str(source.get("blob_sha256", "")),
                }
            )
        if len(expected_artifacts) != 1 or len(artifacts) != 1:
            raise PilotRunnerError(
                f"materialization artifact inventory disagrees with episode {episode_id}"
            )
        for artifact in artifacts:
            if (
                not isinstance(artifact, Mapping)
                or not isinstance(artifact.get("relative_path"), str)
                or not isinstance(artifact.get("source_path"), str)
                or not isinstance(artifact.get("sha256"), str)
                or not isinstance(artifact.get("size_bytes"), int)
                or artifact["size_bytes"] < 0
            ):
                raise PilotRunnerError(f"materialization receipt is incomplete for {episode_id}")
            expected = expected_artifacts[0]
            expected_relative = (
                Path(str(episode_id)) / expected["kind"] / expected["source_path"]
            ).as_posix()
            if (
                artifact.get("kind") != expected["kind"]
                or artifact.get("source_path") != expected["source_path"]
                or artifact.get("relative_path") != expected_relative
                or artifact.get("sha256") != expected["sha256"]
                or not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"])
            ):
                raise PilotRunnerError(
                    f"materialization artifact descriptor disagrees with episode {episode_id}"
                )


def load_run_context(plan_path: Path | str, preparation_path: Path | str, run_id: str) -> RunContext:
    """Load one planned run only when preparation binds its exact plan and receipts."""

    plan_file = Path(plan_path).resolve()
    preparation_file = Path(preparation_path).resolve()
    plan, raw_plan = _load_object(plan_file, "experiment plan")
    preparation, raw_preparation = _load_object(preparation_file, "preparation receipt")
    plan_id = plan.get("plan_id")
    plan_sha256 = hashlib.sha256(raw_plan).hexdigest()
    if not isinstance(plan_id, str) or not _SAFE_COMPONENT.fullmatch(plan_id):
        raise PilotRunnerError("experiment plan has no safe plan_id")
    if preparation.get("status") != "READY" or preparation.get("plan_id") != plan_id:
        raise PilotRunnerError("preparation plan_id/status does not bind the plan")
    if preparation.get("plan_path") != str(plan_file):
        raise PilotRunnerError("preparation plan_path does not bind the plan")
    if preparation.get("plan_sha256") != plan_sha256:
        raise PilotRunnerError("preparation plan_sha256 does not bind the plan")
    runs = plan.get("runs")
    matches = [row for row in runs if isinstance(row, dict) and row.get("run_id") == run_id] if isinstance(runs, list) else []
    if len(matches) != 1:
        raise PilotRunnerError(f"run_id must identify exactly one planned run: {run_id}")
    run = matches[0]
    if not isinstance(run_id, str) or not _SAFE_COMPONENT.fullmatch(run_id):
        raise PilotRunnerError("planned run_id is unsafe")
    episodes = run.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise PilotRunnerError("planned run has no episodes")
    ids = [row.get("episode_id") for row in episodes if isinstance(row, Mapping)]
    if len(ids) != len(episodes) or any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids):
        raise PilotRunnerError("planned run episode IDs are invalid")
    dense_reference = run.get("quality_dense_reference")
    receipt_episodes = list(episodes)
    if dense_reference is not None:
        receipt_episodes.append(dense_reference)
    _validate_receipts(preparation, receipt_episodes)
    return RunContext(
        plan_id=plan_id,
        plan_sha256=plan_sha256,
        run_sha256=_object_sha256(run),
        preparation_sha256=hashlib.sha256(raw_preparation).hexdigest(),
        run=run,
        preparation=preparation,
        plan_path=plan_file,
        preparation_path=preparation_file,
    )


def _revalidate_context(context: RunContext) -> None:
    fresh = load_run_context(context.plan_path, context.preparation_path, str(context.run.get("run_id")))
    if (
        fresh.plan_id != context.plan_id
        or fresh.plan_sha256 != context.plan_sha256
        or fresh.run_sha256 != context.run_sha256
        or fresh.preparation_sha256 != context.preparation_sha256
        or _canonical(fresh.run) != _canonical(context.run)
        or _canonical(fresh.preparation) != _canonical(context.preparation)
    ):
        raise PilotRunnerError("run context files changed after loading")


def _run_namespace(context: RunContext, state_root: Path | str) -> Path:
    run_id = str(context.run.get("run_id"))
    if not _SAFE_COMPONENT.fullmatch(run_id):
        raise PilotRunnerError("planned run_id is unsafe")
    return Path(state_root).resolve() / "plans" / context.plan_id / context.plan_sha256 / run_id / context.run_sha256


def open_run_ledger(context: RunContext, state_root: Path | str) -> EventLedger:
    """Return the plan-hash and run-hash isolated ledger for this context."""

    _revalidate_context(context)
    return EventLedger(_run_namespace(context, state_root) / "events.jsonl")


def _validate_quality_pair(pair: Any, episode_id: str) -> dict[str, Any]:
    if not isinstance(pair, dict):
        raise PilotRunnerError("quality pair is malformed")
    required_strings = ("pair_id", "candidate_id", "dense_artifact_id", "candidate_artifact_id")
    if any(not isinstance(pair.get(key), str) or not pair[key] for key in required_strings):
        raise PilotRunnerError("quality pair is incomplete")
    if pair["candidate_id"] != episode_id:
        raise PilotRunnerError("quality pair candidate does not match its episode")
    if not isinstance(pair.get("metrics"), list) or not pair["metrics"] or any(not isinstance(item, str) or not item for item in pair["metrics"]):
        raise PilotRunnerError("quality pair metrics are incomplete")
    return pair


def expand_run_units(run: Mapping[str, Any]) -> list[Unit]:
    """Expand executable units, treating absent predecessors as frozen history."""

    episodes = run.get("episodes")
    if not isinstance(episodes, list) or any(not isinstance(row, Mapping) for row in episodes):
        raise PilotRunnerError("run episodes are missing or malformed")
    ordered = sorted(episodes, key=lambda row: row.get("global_fifo_index", -1))
    if any(type(row.get("global_fifo_index")) is not int for row in ordered):
        raise PilotRunnerError("planned episode is malformed")
    episode_ids = {row.get("episode_id") for row in ordered}
    if len(episode_ids) != len(ordered) or any(not isinstance(item, str) or not item for item in episode_ids):
        raise PilotRunnerError("planned episode IDs are invalid")

    dense_reference = run.get("quality_dense_reference")
    has_quality = any(row.get("quality_pairs") for row in ordered)
    if has_quality and (
        not isinstance(dense_reference, Mapping)
        or dense_reference.get("episode_id") != DENSE_REFERENCE_ID
        or dense_reference.get("candidate_type") != "dense_reference"
        or dense_reference.get("worker_contract", {}).get("effective_mode") != "one_shot"
    ):
        raise PilotRunnerError("formal quality units require the frozen dense reference")

    quality_by_episode: dict[str, list[dict[str, Any]]] = {}
    terminal_by_episode: dict[str, tuple[str, ...]] = {}
    for episode in ordered:
        episode_id = str(episode["episode_id"])
        pairs = episode.get("quality_pairs", [])
        if not isinstance(pairs, list):
            raise PilotRunnerError("quality_pairs must be a list")
        validated = [_validate_quality_pair(pair, episode_id) for pair in pairs]
        if len({pair["pair_id"] for pair in validated}) != len(validated):
            raise PilotRunnerError("duplicate quality pair ID")
        quality_by_episode[episode_id] = validated
        terminal_by_episode[episode_id] = (
            tuple(f"{episode_id}:quality:{pair['pair_id']}:compare" for pair in validated)
            if validated else (f"{episode_id}:primary",)
        )

    units: list[Unit] = []
    known: set[str] = set()
    dense_generation_by_artifact: dict[str, str] = {}
    for episode in ordered:
        if "golden" in episode:
            raise PilotRunnerError("runner accepts only public planned episodes")
        worker_contract = episode.get("worker_contract")
        if worker_contract is not None and (
            not isinstance(worker_contract, Mapping) or worker_contract.get("effective_mode") != "one_shot"
        ):
            raise PilotRunnerError("persistent workers are not enabled for the pilot runner")
        episode_id = str(episode["episode_id"])
        component = episode.get("component")
        dependencies = episode.get("depends_on", [])
        if not isinstance(component, str) or not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies):
            raise PilotRunnerError("planned episode is malformed")
        internal = tuple(item for item in dependencies if item in episode_ids)
        historical = tuple(item for item in dependencies if item not in episode_ids)
        historical_receipts = episode.get("historical_predecessor_receipts")
        if not isinstance(historical_receipts, list):
            raise PilotRunnerError(
                f"historical predecessor receipts are missing for {episode_id}"
            )
        receipt_ids: list[str] = []
        for receipt in historical_receipts:
            if (
                not isinstance(receipt, Mapping)
                or not isinstance(receipt.get("episode_id"), str)
                or not re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(receipt.get("public_episode_sha256", "")),
                )
                or receipt.get("disposition")
                != "frozen_history_not_replayed_in_this_scope"
            ):
                raise PilotRunnerError(
                    f"historical predecessor receipt is malformed for {episode_id}"
                )
            receipt_ids.append(str(receipt["episode_id"]))
        if tuple(receipt_ids) != historical or len(receipt_ids) != len(set(receipt_ids)):
            raise PilotRunnerError(
                f"historical predecessor receipts disagree with {episode_id} dependencies"
            )
        primary_dependencies = tuple(unit_id for item in internal for unit_id in terminal_by_episode[item])
        affinity = episode.get("worker_affinity", "dynamic")
        index = int(episode["global_fifo_index"])

        def add(unit: Unit) -> None:
            if unit.unit_id in known:
                raise PilotRunnerError(f"duplicate unit ID: {unit.unit_id}")
            known.add(unit.unit_id)
            units.append(unit)

        primary_id = f"{episode_id}:primary"
        # Keep every matched quality measurement for one candidate on the
        # candidate's primary worker.  Besides avoiding cross-GPU score noise,
        # this makes the full pair plan have one unambiguous CUDA UUID.  The
        # dense video may still be generated once on another worker and reused
        # by digest within this benchmark run.
        quality_affinity: int | str = (
            affinity if type(affinity) is int else f"lineage:{episode_id}"
        )
        add(Unit(
            primary_id, episode_id, component, index, primary_dependencies,
            affinity, "primary", historical, None, (episode_id,),
        ))
        for pair in quality_by_episode[episode_id]:
            prefix = f"{episode_id}:quality:{pair['pair_id']}"
            dense_artifact_id = pair["dense_artifact_id"]
            dense_generate = dense_generation_by_artifact.get(dense_artifact_id)
            if dense_generate is None:
                dense_key = hashlib.sha256(dense_artifact_id.encode("utf-8")).hexdigest()[:16]
                dense_generate = f"{DENSE_REFERENCE_ID}:quality:{dense_key}:dense_generate"
                dense_generation_by_artifact[dense_artifact_id] = dense_generate
                add(Unit(
                    dense_generate,
                    DENSE_REFERENCE_ID,
                    "cache",
                    index,
                    (),
                    dense_reference.get("worker_affinity", "dynamic"),
                    "quality_dense_generate",
                    (),
                    pair,
                    (DENSE_REFERENCE_ID,),
                ))
            candidate_generate = f"{prefix}:candidate_generate"
            dense_vbench = f"{prefix}:dense_vbench"
            candidate_vbench = f"{prefix}:candidate_vbench"
            lpips = f"{prefix}:lpips"
            add(Unit(
                candidate_generate, episode_id, component, index, (primary_id,),
                quality_affinity, "quality_candidate_generate", (), pair, (episode_id,),
            ))
            generation_dependencies = (dense_generate, candidate_generate)
            pair_receipts = (DENSE_REFERENCE_ID, episode_id)
            add(Unit(
                dense_vbench, episode_id, component, index, generation_dependencies,
                quality_affinity, "quality_dense_vbench", (), pair, pair_receipts,
            ))
            add(Unit(
                candidate_vbench, episode_id, component, index,
                generation_dependencies, quality_affinity, "quality_candidate_vbench", (),
                pair, pair_receipts,
            ))
            add(Unit(
                lpips, episode_id, component, index, generation_dependencies,
                quality_affinity, "quality_lpips", (), pair, pair_receipts,
            ))
            add(Unit(
                f"{prefix}:compare", episode_id, component, index,
                (dense_vbench, candidate_vbench, lpips), quality_affinity,
                "quality_compare", (),
                pair, pair_receipts,
            ))
    if any(dependency not in known for unit in units for dependency in unit.depends_on):
        raise PilotRunnerError("unit dependency graph refers to an unknown unit")
    return units


def _choose_worker(unit: Unit, workers: list[Mapping[str, Any]], available: list[float], assignments: Mapping[str, DispatchSlot]) -> int:
    affinity = unit.worker_affinity
    if type(affinity) is int:
        if affinity < 0 or affinity >= len(workers):
            raise PilotRunnerError(f"unit affinity is outside planned workers: {unit.unit_id}")
        return affinity
    if isinstance(affinity, str) and affinity.startswith("lineage:"):
        parent = f"{affinity.removeprefix('lineage:')}:primary"
        if parent not in assignments:
            raise PilotRunnerError(f"lineage worker is not assigned: {unit.unit_id}")
        return assignments[parent].worker_id
    if affinity != "dynamic":
        raise PilotRunnerError(f"unit worker affinity is invalid: {unit.unit_id}")
    return min(range(len(workers)), key=lambda index: (available[index], index))


def schedule_run_units(run: Mapping[str, Any], units: list[Unit]) -> dict[str, DispatchSlot]:
    """Build the deterministic policy schedule used as an execution constraint."""

    system, workers = run.get("system"), run.get("workers")
    if system not in {"serial1", "fifo2", "optroll1", "optroll2"} or not isinstance(workers, list) or not workers:
        raise PilotRunnerError("run system/workers are invalid")
    limit = 1 if system in {"serial1", "optroll1"} else 2
    if len(workers) != limit:
        raise PilotRunnerError("planned worker count does not match system")
    pending = {unit.unit_id: unit for unit in units}
    if len(pending) != len(units):
        raise PilotRunnerError("schedule contains duplicate units")
    assignments: dict[str, DispatchSlot] = {}
    available = [0.0] * len(workers)
    while pending:
        ready = [unit for unit in pending.values() if all(item in assignments for item in unit.depends_on)]
        if not ready:
            raise PilotRunnerError("unit dependency graph cannot be scheduled")
        if system in {"serial1", "fifo2"}:
            selected = min(ready, key=lambda unit: (unit.global_fifo_index, _KIND_ORDER[unit.unit_kind], unit.unit_id))
        elif system == "optroll2":
            selected = min(ready, key=lambda unit: (available[_choose_worker(unit, workers, available, assignments)], unit.global_fifo_index, _KIND_ORDER[unit.unit_kind], unit.unit_id))
        else:
            selected = min(ready, key=lambda unit: (unit.component, unit.global_fifo_index, _KIND_ORDER[unit.unit_kind], unit.unit_id))
        worker_id = _choose_worker(selected, workers, available, assignments)
        dependency_end = max((assignments[item].end for item in selected.depends_on), default=0.0)
        start = max(available[worker_id], dependency_end)
        assignments[selected.unit_id] = DispatchSlot(start, start + 1.0, worker_id)
        available[worker_id] = start + 1.0
        del pending[selected.unit_id]
    return assignments


class SubprocessStageExecutor:
    """One-shot executor that streams each subprocess pipe into a durable log."""

    def execute(self, invocation: Mapping[str, Any], *, log_dir: Path) -> ProcessResult:
        argv, cwd, environment = invocation.get("argv"), invocation.get("cwd"), invocation.get("env")
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
            raise PilotRunnerError("invocation argv is invalid")
        if not isinstance(cwd, str) or not isinstance(environment, Mapping):
            raise PilotRunnerError("invocation cwd/env is invalid")
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path, stderr_path = log_dir / "stdout.log", log_dir / "stderr.log"
        env = {str(key): str(value) for key, value in environment.items()}
        started = time.perf_counter()
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            assert process.stdout is not None and process.stderr is not None
            threads = [
                threading.Thread(target=shutil.copyfileobj, args=(process.stdout, stdout)),
                threading.Thread(target=shutil.copyfileobj, args=(process.stderr, stderr)),
            ]
            for thread in threads: thread.start()
            returncode = process.wait()
            for thread in threads: thread.join()
            process.stdout.close(); process.stderr.close()
            stdout.flush(); os.fsync(stdout.fileno()); stderr.flush(); os.fsync(stderr.fileno())
        stdout_hash, stdout_size = _sha256(stdout_path)
        stderr_hash, stderr_size = _sha256(stderr_path)
        return ProcessResult(returncode, time.perf_counter() - started, stdout_path, stderr_path, stdout_hash, stdout_size, stderr_hash, stderr_size)


def _unit_stage(unit: Unit) -> str:
    return unit.unit_kind


def _identity(context: RunContext) -> dict[str, Any]:
    return {
        "plan_id": context.plan_id,
        "plan_sha256": context.plan_sha256,
        "run_id": context.run["run_id"],
        "run_sha256": context.run_sha256,
        "preparation_sha256": context.preparation_sha256,
    }


def _event_stage_state(ledger: EventLedger, unit: Unit, attempt: int) -> str | None:
    current = None
    for event in ledger.read():
        payload = event["payload"]
        if payload.get("episode_id") == unit.unit_id and payload.get("stage") == _unit_stage(unit) and payload.get("attempt") == attempt and event["event_type"].startswith("stage_"):
            current = event["event_type"].removeprefix("stage_")
    return current


def _append_transition(context: RunContext, ledger: EventLedger, unit: Unit, event_type: str, attempt: int, payload: dict[str, Any]) -> None:
    current = _event_stage_state(ledger, unit, attempt)
    previous = _event_stage_state(ledger, unit, attempt - 1) if attempt > 1 else None
    allowed = {
        "stage_queued": current is None and (attempt == 1 or previous in {"queued", "failed", "interrupted", "started"}),
        "stage_started": current == "queued",
        "stage_completed": current == "started",
        "stage_failed": current == "started",
        "stage_interrupted": current == "started",
    }
    if event_type not in allowed or not allowed[event_type]:
        raise PilotRunnerError(f"illegal stage transition {event_type} for {unit.unit_id} attempt {attempt}")
    ledger.append(
        event_type,
        {
            **payload,
            "episode_id": unit.unit_id,
            "stage": _unit_stage(unit),
            "attempt": attempt,
            "parent_episode_id": unit.episode_id,
            "unit_kind": unit.unit_kind,
            **_identity(context),
        },
        idempotency_key=f"unit:{unit.unit_id}:{attempt}:{event_type}",
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name); handle.write(content); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path); temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def _assert_ledger(context: RunContext, ledger: EventLedger, state_root: Path | str) -> Path:
    namespace = _run_namespace(context, state_root)
    if ledger.path.resolve() != (namespace / "events.jsonl").resolve():
        raise PilotRunnerError("ledger path does not bind the run context")
    return namespace


def _planned_episode(context: RunContext, episode_id: str) -> Mapping[str, Any]:
    rows = list(context.run["episodes"])
    dense = context.run.get("quality_dense_reference")
    if isinstance(dense, Mapping):
        rows.append(dense)
    matches = [row for row in rows if row.get("episode_id") == episode_id]
    if len(matches) != 1:
        raise PilotRunnerError("live verification cannot resolve the planned episode")
    return matches[0]


def _receipt_bindings(context: RunContext, unit: Unit) -> dict[str, str]:
    episode_ids = unit.preparation_episode_ids or (unit.episode_id,)
    if len(episode_ids) != len(set(episode_ids)):
        raise PilotRunnerError("unit preparation receipt bindings contain duplicates")
    static_runtime: dict[str, Any] = {}
    static_materialization: dict[str, Any] = {}
    live_runtime: dict[str, Any] = {}
    live_materialization: dict[str, Any] = {}
    for episode_id in episode_ids:
        episode = _planned_episode(context, episode_id)
        try:
            runtime_receipt = context.preparation["runtime_receipts"][episode_id]
            materialization_receipt = context.preparation["materialization_receipts"][episode_id]
            verified_runtime = verify_runtime_receipt(
                Path(__file__).resolve().parents[1],
                runtime_receipt,
                episode["runtime_checkout"],
            )
        except (KeyError, RuntimeError, ValueError, OSError) as exc:
            raise PilotRunnerError("runtime receipt live verification failed") from exc
        static_runtime[episode_id] = runtime_receipt
        static_materialization[episode_id] = materialization_receipt
        live_runtime[episode_id] = verified_runtime
        live_materialization[episode_id] = _verify_materialized_artifacts(
            context, episode_id
        )
    return {
        "runtime_receipt_sha256": _object_sha256(static_runtime),
        "materialization_receipt_sha256": _object_sha256(static_materialization),
        "runtime_live_verification_sha256": _object_sha256(live_runtime),
        "materialization_live_verification_sha256": _object_sha256(live_materialization),
    }


def _assert_no_symlink_chain(path: Path, *, label: str) -> None:
    absolute = path.absolute()
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise PilotRunnerError(f"{label} contains a symlink: {cursor}")


def _verify_materialized_artifacts(context: RunContext, episode_id: str) -> dict[str, Any]:
    derived_root = Path(str(context.preparation.get("derived_root", "")))
    if not derived_root.is_absolute():
        raise PilotRunnerError("preparation derived_root is invalid")
    _assert_no_symlink_chain(derived_root, label="preparation derived_root")
    derived_resolved = derived_root.resolve(strict=False)
    receipt = context.preparation["materialization_receipts"][episode_id]
    verified: list[dict[str, Any]] = []
    for row in receipt["artifacts"]:
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise PilotRunnerError("materialized artifact path is unsafe")
        target = derived_root / relative
        _assert_no_symlink_chain(target, label="materialized artifact path")
        try:
            target.resolve(strict=False).relative_to(derived_resolved)
        except ValueError as exc:
            raise PilotRunnerError("materialized artifact escapes derived_root") from exc
        if not target.is_file() or target.is_symlink():
            raise PilotRunnerError("materialized artifact is missing or not a regular file")
        digest, size = _sha256(target)
        if digest != row.get("sha256") or size != row.get("size_bytes"):
            raise PilotRunnerError("materialized artifact digest or size is stale")
        verified.append(
            {
                "kind": row.get("kind"),
                "relative_path": relative.as_posix(),
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return {
        "episode_id": episode_id,
        "authority_ref": receipt["authority_ref"],
        "derived_root": str(derived_resolved),
        "artifacts": verified,
    }


def _validate_output_path(context: RunContext, ledger: EventLedger, value: Any) -> Path:
    if not isinstance(value, str):
        raise PilotRunnerError("invocation output_path must be absolute")
    output = Path(value)
    if not output.is_absolute() or ".." in output.parts:
        raise PilotRunnerError("invocation output_path must be an absolute normalized path")
    experiment_root = Path(str(context.preparation.get("experiment_root", "")))
    planned_root = (
        experiment_root
        / "runs"
        / context.plan_id
        / context.plan_sha256
        / str(context.run["run_id"])
        / context.run_sha256
    )
    state_namespace = ledger.path.parent
    for root in (planned_root, state_namespace):
        _assert_no_symlink_chain(root, label="allowed output root")
    _assert_no_symlink_chain(output, label="invocation output path")
    resolved = output.resolve(strict=False)
    allowed = False
    for root in (planned_root, state_namespace):
        try:
            resolved.relative_to(root.resolve(strict=False))
            allowed = True
            break
        except ValueError:
            continue
    if not allowed:
        raise PilotRunnerError("invocation output_path is outside allowed run roots")
    return output


def _bound_invocation(context: RunContext, ledger: EventLedger, unit: Unit, invocation: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    if invocation.get("unit_id") != unit.unit_id:
        raise PilotRunnerError("invocation unit_id does not match the planned unit")
    declared_kind = invocation.get("unit_kind")
    if declared_kind != unit.unit_kind:
        raise PilotRunnerError("invocation unit_kind does not match the planned unit")
    if invocation.get("episode_id") != unit.episode_id:
        raise PilotRunnerError("invocation episode_id does not match the planned unit")
    if invocation.get("run_id") != context.run["run_id"]:
        raise PilotRunnerError("invocation run_id does not match the planned run")
    if invocation.get("preparation_episode_ids") != list(
        unit.preparation_episode_ids or (unit.episode_id,)
    ):
        raise PilotRunnerError(
            "invocation preparation_episode_ids do not match the planned unit"
        )
    pair_id = unit.quality_pair["pair_id"] if unit.quality_pair else None
    if invocation.get("quality_pair_id") != pair_id:
        raise PilotRunnerError("invocation quality_pair_id does not match the planned unit")
    _validate_output_path(context, ledger, invocation.get("output_path"))
    bindings = {**_identity(context), **_receipt_bindings(context, unit)}
    bound = {"unit_id": unit.unit_id, "unit_kind": unit.unit_kind, "quality_pair_id": pair_id, "invocation": dict(invocation), "context": bindings}
    return _object_sha256(bound), bindings


def _completion_event(ledger: EventLedger, unit: Unit, attempt: int) -> dict[str, Any]:
    matches = [row for row in ledger.read() if row["event_type"] == "stage_completed" and row["payload"].get("episode_id") == unit.unit_id and row["payload"].get("stage") == _unit_stage(unit) and row["payload"].get("attempt") == attempt]
    if len(matches) != 1:
        raise PilotRunnerError("completed unit has no unique completion event")
    return matches[0]


def _verify_file_evidence(row: Mapping[str, Any], label: str) -> None:
    path = Path(str(row.get("path", "")))
    if not path.is_file() or path.is_symlink():
        raise PilotRunnerError(f"completed unit {label} is missing")
    digest, size = _sha256(path)
    if digest != row.get("sha256") or size != row.get("size_bytes"):
        raise PilotRunnerError(f"completed unit {label} digest or size disagrees with ledger")


def _verify_completion(context: RunContext, ledger: EventLedger, unit: Unit, attempt: int, expected_invocation_sha256: str | None = None) -> dict[str, Any]:
    payload = _completion_event(ledger, unit, attempt)["payload"]
    if any(payload.get(key) != value for key, value in _identity(context).items()) or payload.get("unit_kind") != unit.unit_kind:
        raise PilotRunnerError("completed unit context binding disagrees with the current plan/run")
    if expected_invocation_sha256 is not None and payload.get("invocation_sha256") != expected_invocation_sha256:
        raise PilotRunnerError("completed unit invocation disagrees with the requested invocation")
    _validate_output_path(context, ledger, payload.get("output_path"))
    output = {"path": payload.get("output_path"), "sha256": payload.get("output_sha256"), "size_bytes": payload.get("output_size_bytes")}
    _verify_file_evidence(output, "output")
    logs = payload.get("logs")
    if not isinstance(logs, Mapping):
        raise PilotRunnerError("completed unit log evidence is missing")
    for stream in ("stdout", "stderr"):
        row = logs.get(stream)
        if not isinstance(row, Mapping):
            raise PilotRunnerError("completed unit log evidence is incomplete")
        try:
            Path(str(row.get("path", ""))).resolve().relative_to(
                (ledger.path.parent / "logs").resolve()
            )
        except ValueError as exc:
            raise PilotRunnerError("completed unit log evidence escapes its run namespace") from exc
        _verify_file_evidence(row, f"{stream} log")
    receipt_path = Path(str(payload.get("execution_receipt_path", "")))
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise PilotRunnerError("completed unit execution receipt is missing")
    try:
        receipt_path.resolve().relative_to(ledger.path.parent.resolve())
    except ValueError as exc:
        raise PilotRunnerError("completed unit execution receipt escapes its run namespace") from exc
    receipt_sha, receipt_size = _sha256(receipt_path)
    if receipt_sha != payload.get("execution_receipt_sha256") or receipt_size != payload.get("execution_receipt_size_bytes"):
        raise PilotRunnerError("completed unit execution receipt digest or size disagrees with ledger")
    try: receipt = json.loads(receipt_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc: raise PilotRunnerError("completed unit execution receipt is invalid") from exc
    if not isinstance(receipt, dict) or any(receipt.get(key) != payload.get(key) for key in receipt):
        raise PilotRunnerError("completed unit execution receipt disagrees with the ledger")
    expected_bindings = _receipt_bindings(context, unit)
    if any(payload.get(key) != value for key, value in expected_bindings.items()):
        raise PilotRunnerError("completed unit preparation receipt binding is stale")
    return payload


def resume_unit(context: RunContext, ledger: EventLedger, unit: Unit, invocation: Mapping[str, Any]) -> dict[str, Any]:
    _revalidate_context(context)
    invocation_sha256, _ = _bound_invocation(context, ledger, unit, invocation)
    try:
        action = plan_stage_resume(ledger.reconstruct(), episode_id=unit.unit_id, stage=_unit_stage(unit))
    except ResumePlanError as exc:
        raise PilotRunnerError(str(exc)) from exc
    if action["action"] == "reuse_completed":
        _verify_completion(context, ledger, unit, int(action["attempt"]), invocation_sha256)
    return action


def _unit_completed(context: RunContext, ledger: EventLedger, unit: Unit) -> bool:
    try: action = plan_stage_resume(ledger.reconstruct(), episode_id=unit.unit_id, stage=_unit_stage(unit))
    except ResumePlanError as exc: raise PilotRunnerError(str(exc)) from exc
    if action["action"] != "reuse_completed": return False
    _verify_completion(context, ledger, unit, int(action["attempt"]))
    return True


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_execution_policy(
    context: RunContext,
    ledger: EventLedger,
    unit: Unit,
    worker_id: int,
    dispatch_grant: DispatchGrant | None = None,
) -> tuple[dict[str, Unit], DispatchSlot, list[Unit]]:
    units = expand_run_units(context.run)
    by_id = {item.unit_id: item for item in units}
    if by_id.get(unit.unit_id) != unit:
        raise PilotRunnerError("unit does not exactly match the frozen run plan")
    if dispatch_grant is not None:
        expected_policy = {
            "serial1": "global_fifo_one_shot",
            "fifo2": "global_fifo_two_workers_dependency_aware",
            "optroll1": "typed_validation_decision_aware_one_worker",
            "optroll2": "typed_streams_kernel_cache_one_worker_each",
        }.get(str(context.run.get("system")))
        expected_grant = {
            "plan_id": context.plan_id,
            "plan_sha256": context.plan_sha256,
            "run_id": str(context.run["run_id"]),
            "run_sha256": context.run_sha256,
            "unit_id": unit.unit_id,
            "worker_id": worker_id,
            "dispatch_policy": expected_policy,
        }
        if (
            expected_policy is None
            or context.run.get("dispatch_policy") != expected_policy
            or any(
                getattr(dispatch_grant, key) != value
                for key, value in expected_grant.items()
            )
            or not _SAFE_COMPONENT.fullmatch(dispatch_grant.dispatcher_id)
            or type(dispatch_grant.dispatch_index) is not int
            or dispatch_grant.dispatch_index < 0
        ):
            raise PilotRunnerError("runtime dispatch grant does not bind the frozen run")
        workers = context.run.get("workers")
        if (
            not isinstance(workers, list)
            or worker_id not in {
                row.get("worker_id") for row in workers if isinstance(row, Mapping)
            }
        ):
            raise PilotRunnerError("runtime dispatch worker is outside the frozen run")
        affinity = unit.worker_affinity
        if type(affinity) is int and affinity != worker_id:
            raise PilotRunnerError("runtime dispatch violates exact worker affinity")
        if isinstance(affinity, str) and affinity.startswith("lineage:"):
            parent_id = f"{affinity.removeprefix('lineage:')}:primary"
            parent = by_id.get(parent_id)
            if parent is None or not _unit_completed(context, ledger, parent):
                raise PilotRunnerError("runtime dispatch lineage parent is incomplete")
            parent_action = plan_stage_resume(
                ledger.reconstruct(),
                episode_id=parent.unit_id,
                stage=_unit_stage(parent),
            )
            parent_payload = _completion_event(
                ledger, parent, int(parent_action["attempt"])
            )["payload"]
            if parent_payload.get("worker_id") != worker_id:
                raise PilotRunnerError("runtime dispatch violates lineage worker affinity")
        elif affinity != "dynamic" and type(affinity) is not int:
            raise PilotRunnerError("runtime dispatch worker affinity is invalid")
        prerequisites = [by_id[item] for item in unit.depends_on]
        missing = [
            item.unit_id
            for item in prerequisites
            if not _unit_completed(context, ledger, item)
        ]
        if missing:
            raise PilotRunnerError(
                "runtime dispatch dependency is incomplete: "
                + ", ".join(sorted(missing))
            )
        return (
            by_id,
            DispatchSlot(
                float(dispatch_grant.dispatch_index),
                float(dispatch_grant.dispatch_index + 1),
                worker_id,
            ),
            prerequisites,
        )
    assignments = schedule_run_units(context.run, units)
    slot = assignments[unit.unit_id]
    if type(worker_id) is not int or worker_id != slot.worker_id:
        raise PilotRunnerError("worker assignment does not match the frozen schedule")
    required = [by_id[item] for item in unit.depends_on]
    same_worker_prior = [
        by_id[item_id] for item_id, other in assignments.items()
        if other.worker_id == worker_id and (other.start, other.end, item_id) < (slot.start, slot.end, unit.unit_id)
    ]
    prerequisites = {item.unit_id: item for item in required + same_worker_prior}
    missing = [item.unit_id for item in prerequisites.values() if not _unit_completed(context, ledger, item)]
    if missing:
        raise PilotRunnerError(f"unit dependency/worker-slot predecessor is incomplete: {', '.join(sorted(missing))}")
    return by_id, slot, list(prerequisites.values())


def _archive_preexisting_output(output: Path, unit: Unit, attempt: int) -> dict[str, Any] | None:
    if output.is_symlink():
        raise PilotRunnerError("invocation output_path must not be a symlink")
    if not output.exists(): return None
    if not output.is_file(): raise PilotRunnerError("invocation output_path is not a regular file")
    digest, size = _sha256(output)
    safe = hashlib.sha256(unit.unit_id.encode()).hexdigest()[:16]
    stale = output.parent / f".{output.name}.stale-{safe}-attempt-{attempt}"
    if stale.exists() or stale.is_symlink(): raise PilotRunnerError("stale output archive path already exists")
    os.replace(output, stale)
    directory_fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try: os.fsync(directory_fd)
    finally: os.close(directory_fd)
    return {"path": str(stale), "sha256": digest, "size_bytes": size}


def _logs(
    result: ProcessResult, expected_log_dir: Path
) -> dict[str, dict[str, Any]]:
    _assert_no_symlink_chain(expected_log_dir, label="unit attempt log directory")
    rows = {
        "stdout": {"path": str(result.stdout_path), "sha256": result.stdout_sha256, "size_bytes": result.stdout_size_bytes},
        "stderr": {"path": str(result.stderr_path), "sha256": result.stderr_sha256, "size_bytes": result.stderr_size_bytes},
    }
    for stream, row in rows.items():
        expected = (expected_log_dir / f"{stream}.log").resolve()
        actual = Path(str(row["path"]))
        if not actual.is_absolute() or actual.resolve() != expected:
            raise PilotRunnerError(
                f"completed unit {stream} log escapes its attempt namespace"
            )
        _verify_file_evidence(row, f"{stream} log")
    return rows


def execute_unit(context: RunContext, ledger: EventLedger, unit: Unit, invocation: Mapping[str, Any], executor: StageExecutor, state_root: Path | str, *, worker_id: int, dispatch_grant: DispatchGrant | None = None) -> dict[str, Any]:
    """Atomically claim and execute one frozen one-shot unit without a verdict."""

    _revalidate_context(context)
    namespace = _assert_ledger(context, ledger, state_root)
    invocation_sha256, receipt_bindings = _bound_invocation(context, ledger, unit, invocation)
    safe = hashlib.sha256(unit.unit_id.encode()).hexdigest()
    unit_lock = namespace / "locks" / "units" / f"{safe}.lock"
    with _exclusive_lock(unit_lock):
        resume = resume_unit(context, ledger, unit, invocation)
        if resume["action"] == "reuse_completed":
            return {"status": "REUSED_COMPLETED", "unit_id": unit.unit_id, **resume}
        _, slot, _ = _validate_execution_policy(
            context, ledger, unit, worker_id, dispatch_grant
        )
        planned_workers = context.run.get("workers")
        worker_rows = [
            row
            for row in planned_workers
            if isinstance(row, Mapping) and row.get("worker_id") == worker_id
        ] if isinstance(planned_workers, list) else []
        if len(worker_rows) != 1:
            raise PilotRunnerError("worker cannot be resolved uniquely in the frozen run")
        planned_gpu_uuid = str(worker_rows[0].get("gpu_uuid", ""))
        gpu_stage = unit.unit_kind != "quality_compare"
        if dispatch_grant is not None and gpu_stage:
            invocation_env = invocation.get("env")
            lease_file = Path(str(invocation.get("lease_file", "")))
            if (
                not planned_gpu_uuid.startswith("GPU-")
                or invocation.get("gpu_uuid") != planned_gpu_uuid
                or not isinstance(invocation_env, Mapping)
                or invocation_env.get("CUDA_VISIBLE_DEVICES") != planned_gpu_uuid
                or not lease_file.is_absolute()
            ):
                raise PilotRunnerError(
                    "runtime dispatch invocation does not bind the worker GPU UUID/lease"
                )
            gpu_key = hashlib.sha256(planned_gpu_uuid.encode()).hexdigest()
            worker_lock = Path(state_root).resolve() / "locks" / "gpus" / f"{gpu_key}.lock"
        else:
            worker_lock = namespace / "locks" / "workers" / f"worker-{worker_id}.lock"
        with _exclusive_lock(worker_lock):
            _validate_execution_policy(
                context, ledger, unit, worker_id, dispatch_grant
            )
            live_invocation_sha256, live_receipt_bindings = _bound_invocation(
                context, ledger, unit, invocation
            )
            if live_invocation_sha256 != invocation_sha256:
                raise PilotRunnerError("invocation/live receipt binding changed before execution")
            receipt_bindings = live_receipt_bindings
            attempt = int(resume["attempt"])
            output = _validate_output_path(context, ledger, invocation["output_path"])
            output.parent.mkdir(parents=True, exist_ok=True)
            stale = _archive_preexisting_output(output, unit, attempt)
            common = {
                "worker_mode": slot.worker_mode, "worker_id": worker_id,
                "worker_gpu_uuid": planned_gpu_uuid,
                "invocation_sha256": invocation_sha256, **receipt_bindings,
                "dispatch_grant": (
                    None
                    if dispatch_grant is None
                    else {
                        "dispatcher_id": dispatch_grant.dispatcher_id,
                        "dispatch_index": dispatch_grant.dispatch_index,
                        "dispatch_policy": dispatch_grant.dispatch_policy,
                    }
                ),
            }
            _append_transition(context, ledger, unit, "stage_queued", attempt, {**common, "archived_preexisting_output": stale})
            _append_transition(context, ledger, unit, "stage_started", attempt, common)
            try:
                attempt_log_dir = namespace / "logs" / safe / f"attempt-{attempt}"
                result = executor.execute(invocation, log_dir=attempt_log_dir)
                logs = _logs(result, attempt_log_dir)
            except Exception as exc:
                _append_transition(context, ledger, unit, "stage_failed", attempt, {**common, "reason": f"executor/evidence error: {type(exc).__name__}"})
                raise
            if result.returncode != 0:
                _append_transition(context, ledger, unit, "stage_failed", attempt, {**common, "returncode": result.returncode, "wall_s": result.wall_s, "logs": logs})
                return {"status": "FAILED", "unit_id": unit.unit_id, "attempt": attempt, "returncode": result.returncode, "logs": logs}
            try:
                output = _validate_output_path(context, ledger, invocation["output_path"])
            except PilotRunnerError as exc:
                _append_transition(context, ledger, unit, "stage_failed", attempt, {**common, "returncode": result.returncode, "wall_s": result.wall_s, "logs": logs, "reason": str(exc)})
                raise
            if not output.is_file() or output.is_symlink():
                reason = "successful subprocess did not produce a fresh regular output"
                _append_transition(context, ledger, unit, "stage_failed", attempt, {**common, "returncode": result.returncode, "wall_s": result.wall_s, "logs": logs, "reason": reason})
                return {"status": "FAILED", "unit_id": unit.unit_id, "attempt": attempt, "returncode": result.returncode, "logs": logs}
            output_sha256, output_size = _sha256(output)
            receipt = {
                "schema_version": 1, "unit_id": unit.unit_id, "unit_kind": unit.unit_kind,
                "parent_episode_id": unit.episode_id, "attempt": attempt,
                **_identity(context), **common, "returncode": result.returncode,
                "wall_s": result.wall_s, "output_path": str(output),
                "output_sha256": output_sha256, "output_size_bytes": output_size,
                "logs": logs, "archived_preexisting_output": stale,
            }
            receipt_path = namespace / "unit-receipts" / safe / f"attempt-{attempt}.json"
            receipt_content = _canonical(receipt) + b"\n"
            _atomic_write(receipt_path, receipt_content)
            receipt_sha, receipt_size = _sha256(receipt_path)
            _append_transition(context, ledger, unit, "stage_completed", attempt, {
                **receipt, "execution_receipt_path": str(receipt_path),
                "execution_receipt_sha256": receipt_sha,
                "execution_receipt_size_bytes": receipt_size,
            })
            return {
                "status": "EXECUTED", "unit_id": unit.unit_id, "attempt": attempt,
                "receipt_path": str(receipt_path), "performance_claim": False,
                "decision": "NOT_EMITTED",
            }
