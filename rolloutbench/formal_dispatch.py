from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from .decisions import (
    DecisionError,
    collect_primary_evidence,
    finalize_run_decisions,
)
from .formal_runner import (
    FormalRunnerError,
    FormalStageExecutor,
    _completed_invocation,
    _completed_worker,
    build_formal_invocation,
    verify_formal_compare_output,
)
from .launch_gate import validate_launch_authorization
from .pilot_runner import (
    DispatchGrant,
    PilotRunnerError,
    StageExecutor,
    Unit,
    execute_unit,
    expand_run_units,
    open_run_ledger,
)
from .validators import evaluate_quality_candidate


class FormalDispatchError(RuntimeError):
    """Raised when the authorized completion-driven dispatcher cannot advance."""


_POLICIES = {
    "serial1": "global_fifo_one_shot",
    "fifo2": "global_fifo_two_workers_dependency_aware",
    "optroll1": "typed_validation_decision_aware_one_worker",
    "optroll2": "typed_streams_kernel_cache_one_worker_each",
}
_KIND_ORDER = {
    "primary": 0,
    "quality_dense_generate": 1,
    "quality_candidate_generate": 2,
    "quality_dense_vbench": 3,
    "quality_candidate_vbench": 4,
    "quality_lpips": 5,
    "quality_compare": 6,
}


@contextmanager
def _exclusive_dispatcher(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _priority(system: str, unit: Unit) -> tuple[Any, ...]:
    if system in {"serial1", "fifo2"}:
        return (
            unit.global_fifo_index,
            _KIND_ORDER[unit.unit_kind],
            unit.unit_id,
        )
    # OptRoll first unlocks the one shared dense artifact, then finishes
    # already-open validation chains before releasing another primary.
    phase = (
        0
        if unit.unit_kind == "quality_dense_generate"
        else 1
        if unit.unit_kind != "primary"
        else 2
    )
    return (
        phase,
        unit.global_fifo_index,
        _KIND_ORDER[unit.unit_kind],
        unit.unit_id,
    )


def _lineage_worker(
    context: Any,
    ledger: Any,
    units: Mapping[str, Unit],
    unit: Unit,
) -> int | None:
    affinity = unit.worker_affinity
    if type(affinity) is int:
        return affinity
    if affinity == "dynamic":
        return None
    if isinstance(affinity, str) and affinity.startswith("lineage:"):
        parent = units.get(f"{affinity.removeprefix('lineage:')}:primary")
        if parent is None:
            raise FormalDispatchError("lineage affinity names an unknown primary")
        return int(_completed_worker(context, ledger, parent)["worker_id"])
    raise FormalDispatchError("unit has an invalid worker affinity")


def _choose_dispatches(
    context: Any,
    ledger: Any,
    units: Mapping[str, Unit],
    pending: set[str],
    completed: set[str],
    available_workers: set[int],
) -> list[tuple[Unit, int]]:
    system = str(context.run.get("system"))
    ready = sorted(
        (
            units[unit_id]
            for unit_id in pending
            if set(units[unit_id].depends_on).issubset(completed)
        ),
        key=lambda unit: _priority(system, unit),
    )
    selected: list[tuple[Unit, int]] = []
    free = set(available_workers)
    if system in {"serial1", "fifo2"}:
        while ready and free:
            head = ready[0]
            affinity = _lineage_worker(context, ledger, units, head)
            eligible = sorted(
                worker for worker in free
                if affinity is None or worker == affinity
            )
            if not eligible:
                break
            worker_id = eligible[0]
            selected.append((head, worker_id))
            free.remove(worker_id)
            ready.pop(0)
        return selected

    for worker_id in sorted(free):
        eligible = [
            unit for unit in ready
            if (affinity := _lineage_worker(context, ledger, units, unit)) is None
            or affinity == worker_id
        ]
        if not eligible:
            continue
        chosen = eligible[0]
        selected.append((chosen, worker_id))
        ready.remove(chosen)
    return selected


def _verified_completed_units(
    context: Any,
    ledger: Any,
    units: Mapping[str, Unit],
    *,
    lease_files: Mapping[str, Path | str],
    profile: Mapping[str, Any],
    quality_protocol: Mapping[str, Any],
) -> set[str]:
    physically_completed = {
        str(row["payload"].get("episode_id"))
        for row in ledger.read()
        if row["event_type"] == "stage_completed"
    }
    completed: set[str] = set()
    for unit_id in sorted(physically_completed):
        unit = units.get(unit_id)
        if unit is None:
            raise FormalDispatchError("ledger contains a unit outside the frozen run")
        _completed_invocation(
            context,
            ledger,
            unit,
            lease_files=lease_files,
            profile=profile,
            quality_protocol=quality_protocol,
        )
        completed.add(unit_id)
    return completed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_same_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != content:
            raise FormalDispatchError(
                f"refusing to overwrite conflicting quality decision: {path}"
            )
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_bound_suite(
    profile: Mapping[str, Any], quality_protocol: Mapping[str, Any]
) -> dict[str, Any]:
    suite_path = Path(str(profile.get("suite_path", "")))
    quality_path = suite_path.parent / "quality_protocol.json"
    if (
        not suite_path.is_absolute()
        or not suite_path.is_file()
        or suite_path.is_symlink()
        or not quality_path.is_file()
        or quality_path.is_symlink()
    ):
        raise FormalDispatchError("formal suite/profile files are missing or unsafe")
    try:
        suite_raw = suite_path.read_bytes()
        quality_raw = quality_path.read_bytes()
        suite = json.loads(suite_raw)
        quality = json.loads(quality_raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FormalDispatchError("formal suite/profile files are invalid") from exc
    if (
        not isinstance(suite, dict)
        or not isinstance(quality, dict)
        or hashlib.sha256(suite_raw).hexdigest() != profile.get("suite_sha256")
        or hashlib.sha256(quality_raw).hexdigest()
        != profile.get("quality_protocol_sha256")
        or quality != dict(quality_protocol)
    ):
        raise FormalDispatchError("formal suite or quality protocol binding drifted")
    return suite


def _validate_planned_episode_scope(
    context: Any, profile: Mapping[str, Any], suite: Mapping[str, Any]
) -> list[str]:
    """Bind pilot/full execution to the exact frozen public episode set."""

    suite_path = Path(str(profile.get("suite_path", "")))
    episodes_path = suite_path.parent / "episodes.jsonl"
    expected_hash = (
        suite.get("file_hashes", {})
        .get("episodes.jsonl", {})
        .get("sha256")
    )
    if (
        not episodes_path.is_absolute()
        or not episodes_path.is_file()
        or episodes_path.is_symlink()
        or not isinstance(expected_hash, str)
        or _sha256(episodes_path) != expected_hash
    ):
        raise FormalDispatchError("formal public episode ledger binding drifted")
    try:
        rows = [
            json.loads(line)
            for line in episodes_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FormalDispatchError("formal public episode ledger is invalid") from exc
    public_ids = [
        str(row.get("episode_id", "")) for row in rows if isinstance(row, Mapping)
    ]
    if (
        len(public_ids) != len(rows)
        or len(public_ids) != 35
        or len(set(public_ids)) != 35
        or suite.get("counts", {}).get("total") != 35
    ):
        raise FormalDispatchError("formal public episode ledger is not exactly 35 unique rows")
    scope = context.run.get("scope")
    if scope == "full":
        expected_ids = public_ids
    elif scope == "pilot":
        pilot = suite.get("pilot_episodes")
        if not isinstance(pilot, list):
            raise FormalDispatchError("formal suite has no pilot episode set")
        expected_ids = [str(value) for value in pilot]
    else:
        raise FormalDispatchError("formal run scope is neither pilot nor full")
    planned = context.run.get("episodes")
    planned_ids = [
        str(row.get("episode_id", ""))
        for row in planned
        if isinstance(row, Mapping)
    ] if isinstance(planned, list) else []
    if (
        len(planned_ids) != len(planned or [])
        or len(planned_ids) != len(set(planned_ids))
        or planned_ids != expected_ids
    ):
        raise FormalDispatchError(
            f"{scope} run does not contain the exact frozen episode sequence"
        )
    return expected_ids


def finalize_quality_decisions(
    context: Any,
    ledger: Any,
    units: Mapping[str, Unit],
    quality_protocol: Mapping[str, Any],
    *,
    lease_files: Mapping[str, Path | str],
    profile: Mapping[str, Any],
    primary_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate eight pair receipts and seal one decision per formal candidate."""

    episodes = {
        str(row.get("episode_id")): row
        for row in context.run.get("episodes", [])
        if isinstance(row, Mapping) and row.get("quality_pairs")
    }
    decisions: list[dict[str, Any]] = []
    events = ledger.read()
    for candidate_id, episode in sorted(episodes.items()):
        pairs = episode.get("quality_pairs")
        if not isinstance(pairs, list) or len(pairs) != 8:
            raise FormalDispatchError(
                f"formal candidate {candidate_id} does not have eight pairs"
            )
        score_rows: list[Mapping[str, Any]] = []
        lpips_values: dict[str, list[float]] = {}
        evidence: dict[str, dict[str, Any]] = {}
        for pair in pairs:
            pair_id = str(pair.get("pair_id"))
            unit_id = f"{candidate_id}:quality:{pair_id}:compare"
            unit = units.get(unit_id)
            if unit is None or unit.unit_kind != "quality_compare":
                raise FormalDispatchError("formal compare unit is absent from the run")
            matches = [
                row for row in events
                if row["event_type"] == "stage_completed"
                and row["payload"].get("episode_id") == unit_id
                and row["payload"].get("stage") == "quality_compare"
            ]
            if len(matches) != 1:
                raise FormalDispatchError("formal compare completion is ambiguous")
            completion = matches[0]["payload"]
            try:
                _completed_invocation(
                    context,
                    ledger,
                    unit,
                    lease_files=lease_files,
                    profile=profile,
                    quality_protocol=quality_protocol,
                )
            except (FormalRunnerError, PilotRunnerError) as exc:
                raise FormalDispatchError(
                    "formal compare completion receipt or invocation is stale"
                ) from exc
            path = Path(str(completion.get("output_path", "")))
            if not path.is_file() or path.is_symlink():
                raise FormalDispatchError("formal compare output is missing or unsafe")
            digest = _sha256(path)
            if digest != completion.get("output_sha256"):
                raise FormalDispatchError("formal compare output digest changed")
            try:
                payload = verify_formal_compare_output(
                    path, pair, quality_protocol
                )
            except FormalRunnerError as exc:
                raise FormalDispatchError(
                    "formal compare evidence chain failed replay verification"
                ) from exc
            rows = payload.get("score_rows")
            lpips = payload.get("lpips")
            values = lpips.get("values") if isinstance(lpips, Mapping) else None
            if (
                payload.get("status") != "PARSED"
                or payload.get("pair_id") != pair_id
                or not isinstance(rows, list)
                or not rows
                or not isinstance(values, list)
                or len(values) != 81
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    or value < 0
                    for value in values
                )
            ):
                raise FormalDispatchError("formal compare output is incomplete")
            score_rows.extend(rows)
            lpips_values[pair_id] = values
            evidence[pair_id] = {
                "path": str(path),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        result = evaluate_quality_candidate(
            quality_protocol,
            pairs,
            score_rows,
            lpips_frame_values=lpips_values,
        )
        if result.get("status") not in {"PASS", "FAIL"}:
            raise FormalDispatchError(
                f"quality decision for {candidate_id} failed closed"
            )
        primary = (
            primary_evidence.get(candidate_id)
            if primary_evidence is not None
            else None
        )
        if primary_evidence is not None and not isinstance(primary, Mapping):
            raise FormalDispatchError(
                f"primary evidence for {candidate_id} is missing"
            )
        receipt = {
            "schema_version": 1,
            "record_type": "formal_quality_candidate_decision",
            "plan_id": context.plan_id,
            "plan_sha256": context.plan_sha256,
            "run_id": context.run["run_id"],
            "run_sha256": context.run_sha256,
            "candidate_id": candidate_id,
            "pair_evidence": evidence,
            "primary_evidence": dict(primary) if isinstance(primary, Mapping) else None,
            "result": result,
            "performance_claim": False,
        }
        content = (
            json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        path = (
            Path(context.preparation["experiment_root"])
            / "runs"
            / context.plan_id
            / context.plan_sha256
            / str(context.run["run_id"])
            / context.run_sha256
            / "quality-decisions"
            / f"{candidate_id}.json"
        )
        _write_same_atomic(path, content)
        decision = {
            "outcome": "quality_pass" if result["pass"] else "quality_rejected",
            "component": "cache",
            "frontier_eligible": bool(result["pass"]),
            "contract": str(quality_protocol.get("protocol_id")),
            "receipt_path": str(path),
            "receipt_sha256": _sha256(path),
            "quality_result": result,
            "measured_generation_s": (
                primary.get("generation_s") if isinstance(primary, Mapping) else None
            ),
            "process_wall_s": (
                primary.get("process_wall_s") if isinstance(primary, Mapping) else None
            ),
            "ranking_latency_s": (
                primary.get("ranking_latency_s") if isinstance(primary, Mapping) else None
            ),
            "ranking_latency_contract": (
                primary.get("ranking_latency_contract")
                if isinstance(primary, Mapping)
                else None
            ),
            "primary_evidence_path": (
                primary.get("receipt_path") if isinstance(primary, Mapping) else None
            ),
            "primary_evidence_sha256": (
                primary.get("receipt_sha256") if isinstance(primary, Mapping) else None
            ),
            "decision_semantics": "fresh_evidence_v1_not_historical_oracle",
        }
        ledger.seal_decision(candidate_id, decision)
        decisions.append({"episode_id": candidate_id, "decision": decision})
    return decisions


def dispatch_formal_run(
    context: Any,
    executor: StageExecutor,
    state_root: Path | str,
    *,
    authorization_path: Path | str,
    lease_files: Mapping[str, Path | str],
    profile: Mapping[str, Any],
    quality_protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Run one authorized plan with an actual completion-driven ready queue."""

    system = str(context.run.get("system"))
    policy = _POLICIES.get(system)
    workers = context.run.get("workers")
    if (
        policy is None
        or context.run.get("dispatch_policy") != policy
        or not isinstance(workers, list)
        or len(workers) not in {1, 2}
        or [worker.get("worker_id") for worker in workers] != list(range(len(workers)))
    ):
        raise FormalDispatchError("formal run worker/policy contract is invalid")
    suite = _load_bound_suite(profile, quality_protocol)
    expected_episode_ids = _validate_planned_episode_scope(context, profile, suite)
    ledger = open_run_ledger(context, state_root)
    dispatcher_lock = ledger.path.parent / "locks" / "formal-dispatcher.lock"
    with _exclusive_dispatcher(dispatcher_lock):
        authorization = validate_launch_authorization(
            context,
            authorization_path,
            lease_files,
            preflight_spec=profile,
        )
        unit_rows = expand_run_units(context.run)
        units = {unit.unit_id: unit for unit in unit_rows}
        completed = _verified_completed_units(
            context,
            ledger,
            units,
            lease_files=lease_files,
            profile=profile,
            quality_protocol=quality_protocol,
        )
        pending = set(units) - completed
        dispatcher_id = f"dispatcher-{uuid.uuid4()}"
        ledger.append(
            "run_started",
            {
                "plan_id": context.plan_id,
                "plan_sha256": context.plan_sha256,
                "run_id": context.run["run_id"],
                "run_sha256": context.run_sha256,
                "dispatcher_id": dispatcher_id,
                "dispatch_policy": policy,
                "authorization_sha256": authorization["authorization_sha256"],
                "resumed_completed_unit_count": len(completed),
                "performance_claim": False,
            },
        )
        controlled = FormalStageExecutor(executor)
        outcomes: list[dict[str, Any]] = []
        running: dict[Future[dict[str, Any]], tuple[str, int]] = {}
        available = set(range(len(workers)))
        dispatch_index = 0

        def run_one(unit: Unit, worker_id: int, grant: DispatchGrant) -> dict[str, Any]:
            invocation = build_formal_invocation(
                context,
                unit,
                workers[worker_id],
                lease_files=lease_files,
                profile=profile,
                quality_protocol=quality_protocol,
                ledger=ledger,
            )
            return execute_unit(
                context,
                ledger,
                unit,
                invocation,
                controlled,
                state_root,
                worker_id=worker_id,
                dispatch_grant=grant,
            )

        with ThreadPoolExecutor(
            max_workers=len(workers), thread_name_prefix="rolloutbench-worker"
        ) as pool:
            while pending or running:
                selected = _choose_dispatches(
                    context, ledger, units, pending, completed, available
                )
                for unit, worker_id in selected:
                    refreshed_authorization = validate_launch_authorization(
                        context,
                        authorization_path,
                        lease_files,
                        preflight_spec=profile,
                    )
                    if (
                        refreshed_authorization.get("authorization_sha256")
                        != authorization.get("authorization_sha256")
                    ):
                        raise FormalDispatchError(
                            "launch authorization changed after dispatcher acquisition"
                        )
                    grant = DispatchGrant(
                        plan_id=context.plan_id,
                        plan_sha256=context.plan_sha256,
                        run_id=str(context.run["run_id"]),
                        run_sha256=context.run_sha256,
                        unit_id=unit.unit_id,
                        worker_id=worker_id,
                        dispatch_policy=policy,
                        dispatcher_id=dispatcher_id,
                        dispatch_index=dispatch_index,
                    )
                    dispatch_index += 1
                    pending.remove(unit.unit_id)
                    available.remove(worker_id)
                    future = pool.submit(run_one, unit, worker_id, grant)
                    running[future] = (unit.unit_id, worker_id)
                if not running:
                    raise FormalDispatchError(
                        "completion-driven dispatcher has no runnable unit"
                    )
                finished, _ = wait(tuple(running), return_when=FIRST_COMPLETED)
                for future in finished:
                    unit_id, worker_id = running.pop(future)
                    available.add(worker_id)
                    try:
                        outcome = future.result()
                    except Exception as exc:
                        raise FormalDispatchError(
                            f"formal unit failed: {unit_id}"
                        ) from exc
                    if outcome.get("status") not in {
                        "EXECUTED", "REUSED_COMPLETED"
                    }:
                        raise FormalDispatchError(
                            f"formal unit did not complete: {unit_id}"
                        )
                    completed.add(unit_id)
                    outcomes.append(outcome)

        try:
            primary_evidence = collect_primary_evidence(context, ledger, suite)
            quality_decisions = finalize_quality_decisions(
                context,
                ledger,
                units,
                quality_protocol,
                lease_files=lease_files,
                profile=profile,
                primary_evidence=primary_evidence,
            )
            finalization = finalize_run_decisions(
                context, ledger, suite, primary_evidence
            )
        except DecisionError as exc:
            raise FormalDispatchError("fresh run decision finalization failed") from exc
        if (
            finalization["decision_count"] != len(expected_episode_ids)
            or set(finalization["decisions"]) != set(expected_episode_ids)
        ):
            raise FormalDispatchError("formal run decision count is incomplete")
        frontier_receipt = finalization["frontier_receipt"]
        if not isinstance(frontier_receipt, Mapping):
            raise FormalDispatchError("formal run frontier receipt is missing")
        ledger.append(
            "run_completed",
            {
                "plan_id": context.plan_id,
                "plan_sha256": context.plan_sha256,
                "run_id": context.run["run_id"],
                "run_sha256": context.run_sha256,
                "dispatcher_id": dispatcher_id,
                "dispatch_policy": policy,
                "unit_count": len(units),
                "completed_unit_count": len(completed),
                "decision_count": finalization["decision_count"],
                "quality_decision_count": len(quality_decisions),
                "frontier_receipt_path": frontier_receipt["path"],
                "frontier_receipt_sha256": frontier_receipt["sha256"],
                "performance_claim": False,
            },
        )
        return outcomes
