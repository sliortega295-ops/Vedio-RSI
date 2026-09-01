from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .materialize import materialize_candidate_artifacts
from .runplan import build_experiment_plan
from .runtime_checkout import prepare_runtime_checkout


class PreparationError(RuntimeError):
    """Raised when a formal experiment cannot be prepared exactly."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _load_plan(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        plan = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"cannot read experiment plan {path}: {exc}") from exc
    if not isinstance(plan, dict):
        raise PreparationError("experiment plan must be a JSON object")
    return plan, raw


def _git_source(repository: Path) -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repository,
            text=True,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PreparationError("cannot validate the harness Git checkout") from exc
    return {
        "revision": revision,
        "tree_clean": not dirty,
        "dirty_path_count": len(dirty),
    }


def _plan_gpu_uuids(plan: Mapping[str, Any]) -> tuple[str, str]:
    runs = plan.get("runs")
    if not isinstance(runs, list):
        raise PreparationError("experiment plan has no runs")
    matches = [
        run
        for run in runs
        if isinstance(run, Mapping)
        and run.get("repeat_index") == 1
        and isinstance(run.get("workers"), list)
        and len(run["workers"]) == 2
    ]
    if not matches:
        raise PreparationError("experiment plan has no first-repeat two-GPU mapping")
    workers = sorted(matches[0]["workers"], key=lambda row: row.get("worker_id", -1))
    values = tuple(str(worker.get("gpu_uuid", "")) for worker in workers)
    if len(values) != 2 or len(set(values)) != 2 or any(
        not value.startswith("GPU-") for value in values
    ):
        raise PreparationError("experiment plan GPU UUID mapping is invalid")
    return values


def _validate_canonical_plan(
    plan: Mapping[str, Any], suite_dir: Path, repository: Path, *, require_clean: bool
) -> dict[str, Any]:
    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise PreparationError("experiment plan source receipt is missing")
    live = _git_source(repository)
    if live["revision"] != source.get("revision"):
        raise PreparationError("experiment plan source revision does not match HEAD")
    if require_clean and (source.get("tree_clean") is not True or not live["tree_clean"]):
        raise PreparationError("formal preparation requires a clean planned harness")
    try:
        rebuilt = build_experiment_plan(
            suite_dir,
            scope=str(plan.get("scope")),
            repetitions=int(plan.get("repetitions")),
            gpu_uuids=_plan_gpu_uuids(plan),
            repo_root=repository,
        )
    except (TypeError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        raise PreparationError(f"cannot rebuild canonical plan: {exc}") from exc
    if _canonical(rebuilt) != _canonical(plan):
        raise PreparationError("experiment plan does not match the canonical plan")
    return live


def _unique_episodes(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}

    def add_episode(episode_value: Any) -> None:
        if not isinstance(episode_value, dict):
            raise PreparationError("experiment plan contains a malformed episode")
        episode_id = str(episode_value.get("episode_id", ""))
        if not episode_id:
            raise PreparationError("experiment plan contains an empty episode ID")
        previous = unique.get(episode_id)
        if previous is None:
            unique[episode_id] = episode_value
            return
        stable_fields = (
            "candidate",
            "runtime_checkout",
            "validation",
            "quality_pairs",
            "cache_scope_key",
        )
        if any(
            _canonical(previous.get(field)) != _canonical(episode_value.get(field))
            for field in stable_fields
        ):
            raise PreparationError(
                f"episode {episode_id} changes authority across system runs"
            )

    for run in plan.get("runs", []):
        if not isinstance(run, Mapping) or not isinstance(run.get("episodes"), list):
            raise PreparationError("experiment plan contains a malformed run")
        add_episode(run.get("quality_dense_reference"))
        for episode_value in run["episodes"]:
            add_episode(episode_value)
    return list(unique.values())


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.name}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if path.exists():
                if not path.is_file() or path.read_bytes() != content:
                    raise PreparationError(
                        f"refusing to overwrite conflicting preparation receipt: {path}"
                    )
                return
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=path.parent, prefix=f".{path.name}.", delete=False
                ) as handle:
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
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def prepare_experiment(
    plan_path: Path | str,
    suite_dir: Path | str,
    experiment_root: Path | str,
    *,
    repo_root: Path | str | None = None,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Prepare exact candidate worktrees and authority blobs without using a GPU."""

    repository = (
        Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    )
    experiment = Path(experiment_root).resolve()
    if experiment == repository or experiment == Path(experiment.anchor):
        raise PreparationError("experiment root is too broad")
    plan_file = Path(plan_path).resolve()
    suite_path = Path(suite_dir).resolve()
    plan, raw_plan = _load_plan(plan_file)
    live_source = _validate_canonical_plan(
        plan, suite_path, repository, require_clean=require_clean
    )
    episodes = _unique_episodes(plan)
    worktree_root = experiment / "worktrees"
    derived_root = experiment / "derived"
    runtime_receipts: dict[str, Any] = {}
    materialization_receipts: dict[str, Any] = {}
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        runtime_receipts[episode_id] = prepare_runtime_checkout(
            episode, repository, worktree_root
        )
        materialization_receipts[episode_id] = materialize_candidate_artifacts(
            episode, derived_root, repo_root=repository
        )

    receipt = {
        "schema_version": 1,
        "status": "READY",
        "plan_id": plan["plan_id"],
        "plan_path": str(plan_file),
        "plan_sha256": hashlib.sha256(raw_plan).hexdigest(),
        "suite_path": str(suite_path),
        "source": live_source,
        "experiment_root": str(experiment),
        "worktree_root": str(worktree_root),
        "derived_root": str(derived_root),
        "run_count": len(plan["runs"]),
        "unique_episode_count": len(episodes),
        "runtime_receipts": runtime_receipts,
        "materialization_receipts": materialization_receipts,
        "gpu_execution": False,
        "vbench_execution": False,
        "performance_claim": False,
    }
    content = (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _write_once(experiment / "state" / "preparation.json", content)
    return receipt
