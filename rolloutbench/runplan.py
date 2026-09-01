from __future__ import annotations

import hashlib
import fcntl
import json
import math
import os
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .quality_contract import FORMAL_CACHE_IDS
from .runtime_manifest import build_runtime_manifest
from .scheduler import load_public_episodes
from .schema import validate_suite_directory
from .validators import build_quality_plan


_TWO_GPU_SYSTEMS = {"fifo2", "optroll2"}
_PERSISTENT_REQUEST_SYSTEMS = {"optroll1", "optroll2"}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _source_receipt(repo_root: Path) -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    dirty_rows = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        text=True,
    ).splitlines()
    return {
        "revision": revision,
        "tree_clean": not dirty_rows,
        "dirty_path_count": len(dirty_rows),
    }


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def required_repetitions(latencies: Sequence[float]) -> int:
    """Return the frozen 3+2 repetition requirement from the first three runs."""

    values = [float(value) for value in latencies]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("latencies must be finite and positive")
    if len(values) < 3:
        return 3
    first_three = values[:3]
    coefficient_of_variation = statistics.stdev(first_three) / statistics.fmean(
        first_three
    )
    return 5 if coefficient_of_variation > 0.03 else 3


def _workers(
    system: str, repeat_index: int, gpu_uuids: tuple[str, str]
) -> list[dict[str, Any]]:
    mapping = gpu_uuids if repeat_index % 2 == 1 else tuple(reversed(gpu_uuids))
    if system not in _TWO_GPU_SYSTEMS:
        return [{"worker_id": 0, "gpu_uuid": mapping[0], "component": "any"}]
    if system == "optroll2":
        return [
            {"worker_id": 0, "gpu_uuid": mapping[0], "component": "kernel"},
            {"worker_id": 1, "gpu_uuid": mapping[1], "component": "cache"},
        ]
    return [
        {"worker_id": 0, "gpu_uuid": mapping[0], "component": "any"},
        {"worker_id": 1, "gpu_uuid": mapping[1], "component": "any"},
    ]


def _dispatch_policy(system: str) -> str:
    return {
        "serial1": "global_fifo_one_shot",
        "fifo2": "global_fifo_two_workers_dependency_aware",
        "optroll1": "typed_validation_decision_aware_one_worker",
        "optroll2": "typed_streams_kernel_cache_one_worker_each",
    }[system]


def _episode_plan(
    episode: Mapping[str, Any],
    quality_pairs: Mapping[str, list[dict[str, Any]]],
    runtime_manifest: Mapping[str, Any],
    system: str,
) -> dict[str, Any]:
    episode_id = str(episode["episode_id"])
    declared_inputs = [
        {
            **item,
            "canonical_key": f"{item['episode_id']}:{item['artifact']}",
        }
        for item in episode.get("reuse", {}).get("inputs", [])
    ]
    persistent_requested = system in _PERSISTENT_REQUEST_SYSTEMS
    if episode_id == "K02":
        effective_mode = "one_shot"
        mode_reason = "confirmation_requires_fresh_process"
    else:
        effective_mode = "one_shot"
        mode_reason = (
            "persistent_worker_requires_compatibility_key_and_reset_proof"
            if persistent_requested
            else "system_contract_is_one_shot"
        )
    affinity: int | str
    if system == "optroll2":
        affinity = 0 if episode["component"] == "kernel" else 1
    elif system == "fifo2" and episode_id == "K02":
        affinity = "lineage:K01"
    elif system in {"serial1", "optroll1"}:
        affinity = 0
    else:
        affinity = "dynamic"
    return {
        "episode_id": episode_id,
        "component": episode["component"],
        "round": episode["round"],
        "global_fifo_index": episode["global_fifo_index"],
        "depends_on": list(episode["depends_on"]),
        "candidate_type": episode["candidate_type"],
        "quality_eligibility": episode["quality_eligibility"],
        "candidate": episode["candidate"],
        "runtime_checkout": {
            "mode": "detached_candidate_commit_worktree",
            **runtime_manifest,
            "shared_object_store": True,
            "checkout_mutation_allowed": False,
        },
        "validation": episode["validation"],
        "resources": episode["resources"],
        "declared_artifact_inputs": declared_inputs,
        "cache_scope_key": "K01" if episode_id == "K02" else episode_id,
        "worker_affinity": affinity,
        "worker_contract": {
            "requested_mode": "persistent" if persistent_requested else "one_shot",
            "effective_mode": effective_mode,
            "reason": mode_reason,
            "activation_gate": (
                "compatibility_key_plus_reset_proof"
                if persistent_requested and episode_id != "K02"
                else None
            ),
        },
        "quality_pairs": quality_pairs.get(episode_id, []),
    }


def build_experiment_plan(
    suite_dir: Path | str,
    *,
    scope: str,
    repetitions: int,
    gpu_uuids: Sequence[str],
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Expand the frozen suite into four system plans without executing a GPU."""

    if scope not in {"pilot", "full"}:
        raise ValueError("scope must be pilot or full")
    if repetitions not in {3, 5}:
        raise ValueError("formal plans require exactly three or five repetitions")
    normalized_gpu_uuids = tuple(str(value) for value in gpu_uuids)
    if (
        len(normalized_gpu_uuids) != 2
        or len(set(normalized_gpu_uuids)) != 2
        or any(not value.startswith("GPU-") for value in normalized_gpu_uuids)
    ):
        raise ValueError("exactly two unique NVIDIA GPU UUIDs are required")

    suite_path = Path(suite_dir)
    repository = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    validate_suite_directory(suite_path, repo_root=repository)
    suite = json.loads((suite_path / "suite.json").read_text(encoding="utf-8"))
    protocol = json.loads(
        (suite_path / "quality_protocol.json").read_text(encoding="utf-8")
    )
    public = load_public_episodes(suite_path / "episodes.jsonl")
    by_id = {str(episode["episode_id"]): episode for episode in public}
    selected_ids = (
        list(suite["pilot_episodes"])
        if scope == "pilot"
        else [str(episode["episode_id"]) for episode in public]
    )
    selected = [by_id[episode_id] for episode_id in selected_ids]
    runtime_manifests = {
        episode_id: build_runtime_manifest(repository, by_id[episode_id]["candidate"])
        for episode_id in selected_ids
    }
    source = _source_receipt(repository)
    suite_hashes = {
        name: hashlib.sha256((suite_path / name).read_bytes()).hexdigest()
        for name in (
            "suite.json",
            "episodes.jsonl",
            "artifacts.json",
            "quality_protocol.json",
        )
    }

    quality_pairs: dict[str, list[dict[str, Any]]] = {}
    selected_formal = [
        episode_id for episode_id in selected_ids if episode_id in FORMAL_CACHE_IDS
    ]
    for row in build_quality_plan(protocol, selected_formal):
        quality_pairs.setdefault(str(row["candidate_id"]), []).append(row)

    runs: list[dict[str, Any]] = []
    for system in suite["systems"]:
        for repeat_index in range(1, repetitions + 1):
            cache_namespace = f"cache-namespaces/{system}/repeat-{repeat_index:02d}"
            runs.append(
                {
                    "run_id": f"{scope}-{system}-repeat-{repeat_index:02d}",
                    "system": system,
                    "scope": scope,
                    "repeat_index": repeat_index,
                    "dispatch_policy": _dispatch_policy(system),
                    "workers": _workers(system, repeat_index, normalized_gpu_uuids),
                    "cache_namespace": cache_namespace,
                    "cache_namespace_independent": True,
                    "episodes": [
                        _episode_plan(
                            episode,
                            quality_pairs,
                            runtime_manifests[str(episode["episode_id"])],
                            system,
                        )
                        for episode in selected
                    ],
                }
            )

    identity = {
        "suite_id": suite["suite_id"],
        "scope": scope,
        "repetitions": repetitions,
        "gpu_uuids": normalized_gpu_uuids,
        "systems": suite["systems"],
        "episode_ids": selected_ids,
        "suite_file_sha256": suite_hashes,
        "source_revision": source["revision"],
    }
    return {
        "schema_version": 1,
        "plan_id": hashlib.sha256(_canonical(identity)).hexdigest(),
        "suite_id": suite["suite_id"],
        "suite_file_sha256": suite_hashes,
        "source": source,
        "scope": scope,
        "repetitions": repetitions,
        "adaptive_repeat_rule": "three_repetitions_plus_two_when_first_three_cv_gt_0.03",
        "gpu_uuid_mapping_rule": "swap_worker_to_uuid_mapping_on_even_repetitions",
        "quality_protocol_id": protocol["protocol_id"],
        "quality_pairs_per_formal_candidate": 8,
        "runtime_contract": {
            "mode": "detached_candidate_commit_worktree_replay",
            "harness_source_revision": source["revision"],
            "candidate_semantic_parity": "BOUND_TO_AUTHORITY_COMMIT",
            "integrated_superset_runtime_used_for_execution": False,
            "reason": (
                "historical candidates include semantics later removed or corrected, "
                "including K22 and C01"
            ),
        },
        "execution_status": "NOT_RUN",
        "real_fault_injection_status": "NOT_RUN",
        "performance_claim": False,
        "runs": runs,
    }


def write_experiment_plan(
    path: Path | str,
    plan: Mapping[str, Any],
    *,
    require_clean: bool = True,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Durably write a source-bound plan, refusing dirty formal inputs or drift."""

    source = plan.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("plan has no source receipt")
    repository = (
        Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    )
    live_source = _source_receipt(repository)
    if live_source["revision"] != source.get("revision"):
        raise RuntimeError("experiment plan source revision no longer matches HEAD")
    if require_clean and (
        source.get("tree_clean") is not True or live_source["tree_clean"] is not True
    ):
        raise RuntimeError("formal experiment plans require a clean source tree")
    payload = json.dumps(
        plan, indent=2, sort_keys=True, ensure_ascii=False
    ).encode("utf-8") + b"\n"
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if output.exists():
                existing = output.read_bytes()
                if existing != payload:
                    raise FileExistsError(
                        f"refusing to overwrite a conflicting plan: {output}"
                    )
                return {
                    "path": str(output),
                    "sha256": hashlib.sha256(existing).hexdigest(),
                    "status": "UNCHANGED",
                }
            _atomic_write(output, payload)
            return {
                "path": str(output),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "status": "WRITTEN",
            }
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
