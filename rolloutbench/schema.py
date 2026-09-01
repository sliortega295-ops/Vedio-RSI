from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FILES = {
    "suite.json",
    "episodes.jsonl",
    "artifacts.json",
    "quality_protocol.json",
}
HISTORICAL_STATUSES = {"validated", "not_run", "failed"}
EXPECTED_COMPONENT_COUNTS = {"kernel": 23, "cache": 12}
EXPECTED_HISTORICAL_STATUS_COUNTS = {"validated": 29, "not_run": 4, "failed": 2}
PREFLIGHT_ONLY_IDS = {"K15", "K18", "K21", "K23"}


class SuiteValidationError(ValueError):
    """Raised when a frozen suite violates the v0 contract."""


@dataclass(frozen=True)
class ValidationReport:
    total_episodes: int
    component_counts: dict[str, int]
    historical_status_counts: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "valid",
            "total_episodes": self.total_episodes,
            "component_counts": self.component_counts,
            "historical_status_counts": self.historical_status_counts,
        }


def _load_json_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise SuiteValidationError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise SuiteValidationError(f"{path.name} must contain a JSON object")
    return value, raw


def _load_jsonl_bytes(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SuiteValidationError(f"cannot read {path.name}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SuiteValidationError(f"invalid JSON at episodes.jsonl:{line_number}") from exc
        if not isinstance(row, dict):
            raise SuiteValidationError(f"episodes.jsonl:{line_number} must be an object")
        rows.append(row)
    return rows, raw


@lru_cache(maxsize=256)
def _git_blob(repo_root: Path, ref: str, path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{ref}:{path}"], cwd=repo_root, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise SuiteValidationError(f"Git blob unavailable for {ref}:{path}: {detail}") from exc


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _expected_fifo() -> list[str]:
    order: list[str] = []
    for round_number in range(1, 13):
        order.extend([f"K{round_number:02d}", f"C{round_number:02d}"])
    order.extend(f"K{round_number:02d}" for round_number in range(13, 24))
    return order


def _validate_file_hashes(suite: dict[str, Any], raw_files: dict[str, bytes]) -> None:
    receipts = suite.get("file_hashes")
    if not isinstance(receipts, dict):
        raise SuiteValidationError("suite.json lacks file_hashes")
    for name, raw in raw_files.items():
        receipt = receipts.get(name)
        if not isinstance(receipt, dict) or receipt.get("hash_scope") != "raw_file_bytes":
            raise SuiteValidationError(f"{name} hash scope must be raw_file_bytes")
        if receipt.get("sha256") != _sha256(raw):
            raise SuiteValidationError(f"file hash mismatch: {name}")
    direct_names = {
        "episodes.jsonl": "episodes_sha256",
        "artifacts.json": "artifacts_sha256",
        "quality_protocol.json": "quality_protocol_sha256",
    }
    for filename, field in direct_names.items():
        if suite.get(field) != _sha256(raw_files[filename]):
            raise SuiteValidationError(f"file hash mismatch: {filename}")
    if "self-referential hash cycle" not in str(suite.get("hash_design")):
        raise SuiteValidationError("suite hash design must exclude suite.json and explain the cycle")


def _validate_git_item(
    item: dict[str, Any], *, repo_root: Path, ref: str, label: str, reported_optional: bool = False
) -> None:
    path = item.get("path")
    blob_sha = item.get("blob_sha256")
    reported_sha = item.get("authority_reported_sha256")
    if not isinstance(path, str) or not isinstance(blob_sha, str):
        raise SuiteValidationError(f"{label} Git hash closure is incomplete")
    actual_sha = _sha256(_git_blob(repo_root, ref, path))
    if blob_sha != actual_sha:
        raise SuiteValidationError(f"{label} Git hash mismatch")
    if reported_sha is None and reported_optional:
        return
    if reported_sha != blob_sha:
        raise SuiteValidationError(f"{label} authority hash mismatch")


def _validate_episode_graph(episodes: list[dict[str, Any]]) -> None:
    expected_physical_ids = [f"K{number:02d}" for number in range(1, 24)] + [
        f"C{number:02d}" for number in range(1, 13)
    ]
    episode_ids = [row.get("episode_id") for row in episodes]
    duplicates = sorted(key for key, count in Counter(episode_ids).items() if count > 1)
    if duplicates:
        raise SuiteValidationError(f"duplicate episode_id: {duplicates[0]}")
    if episode_ids != expected_physical_ids:
        raise SuiteValidationError("episodes must contain continuous rounds K01-K23 then C01-C12")

    indices = [row.get("global_fifo_index") for row in episodes]
    if sorted(indices) != list(range(35)):
        raise SuiteValidationError("global_fifo_index must be unique and continuous from 0 to 34")
    fifo_ids = [
        row["episode_id"] for row in sorted(episodes, key=lambda row: row["global_fifo_index"])
    ]
    if fifo_ids != _expected_fifo():
        raise SuiteValidationError("global_fifo_index does not implement the frozen round-robin order")

    previous: dict[str, str] = {}
    for episode in episodes:
        component = episode["component"]
        if episode.get("stream") != component:
            raise SuiteValidationError(f"{episode['episode_id']} stream mismatch")
        expected_dependencies = [previous[component]] if component in previous else []
        if episode.get("depends_on") != expected_dependencies:
            raise SuiteValidationError(f"depends_on mismatch for {episode['episode_id']}")
        previous[component] = episode["episode_id"]


def _validate_episode_contracts(episodes: list[dict[str, Any]], repo_root: Path) -> None:
    for episode in episodes:
        episode_id = episode["episode_id"]
        validation = episode.get("validation", {})
        stages = validation.get("stages", [])
        resources = episode.get("resources", {})
        candidate = episode.get("candidate", {})
        source = episode.get("source", {})
        ref = candidate.get("authority_ref")

        if resources.get("gpu_count") != 1:
            raise SuiteValidationError(f"{episode_id} must reserve exactly one GPU")
        if episode_id.startswith("C"):
            required = {"generate", "collect", "decide"}
            if not required.issubset(stages):
                raise SuiteValidationError(f"Cache validation missing required stages for {episode_id}")
            eligibility = episode.get("quality_eligibility")
            if eligibility == "formal":
                if validation.get("contract") != "full_lossy_quality_v1" or "quality_v1" not in stages:
                    raise SuiteValidationError(f"Cache formal quality contract mismatch for {episode_id}")
            elif (
                validation.get("contract") != "historical_calibration_replay_v1"
                or "legacy_sanity" not in stages
                or "quality_v1" in stages
            ):
                raise SuiteValidationError(f"Cache calibration contract mismatch for {episode_id}")
        if episode_id in PREFLIGHT_ONLY_IDS:
            if validation.get("earliest_legal_exit") != "after_decide":
                raise SuiteValidationError(f"Kernel preflight exit rule mismatch for {episode_id}")
            if validation.get("early_exit_trigger") != "preflight_gate_reject_after_microbenchmark":
                raise SuiteValidationError(f"Kernel preflight trigger mismatch for {episode_id}")
            if "microbenchmark" not in stages or "generate" in stages:
                raise SuiteValidationError(f"Kernel preflight stages mismatch for {episode_id}")
            if validation.get("preflight_completion_requires") != [
                "gpu_lease",
                "probe_source",
                "microbenchmark",
                "probe_result",
            ]:
                raise SuiteValidationError(f"Kernel preflight completion rule mismatch for {episode_id}")
            probe = candidate.get("probe")
            if not isinstance(probe, dict):
                raise SuiteValidationError(f"{episode_id} preflight probe is missing")
            _validate_git_item(probe.get("source", {}), repo_root=repo_root, ref=ref, label=f"{episode_id} probe source")
            _validate_git_item(probe.get("result", {}), repo_root=repo_root, ref=ref, label=f"{episode_id} probe result")
        elif candidate.get("probe") is not None:
            raise SuiteValidationError(f"{episode_id} unexpectedly declares a preflight-only probe")

        config = candidate.get("config")
        if config is None:
            if episode_id not in PREFLIGHT_ONLY_IDS:
                raise SuiteValidationError(f"{episode_id} config Git hash closure is incomplete")
        else:
            _validate_git_item(
                config,
                repo_root=repo_root,
                ref=ref,
                label=f"{episode_id} config",
                reported_optional=episode_id == "C01",
            )
            if episode_id == "C01" and config.get("authority_reported_sha256") is not None:
                raise SuiteValidationError("C01 must preserve the absent authority-reported config hash")

        line_number = source.get("line_number")
        if not isinstance(line_number, int) or line_number < 1:
            raise SuiteValidationError(f"{episode_id} source line is invalid")
        trajectory = _git_blob(repo_root, source.get("git_ref"), source.get("path"))
        lines = trajectory.splitlines()
        if line_number > len(lines) or _sha256(lines[line_number - 1]) != source.get("line_sha256"):
            raise SuiteValidationError(f"{episode_id} replay source Git hash mismatch")

        reuse = episode.get("reuse", {})
        expected_inputs = (
            [{"artifact": "torch_compile_cache", "episode_id": "K01", "required": True}]
            if episode_id == "K02"
            else []
        )
        if reuse.get("inputs") != expected_inputs:
            raise SuiteValidationError(f"{episode_id} reuse contract mismatch")
        golden = episode.get("golden", {})
        if golden.get("scheduler_visible") is not False or golden.get("role") != "acceptance_oracle_only":
            raise SuiteValidationError(f"{episode_id} golden oracle must be hidden from scheduler")
        expected_failure = (
            {
                "kind": "real_fail_closed_layout_mismatch",
                "stage": "generate",
                "deterministic": True,
            }
            if episode_id == "K22"
            else None
        )
        if episode.get("replay", {}).get("failure_contract") != expected_failure:
            raise SuiteValidationError(f"{episode_id} failure contract mismatch")


def _validate_quality(quality: dict[str, Any]) -> None:
    formal = quality.get("formal_cache_candidates")
    excluded = quality.get("excluded_cache_candidates")
    if not isinstance(formal, list) or not isinstance(excluded, dict):
        raise SuiteValidationError("quality protocol candidate partition is missing")
    all_cache = {f"C{number:02d}" for number in range(1, 13)}
    if set(formal) & set(excluded) or set(formal) | set(excluded) != all_cache:
        raise SuiteValidationError("quality candidate partition must cover C01-C12 exactly")
    prompt_suites = quality.get("prompt_selection", {}).get("prompt_suites", [])
    seeds = quality.get("seeds", [])
    if len(prompt_suites) != 4 or len(seeds) != 2 or len(prompt_suites) * len(seeds) != 8:
        raise SuiteValidationError("quality protocol must freeze 4 prompts x 2 seeds = 8 matched pairs")
    if quality.get("matched_pairs_per_candidate") != 8:
        raise SuiteValidationError("matched_pairs_per_candidate must equal 8")
    metric_union = {metric for prompt in prompt_suites for metric in prompt.get("metrics", [])}
    if metric_union != set(quality.get("dimensions", [])) or len(metric_union) != 7:
        raise SuiteValidationError("quality protocol must map exactly seven dimensions")


def validate_suite_directory(
    suite_dir: Path | str, *, repo_root: Path | str | None = None
) -> ValidationReport:
    root = Path(suite_dir)
    repository = (
        Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    )
    missing = sorted(name for name in REQUIRED_FILES if not (root / name).is_file())
    if missing:
        raise SuiteValidationError(f"missing suite files: {', '.join(missing)}")

    suite, _ = _load_json_bytes(root / "suite.json")
    artifacts, artifacts_raw = _load_json_bytes(root / "artifacts.json")
    quality, quality_raw = _load_json_bytes(root / "quality_protocol.json")
    episodes, episodes_raw = _load_jsonl_bytes(root / "episodes.jsonl")
    _validate_file_hashes(
        suite,
        {
            "episodes.jsonl": episodes_raw,
            "artifacts.json": artifacts_raw,
            "quality_protocol.json": quality_raw,
        },
    )

    component_counts = dict(Counter(str(row.get("component")) for row in episodes))
    historical_status_counts = dict(
        Counter(str(row.get("golden", {}).get("historical_status")) for row in episodes)
    )
    if component_counts != EXPECTED_COMPONENT_COUNTS:
        raise SuiteValidationError(
            f"component counts must be {EXPECTED_COMPONENT_COUNTS}, got {component_counts}"
        )
    if historical_status_counts != EXPECTED_HISTORICAL_STATUS_COUNTS:
        raise SuiteValidationError(
            "historical status counts must be "
            f"{EXPECTED_HISTORICAL_STATUS_COUNTS}, got {historical_status_counts}"
        )
    if set(historical_status_counts) - HISTORICAL_STATUSES:
        raise SuiteValidationError("unknown historical status")

    expected_declared = {
        "total": 35,
        "components": EXPECTED_COMPONENT_COUNTS,
        "historical_status": EXPECTED_HISTORICAL_STATUS_COUNTS,
    }
    if suite.get("counts") != expected_declared:
        raise SuiteValidationError("suite.json counts do not match the v0 contract")
    if suite.get("ordering", {}).get("global_fifo_rule") != "round_robin_K_then_C_through_round_12_then_K13_to_K23":
        raise SuiteValidationError("suite ordering rule mismatch")

    artifact_rows = artifacts.get("artifacts")
    if not isinstance(artifact_rows, list) or not artifact_rows:
        raise SuiteValidationError("artifacts.json must declare artifact boundaries")
    allowed_availability = {
        "git_available",
        "remote_only_verified",
        "local_source_verified",
        "missing",
        "regenerate",
    }
    if any(row.get("availability") not in allowed_availability for row in artifact_rows):
        raise SuiteValidationError("artifacts.json contains an invalid availability value")

    _validate_episode_graph(episodes)
    _validate_episode_contracts(episodes, repository)
    _validate_quality(quality)
    return ValidationReport(
        total_episodes=len(episodes),
        component_counts=component_counts,
        historical_status_counts=historical_status_counts,
    )
