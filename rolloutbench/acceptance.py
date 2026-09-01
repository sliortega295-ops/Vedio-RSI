from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .events import EventLedger, atomic_write_stage_output
from .resume import plan_stage_resume
from .scheduler import SYSTEMS, simulate
from .schema import validate_suite_directory
from .validators import HistoricalOracleReplay, compare_historical_oracle


_EXPECTED_SIMULATED_GPU_SLOT_CONCURRENCY = {
    "serial1": 1,
    "fifo2": 2,
    "optroll1": 1,
    "optroll2": 2,
}
_EXPECTED_REUSE_HITS = {
    "serial1": 0,
    "fifo2": 0,
    "optroll1": 1,
    "optroll2": 1,
}
_EXPECTED_ARTIFACT_PATHS = frozenset(
    {
        "historical/oracle-acceptance.json",
        "historical/oracle-replay.json",
        "recovery/C12-before-decision-seal.json",
        "recovery/K20-generation-interrupt.json",
        "recovery/ledgers/C12.jsonl",
        "recovery/ledgers/K20.jsonl",
        "recovery/stage-outputs/.C12-quality-v1-attempt-1.json.lock",
        "recovery/stage-outputs/.K20-generate-attempt-2.json.lock",
        "recovery/stage-outputs/C12-quality-v1-attempt-1.json",
        "recovery/stage-outputs/K20-generate-attempt-2.json",
        "simulations/fifo2.json",
        "simulations/optroll1.json",
        "simulations/optroll2.json",
        "simulations/serial1.json",
    }
)


class EvidenceVerificationError(RuntimeError):
    """Raised when a CPU acceptance evidence pack is incomplete or altered."""


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _count_event(events: tuple[dict[str, Any], ...], event_type: str) -> int:
    return sum(event["event_type"] == event_type for event in events)


def _run_k20_recovery(root: Path) -> dict[str, Any]:
    ledger = EventLedger(root / "ledgers" / "K20.jsonl")
    ledger.append(
        "stage_started",
        {"episode_id": "K20", "stage": "generate", "attempt": 1},
    )
    crashed = ledger.reconstruct()
    interrupted_key = ("K20", "generate", 1)
    if crashed.stage_states.get(interrupted_key) != "interrupted":
        raise AssertionError("K20 interrupted generation was not recoverable")
    resume_plan = plan_stage_resume(crashed, episode_id="K20", stage="generate")
    if resume_plan != {
        "action": "retry",
        "attempt": 2,
        "reason": "prior_attempt_did_not_complete",
    }:
        raise AssertionError("K20 resume planner did not select attempt 2")
    ledger.append(
        "stage_interrupted",
        {"episode_id": "K20", "stage": "generate", "attempt": 1},
    )
    ledger.append(
        "stage_started",
        {
            "episode_id": "K20",
            "stage": "generate",
            "attempt": resume_plan["attempt"],
        },
    )
    output = root / "stage-outputs" / "K20-generate-attempt-2.json"
    atomic_write_stage_output(
        ledger,
        output,
        _json_bytes({"episode_id": "K20", "attempt": 2, "status": "completed"}),
        episode_id="K20",
        stage="generate",
        attempt=resume_plan["attempt"],
    )
    decision = {"outcome": "contract_validated", "resumed_after_attempt": 1}
    first = ledger.seal_decision("K20", decision)
    second = ledger.seal_decision("K20", decision)
    resumed = ledger.reconstruct()
    report = {
        "scenario": "cpu_ledger_contract_K20_interrupted_during_generation",
        "real_process_kill": False,
        "gpu_execution": False,
        "performance_claim": False,
        "resume_planner_action": resume_plan["action"],
        "attempt_1_state": resumed.stage_states[("K20", "generate", 1)],
        "attempt_2_state": resumed.stage_states[("K20", "generate", 2)],
        "physical_decision_records": _count_event(resumed.events, "decision_sealed"),
        "idempotent_decision_event_reused": first["event_id"] == second["event_id"],
        "durable_output_sha256": _file_sha256(output),
    }
    report["status"] = (
        "PASS"
        if report["attempt_1_state"] == "interrupted"
        and report["attempt_2_state"] == "completed"
        and report["physical_decision_records"] == 1
        and report["idempotent_decision_event_reused"]
        else "FAIL"
    )
    return report


def _run_c12_recovery(root: Path) -> dict[str, Any]:
    ledger = EventLedger(root / "ledgers" / "C12.jsonl")
    ledger.append(
        "stage_started",
        {"episode_id": "C12", "stage": "quality_v1", "attempt": 1},
    )
    output = root / "stage-outputs" / "C12-quality-v1-attempt-1.json"
    atomic_write_stage_output(
        ledger,
        output,
        _json_bytes({"episode_id": "C12", "attempt": 1, "status": "completed"}),
        episode_id="C12",
        stage="quality_v1",
        attempt=1,
    )
    before_resume = ledger.reconstruct()
    stage_key = ("C12", "quality_v1", 1)
    if before_resume.stage_states.get(stage_key) != "completed" or before_resume.decisions:
        raise AssertionError("C12 pre-decision recovery boundary is invalid")
    resume_plan = plan_stage_resume(
        before_resume, episode_id="C12", stage="quality_v1"
    )
    if resume_plan["action"] != "reuse_completed":
        raise AssertionError("C12 resume planner did not preserve the completed stage")
    decision = {"outcome": "contract_validated", "reused_completed_stage": True}
    first = ledger.seal_decision("C12", decision)
    second = ledger.seal_decision("C12", decision)
    resumed = ledger.reconstruct()
    completion_records = _count_event(resumed.events, "stage_completed")
    report = {
        "scenario": "cpu_ledger_contract_C12_after_quality_before_decision_seal",
        "real_process_kill": False,
        "gpu_execution": False,
        "performance_claim": False,
        "resume_planner_action": resume_plan["action"],
        "stage_state_before_resume": before_resume.stage_states[stage_key],
        "stage_state_after_resume": resumed.stage_states[stage_key],
        "completed_stage_preserved_after_reconstruct": completion_records == 1,
        "physical_stage_completion_records": completion_records,
        "physical_decision_records": _count_event(resumed.events, "decision_sealed"),
        "idempotent_decision_event_reused": first["event_id"] == second["event_id"],
        "durable_output_sha256": _file_sha256(output),
    }
    report["status"] = (
        "PASS"
        if report["stage_state_before_resume"] == "completed"
        and report["stage_state_after_resume"] == "completed"
        and report["completed_stage_preserved_after_reconstruct"]
        and report["physical_decision_records"] == 1
        and report["idempotent_decision_event_reused"]
        else "FAIL"
    )
    return report


def run_cpu_acceptance(
    suite_dir: Path | str,
    output_dir: Path | str,
    *,
    repo_root: Path | str | None = None,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Write a no-overwrite CPU evidence pack; it makes no GPU performance claim."""

    suite_path = Path(suite_dir)
    output = Path(output_dir)
    repository = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        source_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
        source_tree_clean = not subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repository,
            text=True,
        ).strip()
        if require_clean and not source_tree_clean:
            raise RuntimeError("CPU acceptance requires a clean source tree")
        suite_report = validate_suite_directory(suite_path, repo_root=repository)
        suite = json.loads((suite_path / "suite.json").read_text(encoding="utf-8"))
        episodes = [
            json.loads(line)
            for line in (suite_path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        ]

        simulation_status: dict[str, str] = {}
        oracle_isolated = True
        for system in suite["systems"]:
            if system not in SYSTEMS:
                raise ValueError(f"suite declares unsupported system {system}")
            payload = simulate(system, episodes).as_dict()
            _write_json(output / "simulations" / f"{system}.json", payload)
            summary = payload["summary"]
            passed = (
                summary["synthetic_contract_simulation"] is True
                and summary["performance_claim"] is False
                and summary["released_episodes"] == 35
                and summary["sealed_decisions"] == 35
                and summary["max_gpu_concurrency"]
                == _EXPECTED_SIMULATED_GPU_SLOT_CONCURRENCY[system]
                and summary["declared_reuse_hits"] == _EXPECTED_REUSE_HITS[system]
                and summary["decision_agreement"] is True
                and summary["frontier_agreement"] is True
                and summary["historical_oracle_checked"] is False
                and summary["frontier_semantics"]
                == "fake_contract_only_not_historical"
                and summary["decision_agreement_scope"]
                == "fake_executor_vs_schedule_trace"
            )
            oracle_isolated = oracle_isolated and (
                summary["historical_oracle_checked"] is False
                and summary["frontier_semantics"]
                == "fake_contract_only_not_historical"
                and summary["decision_agreement_scope"]
                == "fake_executor_vs_schedule_trace"
            )
            simulation_status[system] = "PASS" if passed else "FAIL"

        expected_frontier = suite["frontier_contracts"]["legacy_oracle"]
        replay = HistoricalOracleReplay().replay(episodes, expected_frontier)
        oracle = compare_historical_oracle(
            episodes,
            replay["decisions"],
            replay["frontier"],
            expected_frontier,
        )
        _write_json(output / "historical" / "oracle-replay.json", replay)
        _write_json(output / "historical" / "oracle-acceptance.json", oracle)

        k20 = _run_k20_recovery(output / "recovery")
        c12 = _run_c12_recovery(output / "recovery")
        _write_json(output / "recovery" / "K20-generation-interrupt.json", k20)
        _write_json(output / "recovery" / "C12-before-decision-seal.json", c12)

        cpu_contract_status = (
            "PASS"
            if set(simulation_status) == set(_EXPECTED_SIMULATED_GPU_SLOT_CONCURRENCY)
            and set(simulation_status.values()) == {"PASS"}
            and oracle["status"] == "PASS"
            and k20["status"] == "PASS"
            and c12["status"] == "PASS"
            and oracle_isolated
            else "FAIL"
        )
        artifact_hashes = {
            path.relative_to(output).as_posix(): _file_sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "MANIFEST.json"
        }
        suite_hashes = {
            name: _file_sha256(suite_path / name)
            for name in (
                "suite.json",
                "episodes.jsonl",
                "artifacts.json",
                "quality_protocol.json",
            )
        }
        status = (
            "PASS_DIRTY_NONREPRODUCIBLE"
            if cpu_contract_status == "PASS" and not source_tree_clean
            else cpu_contract_status
        )
        manifest = {
            "schema_version": 1,
            "status": status,
            "cpu_contract_status": cpu_contract_status,
            "suite_id": suite["suite_id"],
            "source_revision": source_revision,
            "source_tree_clean": source_tree_clean,
            "suite_file_sha256": suite_hashes,
            "execution_scope": "deterministic_cpu_contract_only",
            "suite_validation": suite_report.as_dict(),
            "simulation_status": simulation_status,
            "historical_oracle_self_consistency_status": oracle["status"],
            "ledger_reconstruction_contract_status": {
                "K20": k20["status"],
                "C12": c12["status"],
            },
            "simulated_gpu_slot_concurrency_contract": (
                _EXPECTED_SIMULATED_GPU_SLOT_CONCURRENCY
            ),
            "gpu_execution_status": "NOT_RUN",
            "quality_execution_status": "NOT_RUN",
            "historical_decisions_used_by_scheduler": not oracle_isolated,
            "real_fault_injection_status": "NOT_RUN",
            "performance_claim": False,
            "required_artifacts": sorted(_EXPECTED_ARTIFACT_PATHS),
            "artifact_sha256_by_path": artifact_hashes,
        }
        _write_json(output / "MANIFEST.json", manifest)
        return manifest
    except Exception as exc:
        _write_json(
            output / "FAILED.json",
            {
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "performance_claim": False,
            },
        )
        raise


def verify_cpu_acceptance_pack(
    pack_dir: Path | str,
    suite_dir: Path | str,
    *,
    repo_root: Path | str | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Independently verify exact inventory, hashes, suite binding, and claim scope."""

    pack = Path(pack_dir)
    suite_path = Path(suite_dir)
    repository = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    if (pack / "FAILED.json").exists():
        raise EvidenceVerificationError("FAILED.json is present")
    manifest_path = pack / "MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceVerificationError("MANIFEST.json is missing or invalid") from exc
    if not isinstance(manifest, dict):
        raise EvidenceVerificationError("MANIFEST.json must contain an object")

    actual_files = {
        path.relative_to(pack).as_posix()
        for path in pack.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    }
    declared = manifest.get("artifact_sha256_by_path")
    required = manifest.get("required_artifacts")
    if not isinstance(declared, dict):
        raise EvidenceVerificationError("artifact hash map is missing")
    if required != sorted(_EXPECTED_ARTIFACT_PATHS):
        raise EvidenceVerificationError("required artifact inventory contract mismatch")
    if actual_files != _EXPECTED_ARTIFACT_PATHS or set(declared) != _EXPECTED_ARTIFACT_PATHS:
        raise EvidenceVerificationError("artifact inventory mismatch")
    for relative_path, expected_sha in declared.items():
        if _file_sha256(pack / relative_path) != expected_sha:
            raise EvidenceVerificationError(f"artifact hash mismatch: {relative_path}")

    validate_suite_directory(suite_path, repo_root=repository)
    expected_suite_hashes = {
        name: _file_sha256(suite_path / name)
        for name in (
            "suite.json",
            "episodes.jsonl",
            "artifacts.json",
            "quality_protocol.json",
        )
    }
    if manifest.get("suite_file_sha256") != expected_suite_hashes:
        raise EvidenceVerificationError("suite hash closure mismatch")
    source_revision = manifest.get("source_revision")
    if not isinstance(source_revision, str) or len(source_revision) != 40:
        raise EvidenceVerificationError("source revision is invalid")
    try:
        subprocess.check_call(
            ["git", "cat-file", "-e", f"{source_revision}^{{commit}}"],
            cwd=repository,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise EvidenceVerificationError("source revision is unavailable") from exc
    if manifest.get("source_tree_clean") is not True and not allow_dirty:
        raise EvidenceVerificationError("dirty source pack is non-reproducible")
    expected_status = "PASS" if manifest.get("source_tree_clean") is True else "PASS_DIRTY_NONREPRODUCIBLE"
    if (
        manifest.get("status") != expected_status
        or manifest.get("cpu_contract_status") != "PASS"
        or manifest.get("performance_claim") is not False
        or manifest.get("gpu_execution_status") != "NOT_RUN"
        or manifest.get("quality_execution_status") != "NOT_RUN"
        or manifest.get("real_fault_injection_status") != "NOT_RUN"
        or manifest.get("historical_decisions_used_by_scheduler") is not False
        or manifest.get("historical_oracle_self_consistency_status") != "PASS"
        or set(manifest.get("ledger_reconstruction_contract_status", {}).values())
        != {"PASS"}
    ):
        raise EvidenceVerificationError("acceptance claim boundary mismatch")
    return {
        "status": "PASS",
        "pack_status": manifest["status"],
        "artifact_count": len(_EXPECTED_ARTIFACT_PATHS),
        "performance_claim": False,
    }
