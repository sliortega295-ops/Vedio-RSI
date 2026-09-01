from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


RESET_PROOF_FIELDS = (
    "controller_history_cleared",
    "rng_reset",
    "scheduler_state_cleared",
    "prompt_state_cleared",
    "output_state_cleared",
    "compile_cache_contract_verified",
    "candidate_cache_contract_verified",
    "fresh_process_structural_equivalence",
    "fresh_process_telemetry_equivalence",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def build_compatibility_key(
    *,
    runtime_file_hashes: Mapping[str, str],
    model_revision: str,
    init_environment: Mapping[str, str],
    dtype: str,
    backend: str,
    workload_fingerprint: str | Mapping[str, Any],
    gpu_arch: str,
    reset_api_version: str,
) -> str:
    """Hash all state that can affect safe persistent-worker compatibility."""

    payload = {
        "schema": "rolloutbench-worker-compatibility-v1",
        "runtime_file_hashes": dict(runtime_file_hashes),
        "model_revision": model_revision,
        "init_environment": dict(init_environment),
        "dtype": dtype,
        "backend": backend,
        "workload_fingerprint": workload_fingerprint,
        "gpu_arch": gpu_arch,
        "reset_api_version": reset_api_version,
    }
    empty = [
        field
        for field in (
            "runtime_file_hashes",
            "model_revision",
            "init_environment",
            "dtype",
            "backend",
            "workload_fingerprint",
            "gpu_arch",
            "reset_api_version",
        )
        if not payload[field]
    ]
    if empty:
        raise ValueError(f"compatibility key fields must be nonempty: {empty}")
    return hashlib.sha256(_canonical(payload)).hexdigest()


def validate_reset_proof(
    expected_compatibility_key: str, proof: Mapping[str, Any] | None
) -> dict[str, Any]:
    errors: list[str] = []
    if proof is None:
        errors.append("reset proof is missing")
    else:
        if proof.get("compatibility_key") != expected_compatibility_key:
            errors.append("compatibility key mismatch")
        for field in RESET_PROOF_FIELDS:
            if field not in proof:
                errors.append(f"missing reset proof field {field}")
            elif proof[field] is not True:
                errors.append(f"reset proof field {field} is not true")
    valid = not errors
    return {
        "valid": valid,
        "worker_mode": "persistent" if valid else "one_shot",
        "fail_closed": not valid,
        "expected_compatibility_key": expected_compatibility_key,
        "errors": errors,
    }


def evaluate_confirmation_reuse(
    episode: Mapping[str, Any],
    *,
    compile_artifact_receipt: Mapping[str, Any] | None,
    expected_compatibility_key: str,
) -> dict[str, Any]:
    """Keep K01->K02 compile lineage distinct from persistent model-state reuse."""

    declared_inputs = episode.get("reuse", {}).get("inputs", [])
    expected = {
        "artifact": "torch_compile_cache",
        "episode_id": "K01",
        "required": True,
    }
    is_k02_confirmation = episode.get("episode_id") == "K02"
    exact_manifest = declared_inputs == [expected]

    errors: list[str] = []
    if not is_k02_confirmation:
        errors.append("episode is not the K02 confirmation")
    if not exact_manifest:
        errors.append("K02 compile-cache lineage manifest is not exactly K01")
    if not isinstance(compile_artifact_receipt, Mapping):
        errors.append("declared K01 compile-cache artifact is unavailable")
    else:
        if compile_artifact_receipt.get("source_episode_id") != "K01":
            errors.append("compile artifact source episode mismatch")
        if compile_artifact_receipt.get("artifact") != "torch_compile_cache":
            errors.append("compile artifact type mismatch")
        if compile_artifact_receipt.get("compatibility_key") != expected_compatibility_key:
            errors.append("compatibility key mismatch")
        digest = compile_artifact_receipt.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            errors.append("compile artifact digest is invalid")
    compile_allowed = not errors
    return {
        "episode_id": episode.get("episode_id"),
        "compile_artifact_reuse": {
            "allowed": compile_allowed,
            "source_episode_id": "K01" if compile_allowed else None,
            "artifact": "torch_compile_cache" if compile_allowed else None,
            "model_state_reuse_implied": False,
        },
        "persistent_model_state_reuse": {
            "allowed": False,
            "reason": "confirmation_requires_fresh_process",
            "requires_reset_proof": False,
            "reset_proof_is_not_sufficient": True,
            "compile_artifact_reuse_implied": False,
        },
        "confirmation_independence_preserved": not errors,
        "errors": errors,
    }
