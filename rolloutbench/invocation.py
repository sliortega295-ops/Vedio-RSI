from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Mapping

from .quality_contract import K22_FAILURE_CONTRACT
from .runtime_checkout import verify_runtime_receipt

_SAFE_ID = re.compile(r"[A-Za-z0-9_.-]+")
_MOTION_SUFFIX = "motion score: 30."
_SYSTEM_EXECUTABLE_PATH = (
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)


class InvocationError(RuntimeError):
    """Raised when an episode cannot be bound to one exact GPU invocation."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_exact(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise InvocationError(f"refusing to overwrite conflicting input: {path}")
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
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def materialize_formal_prompt(
    prompt_root: Path | str, prompt_id: str, prompt_text: str
) -> dict[str, Any]:
    """Write one content-addressed SANA prompt with the frozen motion suffix."""

    if not isinstance(prompt_id, str) or not prompt_id:
        raise InvocationError("prompt_id must be nonempty")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise InvocationError("formal prompt must be nonempty")
    raw = prompt_text.strip()
    if "\n" in raw or "\r" in raw:
        raise InvocationError("formal prompt must contain exactly one line")
    rendered = raw if raw.endswith(_MOTION_SUFFIX) else f"{raw} {_MOTION_SUFFIX}"
    content = (rendered + "\n").encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    target = Path(prompt_root).resolve() / f"{digest}.txt"
    _atomic_write_exact(target, content)
    return {
        "prompt_id": prompt_id,
        "path": str(target),
        "sha256": digest,
        "raw_prompt": raw,
        "rendered_prompt": rendered,
    }


def _validated_artifact(
    receipt: Mapping[str, Any],
    materialized_root: Path,
    *,
    episode_id: str,
    kind: str,
) -> Path:
    if receipt.get("episode_id") != episode_id:
        raise InvocationError("materialization receipt episode mismatch")
    matches = [row for row in receipt.get("artifacts", []) if row.get("kind") == kind]
    if len(matches) != 1:
        raise InvocationError(f"expected one materialized {kind} artifact")
    row = matches[0]
    relative = Path(str(row.get("relative_path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise InvocationError("materialized artifact path is unsafe")
    target = (materialized_root / relative).resolve()
    try:
        target.relative_to(materialized_root.resolve())
    except ValueError as exc:
        raise InvocationError("materialized artifact escapes its root") from exc
    if not target.is_file() or _sha256(target) != row.get("sha256"):
        raise InvocationError(f"materialized {kind} artifact is missing or corrupt")
    return target


def _load_config_environment(config_path: Path, repo_root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    profile_name = config.get("model_profile")
    if not isinstance(profile_name, str) or not _SAFE_ID.fullmatch(profile_name):
        raise InvocationError("candidate config has an invalid model_profile")
    profile_path = repo_root / "models" / f"{profile_name}.toml"
    try:
        with profile_path.open("rb") as handle:
            profile = tomllib.load(handle)
    except OSError as exc:
        raise InvocationError(f"model profile is unavailable: {profile_path}") from exc
    environment: dict[str, str] = {}
    for source in (profile.get("env", {}), config.get("env", {})):
        if not isinstance(source, Mapping):
            raise InvocationError("config environment must be a mapping")
        for key, value in source.items():
            if not isinstance(key, str) or not isinstance(value, (str, int, float, bool)):
                raise InvocationError("config environment contains a non-scalar value")
            environment[key] = str(value)
    return environment, {**profile, **config}


def _safe_component(value: Any, *, label: str) -> str:
    text = str(value)
    if not _SAFE_ID.fullmatch(text):
        raise InvocationError(f"{label} is not a safe path component")
    return text


def _validate_harness(
    repository: Path,
    plan_source: Mapping[str, Any],
    *,
    require_clean: bool,
) -> dict[str, Any]:
    expected_revision = plan_source.get("revision")
    if not isinstance(expected_revision, str) or not re.fullmatch(
        r"[0-9a-f]{40,64}", expected_revision
    ):
        raise InvocationError("plan source revision is invalid")
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
        raise InvocationError("cannot validate the harness Git checkout") from exc
    if revision != expected_revision:
        raise InvocationError("harness HEAD does not match the experiment plan")
    if require_clean and (plan_source.get("tree_clean") is not True or dirty):
        raise InvocationError("formal invocation requires the clean planned harness")
    relative_paths = (
        "models/sana_video_2b_h100.toml",
        "models/sana_video_2b_h100/baseline/scripts/run_sana_video_2b_gpu.sh",
        "models/sana_video_2b_h100/baseline/gpu_infer.py",
        "models/sana_video_2b_h100/baseline/gpu_guard.py",
    )
    hashes: dict[str, str] = {}
    for relative in relative_paths:
        path = repository / relative
        if not path.is_file():
            raise InvocationError(f"planned harness file is unavailable: {relative}")
        hashes[relative] = _sha256(path)
    return {
        "revision": revision,
        "tree_clean_at_invocation": not dirty,
        "critical_file_sha256": hashes,
    }


def build_episode_invocation(
    *,
    repo_root: Path | str,
    experiment_root: Path | str,
    plan_id: str,
    plan_sha256: str,
    run_sha256: str,
    run: Mapping[str, Any],
    episode: Mapping[str, Any],
    worker: Mapping[str, Any],
    materialized_root: Path | str,
    materialization_receipt: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
    lease_files: Mapping[str, Path | str],
    plan_source: Mapping[str, Any],
    quality_pair: Mapping[str, Any] | None = None,
    require_clean_harness: bool = True,
) -> dict[str, Any]:
    """Bind one public episode or quality pair to an argv/env/output contract."""

    if "golden" in episode:
        raise InvocationError("invocation accepts only a public episode")
    repository = Path(repo_root).resolve()
    harness = _validate_harness(
        repository, plan_source, require_clean=require_clean_harness
    )
    experiment = Path(experiment_root).resolve()
    derived = Path(materialized_root).resolve()
    episode_id = _safe_component(episode.get("episode_id"), label="episode_id")
    run_id = _safe_component(run.get("run_id"), label="run_id")
    safe_plan_id = _safe_component(plan_id, label="plan_id")
    if not re.fullmatch(r"[0-9a-f]{64}", plan_sha256):
        raise InvocationError("plan_sha256 must be a lowercase SHA-256 digest")
    if not re.fullmatch(r"[0-9a-f]{64}", run_sha256):
        raise InvocationError("run_sha256 must be a lowercase SHA-256 digest")
    candidate = episode.get("candidate")
    if not isinstance(candidate, Mapping):
        raise InvocationError("episode candidate is missing")
    runtime_checkout = episode.get("runtime_checkout")
    if not isinstance(runtime_checkout, Mapping):
        raise InvocationError("episode runtime checkout contract is missing")
    runtime_ref = runtime_checkout.get("git_ref")
    try:
        verified_runtime_receipt = verify_runtime_receipt(
            repository, runtime_receipt, runtime_checkout
        )
    except (RuntimeError, ValueError) as exc:
        raise InvocationError("runtime checkout receipt cannot be reverified") from exc
    checkout = Path(verified_runtime_receipt["worktree_path"])
    runtime_root = checkout / "external" / "sol_runtime"
    if not runtime_root.is_dir():
        raise InvocationError("candidate runtime checkout is incomplete")

    gpu_uuid = str(worker.get("gpu_uuid", ""))
    if gpu_uuid not in lease_files:
        raise InvocationError(f"no cooperative lease is declared for {gpu_uuid}")
    lease_file = Path(lease_files[gpu_uuid]).resolve()
    cache_namespace = Path(str(run.get("cache_namespace", "")))
    if cache_namespace.is_absolute() or ".." in cache_namespace.parts:
        raise InvocationError("run cache namespace is unsafe")
    cache_scope = _safe_component(episode.get("cache_scope_key"), label="cache_scope_key")
    cache_root = experiment / cache_namespace / cache_scope

    variant_parts = ["primary"]
    pair_id: str | None = None
    quality_role: str | None = None
    seed = 42
    if quality_pair is not None:
        candidate_type = episode.get("candidate_type")
        if candidate_type not in {"lossy_cache", "dense_reference"}:
            raise InvocationError(
                "quality pairs are valid only for Cache candidates or the dense reference"
            )
        pair_id = str(quality_pair.get("pair_id", ""))
        if candidate_type == "dense_reference":
            declared_pairs = [
                row
                for planned_episode in run.get("episodes", [])
                if isinstance(planned_episode, Mapping)
                for row in planned_episode.get("quality_pairs", [])
                if row.get("pair_id") == pair_id
            ]
            quality_role = "dense"
        else:
            declared_pairs = [
                row
                for row in episode.get("quality_pairs", [])
                if row.get("pair_id") == pair_id
            ]
            quality_role = "candidate"
        if len(declared_pairs) != 1 or _canonical(declared_pairs[0]) != _canonical(
            quality_pair
        ):
            raise InvocationError("quality pair is not declared for this episode")
        quality_pair = declared_pairs[0]
        seed = int(quality_pair.get("seed", -1))
        if seed not in {42, 12345}:
            raise InvocationError("quality pair has an unsupported seed")
        suite = _safe_component(quality_pair.get("prompt_suite"), label="prompt_suite")
        variant_parts = ["quality-v1", suite, f"seed-{seed}"]
    output_dir = (
        experiment
        / "runs"
        / safe_plan_id
        / plan_sha256
        / run_id
        / run_sha256
        / episode_id
    )
    for part in variant_parts:
        output_dir /= part
    output_dir.mkdir(parents=True, exist_ok=True)

    config_descriptor = candidate.get("config")
    probe_descriptor = candidate.get("probe")
    failure_contract = episode.get("expected_failure_contract")
    if failure_contract is not None and (
        not isinstance(failure_contract, Mapping)
        or dict(failure_contract) != dict(K22_FAILURE_CONTRACT)
        or episode_id != "K22"
        or quality_pair is not None
    ):
        raise InvocationError("episode deterministic failure contract is invalid")
    common_env = {
        # SubprocessStageExecutor intentionally replaces, rather than inherits,
        # the parent environment. Keep the system compiler toolchain reachable
        # without admitting an operator-specific login PATH.
        "PATH": _SYSTEM_EXECUTABLE_PATH,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
        "TRITON_CACHE_DIR": str(cache_root / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "torchinductor"),
        "TMPDIR": str(cache_root / "tmp"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    prompt_receipt: dict[str, Any] | None = None

    if config_descriptor is not None:
        config_path = _validated_artifact(
            materialization_receipt,
            derived,
            episode_id=episode_id,
            kind="config",
        )
        environment, merged = _load_config_environment(config_path, repository)
        config_id = merged.get("id")
        if not isinstance(config_id, str) or not config_id:
            raise InvocationError("candidate config has no id")
        if failure_contract is not None and (
            config_id != failure_contract["config_id"]
            or config_descriptor.get("blob_sha256")
            != failure_contract["config_sha256"]
            or runtime_ref != failure_contract["runtime_ref"]
        ):
            raise InvocationError("expected failure config/runtime binding drifted")
        submodule = Path(str(merged.get("submodule", "")))
        run_script = Path(str(merged.get("run_script", "")))
        if (
            submodule.is_absolute()
            or run_script.is_absolute()
            or ".." in submodule.parts
            or ".." in run_script.parts
        ):
            raise InvocationError("model run script path is unsafe")
        script = (repository / submodule / run_script).resolve()
        if not script.is_file():
            raise InvocationError(f"model run script is unavailable: {script}")
        if quality_pair is None:
            prompt_raw = environment.get("SANA_PROMPT_FILE", "")
            prompt_path = Path(prompt_raw)
            prompt_path = (
                prompt_path if prompt_path.is_absolute() else repository / prompt_path
            ).resolve()
            if not prompt_path.is_file():
                raise InvocationError("default formal prompt is unavailable")
        else:
            prompt_receipt = materialize_formal_prompt(
                experiment / "prompts",
                pair_id or "",
                str(quality_pair.get("prompt", "")),
            )
            prompt_path = Path(prompt_receipt["path"])
        environment.update(common_env)
        environment.update(
            {
                "OUT_DIR": str(output_dir),
                "AUTOVIDEO_REPO_ROOT": str(repository),
                "AUTOVIDEO_CONFIG_ID": config_id,
                "SANA_RUNTIME_ROOT": str(runtime_root),
                "SANA_GPU_LEASE_FILE": str(lease_file),
                "SANA_PROMPT_FILE": str(prompt_path),
                "SANA_WORKLOAD_SEED": str(seed),
                "ROLLOUTBENCH_RUNTIME_REF": str(runtime_ref),
                "ROLLOUTBENCH_REQUIRED_RUNTIME_PATHS_JSON": json.dumps(
                    runtime_checkout["required_runtime_paths"],
                    separators=(",", ":"),
                ),
            }
        )
        if failure_contract is not None:
            environment.update(
                {
                    "ROLLOUTBENCH_EXPECTED_FAILURE_EPISODE": str(
                        failure_contract["episode_id"]
                    ),
                    "ROLLOUTBENCH_EXPECTED_FAILURE_CODE": str(
                        failure_contract["failure_code"]
                    ),
                    "ROLLOUTBENCH_EXPECTED_FAILURE_MARKER": str(
                        failure_contract["expected_log_marker"]
                    ),
                    "ROLLOUTBENCH_EXPECTED_FAILURE_CONFIG_SHA256": str(
                        failure_contract["config_sha256"]
                    ),
                    "ROLLOUTBENCH_EXPECTED_FAILURE_RUNTIME_REF": str(
                        failure_contract["runtime_ref"]
                    ),
                }
            )
        argv = ["bash", str(script)]
        kind = (
            "expected_fail_closed_generation"
            if failure_contract is not None
            else "config_generation"
        )
        output_path = (
            output_dir / "benchmark.json"
            if failure_contract is not None
            else output_dir / "out.mp4"
        )
        cwd = repository
    elif probe_descriptor is not None:
        if quality_pair is not None:
            raise InvocationError("preflight probes cannot receive quality pairs")
        source = _validated_artifact(
            materialization_receipt,
            derived,
            episode_id=episode_id,
            kind="probe_source",
        )
        with (repository / "models" / "sana_video_2b_h100.toml").open("rb") as handle:
            profile = tomllib.load(handle)
        runtime_python = str(profile.get("env", {}).get("SANA_PYTHON_BIN", ""))
        if not runtime_python:
            raise InvocationError("SANA runtime Python is missing from the model profile")
        output_path = output_dir / "probe-result.json"
        guard_dir = repository / "models" / "sana_video_2b_h100" / "baseline"
        argv = [
            runtime_python,
            str(source),
            "--lease-file",
            str(lease_file),
            "--guard-dir",
            str(guard_dir),
        ]
        if "--runtime-python-root" in source.read_text(encoding="utf-8"):
            argv.extend(["--runtime-python-root", str(runtime_root / "python")])
        argv.extend(["--out", str(output_path)])
        environment = {
            **common_env,
            "ROLLOUTBENCH_RUNTIME_REF": str(runtime_ref),
        }
        kind = "gpu_preflight_probe"
        cwd = checkout
    else:
        raise InvocationError("candidate has neither a config nor a probe")

    identity = {
        "argv": argv,
        "env": environment,
        "cwd": str(cwd),
        "output_path": str(output_path),
        "episode_id": episode_id,
        "run_id": run_id,
        "quality_pair_id": pair_id,
        "quality_role": quality_role,
        "gpu_uuid": gpu_uuid,
        "runtime_ref": runtime_ref,
        "runtime_tree_oid": runtime_checkout["runtime_tree_oid"],
        "expected_failure_contract": (
            dict(failure_contract) if failure_contract is not None else None
        ),
        "harness": harness,
    }
    return {
        "schema_version": 1,
        "kind": kind,
        **identity,
        "output_dir": str(output_dir),
        "cache_root": str(cache_root),
        "lease_file": str(lease_file),
        "prompt": prompt_receipt,
        "quality_artifact_id": (
            quality_pair[f"{quality_role}_artifact_id"]
            if quality_pair is not None and quality_role is not None
            else None
        ),
        "command_fingerprint": hashlib.sha256(_canonical(identity)).hexdigest(),
        "execution_status": "NOT_RUN",
        "performance_claim": False,
    }
