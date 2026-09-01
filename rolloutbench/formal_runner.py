"""Controlled formal-stage construction above :mod:`pilot_runner`.

This module deliberately has no caller-supplied argv interface.  GPU dispatch
is owned by :mod:`rolloutbench.formal_dispatch`, after external authorization.
"""

from __future__ import annotations

import hashlib
import fcntl
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from models.sana_video_2b_h100.baseline.gpu_guard import locked_idle_lease

from .invocation import InvocationError, build_episode_invocation
from .quality_contract import DENSE_REFERENCE_ID, K22_FAILURE_CONTRACT
from .pilot_runner import (
    RunContext,
    StageExecutor,
    Unit,
    expand_run_units,
    resume_unit,
)
from .vbench_runner import VBenchContractError, build_vbench_pair_plan, parse_vbench_pair_results


class FormalRunnerError(RuntimeError):
    """Raised when a stage cannot be derived from frozen formal evidence."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _fingerprint(invocation: Mapping[str, Any]) -> str:
    keys = (
        "argv", "env", "cwd", "output_path", "episode_id", "run_id",
        "quality_pair_id", "quality_role", "gpu_uuid", "runtime_ref",
        "runtime_tree_oid", "expected_failure_contract", "harness",
    )
    return hashlib.sha256(_canonical({key: invocation.get(key) for key in keys})).hexdigest()


def _episode(context: RunContext, episode_id: str) -> Mapping[str, Any]:
    candidates = [row for row in context.run.get("episodes", []) if row.get("episode_id") == episode_id]
    dense = context.run.get("quality_dense_reference")
    if isinstance(dense, Mapping) and dense.get("episode_id") == episode_id:
        candidates.append(dense)
    if len(candidates) != 1:
        raise FormalRunnerError("planned episode cannot be resolved uniquely")
    return candidates[0]


def _wrap(unit: Unit, base: Mapping[str, Any]) -> dict[str, Any]:
    if base.get("command_fingerprint") != _fingerprint(base):
        raise FormalRunnerError("canonical invocation command fingerprint mismatch")
    pair_id = unit.quality_pair["pair_id"] if unit.quality_pair else None
    if base.get("episode_id") != unit.episode_id or base.get("quality_pair_id") != pair_id:
        raise FormalRunnerError("canonical invocation does not bind its planned unit")
    return {
        **base,
        "unit_id": unit.unit_id,
        "unit_kind": unit.unit_kind,
        "episode_id": unit.episode_id,
        "run_id": base.get("run_id"),
        "preparation_episode_ids": list(unit.preparation_episode_ids or (unit.episode_id,)),
    }


def _profile_value(profile: Mapping[str, Any], name: str) -> str:
    value = profile.get(name)
    if not isinstance(value, str) or not value:
        raise FormalRunnerError(f"formal profile is missing {name}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    """Seal bytes once, permitting only an identical idempotent replay."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.write.lock")
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        temporary: Path | None = None
        try:
            if path.exists():
                if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
                    raise FormalRunnerError(
                        f"refusing to overwrite conflicting formal evidence: {path}"
                    )
                return
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
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
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _load_regular_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise FormalRunnerError(f"{label} must be an absolute regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FormalRunnerError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise FormalRunnerError(f"{label} must be an object")
    return value


def _validate_lpips_receipt(
    path: Path, pair: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    lpips = _load_regular_json(path, "LPIPS receipt")
    identity = {key: value for key, value in lpips.items() if key != "result_fingerprint"}
    values = lpips.get("values")
    mean = lpips.get("mean")
    videos = plan.get("videos")
    if not isinstance(videos, Mapping):
        raise FormalRunnerError("formal VBench plan has no video bindings")
    dense = videos.get("dense")
    candidate = videos.get("candidate")
    if (
        lpips.get("pair_id") != pair.get("pair_id")
        or lpips.get("status") != "COMPLETED"
        or lpips.get("metric") != "lpips_v0.1_alex"
        or lpips.get("frame_count") != 81
        or lpips.get("frame_shape_hwc") != [480, 832, 3]
        or lpips.get("result_fingerprint")
        != hashlib.sha256(_canonical(identity)).hexdigest()
        or not isinstance(dense, Mapping)
        or not isinstance(candidate, Mapping)
        or lpips.get("dense_video", {}).get("path") != dense.get("path")
        or lpips.get("dense_video", {}).get("sha256") != dense.get("sha256")
        or lpips.get("candidate_video", {}).get("path") != candidate.get("path")
        or lpips.get("candidate_video", {}).get("sha256") != candidate.get("sha256")
        or not isinstance(values, list)
        or len(values) != 81
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in values
        )
        or not isinstance(mean, (int, float))
        or isinstance(mean, bool)
        or not math.isfinite(float(mean))
        or not math.isclose(
            float(mean),
            sum(float(value) for value in values) / 81,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise FormalRunnerError("LPIPS receipt is not a finite matched 81-frame result")
    for role, video in (("dense", dense), ("candidate", candidate)):
        video_path = Path(str(video.get("path", "")))
        if (
            not video_path.is_absolute()
            or not video_path.is_file()
            or video_path.is_symlink()
            or _sha256(video_path) != video.get("sha256")
        ):
            raise FormalRunnerError(f"{role} video changed after quality planning")
    return lpips


def verify_formal_compare_output(
    path: Path | str,
    pair: Mapping[str, Any],
    quality_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild and verify the complete VBench/LPIPS evidence chain."""

    output_path = Path(path)
    payload = _load_regular_json(output_path, "formal compare output")
    chain = payload.get("evidence_chain")
    if not isinstance(chain, Mapping) or chain.get("schema_version") != 1:
        raise FormalRunnerError("formal compare evidence chain is missing")
    plan = chain.get("vbench_plan")
    vbench_entry = chain.get("vbench_execution_receipt")
    lpips_entry = chain.get("lpips_receipt")
    if (
        not isinstance(plan, Mapping)
        or not isinstance(vbench_entry, Mapping)
        or not isinstance(lpips_entry, Mapping)
        or chain.get("plan_fingerprint") != plan.get("plan_fingerprint")
        or chain.get("quality_protocol_fingerprint")
        != plan.get("quality_protocol_fingerprint")
    ):
        raise FormalRunnerError("formal compare evidence bindings are incomplete")

    receipt_path = Path(str(vbench_entry.get("path", "")))
    receipt = _load_regular_json(receipt_path, "VBench execution receipt")
    if (
        _sha256(receipt_path) != vbench_entry.get("sha256")
        or receipt.get("receipt_fingerprint") != vbench_entry.get("receipt_fingerprint")
    ):
        raise FormalRunnerError("VBench execution receipt binding changed")
    try:
        parsed = parse_vbench_pair_results(
            pair,
            quality_protocol=quality_protocol,
            plan=plan,
            execution_receipt_path=receipt_path,
        )
    except (KeyError, TypeError, ValueError, VBenchContractError) as exc:
        raise FormalRunnerError("formal VBench evidence chain is invalid") from exc

    lpips_path = Path(str(lpips_entry.get("path", "")))
    lpips = _validate_lpips_receipt(lpips_path, pair, plan)
    if (
        _sha256(lpips_path) != lpips_entry.get("sha256")
        or lpips.get("result_fingerprint") != lpips_entry.get("result_fingerprint")
    ):
        raise FormalRunnerError("LPIPS receipt binding changed")
    expected_chain = {
        "schema_version": 1,
        "plan_fingerprint": plan["plan_fingerprint"],
        "quality_protocol_fingerprint": plan["quality_protocol_fingerprint"],
        "vbench_plan": dict(plan),
        "vbench_execution_receipt": {
            "path": str(receipt_path),
            "sha256": _sha256(receipt_path),
            "receipt_fingerprint": receipt["receipt_fingerprint"],
        },
        "lpips_receipt": {
            "path": str(lpips_path),
            "sha256": _sha256(lpips_path),
            "result_fingerprint": lpips["result_fingerprint"],
        },
    }
    expected = {
        **parsed,
        "evidence_chain": expected_chain,
        "lpips": {
            "path": str(lpips_path),
            "sha256": _sha256(lpips_path),
            "result_fingerprint": lpips["result_fingerprint"],
            "mean": lpips["mean"],
            "values": lpips["values"],
        },
        "threshold_decision": "NOT_EMITTED",
    }
    if payload != expected:
        raise FormalRunnerError("formal compare output disagrees with fresh evidence parsing")
    return payload


def _stage_output(context: RunContext, unit: Unit, suffix: str) -> Path:
    safe = hashlib.sha256(unit.unit_id.encode()).hexdigest()
    return (
        Path(context.preparation["experiment_root"])
        / "runs"
        / context.plan_id
        / context.plan_sha256
        / str(context.run["run_id"])
        / context.run_sha256
        / "formal-stages"
        / f"{safe}{suffix}"
    )


def _completed_video(context: RunContext, ledger: Any, unit: Unit, invocation: Mapping[str, Any], artifact_id: str) -> tuple[Path, dict[str, str]]:
    action = resume_unit(context, ledger, unit, invocation)
    if action["action"] != "reuse_completed":
        raise FormalRunnerError("required generation evidence is incomplete")
    attempt = int(action["attempt"])
    rows = [row for row in ledger.read() if row["event_type"] == "stage_completed" and row["payload"].get("episode_id") == unit.unit_id and row["payload"].get("attempt") == attempt]
    if len(rows) != 1:
        raise FormalRunnerError("generation completion evidence is ambiguous")
    payload = rows[0]["payload"]
    path = Path(str(payload.get("output_path", "")))
    return path, {"artifact_id": artifact_id, "video_path": str(path), "sha256": str(payload.get("output_sha256", ""))}


def _completed_worker(context: RunContext, ledger: Any, unit: Unit) -> Mapping[str, Any]:
    rows = [
        row
        for row in ledger.read()
        if row["event_type"] == "stage_completed"
        and row["payload"].get("episode_id") == unit.unit_id
        and row["payload"].get("stage") == unit.unit_kind
    ]
    if len(rows) != 1:
        raise FormalRunnerError("completed dependency has no unique worker evidence")
    worker_id = rows[0]["payload"].get("worker_id")
    workers = context.run.get("workers")
    matches = [
        worker
        for worker in workers if isinstance(worker, Mapping)
        and worker.get("worker_id") == worker_id
    ] if isinstance(workers, list) else []
    if len(matches) != 1:
        raise FormalRunnerError("completed dependency worker is outside the run plan")
    return matches[0]


def _completed_invocation(
    context: RunContext,
    ledger: Any,
    unit: Unit,
    *,
    lease_files: Mapping[str, Path | str],
    profile: Mapping[str, Any],
    quality_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    invocation = build_formal_invocation(
        context,
        unit,
        _completed_worker(context, ledger, unit),
        lease_files=lease_files,
        profile=profile,
        quality_protocol=quality_protocol,
        ledger=ledger,
    )
    if resume_unit(context, ledger, unit, invocation)["action"] != "reuse_completed":
        raise FormalRunnerError("dependency completion evidence is incomplete")
    return invocation


def build_formal_invocation(
    context: RunContext,
    unit: Unit,
    worker: Mapping[str, Any],
    *,
    lease_files: Mapping[str, Path | str],
    profile: Mapping[str, Any],
    quality_protocol: Mapping[str, Any],
    ledger: Any | None = None,
) -> dict[str, Any]:
    """Derive one stage invocation from frozen context; never accept argv input."""

    if unit.unit_kind in {"primary", "quality_candidate_generate", "quality_dense_generate"}:
        episode = _episode(
            context,
            DENSE_REFERENCE_ID
            if unit.unit_kind == "quality_dense_generate"
            else unit.episode_id,
        )
        try:
            base = build_episode_invocation(
                repo_root=Path(__file__).resolve().parents[1],
                experiment_root=context.preparation["experiment_root"],
                plan_id=context.plan_id,
                plan_sha256=context.plan_sha256,
                run_sha256=context.run_sha256,
                run=context.run,
                episode=episode,
                worker=worker,
                materialized_root=context.preparation["derived_root"],
                materialization_receipt=context.preparation["materialization_receipts"][episode["episode_id"]],
                runtime_receipt=context.preparation["runtime_receipts"][episode["episode_id"]],
                lease_files=lease_files,
                plan_source=context.run.get("source", context.preparation.get("source", {})),
                quality_pair=unit.quality_pair,
            )
        except (KeyError, InvocationError, TypeError, ValueError) as exc:
            raise FormalRunnerError("cannot build canonical generation invocation") from exc
        return _wrap(unit, base)

    if unit.unit_kind in {"quality_dense_vbench", "quality_candidate_vbench"}:
        if ledger is None or unit.quality_pair is None:
            raise FormalRunnerError("VBench construction requires verified generation evidence")
        units = {item.unit_id: item for item in expand_run_units(context.run)}
        dense_id, candidate_id = unit.depends_on
        dense_unit, candidate_unit = units[dense_id], units[candidate_id]
        dense_invocation = _completed_invocation(
            context, ledger, dense_unit, lease_files=lease_files,
            profile=profile, quality_protocol=quality_protocol,
        )
        candidate_invocation = _completed_invocation(
            context, ledger, candidate_unit, lease_files=lease_files,
            profile=profile, quality_protocol=quality_protocol,
        )
        dense_video, dense_receipt = _completed_video(context, ledger, dense_unit, dense_invocation, unit.quality_pair["dense_artifact_id"])
        candidate_video, candidate_receipt = _completed_video(context, ledger, candidate_unit, candidate_invocation, unit.quality_pair["candidate_artifact_id"])
        plan = build_vbench_role_plan(
            context,
            unit,
            dense_video_path=dense_video,
            candidate_video_path=candidate_video,
            dense_receipt=dense_receipt,
            candidate_receipt=candidate_receipt,
            profile=profile,
            quality_protocol=quality_protocol,
            gpu_uuid=str(worker.get("gpu_uuid", "")),
        )
        role = "dense" if unit.unit_kind == "quality_dense_vbench" else "candidate"
        official = plan["invocations"][role]
        gpu_uuid = str(worker.get("gpu_uuid", ""))
        if gpu_uuid not in lease_files:
            raise FormalRunnerError("VBench stage has no exact GPU lease file")
        return {**official, "unit_id": unit.unit_id, "unit_kind": unit.unit_kind, "episode_id": unit.episode_id, "run_id": context.run["run_id"], "quality_pair_id": unit.quality_pair["pair_id"], "preparation_episode_ids": list(unit.preparation_episode_ids), "output_path": str(_stage_output(context, unit, ".vbench.json")), "gpu_uuid": gpu_uuid, "lease_file": str(Path(lease_files[gpu_uuid]).resolve()), "executor_gpu_lock": True, "formal_vbench_plan": plan, "formal_vbench_role": role}
    if unit.unit_kind == "quality_compare":
        if ledger is None or unit.quality_pair is None:
            raise FormalRunnerError("compare requires immutable quality-stage evidence")
        units = {item.unit_id: item for item in expand_run_units(context.run)}
        dense_id, candidate_id, lpips_id = unit.depends_on
        vbench_units = (units[dense_id], units[candidate_id])
        vbench_invocations = [
            _completed_invocation(
                context, ledger, item, lease_files=lease_files,
                profile=profile, quality_protocol=quality_protocol,
            )
            for item in vbench_units
        ]
        if (
            vbench_invocations[0]["formal_vbench_plan"]["plan_fingerprint"]
            != vbench_invocations[1]["formal_vbench_plan"]["plan_fingerprint"]
        ):
            raise FormalRunnerError("paired VBench roles disagree on their frozen plan")
        lpips_unit = units[lpips_id]
        lpips_invocation = _completed_invocation(
            context, ledger, lpips_unit, lease_files=lease_files,
            profile=profile, quality_protocol=quality_protocol,
        )
        return {"argv": ["formal-compare"], "cwd": str(Path(context.preparation["experiment_root"])), "env": {}, "output_path": str(_stage_output(context, unit, ".scores.json")), "unit_id": unit.unit_id, "unit_kind": unit.unit_kind, "episode_id": unit.episode_id, "run_id": context.run["run_id"], "quality_pair_id": unit.quality_pair["pair_id"], "preparation_episode_ids": list(unit.preparation_episode_ids), "formal_compare": {"pair": unit.quality_pair, "quality_protocol": quality_protocol, "plan": vbench_invocations[0]["formal_vbench_plan"], "dense_result": vbench_invocations[0]["output_path"], "candidate_result": vbench_invocations[1]["output_path"], "lpips": lpips_invocation["output_path"]}}
    if unit.unit_kind == "quality_lpips":
        if ledger is None or unit.quality_pair is None:
            raise FormalRunnerError("LPIPS construction requires verified generation evidence")
        units = {item.unit_id: item for item in expand_run_units(context.run)}
        dense_id, candidate_id = unit.depends_on
        dense_unit, candidate_unit = units[dense_id], units[candidate_id]
        dense_invocation = _completed_invocation(
            context, ledger, dense_unit, lease_files=lease_files,
            profile=profile, quality_protocol=quality_protocol,
        )
        candidate_invocation = _completed_invocation(
            context, ledger, candidate_unit, lease_files=lease_files,
            profile=profile, quality_protocol=quality_protocol,
        )
        dense_video, _ = _completed_video(context, ledger, dense_unit, dense_invocation, unit.quality_pair["dense_artifact_id"])
        candidate_video, _ = _completed_video(context, ledger, candidate_unit, candidate_invocation, unit.quality_pair["candidate_artifact_id"])
        cache = _profile_value(profile, "vbench_cache_path")
        output = _stage_output(context, unit, ".lpips.json")
        gpu_uuid = str(worker.get("gpu_uuid", ""))
        if gpu_uuid not in lease_files:
            raise FormalRunnerError("LPIPS stage has no exact GPU lease file")
        return {"argv": [_profile_value(profile, "vbench_python_bin"), "-m", "rolloutbench.lpips_cli", "--dense-video", str(dense_video), "--candidate-video", str(candidate_video), "--output", str(output), "--pair-id", unit.quality_pair["pair_id"]], "cwd": str(Path(__file__).resolve().parents[1]), "env": {"PYTHONPATH": str(Path(__file__).resolve().parents[1]), "CUDA_VISIBLE_DEVICES": gpu_uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "TORCH_HOME": str(Path(cache) / "torch_home"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONNOUSERSITE": "1"}, "output_path": str(output), "unit_id": unit.unit_id, "unit_kind": unit.unit_kind, "episode_id": unit.episode_id, "run_id": context.run["run_id"], "quality_pair_id": unit.quality_pair["pair_id"], "preparation_episode_ids": list(unit.preparation_episode_ids), "gpu_uuid": gpu_uuid, "lease_file": str(Path(lease_files[gpu_uuid]).resolve()), "executor_gpu_lock": True, "formal_lpips": True}
    raise FormalRunnerError("unknown formal unit kind")


def build_vbench_role_plan(
    context: RunContext,
    unit: Unit,
    *,
    dense_video_path: Path | str,
    candidate_video_path: Path | str,
    dense_receipt: Mapping[str, Any],
    candidate_receipt: Mapping[str, Any],
    profile: Mapping[str, Any],
    quality_protocol: Mapping[str, Any],
    gpu_uuid: str,
) -> dict[str, Any]:
    """Build the official VBench plan from already verified generation evidence."""

    if unit.unit_kind not in {"quality_dense_vbench", "quality_candidate_vbench"} or unit.quality_pair is None:
        raise FormalRunnerError("unit is not a VBench quality role")
    try:
        return build_vbench_pair_plan(
            unit.quality_pair,
            quality_protocol=quality_protocol,
            dense_video_path=dense_video_path,
            candidate_video_path=candidate_video_path,
            dense_receipt=dense_receipt,
            candidate_receipt=candidate_receipt,
            vbench_source_path=_profile_value(profile, "vbench_source_path"),
            vbench_source_ref=str(quality_protocol.get("vbench", {}).get("git_ref", "")),
            vbench_cache_path=_profile_value(profile, "vbench_cache_path"),
            python_bin=_profile_value(profile, "vbench_python_bin"),
            output_path=(
                Path(context.preparation["experiment_root"])
                / "runs"
                / context.plan_id
                / context.plan_sha256
                / str(context.run["run_id"])
                / context.run_sha256
                / "vbench"
                / unit.quality_pair["pair_id"]
            ),
            gpu_uuid=gpu_uuid,
        )
    except (VBenchContractError, TypeError, ValueError) as exc:
        raise FormalRunnerError("cannot build pinned official VBench plan") from exc


class FormalStageExecutor:
    """Adapter preserving official VBench argv while materializing one stage file."""

    def __init__(self, delegate: StageExecutor):
        self._delegate = delegate

    def execute(self, invocation: Mapping[str, Any], *, log_dir: Path):
        if invocation.get("executor_gpu_lock") is True:
            lease_file = invocation.get("lease_file")
            gpu_uuid = invocation.get("gpu_uuid")
            if not isinstance(lease_file, str) or not isinstance(gpu_uuid, str):
                raise FormalRunnerError("GPU-locked formal stage lacks lease identity")
            with locked_idle_lease(lease_file) as (lease, _gpu):
                if lease.gpu_uuid != gpu_uuid:
                    raise FormalRunnerError("formal stage lease GPU UUID mismatch")
                return self._execute_unlocked(invocation, log_dir=log_dir)
        return self._execute_unlocked(invocation, log_dir=log_dir)

    def _execute_unlocked(self, invocation: Mapping[str, Any], *, log_dir: Path):
        compare = invocation.get("formal_compare")
        if isinstance(compare, Mapping):
            return self._compare(invocation, compare, log_dir)
        plan = invocation.get("formal_vbench_plan")
        if not isinstance(plan, Mapping):
            result = self._delegate.execute(invocation, log_dir=log_dir)
            failure_contract = invocation.get("expected_failure_contract")
            if isinstance(failure_contract, Mapping):
                return self._accept_expected_failure(
                    invocation, failure_contract, result
                )
            return result
        role = invocation.get("formal_vbench_role")
        official = plan.get("invocations", {}).get(role)
        if role not in {"dense", "candidate"} or not isinstance(official, Mapping):
            raise FormalRunnerError("VBench adapter role is invalid")
        result = self._delegate.execute(official, log_dir=log_dir)
        if result.returncode != 0:
            return result
        matches = [path for path in Path(str(official["output_path"])).glob("results_*_eval_results.json") if path.is_file() and not path.is_symlink()]
        if len(matches) != 1:
            raise FormalRunnerError("official VBench result_glob must resolve exactly one regular JSON")
        source = matches[0]
        _write_atomic(Path(str(invocation["output_path"])), source.read_bytes())
        return result

    def _accept_expected_failure(
        self,
        invocation: Mapping[str, Any],
        failure_contract: Mapping[str, Any],
        result: Any,
    ):
        """Turn only the frozen K22 fail-closed outcome into contract success."""

        if (
            dict(failure_contract) != dict(K22_FAILURE_CONTRACT)
            or invocation.get("episode_id") != "K22"
            or invocation.get("runtime_ref") != K22_FAILURE_CONTRACT["runtime_ref"]
            or result.returncode == 0
        ):
            raise FormalRunnerError("deterministic failure did not fail as declared")
        output = Path(str(invocation.get("output_path", "")))
        if (
            output.name != "benchmark.json"
            or not output.is_file()
            or output.is_symlink()
            or (output.parent / "out.mp4").exists()
        ):
            raise FormalRunnerError("expected failure artifacts are unsafe or inconsistent")
        try:
            benchmark = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FormalRunnerError("expected failure benchmark is invalid") from exc
        if not isinstance(benchmark, Mapping):
            raise FormalRunnerError("expected failure benchmark must be an object")
        child_returncode = benchmark.get("returncode")
        failure = benchmark.get("failure")
        run_log = output.parent / "run.log"
        run_config_path = output.parent / "run_config.json"
        run_config = _load_regular_json(run_config_path, "K22 run configuration")
        expected_failure = {
            "episode_id": K22_FAILURE_CONTRACT["episode_id"],
            "failure_code": K22_FAILURE_CONTRACT["failure_code"],
            "stage": K22_FAILURE_CONTRACT["stage"],
            "expected_log_marker": K22_FAILURE_CONTRACT["expected_log_marker"],
            "observed_marker_count": 1,
            "marker_matched": True,
            "child_returncode": K22_FAILURE_CONTRACT["child_returncode"],
            "config_id": K22_FAILURE_CONTRACT["config_id"],
            "config_sha256": K22_FAILURE_CONTRACT["config_sha256"],
            "runtime_ref": K22_FAILURE_CONTRACT["runtime_ref"],
            "run_log": {
                "path": str(run_log),
                "sha256": _sha256(run_log) if run_log.is_file() else None,
            },
        }
        if (
            benchmark.get("status") != "FAILED"
            or not isinstance(child_returncode, int)
            or isinstance(child_returncode, bool)
            or child_returncode != K22_FAILURE_CONTRACT["child_returncode"]
            or benchmark.get("generation_s") is not None
            or benchmark.get("residual_compute_apps") != []
            or failure != expected_failure
            or not run_log.is_file()
            or run_log.is_symlink()
            or run_log.read_bytes().count(
                K22_FAILURE_CONTRACT["expected_log_marker"].encode("utf-8")
            )
            != 1
            or run_config.get("config_id") != K22_FAILURE_CONTRACT["config_id"]
            or run_config.get("source", {}).get("runtime_authority_sha")
            != K22_FAILURE_CONTRACT["runtime_ref"]
            or run_config.get("expected_failure_contract")
            != {
                key: K22_FAILURE_CONTRACT[key]
                for key in (
                    "episode_id",
                    "failure_code",
                    "expected_log_marker",
                    "config_sha256",
                    "runtime_ref",
                )
            }
        ):
            raise FormalRunnerError("expected failure did not fail closed during generation")
        from .pilot_runner import ProcessResult

        return ProcessResult(
            0,
            result.wall_s,
            result.stdout_path,
            result.stderr_path,
            result.stdout_sha256,
            result.stdout_size_bytes,
            result.stderr_sha256,
            result.stderr_size_bytes,
        )

    def _compare(self, invocation: Mapping[str, Any], compare: Mapping[str, Any], log_dir: Path):
        pair, plan = compare.get("pair"), compare.get("plan")
        dense, candidate, lpips_path = (Path(str(compare.get(key, ""))) for key in ("dense_result", "candidate_result", "lpips"))
        if not isinstance(pair, Mapping) or not isinstance(plan, Mapping) or not all(path.is_file() and not path.is_symlink() for path in (dense, candidate, lpips_path)):
            raise FormalRunnerError("compare inputs are missing or unsafe")
        try:
            lpips = _validate_lpips_receipt(lpips_path, pair, plan)
            values = lpips["values"]
            execution = {"schema_version": 1, "record_type": "vbench_execution_receipt", "status": "COMPLETED", "formality": "FORMAL", "plan_fingerprint": plan["plan_fingerprint"], "quality_protocol_fingerprint": plan["quality_protocol_fingerprint"], "vbench_source": {"path": plan["vbench_source_path"], "ref": plan["vbench_source_ref"], "verification": "FORMAL"}, "invocations": {role: {"command_fingerprint": plan["invocations"][role]["command_fingerprint"], "video_path": plan["videos"][role]["path"], "video_sha256": plan["videos"][role]["sha256"], "result_path": str(path), "result_sha256": _sha256(path)} for role, path in (("dense", dense), ("candidate", candidate))}}
            execution["receipt_fingerprint"] = hashlib.sha256(_canonical(execution)).hexdigest()
            receipt_path = Path(str(invocation["output_path"])).with_suffix(".vbench-receipt.json")
            _write_atomic(receipt_path, json.dumps(execution, sort_keys=True).encode("utf-8") + b"\n")
            protocol = compare.get("quality_protocol")
            parsed = parse_vbench_pair_results(pair, quality_protocol=protocol, plan=plan, execution_receipt_path=receipt_path)
        except (KeyError, TypeError, ValueError, VBenchContractError) as exc:
            raise FormalRunnerError("formal compare evidence is invalid") from exc
        evidence_chain = {
            "schema_version": 1,
            "plan_fingerprint": plan["plan_fingerprint"],
            "quality_protocol_fingerprint": plan["quality_protocol_fingerprint"],
            "vbench_plan": dict(plan),
            "vbench_execution_receipt": {
                "path": str(receipt_path),
                "sha256": _sha256(receipt_path),
                "receipt_fingerprint": execution["receipt_fingerprint"],
            },
            "lpips_receipt": {
                "path": str(lpips_path),
                "sha256": _sha256(lpips_path),
                "result_fingerprint": lpips["result_fingerprint"],
            },
        }
        payload = {
            **parsed,
            "evidence_chain": evidence_chain,
            "lpips": {
                "path": str(lpips_path),
                "sha256": _sha256(lpips_path),
                "result_fingerprint": lpips["result_fingerprint"],
                "mean": lpips["mean"],
                "values": values,
            },
            "threshold_decision": "NOT_EMITTED",
        }
        output_path = Path(str(invocation["output_path"]))
        _write_atomic(output_path, json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n")
        verify_formal_compare_output(output_path, pair, protocol)
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout, stderr = log_dir / "stdout.log", log_dir / "stderr.log"; stdout.write_bytes(b"formal compare\n"); stderr.write_bytes(b"")
        from .pilot_runner import ProcessResult
        return ProcessResult(0, 0.0, stdout, stderr, _sha256(stdout), stdout.stat().st_size, _sha256(stderr), 0)


def dispatch_formal_serial(
    context: RunContext,
    executor: StageExecutor,
    state_root: Path | str,
    lease_files: Mapping[str, Path | str],
    profile: Mapping[str, Any],
    quality_protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Legacy entry point retained only to fail closed.

    All GPU systems, including one-GPU baselines, must go through
    :func:`rolloutbench.formal_dispatch.dispatch_formal_run`, which validates
    an external ownership authorization and active UUID-scoped leases.
    """

    del context, executor, state_root, lease_files, profile, quality_protocol
    raise FormalRunnerError(
        "legacy serial dispatch is disabled; use the externally authorized formal dispatcher"
    )
