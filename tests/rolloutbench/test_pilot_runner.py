from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from rolloutbench.pilot_runner import (
    DispatchGrant, PilotRunnerError, ProcessResult, _append_transition, execute_unit,
    expand_run_units, load_run_context, open_run_ledger, resume_unit,
    schedule_run_units,
)


def _episode(episode_id: str, index: int, component: str, deps=None) -> dict:
    authority = hashlib.sha256(episode_id.encode()).hexdigest()
    runtime_ref = hashlib.sha256(f"runtime:{episode_id}".encode()).hexdigest()
    artifact_sha = hashlib.sha256(b"x").hexdigest()
    return {
        "episode_id": episode_id, "global_fifo_index": index,
        "component": component, "depends_on": deps or [],
        "historical_predecessor_receipts": [],
        "candidate_type": "lossy_cache" if component == "cache" else "exact_kernel",
        "candidate": {
            "authority_ref": authority,
            "config": {
                "path": "config.toml",
                "blob_sha256": artifact_sha,
                "authority_reported_sha256": artifact_sha,
                "hash_scope": "raw_git_blob_bytes",
            },
            "probe": None,
        },
        "runtime_checkout": {
            "git_ref": runtime_ref, "ref_role": "candidate_commit",
            "runtime_tree_oid": hashlib.sha256(f"tree:{episode_id}".encode()).hexdigest(),
            "required_runtime_paths": ["run.py"],
        },
        "worker_affinity": "lineage:K01" if episode_id == "K02" else (0 if component == "kernel" else 1),
        "quality_pairs": ([{
            "pair_id": f"C02:{suite}:seed-{seed}", "candidate_id": "C02",
            "seed": seed, "prompt_suite": suite, "metrics": ["subject_consistency"],
            "dense_artifact_id": f"dense:{suite}:{seed}",
            "candidate_artifact_id": f"C02:{suite}:{seed}",
        } for suite in ("subject", "scene", "temporal", "overall") for seed in (42, 12345)]
        if episode_id == "C02" else []),
    }


def _run(system: str, run_id: str | None = None) -> dict:
    workers = [{"worker_id": 0, "gpu_uuid": "GPU-A", "component": "any"}]
    if system in {"fifo2", "optroll2"}:
        workers = [
            {"worker_id": 0, "gpu_uuid": "GPU-A", "component": "kernel" if system == "optroll2" else "any"},
            {"worker_id": 1, "gpu_uuid": "GPU-B", "component": "cache" if system == "optroll2" else "any"},
        ]
    episodes = [_episode("K01", 0, "kernel"), _episode("C02", 1, "cache"), _episode("K02", 2, "kernel", ["K01"])]
    if len(workers) == 1:
        for episode in episodes:
            if episode["episode_id"] != "K02":
                episode["worker_affinity"] = 0
    elif system == "fifo2":
        for episode in episodes:
            episode["worker_affinity"] = (
                "lineage:K01" if episode["episode_id"] == "K02" else "dynamic"
            )
    dense = _episode("DENSE", -1, "cache")
    dense["candidate_type"] = "dense_reference"
    dense["worker_contract"] = {"effective_mode": "one_shot"}
    dense["worker_affinity"] = 0 if len(workers) == 1 else (
        1 if system == "optroll2" else "dynamic"
    )
    return {
        "run_id": run_id or f"pilot-{system}-repeat-01",
        "system": system,
        "dispatch_policy": {
            "serial1": "global_fifo_one_shot",
            "fifo2": "global_fifo_two_workers_dependency_aware",
            "optroll1": "typed_validation_decision_aware_one_worker",
            "optroll2": "typed_streams_kernel_cache_one_worker_each",
        }[system],
        "workers": workers,
        "quality_dense_reference": dense,
        "episodes": episodes,
    }


def _preparation(plan_path: Path, raw: bytes, episodes: list[dict], experiment_root: Path) -> dict:
    runtime, material = {}, {}
    derived_root = experiment_root / "derived"
    for episode in episodes:
        eid, contract = episode["episode_id"], episode["runtime_checkout"]
        runtime[eid] = {
            "status": "READY", "runtime_ref": contract["git_ref"],
            "ref_role": contract["ref_role"], "runtime_tree_oid": contract["runtime_tree_oid"],
            "required_runtime_paths": contract["required_runtime_paths"],
            "worktree_path": f"/prepared/{eid}",
            "critical_runtime_file_sha256": {"run.py": "a" * 64},
        }
        relative = Path(eid) / "config" / "config.toml"
        artifact_path = derived_root / relative
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"x")
        material[eid] = {
            "episode_id": eid, "authority_ref": episode["candidate"]["authority_ref"],
            "artifacts": [{
                "kind": "config", "source_path": "config.toml",
                "relative_path": relative.as_posix(),
                "sha256": hashlib.sha256(b"x").hexdigest(), "size_bytes": 1,
            }],
        }
    return {
        "status": "READY", "plan_id": "plan-1", "plan_path": str(plan_path.resolve()),
        "plan_sha256": hashlib.sha256(raw).hexdigest(),
        "experiment_root": str(experiment_root.resolve()),
        "derived_root": str(derived_root.resolve()),
        "runtime_receipts": runtime, "materialization_receipts": material,
    }


def _context(root: Path, run: dict, suffix=""):
    plan = {"plan_id": "plan-1", "runs": [run], "salt": suffix}
    plan_path = root / f"plan{suffix}.json"
    raw = json.dumps(plan, sort_keys=True).encode()
    plan_path.write_bytes(raw)
    prep = _preparation(
        plan_path,
        raw,
        [run["quality_dense_reference"], *run["episodes"]],
        root / f"experiment{suffix}",
    )
    prep_path = root / f"prep{suffix}.json"
    prep_path.write_text(json.dumps(prep))
    return load_run_context(plan_path, prep_path, run["run_id"]), prep_path, prep


def _output(context, name: str) -> Path:
    return (
        Path(context.preparation["experiment_root"])
        / "runs"
        / context.plan_id
        / context.plan_sha256
        / context.run["run_id"]
        / context.run_sha256
        / name
    )


def _invocation(context, unit, output: Path) -> dict:
    value = {
        "unit_id": unit.unit_id, "unit_kind": unit.unit_kind,
        "episode_id": unit.episode_id, "run_id": context.run["run_id"],
        "preparation_episode_ids": list(
            unit.preparation_episode_ids or (unit.episode_id,)
        ),
        "output_path": str(output),
    }
    if unit.quality_pair is not None:
        value["quality_pair_id"] = unit.quality_pair["pair_id"]
    return value


class _Executor:
    def __init__(self, write=True, started=None, release=None):
        self.write, self.started, self.release, self.calls = write, started, release, 0

    def execute(self, invocation, *, log_dir):
        self.calls += 1
        if self.started: self.started.set()
        if self.release: self.release.wait(timeout=2)
        log_dir.mkdir(parents=True, exist_ok=True)
        output = Path(invocation["output_path"])
        if self.write:
            output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(b"result")
        stdout, stderr = log_dir / "stdout.log", log_dir / "stderr.log"
        stdout.write_bytes(b"out"); stderr.write_bytes(b"err")
        return ProcessResult(0, .01, stdout, stderr, hashlib.sha256(b"out").hexdigest(), 3, hashlib.sha256(b"err").hexdigest(), 3)


class _ExternalLogExecutor(_Executor):
    def __init__(self, external_root: Path):
        super().__init__()
        self.external_root = external_root

    def execute(self, invocation, *, log_dir):
        self.calls += 1
        output = Path(invocation["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"result")
        self.external_root.mkdir(parents=True, exist_ok=True)
        stdout = self.external_root / "stdout.log"
        stderr = self.external_root / "stderr.log"
        stdout.write_bytes(b"out")
        stderr.write_bytes(b"err")
        return ProcessResult(
            0, .01, stdout, stderr,
            hashlib.sha256(b"out").hexdigest(), 3,
            hashlib.sha256(b"err").hexdigest(), 3,
        )


class PilotRunnerTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch(
            "rolloutbench.pilot_runner.verify_runtime_receipt",
            side_effect=lambda _repository, receipt, _contract: dict(receipt),
        )
        self.verify_runtime = patcher.start()
        self.addCleanup(patcher.stop)

    def test_context_binds_plan_run_and_receipt_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); context, prep_path, prep = _context(root, _run("serial1"))
            self.assertEqual(hashlib.sha256(context.plan_path.read_bytes()).hexdigest(), context.plan_sha256)
            self.assertEqual(hashlib.sha256(json.dumps(context.run, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), context.run_sha256)
            prep["runtime_receipts"]["K01"]["runtime_ref"] = "0" * 64
            prep_path.write_text(json.dumps(prep))
            with self.assertRaisesRegex(PilotRunnerError, "runtime receipt"):
                load_run_context(context.plan_path, prep_path, context.run["run_id"])
            prep = _preparation(
                context.plan_path,
                context.plan_path.read_bytes(),
                [context.run["quality_dense_reference"], *context.run["episodes"]],
                Path(context.preparation["experiment_root"]),
            )
            prep["materialization_receipts"]["K01"]["episode_id"] = "K99"
            prep_path.write_text(json.dumps(prep))
            with self.assertRaisesRegex(PilotRunnerError, "materialization receipt"):
                load_run_context(context.plan_path, prep_path, context.run["run_id"])

            prep = _preparation(
                context.plan_path,
                context.plan_path.read_bytes(),
                [context.run["quality_dense_reference"], *context.run["episodes"]],
                Path(context.preparation["experiment_root"]),
            )
            prep["materialization_receipts"]["K01"]["artifacts"][0][
                "source_path"
            ] = "other.toml"
            prep_path.write_text(json.dumps(prep))
            with self.assertRaisesRegex(PilotRunnerError, "artifact descriptor"):
                load_run_context(context.plan_path, prep_path, context.run["run_id"])

    def test_external_predecessor_is_historical_but_k02_dependency_remains(self):
        run = _run("serial1"); run["episodes"].append(_episode("K15", 3, "kernel", ["K14"])); run["episodes"][-1]["worker_affinity"] = 0
        run["episodes"][-1]["historical_predecessor_receipts"] = [{
            "episode_id": "K14",
            "public_episode_sha256": "a" * 64,
            "disposition": "frozen_history_not_replayed_in_this_scope",
        }]
        units = {unit.unit_id: unit for unit in expand_run_units(run)}
        self.assertEqual((), units["K15:primary"].depends_on)
        self.assertEqual(("K14",), units["K15:primary"].historical_predecessors)
        self.assertEqual(("K01:primary",), units["K02:primary"].depends_on)

    def test_quality_fanout_is_complete_matched_pair_graph(self):
        for system in ("serial1", "fifo2", "optroll1", "optroll2"):
            units = expand_run_units(_run(system)); quality = [u for u in units if u.quality_pair]
            self.assertEqual(48, len(quality)); by_pair = {}
            for unit in quality: by_pair.setdefault(unit.quality_pair["pair_id"], {})[unit.unit_kind] = unit
            self.assertEqual(8, len(by_pair))
            expected = {
                "quality_dense_generate", "quality_candidate_generate",
                "quality_dense_vbench", "quality_candidate_vbench",
                "quality_lpips", "quality_compare",
            }
            for roles in by_pair.values():
                self.assertEqual(expected, set(roles))
                generated = {roles["quality_dense_generate"].unit_id, roles["quality_candidate_generate"].unit_id}
                self.assertEqual(generated, set(roles["quality_dense_vbench"].depends_on))
                self.assertEqual(generated, set(roles["quality_candidate_vbench"].depends_on))
                self.assertEqual(generated, set(roles["quality_lpips"].depends_on))
                self.assertEqual(
                    {
                        roles["quality_dense_vbench"].unit_id,
                        roles["quality_candidate_vbench"].unit_id,
                        roles["quality_lpips"].unit_id,
                    },
                    set(roles["quality_compare"].depends_on),
                )
            schedule = schedule_run_units(_run(system), units)
            self.assertEqual(len(units), len(schedule))
            for roles in by_pair.values():
                primary_worker = schedule[
                    f"{roles['quality_compare'].episode_id}:primary"
                ].worker_id
                self.assertTrue(
                    all(
                        schedule[role.unit_id].worker_id == primary_worker
                        for kind, role in roles.items()
                        if kind != "quality_dense_generate"
                    )
                )
                self.assertEqual((), roles["quality_dense_generate"].depends_on)

    def test_evidence_namespace_binds_plan_sha_and_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts = [_context(root, _run("serial1"), "-a")[0], _context(root, _run("serial1"), "-b")[0], _context(root, _run("serial1", "other"), "-c")[0]]
            paths = {open_run_ledger(c, root / "state").path for c in contexts}
            self.assertEqual(3, len(paths))
            self.assertTrue(all("plan-1" in path.parts for path in paths))

    def test_execution_enforces_worker_assignment_and_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); context = _context(root, _run("fifo2"))[0]; ledger = open_run_ledger(context, root / "state")
            units = {u.unit_id: u for u in expand_run_units(context.run)}; executor = _Executor()
            with self.assertRaisesRegex(PilotRunnerError, "worker assignment"):
                execute_unit(context, ledger, units["K01:primary"], _invocation(context, units["K01:primary"], _output(context, "k1")), executor, root / "state", worker_id=1)
            with self.assertRaisesRegex(PilotRunnerError, "dependency"):
                execute_unit(context, ledger, units["K02:primary"], _invocation(context, units["K02:primary"], _output(context, "k2")), executor, root / "state", worker_id=0)
            self.assertEqual(0, executor.calls)
            serial = _context(root, _run("serial1"), "-serial")[0]
            serial_ledger = open_run_ledger(serial, root / "state")
            serial_units = {u.unit_id: u for u in expand_run_units(serial.run)}
            with self.assertRaisesRegex(PilotRunnerError, "worker-slot predecessor"):
                execute_unit(serial, serial_ledger, serial_units["C02:primary"], _invocation(serial, serial_units["C02:primary"], _output(serial, "c2")), executor, root / "state", worker_id=0)

    def test_runtime_dispatch_grant_allows_completion_driven_dynamic_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = _context(root, _run("fifo2"))[0]
            ledger = open_run_ledger(context, root / "state")
            unit = next(
                item for item in expand_run_units(context.run)
                if item.unit_id == "K01:primary"
            )
            grant = DispatchGrant(
                plan_id=context.plan_id,
                plan_sha256=context.plan_sha256,
                run_id=context.run["run_id"],
                run_sha256=context.run_sha256,
                unit_id=unit.unit_id,
                worker_id=1,
                dispatch_policy=context.run["dispatch_policy"],
                dispatcher_id="dispatcher-test",
                dispatch_index=0,
            )
            with self.assertRaisesRegex(PilotRunnerError, "worker GPU UUID"):
                execute_unit(
                    context,
                    ledger,
                    unit,
                    {
                        **_invocation(context, unit, _output(context, "wrong-gpu.bin")),
                        "gpu_uuid": "GPU-A",
                        "lease_file": "/leases/GPU-A.json",
                        "env": {"CUDA_VISIBLE_DEVICES": "GPU-A"},
                    },
                    _Executor(),
                    root / "state",
                    worker_id=1,
                    dispatch_grant=grant,
                )
            result = execute_unit(
                context,
                ledger,
                unit,
                {
                    **_invocation(context, unit, _output(context, "dynamic.bin")),
                    "gpu_uuid": "GPU-B",
                    "lease_file": "/leases/GPU-B.json",
                    "env": {"CUDA_VISIBLE_DEVICES": "GPU-B"},
                },
                _Executor(),
                root / "state",
                worker_id=1,
                dispatch_grant=grant,
            )
            self.assertEqual("EXECUTED", result["status"])

            bad = DispatchGrant(**{**grant.__dict__, "plan_sha256": "0" * 64})
            other = next(
                item for item in expand_run_units(context.run)
                if item.unit_id == "C02:primary"
            )
            bad = DispatchGrant(
                **{
                    **bad.__dict__,
                    "unit_id": other.unit_id,
                    "dispatch_index": 1,
                }
            )
            with self.assertRaisesRegex(PilotRunnerError, "dispatch grant"):
                execute_unit(
                    context,
                    ledger,
                    other,
                    {
                        **_invocation(context, other, _output(context, "bad.bin")),
                        "gpu_uuid": "GPU-B",
                        "lease_file": "/leases/GPU-B.json",
                        "env": {"CUDA_VISIBLE_DEVICES": "GPU-B"},
                    },
                    _Executor(),
                    root / "state",
                    worker_id=1,
                    dispatch_grant=bad,
                )

    def test_invocation_requires_all_four_exact_identity_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); context = _context(root, _run("serial1"))[0]; ledger = open_run_ledger(context, root / "state")
            unit = next(u for u in expand_run_units(context.run) if u.unit_id == "K01:primary"); base = _invocation(context, unit, _output(context, "output.bin")); executor = _Executor()
            for field in ("unit_id", "unit_kind", "episode_id", "run_id"):
                with self.subTest(missing=field), self.assertRaisesRegex(PilotRunnerError, field):
                    execute_unit(context, ledger, unit, {key: value for key, value in base.items() if key != field}, executor, root / "state", worker_id=0)
                with self.subTest(mismatch=field), self.assertRaisesRegex(PilotRunnerError, field):
                    execute_unit(context, ledger, unit, {**base, field: "wrong"}, executor, root / "state", worker_id=0)
            self.assertEqual(0, executor.calls)

    def test_output_must_remain_in_experiment_or_state_root_without_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); context = _context(root, _run("serial1"))[0]; ledger = open_run_ledger(context, root / "state")
            unit = next(u for u in expand_run_units(context.run) if u.unit_id == "K01:primary"); executor = _Executor()
            outside = root / "outside.bin"; outside.write_bytes(b"preserve")
            with self.assertRaisesRegex(PilotRunnerError, "allowed run roots"):
                execute_unit(context, ledger, unit, _invocation(context, unit, outside), executor, root / "state", worker_id=0)
            self.assertEqual(b"preserve", outside.read_bytes())

            old_unhashed_root = (
                Path(context.preparation["experiment_root"])
                / "runs" / context.plan_id / context.run["run_id"] / "old.bin"
            )
            with self.assertRaisesRegex(PilotRunnerError, "allowed run roots"):
                execute_unit(
                    context, ledger, unit,
                    _invocation(context, unit, old_unhashed_root),
                    executor, root / "state", worker_id=0,
                )

            allowed = _output(context, "placeholder").parent; allowed.mkdir(parents=True, exist_ok=True)
            linked_parent = allowed / "linked"; linked_parent.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(PilotRunnerError, "symlink|allowed run roots"):
                execute_unit(context, ledger, unit, _invocation(context, unit, linked_parent / "escape.bin"), executor, root / "state", worker_id=0)
            linked_output = allowed / "linked-output.bin"; linked_output.symlink_to(outside)
            with self.assertRaisesRegex(PilotRunnerError, "symlink|allowed run roots"):
                execute_unit(context, ledger, unit, _invocation(context, unit, linked_output), executor, root / "state", worker_id=0)
            self.assertEqual(b"preserve", outside.read_bytes()); self.assertEqual(0, executor.calls)

            state_output = ledger.path.parent / "outputs" / "result.json"
            result = execute_unit(context, ledger, unit, _invocation(context, unit, state_output), executor, root / "state", worker_id=0)
            self.assertEqual("EXECUTED", result["status"])

    def test_executor_logs_must_be_created_in_the_exact_attempt_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = _context(root, _run("serial1"))[0]
            ledger = open_run_ledger(context, root / "state")
            unit = next(
                item for item in expand_run_units(context.run)
                if item.unit_id == "K01:primary"
            )
            executor = _ExternalLogExecutor(root / "external-logs")
            with self.assertRaisesRegex(PilotRunnerError, "log escapes"):
                execute_unit(
                    context,
                    ledger,
                    unit,
                    _invocation(context, unit, _output(context, "output.bin")),
                    executor,
                    root / "state",
                    worker_id=0,
                )
            self.assertEqual(1, executor.calls)
            self.assertFalse(
                any(
                    row["event_type"] == "stage_completed"
                    for row in ledger.read()
                )
            )

    def test_live_runtime_and_materialized_artifact_tamper_fail_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); context = _context(root, _run("serial1"))[0]; ledger = open_run_ledger(context, root / "state")
            unit = next(u for u in expand_run_units(context.run) if u.unit_id == "K01:primary"); invocation = _invocation(context, unit, _output(context, "out")); executor = _Executor()
            self.verify_runtime.side_effect = RuntimeError("dirty checkout")
            with self.assertRaisesRegex(PilotRunnerError, "runtime receipt live verification"):
                execute_unit(context, ledger, unit, invocation, executor, root / "state", worker_id=0)
            self.verify_runtime.side_effect = lambda _repository, receipt, _contract: dict(receipt)
            artifact = context.preparation["materialization_receipts"]["K01"]["artifacts"][0]
            artifact_path = Path(context.preparation["derived_root"]) / artifact["relative_path"]
            external = root / "external-config"; external.write_bytes(b"x")
            artifact_path.unlink(); artifact_path.symlink_to(external)
            with self.assertRaisesRegex(PilotRunnerError, "materialized artifact path contains a symlink"):
                execute_unit(context, ledger, unit, invocation, executor, root / "state", worker_id=0)
            artifact_path.unlink(); artifact_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(PilotRunnerError, "materialized artifact"):
                execute_unit(context, ledger, unit, invocation, executor, root / "state", worker_id=0)
            self.assertEqual(0, executor.calls)

    def test_atomic_claim_allows_only_one_executor_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); context = _context(root, _run("serial1"))[0]; ledger = open_run_ledger(context, root / "state")
            unit = next(u for u in expand_run_units(context.run) if u.unit_id == "K01:primary"); invocation = _invocation(context, unit, _output(context, "output.bin"))
            started, release, executor = threading.Event(), threading.Event(), _Executor(started=threading.Event(), release=None)
            executor.started, executor.release = started, release
            results, errors = [], []
            def run():
                try: results.append(execute_unit(context, ledger, unit, invocation, executor, root / "state", worker_id=0))
                except BaseException as exc: errors.append(exc)
            one, two = threading.Thread(target=run), threading.Thread(target=run)
            one.start(); self.assertTrue(started.wait(1)); two.start(); time.sleep(.05)
            self.assertEqual(1, executor.calls); release.set(); one.join(2); two.join(2)
            self.assertFalse(errors); self.assertEqual(1, executor.calls)
            self.assertEqual({"EXECUTED", "REUSED_COMPLETED"}, {r["status"] for r in results})

    def test_old_output_cannot_satisfy_new_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); context = _context(root, _run("serial1"))[0]; ledger = open_run_ledger(context, root / "state")
            unit = next(u for u in expand_run_units(context.run) if u.unit_id == "K01:primary"); output = _output(context, "output.bin"); output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(b"old")
            result = execute_unit(context, ledger, unit, _invocation(context, unit, output), _Executor(write=False), root / "state", worker_id=0)
            self.assertEqual("FAILED", result["status"]); self.assertFalse(output.exists()); self.assertTrue(list(output.parent.glob(".output.bin.stale-*")))

    def test_queued_crash_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); context = _context(root, _run("serial1"))[0]; ledger = open_run_ledger(context, root / "state")
            unit = next(u for u in expand_run_units(context.run) if u.unit_id == "K01:primary")
            _append_transition(context, ledger, unit, "stage_queued", 1, {"worker_mode": "one_shot"})
            result = execute_unit(context, ledger, unit, _invocation(context, unit, _output(context, "out")), _Executor(), root / "state", worker_id=0)
            self.assertEqual(("EXECUTED", 2), (result["status"], result["attempt"]))

    def test_completion_revalidates_context_invocation_receipt_logs_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); context = _context(root, _run("serial1"))[0]; ledger = open_run_ledger(context, root / "state")
            unit = next(u for u in expand_run_units(context.run) if u.unit_id == "K01:primary"); invocation = _invocation(context, unit, _output(context, "output.bin"))
            result = execute_unit(context, ledger, unit, invocation, _Executor(), root / "state", worker_id=0)
            receipt_path = Path(result["receipt_path"]); receipt_bytes = receipt_path.read_bytes(); receipt = json.loads(receipt_bytes)
            self.assertEqual((context.plan_id, context.plan_sha256, context.run_sha256), (receipt["plan_id"], receipt["plan_sha256"], receipt["run_sha256"]))
            self.assertRegex(receipt["runtime_live_verification_sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(receipt["materialization_live_verification_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual("reuse_completed", resume_unit(context, ledger, unit, invocation)["action"])
            with self.assertRaisesRegex(PilotRunnerError, "invocation"):
                resume_unit(context, ledger, unit, {**invocation, "different": True})
            receipt_path.write_bytes(b"{}\n")
            with self.assertRaisesRegex(PilotRunnerError, "receipt digest"):
                resume_unit(context, ledger, unit, invocation)
            receipt_path.write_bytes(receipt_bytes)
            Path(receipt["logs"]["stdout"]["path"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(PilotRunnerError, "log"):
                resume_unit(context, ledger, unit, invocation)

    def test_quality_role_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); context = _context(root, _run("serial1"))[0]; ledger = open_run_ledger(context, root / "state")
            unit = next(u for u in expand_run_units(context.run) if u.unit_kind == "quality_dense_generate"); invocation = _invocation(context, unit, _output(context, "dense.mp4")); invocation["unit_kind"] = "quality_candidate_generate"; executor = _Executor()
            with self.assertRaisesRegex(PilotRunnerError, "unit_kind"):
                execute_unit(context, ledger, unit, invocation, executor, root / "state", worker_id=0)
            self.assertEqual(0, executor.calls)

    def test_persistent_worker_contract_fails_closed(self):
        run = _run("optroll1"); run["episodes"][0]["worker_contract"] = {"effective_mode": "persistent"}
        with self.assertRaisesRegex(PilotRunnerError, "persistent workers"): expand_run_units(run)

    def test_subprocess_executor_writes_separate_logs(self):
        from rolloutbench.pilot_runner import SubprocessStageExecutor
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); command = [sys.executable, "-c", "import sys; print('stdout'); print('stderr', file=sys.stderr)"]
            result = SubprocessStageExecutor().execute({"argv": command, "cwd": str(root), "env": {}, "output_path": str(root / "output.bin")}, log_dir=root / "logs")
            self.assertEqual((0, "stdout\n", "stderr\n"), (result.returncode, result.stdout_path.read_text(), result.stderr_path.read_text()))


if __name__ == "__main__": unittest.main()
