from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rolloutbench.cli import main
from rolloutbench.events import EventLedger
from rolloutbench.formal_dispatch import (
    FormalDispatchError,
    _priority,
    _validate_planned_episode_scope,
    dispatch_formal_run,
    finalize_quality_decisions,
)
from rolloutbench.pilot_runner import RunContext, Unit
from rolloutbench.validators import build_quality_plan


class FormalDispatchTests(unittest.TestCase):
    def _context(self, root: Path, system: str = "fifo2") -> RunContext:
        policy = {
            "serial1": "global_fifo_one_shot",
            "fifo2": "global_fifo_two_workers_dependency_aware",
            "optroll1": "typed_validation_decision_aware_one_worker",
            "optroll2": "typed_streams_kernel_cache_one_worker_each",
        }[system]
        count = 2 if system in {"fifo2", "optroll2"} else 1
        return RunContext(
            plan_id="plan-1",
            plan_sha256="a" * 64,
            run_sha256="b" * 64,
            preparation_sha256="c" * 64,
            run={
                "run_id": f"pilot-{system}-repeat-01",
                "system": system,
                "dispatch_policy": policy,
                "workers": [
                    {"worker_id": index, "gpu_uuid": f"GPU-{index}"}
                    for index in range(count)
                ],
            },
            preparation={},
            plan_path=root / "plan.json",
            preparation_path=root / "preparation.json",
        )

    def test_fifo2_dispatches_the_next_unit_to_the_first_actually_free_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = self._context(root)
            units = [
                Unit(f"U{index}:primary", f"U{index}", "kernel", index, (), "dynamic")
                for index in range(3)
            ]
            ledger = EventLedger(root / "state" / "events.jsonl")
            durations = {"U0:primary": 0.12, "U1:primary": 0.02, "U2:primary": 0.01}
            calls: list[tuple[str, int, float]] = []
            active = 0
            maximum = 0
            guard = threading.Lock()

            def fake_execute(
                _context, _ledger, unit, _invocation, _executor, _state_root,
                *, worker_id, dispatch_grant,
            ):
                nonlocal active, maximum
                with guard:
                    active += 1
                    maximum = max(maximum, active)
                    calls.append((unit.unit_id, worker_id, time.monotonic()))
                time.sleep(durations[unit.unit_id])
                with guard:
                    active -= 1
                return {"status": "EXECUTED", "unit_id": unit.unit_id}

            with (
                patch(
                    "rolloutbench.formal_dispatch.validate_launch_authorization",
                    return_value={"authorization_sha256": "d" * 64},
                ),
                patch(
                    "rolloutbench.formal_dispatch._load_bound_suite",
                    return_value={},
                ),
                patch(
                    "rolloutbench.formal_dispatch._validate_planned_episode_scope",
                    return_value=[],
                ),
                patch(
                    "rolloutbench.formal_dispatch.open_run_ledger",
                    return_value=ledger,
                ),
                patch(
                    "rolloutbench.formal_dispatch.expand_run_units",
                    return_value=units,
                ),
                patch(
                    "rolloutbench.formal_dispatch.build_formal_invocation",
                    return_value={},
                ),
                patch(
                    "rolloutbench.formal_dispatch.execute_unit",
                    side_effect=fake_execute,
                ),
                patch(
                    "rolloutbench.formal_dispatch.collect_primary_evidence",
                    return_value={},
                ),
                patch(
                    "rolloutbench.formal_dispatch.finalize_quality_decisions",
                    return_value=[],
                ),
                patch(
                    "rolloutbench.formal_dispatch.finalize_run_decisions",
                    return_value={
                        "decision_count": 0,
                        "decisions": {},
                        "frontier_receipt": {
                            "path": str(root / "frontier.json"),
                            "sha256": "e" * 64,
                        },
                    },
                ),
            ):
                outcomes = dispatch_formal_run(
                    context,
                    object(),
                    root / "state-root",
                    authorization_path=root / "authorization.json",
                    lease_files={},
                    profile={},
                    quality_protocol={},
                )

        self.assertEqual(3, len(outcomes))
        self.assertEqual(2, maximum)
        by_unit = {unit_id: worker_id for unit_id, worker_id, _ in calls}
        self.assertEqual(0, by_unit["U0:primary"])
        self.assertEqual(1, by_unit["U1:primary"])
        self.assertEqual(1, by_unit["U2:primary"])

    def test_optroll_prioritizes_shared_dense_unlock_over_new_primary(self):
        dense = Unit(
            "DENSE:quality:x:dense_generate", "DENSE", "cache", 10, (), 0,
            "quality_dense_generate",
        )
        primary = Unit("K01:primary", "K01", "kernel", 0, (), 0)
        self.assertLess(_priority("optroll1", dense), _priority("optroll1", primary))
        self.assertLess(_priority("serial1", primary), _priority("serial1", dense))

    def test_dispatch_revalidates_authorization_before_each_new_unit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = self._context(root, "serial1")
            units = [
                Unit(f"U{index}:primary", f"U{index}", "kernel", index, (), 0)
                for index in range(2)
            ]
            ledger = EventLedger(root / "state" / "events.jsonl")
            authorization_calls = 0
            execution_calls: list[str] = []

            def validate(*_args, **_kwargs):
                nonlocal authorization_calls
                authorization_calls += 1
                if authorization_calls >= 3:
                    raise RuntimeError("authorization revoked")
                return {"authorization_sha256": "d" * 64}

            def execute(
                _context, _ledger, unit, _invocation, _executor, _state_root,
                *, worker_id, dispatch_grant,
            ):
                del worker_id, dispatch_grant
                execution_calls.append(unit.unit_id)
                return {"status": "EXECUTED", "unit_id": unit.unit_id}

            with (
                patch(
                    "rolloutbench.formal_dispatch.validate_launch_authorization",
                    side_effect=validate,
                ),
                patch("rolloutbench.formal_dispatch._load_bound_suite", return_value={}),
                patch(
                    "rolloutbench.formal_dispatch._validate_planned_episode_scope",
                    return_value=[],
                ),
                patch("rolloutbench.formal_dispatch.open_run_ledger", return_value=ledger),
                patch("rolloutbench.formal_dispatch.expand_run_units", return_value=units),
                patch("rolloutbench.formal_dispatch.build_formal_invocation", return_value={}),
                patch("rolloutbench.formal_dispatch.execute_unit", side_effect=execute),
            ):
                with self.assertRaisesRegex(RuntimeError, "revoked"):
                    dispatch_formal_run(
                        context,
                        object(),
                        root / "state-root",
                        authorization_path=root / "authorization.json",
                        lease_files={},
                        profile={},
                        quality_protocol={},
                    )
        self.assertEqual(["U0:primary"], execution_calls)
        self.assertEqual(3, authorization_calls)

    def test_scope_validator_requires_exact_frozen_sequence(self):
        repo = Path(__file__).resolve().parents[2]
        suite_dir = repo / "benchmarks/sana_video_2b_h100_v0"
        suite = json.loads((suite_dir / "suite.json").read_text())
        episode_ids = [
            json.loads(line)["episode_id"]
            for line in (suite_dir / "episodes.jsonl").read_text().splitlines()
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = self._context(root, "serial1")
            context = RunContext(
                **{
                    **context.__dict__,
                    "run": {
                        **context.run,
                        "scope": "full",
                        "episodes": [
                            {"episode_id": episode_id} for episode_id in episode_ids
                        ],
                    },
                }
            )
            profile = {"suite_path": str(suite_dir / "suite.json")}
            self.assertEqual(
                episode_ids,
                _validate_planned_episode_scope(context, profile, suite),
            )
            shortened = RunContext(
                **{
                    **context.__dict__,
                    "run": {
                        **context.run,
                        "episodes": context.run["episodes"][:-1],
                    },
                }
            )
            with self.assertRaisesRegex(FormalDispatchError, "exact frozen"):
                _validate_planned_episode_scope(shortened, profile, suite)

    def test_cli_requires_an_explicit_gpu_execution_acknowledgement(self):
        with self.assertRaisesRegex(RuntimeError, "execute-authorized"):
            main(
                [
                    "run-formal",
                    "--plan", "/missing/plan.json",
                    "--preparation", "/missing/preparation.json",
                    "--run-id", "pilot-serial1-repeat-01",
                    "--state-root", "/missing/state",
                    "--authorization", "/missing/authorization.json",
                ]
            )

    def test_eight_pair_outputs_seal_one_idempotent_quality_decision(self):
        repo = Path(__file__).resolve().parents[2]
        protocol = json.loads(
            (repo / "benchmarks/sana_video_2b_h100_v0/quality_protocol.json").read_text()
        )
        pairs = build_quality_plan(protocol, ["C02"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = self._context(root, "serial1")
            context = RunContext(
                **{
                    **context.__dict__,
                    "run": {
                        **context.run,
                        "episodes": [{"episode_id": "C02", "quality_pairs": pairs}],
                    },
                    "preparation": {"experiment_root": str(root / "experiment")},
                }
            )
            ledger = EventLedger(root / "events.jsonl")
            units = {}
            for pair in pairs:
                unit_id = f"C02:quality:{pair['pair_id']}:compare"
                unit = Unit(
                    unit_id, "C02", "cache", 0, (), 0,
                    "quality_compare", (), pair,
                )
                units[unit_id] = unit
                output = root / "pairs" / f"{hashlib.sha256(unit_id.encode()).hexdigest()}.json"
                output.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "status": "PARSED",
                    "pair_id": pair["pair_id"],
                    "score_rows": [
                        {
                            "pair_id": pair["pair_id"],
                            "metric": metric,
                            "dense_score": 1.0,
                            "candidate_score": 0.999,
                        }
                        for metric in pair["metrics"]
                    ],
                    "lpips": {"values": [0.1] * 81},
                }
                output.write_text(json.dumps(payload), encoding="utf-8")
                ledger.append(
                    "stage_completed",
                    {
                        "episode_id": unit_id,
                        "stage": "quality_compare",
                        "attempt": 1,
                        "output_path": str(output),
                        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    },
                )
            with (
                patch(
                    "rolloutbench.formal_dispatch._completed_invocation",
                    return_value={},
                ),
                patch(
                    "rolloutbench.formal_dispatch.verify_formal_compare_output",
                    side_effect=lambda path, _pair, _protocol: json.loads(
                        Path(path).read_text(encoding="utf-8")
                    ),
                ),
            ):
                first = finalize_quality_decisions(
                    context,
                    ledger,
                    units,
                    protocol,
                    lease_files={},
                    profile={},
                )
                second = finalize_quality_decisions(
                    context,
                    ledger,
                    units,
                    protocol,
                    lease_files={},
                    profile={},
                )
            decision_count = len(ledger.reconstruct().decisions)

        self.assertEqual(first, second)
        self.assertEqual("quality_pass", first[0]["decision"]["outcome"])
        self.assertEqual(1, decision_count)


if __name__ == "__main__":
    unittest.main()
