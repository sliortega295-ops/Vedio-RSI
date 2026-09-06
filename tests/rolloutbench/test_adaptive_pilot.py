from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rolloutbench.adaptive_pilot import (
    AdaptivePilotError,
    _analysis_implementation_receipt,
    _formal_repeat_rule,
    analyze_adaptive_pilot,
    load_contexts_by_system,
    parse_completed_repetitions,
    semantic_decisions,
    write_adaptive_pilot_result,
)
from rolloutbench.aggregation import AggregationError
from rolloutbench.cli import main
from rolloutbench.pilot_runner import RunContext


class AdaptivePilotTests(unittest.TestCase):
    def test_completed_matrix_is_exact(self) -> None:
        self.assertEqual(
            {"serial1": 3, "fifo2": 3, "optroll1": 2, "optroll2": 2},
            parse_completed_repetitions(
                ["serial1=3", "fifo2=3", "optroll1=2", "optroll2=2"]
            ),
        )
        with self.assertRaisesRegex(AdaptivePilotError, "requires exactly"):
            parse_completed_repetitions(
                ["serial1=3", "fifo2=3", "optroll1=3", "optroll2=2"]
            )

    def test_semantic_signature_ignores_measurements_and_receipts(self) -> None:
        base = {
            "K20": {
                "component": "kernel",
                "outcome": "exact_validated",
                "frontier_eligible": True,
                "decision_semantics": "fresh",
                "ranking_latency_contract": "one_shot",
                "ranking_latency_s": 91.0,
                "decision_receipt_path": "/first",
            }
        }
        changed_measurements = json.loads(json.dumps(base))
        changed_measurements["K20"]["ranking_latency_s"] = 99.0
        changed_measurements["K20"]["decision_receipt_path"] = "/second"
        self.assertEqual(
            semantic_decisions(base), semantic_decisions(changed_measurements)
        )
        changed_outcome = json.loads(json.dumps(base))
        changed_outcome["K20"]["outcome"] = "rejected"
        self.assertNotEqual(
            semantic_decisions(base), semantic_decisions(changed_outcome)
        )

    def _contexts(
        self, root: Path
    ) -> tuple[
        dict[str, list[RunContext]],
        Path,
        dict[str, object],
        dict[str, object],
    ]:
        suite_path = root / "suite"
        suite_path.mkdir()
        protocol = {"protocol_id": "quality-v1"}
        episode_ids = ["K20", "C02"]
        (suite_path / "episodes.jsonl").write_text(
            "".join(
                json.dumps({"episode_id": episode_id}, sort_keys=True) + "\n"
                for episode_id in episode_ids
            ),
            encoding="utf-8",
        )
        (suite_path / "artifacts.json").write_text("{}\n", encoding="utf-8")
        (suite_path / "quality_protocol.json").write_text(
            json.dumps(protocol, sort_keys=True) + "\n", encoding="utf-8"
        )
        suite = {
            "suite_id": "test-suite",
            "systems": ["serial1", "fifo2", "optroll1", "optroll2"],
            "pilot_episodes": episode_ids,
        }
        (suite_path / "suite.json").write_text(
            json.dumps(suite, sort_keys=True) + "\n", encoding="utf-8"
        )
        suite_hashes = {
            name: hashlib.sha256((suite_path / name).read_bytes()).hexdigest()
            for name in (
                "suite.json",
                "episodes.jsonl",
                "artifacts.json",
                "quality_protocol.json",
            )
        }
        counts = {"serial1": 3, "fifo2": 3, "optroll1": 2, "optroll2": 2}
        planned_runs = [
            {
                "run_id": f"pilot-{system}-repeat-{index:02d}",
                "system": system,
                "scope": "pilot",
                "repeat_index": index,
                "episodes": [
                    {"episode_id": "K20"},
                    {"episode_id": "C02"},
                ],
            }
            for system in counts
            for index in range(1, 6)
        ]
        plan = root / "plan.json"
        preparation = root / "preparation.json"
        state = root / "state"
        plan.write_text(
            json.dumps(
                {
                    "plan_id": "plan",
                    "suite_id": suite["suite_id"],
                    "suite_file_sha256": suite_hashes,
                    "quality_protocol_id": protocol["protocol_id"],
                    "scope": "pilot",
                    "repetitions": 5,
                    "runs": planned_runs,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        preparation.write_text("{}\n", encoding="utf-8")
        state.mkdir()
        plan_sha = hashlib.sha256(plan.read_bytes()).hexdigest()
        preparation_sha = hashlib.sha256(preparation.read_bytes()).hexdigest()
        contexts: dict[str, list[RunContext]] = {}
        for system, count in counts.items():
            contexts[system] = [
                RunContext(
                    plan_id="plan",
                    plan_sha256=plan_sha,
                    run_sha256=str(index) * 64,
                    preparation_sha256=preparation_sha,
                    run=next(
                        row
                        for row in planned_runs
                        if row["system"] == system and row["repeat_index"] == index
                    ),
                    preparation={},
                    plan_path=plan,
                    preparation_path=preparation,
                )
                for index in range(1, count + 1)
            ]
        return contexts, state, suite, protocol

    @staticmethod
    def _record(context: RunContext, ttvf_s: float, frontier: dict[str, str]) -> dict:
        decisions = {
            "K20": {
                "component": "kernel",
                "outcome": "exact_validated",
                "frontier_eligible": True,
                "decision_semantics": "fresh",
                "ranking_latency_contract": "one_shot",
                "ranking_latency_s": 90.0,
            },
            "C02": {
                "component": "cache",
                "outcome": "quality_pass",
                "frontier_eligible": True,
                "decision_semantics": "fresh",
                "ranking_latency_contract": "one_shot",
                "contract": "quality-v1",
                "ranking_latency_s": 80.0,
                "quality_result": {
                    "protocol_id": "quality-v1",
                    "eligibility": "formal",
                    "status": "PASS",
                    "pass": True,
                    "errors": [],
                    "thresholds": {"drop": 0.02},
                    "lpips": {
                        "status": "COMPLETE",
                        "role": "secondary",
                        "hard_acceptance_effect": False,
                        "errors": [],
                    },
                },
            },
        }
        workers = 2 if context.run["system"] in {"fifo2", "optroll2"} else 1
        busy = ttvf_s * workers * 0.5
        return {
            "run_id": context.run["run_id"],
            "repeat_index": context.run["repeat_index"],
            "ttvf_s": ttvf_s,
            "ttvf_clock": "same_boot_monotonic",
            "decisions": decisions,
            "frontier": {"frontier": frontier},
            "ledger_receipt": {
                "path": f"/{context.run['run_id']}.jsonl",
                "sha256": "a" * 64,
                "size_bytes": 1,
            },
            "gpu_busy_s": busy,
            "gpu_capacity_s": ttvf_s * workers,
            "gpu_queue_idle_s": ttvf_s * workers - busy,
            "scheduler_gpu_utilization": 0.5,
            "quality_wall_s": 50.0,
            "measured_generation_s": 20.0,
            "model_load_compile_warmup_s": 10.0,
        }

    def test_two_stable_optroll_runs_stop_and_remain_non_formal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts, state, suite, protocol = self._contexts(root)
            samples = {
                "serial1": [100.0, 102.0, 98.0],
                "fifo2": [90.0, 91.0, 92.0],
                "optroll1": [103.0, 104.0],
                "optroll2": [85.0, 86.0],
            }
            records = {
                context.run["run_id"]: self._record(
                    context,
                    samples[system][index],
                    {"kernel": "K20", "cache": "C02"},
                )
                for system, rows in contexts.items()
                for index, context in enumerate(rows)
            }
            with patch(
                "rolloutbench.adaptive_pilot._run_record",
                side_effect=lambda context, _state, **_kwargs: records[
                    context.run["run_id"]
                ],
            ) as replay, patch(
                "rolloutbench.adaptive_pilot._verify_file_receipt",
                side_effect=lambda receipt, _label: receipt,
            ):
                result = analyze_adaptive_pilot(
                    contexts,
                    state,
                    suite,
                    protocol,
                    suite_path=root / "suite",
                )
        self.assertEqual(10, replay.call_count)
        self.assertTrue(
            all(call.kwargs == {"repair_ledger_tail": False} for call in replay.call_args_list)
        )
        self.assertEqual("EXPLORATORY_ADAPTIVE_PILOT_COMPLETE", result["status"])
        self.assertFalse(result["formal_compare_systems_compatible"])
        self.assertTrue(result["adaptive_policy"]["optroll_stop_after_two"])
        self.assertFalse(result["adaptive_policy"]["optroll_repeat_3_included"])
        self.assertNotIn("optroll_repeat_3_launched", result["adaptive_policy"])
        self.assertIn(
            "adaptive_pilot.py",
            result["source_receipts"]["analysis_implementation"]["disk_source"][
                "files"
            ],
        )
        self.assertTrue(
            result["semantic_candidate_decision_agreement_across_all_runs"]
        )
        self.assertEqual("optroll2", result["comparison"]["ranking_by_median_ttvf"][0])

    def test_optroll_frontier_disagreement_requires_third_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts, state, suite, protocol = self._contexts(root)
            records = {}
            for system, rows in contexts.items():
                for context in rows:
                    frontier = {"kernel": "K20", "cache": "C02"}
                    if context.run["run_id"] == "pilot-optroll2-repeat-02":
                        frontier = {"kernel": "K20", "cache": "C09"}
                    records[context.run["run_id"]] = self._record(
                        context, 100.0, frontier
                    )
            with patch(
                "rolloutbench.adaptive_pilot._run_record",
                side_effect=lambda context, _state, **_kwargs: records[
                    context.run["run_id"]
                ],
            ), patch(
                "rolloutbench.adaptive_pilot._verify_file_receipt",
                side_effect=lambda receipt, _label: receipt,
            ):
                result = analyze_adaptive_pilot(
                    contexts,
                    state,
                    suite,
                    protocol,
                    suite_path=root / "suite",
                )
        self.assertEqual(
            "EXPLORATORY_ADAPTIVE_PILOT_NEEDS_OPTROLL_REPEAT_3",
            result["status"],
        )
        self.assertFalse(
            result["systems"]["optroll2"]["adaptive_two_repetition_gate"][
                "stop_after_two"
            ]
        )

    def test_suite_contract_must_match_the_bound_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts, state, suite, protocol = self._contexts(root)
            mismatched = dict(suite)
            mismatched["suite_id"] = "wrong-suite"
            with self.assertRaisesRegex(AdaptivePilotError, "does not match suite.json"):
                analyze_adaptive_pilot(
                    contexts,
                    state,
                    mismatched,
                    protocol,
                    suite_path=root / "suite",
                )

    def test_plan_suite_hashes_must_match_the_suite_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts, state, suite, protocol = self._contexts(root)
            plan_path = contexts["serial1"][0].plan_path
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["suite_file_sha256"]["artifacts.json"] = "0" * 64
            plan_path.write_text(
                json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8"
            )
            plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            rebound = {
                system: [replace(context, plan_sha256=plan_sha) for context in rows]
                for system, rows in contexts.items()
            }
            with self.assertRaisesRegex(
                AdaptivePilotError, "plan, suite and pilot episode contract"
            ):
                analyze_adaptive_pilot(
                    rebound,
                    state,
                    suite,
                    protocol,
                    suite_path=root / "suite",
                )

    def test_plan_matrix_rejects_malformed_unselected_repetitions(self) -> None:
        for mutation in ("non_mapping_episode", "boolean_repeat_index"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                contexts, state, suite, protocol = self._contexts(root)
                plan_path = contexts["serial1"][0].plan_path
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                if mutation == "non_mapping_episode":
                    target = next(
                        row
                        for row in plan["runs"]
                        if row["system"] == "optroll2" and row["repeat_index"] == 5
                    )
                    target["episodes"].append("not-an-episode-object")
                else:
                    target = next(
                        row
                        for row in plan["runs"]
                        if row["system"] == "serial1" and row["repeat_index"] == 1
                    )
                    target["repeat_index"] = True
                plan_path.write_text(
                    json.dumps(plan, sort_keys=True) + "\n", encoding="utf-8"
                )
                plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
                rebound = {
                    system: [
                        replace(context, plan_sha256=plan_sha) for context in rows
                    ]
                    for system, rows in contexts.items()
                }
                with self.assertRaisesRegex(
                    AdaptivePilotError, "plan, suite and pilot episode contract"
                ):
                    analyze_adaptive_pilot(
                        rebound,
                        state,
                        suite,
                        protocol,
                        suite_path=root / "suite",
                    )

    def test_missing_eligible_latency_is_formally_not_evaluable(self) -> None:
        records = [
            {
                "decisions": {
                    "K20": {
                        "component": "kernel",
                        "outcome": "exact_validated",
                        "frontier_eligible": True,
                        **({"ranking_latency_s": 40.0} if index < 3 else {}),
                    }
                }
            }
            for index in range(1, 4)
        ]
        result = _formal_repeat_rule(records)
        self.assertEqual("NOT_EVALUABLE_INCOMPLETE_LATENCIES", result["status"])
        self.assertEqual(["K20"], result["incomplete_candidates"])
        self.assertIsNone(result["additional_repetitions_required"])

    def test_formal_rule_preserves_all_three_latency_candidate_universe(self) -> None:
        records = [
            {
                "decisions": {
                    "C01": {
                        "component": "cache",
                        "outcome": "excluded_provenance_failed",
                        "frontier_eligible": False,
                        "ranking_latency_s": latency,
                    }
                }
            }
            for latency in (10.0, 10.0, 20.0)
        ]
        result = _formal_repeat_rule(records)
        self.assertEqual("NEEDS_TWO_ADDITIONAL_REPETITIONS", result["status"])
        self.assertEqual(["C01"], result["candidate_cv_universe"])
        self.assertIn("C01", result["candidate_cv_first_three"])
        self.assertEqual(["C01"], result["candidates_above_threshold"])

    def test_formal_rule_rejects_outcome_flag_disagreement(self) -> None:
        records = [
            {
                "decisions": {
                    "K20": {
                        "component": "kernel",
                        "outcome": "exact_validated",
                        "frontier_eligible": False,
                        "ranking_latency_s": latency,
                    }
                }
            }
            for latency in (10.0, 10.0, 10.0)
        ]
        result = _formal_repeat_rule(records)
        self.assertEqual(
            "NOT_EVALUABLE_INCONSISTENT_FRONTIER_ELIGIBILITY",
            result["status"],
        )
        self.assertEqual(
            ["K20"],
            result["inconsistent_frontier_eligibility_candidates"],
        )

    def test_analysis_receipt_distinguishes_disk_source_and_loaded_code(self) -> None:
        receipt = _analysis_implementation_receipt()
        self.assertEqual(
            "separate_disk_source_and_live_loaded_code_receipts",
            receipt["scope"],
        )
        self.assertIn("disk_source", receipt)
        self.assertIn("loaded_code", receipt)
        self.assertNotEqual(
            receipt["disk_source"]["scope"], receipt["loaded_code"]["scope"]
        )

    def test_load_contexts_and_writer_enforce_frozen_contracts(self) -> None:
        completed = {"serial1": 3, "fifo2": 3, "optroll1": 2, "optroll2": 2}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "run_id": f"pilot-{system}-repeat-{index:02d}",
                                "system": system,
                                "repeat_index": index,
                            }
                            for system in completed
                            for index in range(1, 6)
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            loaded: list[str] = []
            with patch(
                "rolloutbench.adaptive_pilot.load_run_context",
                side_effect=lambda _plan, _preparation, run_id: loaded.append(run_id)
                or run_id,
            ):
                result = load_contexts_by_system(
                    plan, root / "preparation.json", completed
                )
            self.assertEqual(10, len(loaded))
            self.assertEqual(2, len(result["optroll2"]))

            output = root / "result.json"
            first = write_adaptive_pilot_result(output, {"status": "one"})
            second = write_adaptive_pilot_result(output, {"status": "one"})
            self.assertEqual(first["sha256"], second["sha256"])
            with self.assertRaisesRegex(AggregationError, "conflicting"):
                write_adaptive_pilot_result(output, {"status": "two"})

    def test_cli_routes_suite_path_and_adaptive_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _contexts, _state, suite, protocol = self._contexts(root)
            output = root / "adaptive.json"
            with (
                patch("rolloutbench.cli.validate_suite_directory"),
                patch(
                    "rolloutbench.adaptive_pilot.load_contexts_by_system",
                    return_value={"contexts": True},
                ),
                patch(
                    "rolloutbench.adaptive_pilot.analyze_adaptive_pilot",
                    return_value={"status": "EXPLORATORY_ADAPTIVE_PILOT_COMPLETE"},
                ) as analyze,
                patch(
                    "rolloutbench.adaptive_pilot.write_adaptive_pilot_result",
                    return_value={"status": "WRITTEN", "sha256": "a" * 64},
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    0,
                    main(
                        [
                            "summarize-adaptive-pilot",
                            "--plan",
                            str(root / "plan.json"),
                            "--preparation",
                            str(root / "preparation.json"),
                            "--state-root",
                            str(root / "state"),
                            "--completed",
                            "serial1=3",
                            "--completed",
                            "fifo2=3",
                            "--completed",
                            "optroll1=2",
                            "--completed",
                            "optroll2=2",
                            "--suite",
                            str(root / "suite"),
                            "--repo-root",
                            str(root),
                            "--output",
                            str(output),
                        ]
                    ),
                )
            self.assertEqual(suite, analyze.call_args.args[2])
            self.assertEqual(protocol, analyze.call_args.args[3])
            self.assertEqual(root / "suite", analyze.call_args.kwargs["suite_path"])


if __name__ == "__main__":
    unittest.main()
