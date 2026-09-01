from __future__ import annotations

import json
import hashlib
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from rolloutbench.aggregation import (
    AggregationError,
    _replay_system_result,
    aggregate_system,
    compare_system_results,
    write_system_comparison,
    write_system_result,
)
from rolloutbench.pilot_runner import RunContext
from rolloutbench.quality_contract import FORMAL_CACHE_IDS


class SystemAggregationTests(unittest.TestCase):
    _EPISODE_IDS = ["K19", "K20", *FORMAL_CACHE_IDS]

    def _contexts(self, root: Path) -> list[RunContext]:
        plan_path = root / "plan.json"
        preparation_path = root / "preparation.json"
        plan_path.write_text('{"plan_id":"plan"}\n', encoding="utf-8")
        preparation_path.write_text('{"status":"READY"}\n', encoding="utf-8")
        plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        preparation_sha256 = hashlib.sha256(
            preparation_path.read_bytes()
        ).hexdigest()
        return [
            RunContext(
                plan_id="plan",
                plan_sha256=plan_sha256,
                run_sha256=str(index) * 64,
                preparation_sha256=preparation_sha256,
                run={
                    "run_id": f"full-serial1-repeat-{index:02d}",
                    "system": "serial1",
                    "scope": "full",
                    "repeat_index": index,
                    "episodes": [
                        {"episode_id": episode_id}
                        for episode_id in self._EPISODE_IDS
                    ],
                },
                preparation={},
                plan_path=plan_path,
                preparation_path=preparation_path,
            )
            for index in range(1, 4)
        ]

    def _record(self, root: Path, index: int, *, unstable: bool = False) -> dict:
        kernel_latency = [40.0, 40.2, 45.0 if unstable else 39.9][index - 1]
        decisions = {
            "K19": {
                "component": "kernel",
                "outcome": "exact_validated",
                "measured_generation_s": 10.0,
                "ranking_latency_s": 41.0 + index / 10,
            },
            "K20": {
                "component": "kernel",
                "outcome": "exact_validated",
                "measured_generation_s": 20.0,
                "ranking_latency_s": kernel_latency,
            },
        }
        for offset, candidate in enumerate(FORMAL_CACHE_IDS):
            decisions[candidate] = {
                "component": "cache",
                "outcome": "quality_pass",
                "measured_generation_s": 25.0 + offset,
                "ranking_latency_s": 50.0 + offset,
            }
        decisions["C12"]["ranking_latency_s"] = 38.0 + index / 10
        ledger_path = root / f"ledger-{index}.jsonl"
        ledger_path.write_text(f'{{"repeat":{index}}}\n', encoding="utf-8")
        return {
            "run_id": f"full-serial1-repeat-{index:02d}",
            "repeat_index": index,
            "run_sha256": str(index) * 64,
            "ttvf_s": 100.0,
            "decision_count": len(decisions),
            "decisions": decisions,
            "frontier": {},
            "ledger_receipt": {
                "path": str(ledger_path),
                "sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
                "size_bytes": ledger_path.stat().st_size,
            },
            "stage_interval_count": 1,
            "gpu_busy_s": 90.0,
            "gpu_capacity_s": 100.0,
            "gpu_queue_idle_s": 10.0,
            "scheduler_gpu_utilization": 0.9,
            "worker_busy_s": {"0": 90.0},
            "quality_wall_s": 30.0,
            "measured_generation_s": 80.0,
            "model_load_compile_warmup_s": 10.0,
            "phase_receipt_coverage": {},
            "candidate_ranking_latency_p50_s": 40.0,
            "candidate_ranking_latency_p95_s": 55.0,
        }

    def test_three_stable_repetitions_select_fresh_k20_c12_frontier(self):
        repo = Path(__file__).resolve().parents[2]
        suite = json.loads(
            (repo / "benchmarks/sana_video_2b_h100_v0/suite.json").read_text()
        )
        protocol = json.loads(
            (repo / "benchmarks/sana_video_2b_h100_v0/quality_protocol.json").read_text()
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "state").mkdir()
            contexts = self._contexts(root)
            records = [self._record(root, index) for index in range(1, 4)]
            with patch("rolloutbench.aggregation._run_record", side_effect=records):
                result = aggregate_system(contexts, root / "state", suite, protocol)
        self.assertEqual("FULL_AGGREGATED", result["status"])
        self.assertEqual({"kernel": "K20", "cache": "C12"}, result["frontier"])
        self.assertTrue(result["frontier_agreement_with_legacy_oracle"])
        self.assertEqual(0.9, result["metrics"]["scheduler_gpu_utilization"])
        self.assertEqual(
            "fresh_repetition_median_ranking_latency_s_one_shot_process_wall",
            result["kernel_frontier"]["selection_basis"],
        )

    def test_high_cv_requires_two_more_repetitions(self):
        repo = Path(__file__).resolve().parents[2]
        suite = json.loads(
            (repo / "benchmarks/sana_video_2b_h100_v0/suite.json").read_text()
        )
        protocol = json.loads(
            (repo / "benchmarks/sana_video_2b_h100_v0/quality_protocol.json").read_text()
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "state").mkdir()
            contexts = self._contexts(root)
            records = [
                self._record(root, index, unstable=True) for index in range(1, 4)
            ]
            with patch("rolloutbench.aggregation._run_record", side_effect=records):
                result = aggregate_system(contexts, root / "state", suite, protocol)
        self.assertEqual("NEEDS_TWO_ADDITIONAL_REPETITIONS", result["status"])
        self.assertIsNone(result["frontier"])
        self.assertIn("K20", result["adaptive_repeat_rule"]["candidates_above_threshold"])

    def test_source_replay_selects_first_three_from_predeclared_five(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_root = (root / "state").resolve()
            state_root.mkdir()
            planned_runs = [
                {
                    "run_id": f"full-serial1-repeat-{index:02d}",
                    "system": "serial1",
                    "scope": "full",
                    "repeat_index": index,
                    "episodes": [{"episode_id": "K20"}],
                }
                for index in range(1, 6)
            ]
            plan_path = (root / "plan.json").resolve()
            plan_path.write_text(
                json.dumps(
                    {
                        "plan_id": "plan",
                        "repetitions": 5,
                        "runs": planned_runs,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            preparation_path = (root / "preparation.json").resolve()
            preparation_path.write_text('{"status":"READY"}\n', encoding="utf-8")
            plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            preparation_sha256 = hashlib.sha256(
                preparation_path.read_bytes()
            ).hexdigest()
            contexts: list[RunContext] = []
            ledger_receipts: dict[str, dict] = {}
            for planned in planned_runs[:3]:
                run_sha256 = hashlib.sha256(
                    json.dumps(
                        planned,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest()
                context = RunContext(
                    plan_id="plan",
                    plan_sha256=plan_sha256,
                    run_sha256=run_sha256,
                    preparation_sha256=preparation_sha256,
                    run=planned,
                    preparation={},
                    plan_path=plan_path,
                    preparation_path=preparation_path,
                )
                contexts.append(context)
                ledger_path = (
                    state_root
                    / "plans"
                    / "plan"
                    / plan_sha256
                    / planned["run_id"]
                    / run_sha256
                    / "events.jsonl"
                )
                ledger_path.parent.mkdir(parents=True)
                ledger_path.write_text("{}\n", encoding="utf-8")
                ledger_receipts[planned["run_id"]] = {
                    "path": str(ledger_path),
                    "sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
                    "size_bytes": ledger_path.stat().st_size,
                }
            result = {
                "plan_id": "plan",
                "plan_sha256": plan_sha256,
                "system": "serial1",
                "repetitions": 3,
                "runs": [{"run_id": row.run["run_id"]} for row in contexts],
                "source_receipts": {
                    "plan": {
                        "path": str(plan_path),
                        "sha256": plan_sha256,
                        "size_bytes": plan_path.stat().st_size,
                    },
                    "preparation": {
                        "path": str(preparation_path),
                        "sha256": preparation_sha256,
                        "size_bytes": preparation_path.stat().st_size,
                    },
                    "state_root": str(state_root),
                    "run_ledgers": ledger_receipts,
                },
            }
            with (
                patch(
                    "rolloutbench.aggregation.load_run_context",
                    side_effect=contexts,
                ) as load_context,
                patch(
                    "rolloutbench.aggregation.aggregate_system",
                    return_value=result,
                ) as aggregate,
            ):
                self.assertEqual(result, _replay_system_result(result, {}, {}))
            self.assertEqual(3, load_context.call_count)
            self.assertEqual(3, len(aggregate.call_args.args[0]))


class FourSystemComparisonTests(unittest.TestCase):
    def _suite(self) -> tuple[Path, dict, dict, list[str]]:
        repo = Path(__file__).resolve().parents[2]
        suite_dir = repo / "benchmarks/sana_video_2b_h100_v0"
        suite = json.loads((suite_dir / "suite.json").read_text())
        protocol = json.loads((suite_dir / "quality_protocol.json").read_text())
        episode_ids = [
            json.loads(line)["episode_id"]
            for line in (suite_dir / "episodes.jsonl").read_text().splitlines()
        ]
        return suite_dir, suite, protocol, episode_ids

    def _write_result(
        self,
        root: Path,
        system: str,
        *,
        total_s: float,
        plan_sha256: str = "a" * 64,
        status: str = "FULL_AGGREGATED",
        drop_episode: bool = False,
    ) -> Path:
        suite_dir, suite, protocol, episode_ids = self._suite()
        del suite_dir
        workers = 2 if system in {"fifo2", "optroll2"} else 1
        run_ttvf = total_s / 3.0
        busy_per_run = run_ttvf * workers * 0.8
        capacity_per_run = run_ttvf * workers
        runs = []
        for repeat_index in range(1, 4):
            run_id = f"full-{system}-repeat-{repeat_index:02d}"
            run_sha256 = str(repeat_index) * 64
            frontier = {
                "schema_version": 1,
                "record_type": "provisional_single_repetition_frontier",
                "status": "PROVISIONAL_SINGLE_REPETITION",
                "plan_id": "plan",
                "plan_sha256": plan_sha256,
                "run_id": run_id,
                "run_sha256": run_sha256,
                "decision_count": len(episode_ids),
                "frontier": {"kernel": "K20", "cache": "C12"},
                "ranked": {
                    "kernel": [{"episode_id": "K20", "ranking_latency_s": 1.0}],
                    "cache": [{"episode_id": "C12", "ranking_latency_s": 1.0}],
                },
                "legacy_oracle": {"kernel": "K20", "cache": "C12"},
                "selection_basis": (
                    "fresh_single_run_ranking_latency_s_one_shot_process_wall"
                ),
                "final_frontier_requires_repetition_aggregation": True,
                "performance_claim": False,
            }
            frontier_path = root / system / f"frontier-{repeat_index}.json"
            frontier_path.parent.mkdir(parents=True, exist_ok=True)
            frontier_path.write_text(json.dumps(frontier, sort_keys=True) + "\n")
            decisions = {episode_id: {"outcome": "observed"} for episode_id in episode_ids}
            if drop_episode:
                decisions.pop(episode_ids[-1])
            runs.append(
                {
                    "run_id": run_id,
                    "repeat_index": repeat_index,
                    "run_sha256": run_sha256,
                    "ttvf_s": run_ttvf,
                    "decision_count": len(episode_ids),
                    "decisions": decisions,
                    "frontier": frontier,
                    "frontier_receipt": {
                        "path": str(frontier_path),
                        "sha256": hashlib.sha256(frontier_path.read_bytes()).hexdigest(),
                        "size_bytes": frontier_path.stat().st_size,
                    },
                    "gpu_busy_s": busy_per_run,
                    "gpu_capacity_s": capacity_per_run,
                    "gpu_queue_idle_s": capacity_per_run - busy_per_run,
                    "scheduler_gpu_utilization": busy_per_run / capacity_per_run,
                }
            )
        result = {
            "schema_version": 1,
            "record_type": "sol_rolloutbench_system_result",
            "status": status,
            "plan_id": "plan",
            "plan_sha256": plan_sha256,
            "system": system,
            "scope": "full",
            "repetitions": 3,
            "episode_ids": episode_ids,
            "suite_contract": {
                "suite_id": suite["suite_id"],
                "episodes_sha256": suite["episodes_sha256"],
                "quality_protocol_sha256": suite["quality_protocol_sha256"],
                "quality_protocol_id": protocol["protocol_id"],
            },
            "frontier": {"kernel": "K20", "cache": "C12"},
            "kernel_frontier": {"winner": "K20"},
            "cache_frontier": {"winner": "C12"},
            "frontier_agreement_with_legacy_oracle": True,
            "metrics": {
                "time_to_validated_frontier": {"total_s": total_s},
                "gpu_hours": busy_per_run * 3 / 3600.0,
                "validated_decisions_per_hour": len(episode_ids) * 3 / (total_s / 3600.0),
                "scheduler_gpu_utilization": 0.8,
                "gpu_queue_idle_s": (capacity_per_run - busy_per_run) * 3,
            },
            "runs": runs,
            "performance_claim": False,
        }
        path = root / f"{system}.json"
        write_system_result(path, result)
        return path

    def _four(self, root: Path) -> list[Path]:
        totals = {"serial1": 400.0, "fifo2": 250.0, "optroll1": 300.0, "optroll2": 200.0}
        return [
            self._write_result(root, system, total_s=totals[system])
            for system in ("serial1", "fifo2", "optroll1", "optroll2")
        ]

    def _compare(self, paths: list[Path], suite_dir: Path, repo: Path) -> dict:
        with patch(
            "rolloutbench.aggregation._replay_system_result",
            side_effect=lambda result, _suite, _protocol: result,
        ):
            return compare_system_results(paths, suite_dir, repo_root=repo)

    def test_exact_four_system_comparison_is_sealed_and_ranked_by_ttvf(self):
        suite_dir, _, _, _ = self._suite()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._four(root)
            comparison = self._compare(
                paths, suite_dir, Path(__file__).resolve().parents[2]
            )
            receipt = write_system_comparison(root / "comparison.json", comparison)
            replay = write_system_comparison(root / "comparison.json", comparison)
        self.assertEqual("FULL_COMPARISON_VALIDATED", comparison["status"])
        self.assertEqual("optroll2", comparison["fastest_system"])
        self.assertEqual(
            ["optroll2", "fifo2", "optroll1", "serial1"],
            comparison["ttvf_ranking"],
        )
        self.assertEqual(receipt["sha256"], replay["sha256"])

    def test_comparison_rejects_missing_duplicate_cross_plan_and_partial_inputs(self):
        suite_dir, _, _, _ = self._suite()
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._four(root)
            with self.assertRaisesRegex(AggregationError, "exactly four"):
                self._compare(paths[:3], suite_dir, repo)
            with self.assertRaisesRegex(AggregationError, "duplicate"):
                self._compare([paths[0], paths[0], paths[2], paths[3]], suite_dir, repo)

            paths[3] = self._write_result(
                root / "cross-plan", "optroll2", total_s=200.0, plan_sha256="b" * 64
            )
            with self.assertRaisesRegex(AggregationError, "one plan"):
                self._compare(paths, suite_dir, repo)

            paths = self._four(root / "partial")
            paths[3] = self._write_result(
                root / "partial-replacement",
                "optroll2",
                total_s=200.0,
                status="NEEDS_TWO_ADDITIONAL_REPETITIONS",
            )
            with self.assertRaisesRegex(AggregationError, "header"):
                self._compare(paths, suite_dir, repo)

    def test_comparison_rejects_episode_or_frontier_receipt_tamper(self):
        suite_dir, _, _, _ = self._suite()
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._four(root)
            broken = self._write_result(
                root / "missing-episode",
                "optroll2",
                total_s=200.0,
                drop_episode=True,
            )
            with self.assertRaisesRegex(AggregationError, "exact episode"):
                self._compare([*paths[:3], broken], suite_dir, repo)

            frontier_path = root / "optroll2" / "frontier-1.json"
            frontier_path.write_text("{}\n")
            with self.assertRaisesRegex(AggregationError, "digest"):
                self._compare(paths, suite_dir, repo)

    def test_comparison_rejects_hand_forged_unreplayable_results(self):
        suite_dir, _, _, _ = self._suite()
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            paths = self._four(Path(directory))
            with self.assertRaisesRegex(AggregationError, "replayable source"):
                compare_system_results(paths, suite_dir, repo_root=repo)

    def test_comparison_rejects_per_run_busy_time_over_capacity(self):
        suite_dir, _, _, _ = self._suite()
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._four(root)
            value = json.loads(paths[-1].read_text(encoding="utf-8"))
            capacity = value["runs"][0]["gpu_capacity_s"]
            delta = capacity * 0.25
            value["runs"][0]["gpu_busy_s"] += delta
            value["runs"][0]["scheduler_gpu_utilization"] = (
                value["runs"][0]["gpu_busy_s"] / capacity
            )
            value["runs"][1]["gpu_busy_s"] -= delta
            value["runs"][1]["gpu_queue_idle_s"] += delta
            value["runs"][1]["scheduler_gpu_utilization"] = (
                value["runs"][1]["gpu_busy_s"] / capacity
            )
            forged = root / "optroll2-over-capacity.json"
            forged.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(AggregationError, "per-run GPU metrics"):
                self._compare([*paths[:3], forged], suite_dir, repo)

    def test_locked_writer_allows_only_one_of_two_conflicting_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            barrier = threading.Barrier(2)
            successes: list[str] = []
            errors: list[Exception] = []

            def writer(status: str) -> None:
                barrier.wait()
                try:
                    write_system_result(path, {"status": status})
                    successes.append(status)
                except Exception as exc:  # the losing conflicting writer must fail
                    errors.append(exc)

            threads = [
                threading.Thread(target=writer, args=(status,))
                for status in ("one", "two")
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(1, len(successes))
        self.assertEqual(1, len(errors))
        self.assertIsInstance(errors[0], AggregationError)


if __name__ == "__main__":
    unittest.main()
