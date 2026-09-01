from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from rolloutbench.executor import FakeExecutor
from rolloutbench.scheduler import load_public_episodes, simulate
from rolloutbench.schema import SuiteValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_DIR = REPO_ROOT / "benchmarks" / "sana_video_2b_h100_v0"


class SchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.episodes = load_public_episodes(SUITE_DIR / "episodes.jsonl")

    def test_loader_removes_golden_and_all_systems_release_and_decide_all_35(self) -> None:
        self.assertEqual(35, len(self.episodes))
        self.assertTrue(all("golden" not in episode for episode in self.episodes))
        for system in ("serial1", "fifo2", "optroll1", "optroll2"):
            with self.subTest(system=system):
                result = simulate(system, self.episodes, FakeExecutor())
                self.assertEqual(35, result.summary["released_episodes"])
                self.assertEqual(35, result.summary["sealed_decisions"])
                self.assertEqual(35, len({row["episode_id"] for row in result.decisions}))

    def test_concurrency_limits_global_fifo_and_stream_dependencies(self) -> None:
        expected_fifo = [
            episode["episode_id"]
            for episode in sorted(self.episodes, key=lambda row: row["global_fifo_index"])
        ]
        limits = {"serial1": 1, "fifo2": 2, "optroll1": 1, "optroll2": 2}
        for system, limit in limits.items():
            with self.subTest(system=system):
                result = simulate(system, self.episodes, FakeExecutor())
                self.assertLessEqual(result.summary["max_gpu_concurrency"], limit)
                starts = [row for row in result.trace if row["event"] == "episode_started"]
                if system in {"serial1", "fifo2"}:
                    self.assertEqual(expected_fifo, [row["episode_id"] for row in starts])
                by_id = {row["episode_id"]: row for row in starts}
                completed = {
                    row["episode_id"]: row
                    for row in result.trace
                    if row["event"] == "episode_completed"
                }
                for episode in self.episodes:
                    for dependency in episode["depends_on"]:
                        self.assertLessEqual(completed[dependency]["time"], by_id[episode["episode_id"]]["time"])

    def test_optroll_uses_only_declared_reuse_and_typed_validation_exit(self) -> None:
        for system in ("optroll1", "optroll2"):
            result = simulate(system, self.episodes, FakeExecutor())
            cache_hits = [row for row in result.trace if row["event"] == "cache_hit"]
            self.assertEqual([("K02", "K01", "torch_compile_cache")], [
                (row["episode_id"], row["source_episode_id"], row["artifact"])
                for row in cache_hits
            ])
            stages = {
                episode_id: [row["stage"] for row in result.trace if row.get("episode_id") == episode_id and row["event"] == "stage_completed"]
                for episode_id in ("K15", "C01", "C02")
            }
            self.assertEqual(["acquire_gpu", "preflight", "microbenchmark", "decide"], stages["K15"])
            self.assertIn("legacy_sanity", stages["C01"])
            self.assertNotIn("quality_v1", stages["C01"])
            self.assertIn("quality_v1", stages["C02"])

        for system in ("serial1", "fifo2"):
            result = simulate(system, self.episodes, FakeExecutor())
            self.assertFalse(any(row["event"] == "cache_hit" for row in result.trace))

    def test_scheduler_result_is_identical_after_golden_is_removed_or_changed(self) -> None:
        raw = [json.loads(line) for line in (SUITE_DIR / "episodes.jsonl").read_text().splitlines()]
        public = load_public_episodes(raw)
        tampered = copy.deepcopy(raw)
        for episode in tampered:
            episode["golden"] = {"scheduler_visible": False, "decision": {"outcome": "tampered"}}
        for system in ("serial1", "fifo2", "optroll1", "optroll2"):
            expected = simulate(system, public, FakeExecutor()).as_dict()
            actual = simulate(system, load_public_episodes(tampered), FakeExecutor()).as_dict()
            self.assertEqual(expected, actual)

    def test_fake_executor_decisions_and_frontier_agree_with_schedule_summary(self) -> None:
        result = simulate("optroll2", self.episodes, FakeExecutor())
        self.assertTrue(result.summary["decision_agreement"])
        self.assertTrue(result.summary["frontier_agreement"])
        self.assertFalse(result.summary["historical_oracle_checked"])
        self.assertEqual("fake_contract_only_not_historical", result.summary["frontier_semantics"])
        self.assertEqual(result.summary["frontier"], result.frontier)
        self.assertEqual({"kernel": "K20", "cache": "C12"}, result.frontier)
        decisions = {row["episode_id"]: row for row in result.decisions}
        self.assertEqual("recorded_failure", decisions["K22"]["outcome"])

    def test_cli_simulate_writes_deterministic_json(self) -> None:
        from rolloutbench.cli import main

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            args = ["simulate", "--system", "optroll2", "--suite", str(SUITE_DIR)]
            self.assertEqual(0, main(args + ["--output", str(first)]))
            self.assertEqual(0, main(args + ["--output", str(second)]))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            payload = json.loads(first.read_text())
            self.assertTrue(payload["summary"]["synthetic_contract_simulation"])

    def test_cli_simulate_validates_suite_before_scheduling(self) -> None:
        from rolloutbench.cli import main

        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "suite"
            shutil.copytree(SUITE_DIR, tampered)
            episodes_path = tampered / "episodes.jsonl"
            rows = episodes_path.read_text(encoding="utf-8").splitlines()
            first = json.loads(rows[0])
            first["hypothesis"] = "tampered"
            rows[0] = json.dumps(first, sort_keys=True)
            episodes_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SuiteValidationError, "file hash mismatch"):
                main(["simulate", "--system", "serial1", "--suite", str(tampered)])


if __name__ == "__main__":
    unittest.main()
