from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from rolloutbench.pilot_runner import expand_run_units
from rolloutbench.runplan import (
    build_experiment_plan,
    required_repetitions,
    write_experiment_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_DIR = REPO_ROOT / "benchmarks" / "sana_video_2b_h100_v0"
GPU_UUIDS = (
    "GPU-83ed65f8-62e5-2a01-3471-8bfc752971d3",
    "GPU-847305ce-670b-91ee-e0a9-aa3b7833df23",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class H100RunPlanTests(unittest.TestCase):
    def test_pilot_predeclares_four_systems_five_repetitions_without_golden(self) -> None:
        plan = build_experiment_plan(
            SUITE_DIR,
            scope="pilot",
            repetitions=5,
            gpu_uuids=GPU_UUIDS,
            repo_root=REPO_ROOT,
        )
        self.assertEqual("NOT_RUN", plan["execution_status"])
        self.assertFalse(plan["performance_claim"])
        self.assertEqual(
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            plan["source"]["revision"],
        )
        self.assertEqual(
            "detached_candidate_commit_worktree_replay",
            plan["runtime_contract"]["mode"],
        )
        self.assertEqual(
            "BOUND_TO_AUTHORITY_COMMIT",
            plan["runtime_contract"]["candidate_semantic_parity"],
        )
        k22 = next(
            episode
            for run in plan["runs"]
            for episode in run["episodes"]
            if episode["episode_id"] == "K22"
        )
        self.assertEqual(
            k22["candidate"]["candidate_commit"],
            k22["runtime_checkout"]["git_ref"],
        )
        k15 = next(
            episode
            for run in plan["runs"]
            for episode in run["episodes"]
            if episode["episode_id"] == "K15"
        )
        self.assertEqual(
            k15["candidate"]["parent_sha"], k15["runtime_checkout"]["git_ref"]
        )
        self.assertEqual(
            "parent_for_preflight_rejection", k15["runtime_checkout"]["ref_role"]
        )
        self.assertEqual(4, len(plan["suite_file_sha256"]))
        self.assertEqual(20, len(plan["runs"]))
        expected_ids = _json(SUITE_DIR / "suite.json")["pilot_episodes"]
        namespaces: set[str] = set()
        for run in plan["runs"]:
            self.assertEqual(expected_ids, [row["episode_id"] for row in run["episodes"]])
            self.assertNotIn("golden", json.dumps(run))
            self.assertNotIn(run["cache_namespace"], namespaces)
            namespaces.add(run["cache_namespace"])
            expected_workers = 1 if run["system"] in {"serial1", "optroll1"} else 2
            self.assertEqual(expected_workers, len(run["workers"]))
            dense = run["quality_dense_reference"]
            self.assertEqual("DENSE", dense["episode_id"])
            self.assertEqual(
                "8bd01c6898f920c140a9c74197676debbcaff1fe",
                dense["runtime_checkout"]["git_ref"],
            )
            self.assertEqual("dense_reference", dense["candidate_type"])
            self.assertEqual(138, len(expand_run_units(run)))
            selected = {row["episode_id"] for row in run["episodes"]}
            for episode in run["episodes"]:
                expected_historical = [
                    predecessor
                    for predecessor in episode["depends_on"]
                    if predecessor not in selected
                ]
                self.assertEqual(
                    expected_historical,
                    [
                        row["episode_id"]
                        for row in episode["historical_predecessor_receipts"]
                    ],
                )

        optroll2 = {
            run["repeat_index"]: run
            for run in plan["runs"]
            if run["system"] == "optroll2"
        }
        self.assertEqual(GPU_UUIDS[0], optroll2[1]["workers"][0]["gpu_uuid"])
        self.assertEqual(GPU_UUIDS[1], optroll2[2]["workers"][0]["gpu_uuid"])

    def test_confirmation_and_quality_routes_are_explicit(self) -> None:
        plan = build_experiment_plan(
            SUITE_DIR,
            scope="pilot",
            repetitions=5,
            gpu_uuids=GPU_UUIDS,
            repo_root=REPO_ROOT,
        )
        run = next(row for row in plan["runs"] if row["system"] == "optroll2")
        episodes = {row["episode_id"]: row for row in run["episodes"]}
        k02 = episodes["K02"]
        self.assertEqual("one_shot", k02["worker_contract"]["effective_mode"])
        self.assertEqual(
            "K01:torch_compile_cache",
            k02["declared_artifact_inputs"][0]["canonical_key"],
        )
        self.assertEqual("K01", k02["cache_scope_key"])
        fifo2 = next(row for row in plan["runs"] if row["system"] == "fifo2")
        fifo_k02 = next(row for row in fifo2["episodes"] if row["episode_id"] == "K02")
        self.assertEqual("lineage:K01", fifo_k02["worker_affinity"])
        self.assertEqual([], episodes["C01"]["quality_pairs"])
        self.assertEqual(8, len(episodes["C02"]["quality_pairs"]))
        self.assertEqual(8, len(episodes["C09"]["quality_pairs"]))
        self.assertEqual(8, len(episodes["C12"]["quality_pairs"]))
        self.assertEqual(1, run["quality_dense_reference"]["worker_affinity"])

    def test_full_scope_contains_all_35_and_adaptive_repeat_rule(self) -> None:
        plan = build_experiment_plan(
            SUITE_DIR,
            scope="full",
            repetitions=5,
            gpu_uuids=GPU_UUIDS,
            repo_root=REPO_ROOT,
        )
        self.assertTrue(all(len(run["episodes"]) == 35 for run in plan["runs"]))
        self.assertTrue(
            all(
                not episode["historical_predecessor_receipts"]
                for run in plan["runs"]
                for episode in run["episodes"]
            )
        )
        self.assertEqual(3, required_repetitions([10.0, 10.1, 9.9]))
        self.assertEqual(5, required_repetitions([10.0, 11.0, 9.0]))
        with self.assertRaises(ValueError):
            required_repetitions([10.0, 0.0, 9.0])

    def test_plan_writer_is_atomic_idempotent_and_dirty_fail_closed(self) -> None:
        plan = build_experiment_plan(
            SUITE_DIR,
            scope="pilot",
            repetitions=5,
            gpu_uuids=GPU_UUIDS,
            repo_root=REPO_ROOT,
        )
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "pilot-plan.json"
            dirty_plan = copy.deepcopy(plan)
            dirty_plan["source"]["tree_clean"] = False
            with self.assertRaises(RuntimeError):
                write_experiment_plan(target, dirty_plan)
            first = write_experiment_plan(target, plan, require_clean=False)
            second = write_experiment_plan(target, plan, require_clean=False)
            self.assertEqual("WRITTEN", first["status"])
            self.assertEqual("UNCHANGED", second["status"])
            self.assertEqual(first["sha256"], second["sha256"])
            changed = dict(plan)
            changed["execution_status"] = "INVALID"
            with self.assertRaises(FileExistsError):
                write_experiment_plan(target, changed, require_clean=False)
            wrong_source = copy.deepcopy(plan)
            wrong_source["source"]["revision"] = "0" * 40
            with self.assertRaisesRegex(RuntimeError, "revision"):
                write_experiment_plan(
                    Path(temporary) / "wrong.json",
                    wrong_source,
                    require_clean=False,
                    repo_root=REPO_ROOT,
                )

    def test_formal_plan_rejects_only_three_predeclared_repetitions(self) -> None:
        with self.assertRaisesRegex(ValueError, "predeclare five"):
            build_experiment_plan(
                SUITE_DIR,
                scope="pilot",
                repetitions=3,
                gpu_uuids=GPU_UUIDS,
                repo_root=REPO_ROOT,
            )


if __name__ == "__main__":
    unittest.main()
