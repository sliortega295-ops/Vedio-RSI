from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rolloutbench.preparation import PreparationError, prepare_experiment
from rolloutbench.runplan import build_experiment_plan


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_DIR = REPO_ROOT / "benchmarks" / "sana_video_2b_h100_v0"
GPU_UUIDS = (
    "GPU-83ed65f8-62e5-2a01-3471-8bfc752971d3",
    "GPU-847305ce-670b-91ee-e0a9-aa3b7833df23",
)


def _write_plan(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


class ExperimentPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = build_experiment_plan(
            SUITE_DIR,
            scope="pilot",
            repetitions=3,
            gpu_uuids=GPU_UUIDS,
            repo_root=REPO_ROOT,
        )

    def test_prepares_each_unique_episode_once_and_is_idempotent(self) -> None:
        runtime_calls: list[str] = []
        artifact_calls: list[str] = []

        def fake_runtime(episode, repository_root, worktree_root):
            del repository_root
            episode_id = episode["episode_id"]
            runtime_calls.append(episode_id)
            manifest = episode["runtime_checkout"]
            return {
                "schema_version": 1,
                "status": "READY",
                "runtime_ref": manifest["git_ref"],
                "ref_role": manifest["ref_role"],
                "worktree_path": str(Path(worktree_root) / manifest["git_ref"]),
                "runtime_tree_oid": manifest["runtime_tree_oid"],
                "required_runtime_paths": manifest["required_runtime_paths"],
                "critical_runtime_file_sha256": {
                    path: "0" * 64 for path in manifest["required_runtime_paths"]
                },
            }

        def fake_materialize(episode, derived_root, *, repo_root):
            del derived_root, repo_root
            episode_id = episode["episode_id"]
            artifact_calls.append(episode_id)
            return {
                "schema_version": 1,
                "episode_id": episode_id,
                "authority_ref": episode["candidate"]["authority_ref"],
                "artifacts": [{"kind": "fixture", "sha256": "1" * 64}],
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            _write_plan(plan_path, self.plan)
            with (
                mock.patch(
                    "rolloutbench.preparation.prepare_runtime_checkout",
                    side_effect=fake_runtime,
                ),
                mock.patch(
                    "rolloutbench.preparation.materialize_candidate_artifacts",
                    side_effect=fake_materialize,
                ),
            ):
                first = prepare_experiment(
                    plan_path,
                    SUITE_DIR,
                    root / "experiment",
                    repo_root=REPO_ROOT,
                    require_clean=False,
                )
                second = prepare_experiment(
                    plan_path,
                    SUITE_DIR,
                    root / "experiment",
                    repo_root=REPO_ROOT,
                    require_clean=False,
                )

            pilot_ids = list(self.plan["runs"][0]["episodes"])
            expected_ids = ["DENSE"] + [episode["episode_id"] for episode in pilot_ids]
            self.assertEqual(expected_ids * 2, runtime_calls)
            self.assertEqual(expected_ids * 2, artifact_calls)
            self.assertEqual("READY", first["status"])
            self.assertEqual(first, second)
            self.assertEqual(11, first["unique_episode_count"])
            self.assertEqual(12, first["run_count"])
            self.assertFalse(first["gpu_execution"])
            self.assertFalse(first["vbench_execution"])
            receipt_path = root / "experiment" / "state" / "preparation.json"
            self.assertEqual(first, json.loads(receipt_path.read_text()))

    def test_tampered_plan_and_conflicting_existing_receipt_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            tampered = copy.deepcopy(self.plan)
            tampered["runs"][0]["episodes"][0]["candidate"]["parent_sha"] = "0" * 40
            _write_plan(plan_path, tampered)
            with self.assertRaisesRegex(PreparationError, "canonical plan"):
                prepare_experiment(
                    plan_path,
                    SUITE_DIR,
                    root / "experiment",
                    repo_root=REPO_ROOT,
                    require_clean=False,
                )

            _write_plan(plan_path, self.plan)
            receipt_path = root / "experiment" / "state" / "preparation.json"
            receipt_path.parent.mkdir(parents=True)
            receipt_path.write_text("{}\n")
            with (
                mock.patch(
                    "rolloutbench.preparation.prepare_runtime_checkout",
                    return_value={"status": "READY"},
                ),
                mock.patch(
                    "rolloutbench.preparation.materialize_candidate_artifacts",
                    return_value={"artifacts": []},
                ),
                self.assertRaisesRegex(PreparationError, "overwrite"),
            ):
                prepare_experiment(
                    plan_path,
                    SUITE_DIR,
                    root / "experiment",
                    repo_root=REPO_ROOT,
                    require_clean=False,
                )


if __name__ == "__main__":
    unittest.main()
