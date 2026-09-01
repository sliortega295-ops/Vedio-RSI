from __future__ import annotations

import json
import copy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rolloutbench.invocation import (
    InvocationError,
    build_episode_invocation,
    materialize_formal_prompt,
)
from rolloutbench.materialize import materialize_candidate_artifacts
from rolloutbench.runplan import build_experiment_plan
from rolloutbench.validators import build_quality_plan


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_DIR = REPO_ROOT / "benchmarks" / "sana_video_2b_h100_v0"
GPU_UUIDS = (
    "GPU-83ed65f8-62e5-2a01-3471-8bfc752971d3",
    "GPU-847305ce-670b-91ee-e0a9-aa3b7833df23",
)
PLAN_SHA256 = "a" * 64
RUN_SHA256 = "b" * 64
PLAN_SOURCE = {
    "revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip(),
    "tree_clean": False,
}


def _pilot_run(system: str = "optroll2") -> dict:
    plan = build_experiment_plan(
        SUITE_DIR,
        scope="pilot",
        repetitions=3,
        gpu_uuids=GPU_UUIDS,
        repo_root=REPO_ROOT,
    )
    return next(row for row in plan["runs"] if row["system"] == system)


def _episode(run: dict, episode_id: str) -> dict:
    return next(row for row in run["episodes"] if row["episode_id"] == episode_id)


def _runtime_receipt(episode: dict) -> dict:
    return {
        "runtime_ref": episode["runtime_checkout"]["git_ref"],
        "runtime_tree_oid": episode["runtime_checkout"]["runtime_tree_oid"],
        "required_runtime_paths": episode["runtime_checkout"]["required_runtime_paths"],
        "worktree_path": str(REPO_ROOT),
        "status": "VALIDATED_EXISTING",
    }


class EpisodeInvocationTests(unittest.TestCase):
    def setUp(self) -> None:
        def verify_stub(_repository: Path, receipt: dict, expected: dict) -> dict:
            if receipt.get("runtime_ref") != expected.get("git_ref"):
                raise RuntimeError("runtime commit mismatch")
            return receipt

        self.runtime_verifier = mock.patch(
            "rolloutbench.invocation.verify_runtime_receipt",
            side_effect=verify_stub,
        )
        self.runtime_verifier.start()
        self.addCleanup(self.runtime_verifier.stop)

    def test_config_invocation_uses_exact_runtime_uuid_and_isolated_cache(self) -> None:
        run = _pilot_run()
        episode = _episode(run, "K01")
        worker = run["workers"][0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            derived = root / "derived"
            materialized = materialize_candidate_artifacts(
                episode, derived, repo_root=REPO_ROOT
            )
            invocation = build_episode_invocation(
                repo_root=REPO_ROOT,
                experiment_root=root,
                plan_id="plan-123",
                plan_sha256=PLAN_SHA256,
                run_sha256=RUN_SHA256,
                run=run,
                episode=episode,
                worker=worker,
                materialized_root=derived,
                materialization_receipt=materialized,
                runtime_receipt=_runtime_receipt(episode),
                lease_files={worker["gpu_uuid"]: root / "lease-0.json"},
                plan_source=PLAN_SOURCE,
                require_clean_harness=False,
            )
            self.assertEqual("config_generation", invocation["kind"])
            self.assertEqual(
                str(REPO_ROOT / "models/sana_video_2b_h100/baseline/scripts/run_sana_video_2b_gpu.sh"),
                invocation["argv"][1],
            )
            self.assertEqual("1", invocation["env"]["SANA_ENABLE_COMPILE"])
            self.assertEqual(worker["gpu_uuid"], invocation["env"]["CUDA_VISIBLE_DEVICES"])
            self.assertEqual("42", invocation["env"]["SANA_WORKLOAD_SEED"])
            self.assertEqual(
                str(REPO_ROOT / "external/sol_runtime"),
                invocation["env"]["SANA_RUNTIME_ROOT"],
            )
            self.assertIn("optroll2/repeat-01/K01", invocation["env"]["TRITON_CACHE_DIR"])
            self.assertNotIn("golden", json.dumps(invocation))

    def test_k02_reuses_only_k01_compile_namespace(self) -> None:
        run = _pilot_run("fifo2")
        episode = _episode(run, "K02")
        worker = run["workers"][0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            derived = root / "derived"
            materialized = materialize_candidate_artifacts(
                episode, derived, repo_root=REPO_ROOT
            )
            invocation = build_episode_invocation(
                repo_root=REPO_ROOT,
                experiment_root=root,
                plan_id="plan-123",
                plan_sha256=PLAN_SHA256,
                run_sha256=RUN_SHA256,
                run=run,
                episode=episode,
                worker=worker,
                materialized_root=derived,
                materialization_receipt=materialized,
                runtime_receipt=_runtime_receipt(episode),
                lease_files={worker["gpu_uuid"]: root / "lease-0.json"},
                plan_source=PLAN_SOURCE,
                require_clean_harness=False,
            )
            self.assertIn("fifo2/repeat-01/K01", invocation["env"]["TORCHINDUCTOR_CACHE_DIR"])
            self.assertNotIn("/K02/", invocation["env"]["TORCHINDUCTOR_CACHE_DIR"])

    def test_quality_pair_renders_motion_suffix_and_routes_seed(self) -> None:
        run = _pilot_run()
        episode = _episode(run, "C02")
        worker = run["workers"][1]
        protocol = json.loads((SUITE_DIR / "quality_protocol.json").read_text())
        pair = next(
            row
            for row in build_quality_plan(protocol, ["C02"])
            if row["seed"] == 12345
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            derived = root / "derived"
            materialized = materialize_candidate_artifacts(
                episode, derived, repo_root=REPO_ROOT
            )
            invocation = build_episode_invocation(
                repo_root=REPO_ROOT,
                experiment_root=root,
                plan_id="plan-123",
                plan_sha256=PLAN_SHA256,
                run_sha256=RUN_SHA256,
                run=run,
                episode=episode,
                worker=worker,
                materialized_root=derived,
                materialization_receipt=materialized,
                runtime_receipt=_runtime_receipt(episode),
                lease_files={worker["gpu_uuid"]: root / "lease-1.json"},
                plan_source=PLAN_SOURCE,
                quality_pair=pair,
                require_clean_harness=False,
            )
            prompt = Path(invocation["env"]["SANA_PROMPT_FILE"])
            self.assertEqual(
                f'{pair["prompt"]} motion score: 30.\n',
                prompt.read_text(),
            )
            self.assertEqual("12345", invocation["env"]["SANA_WORKLOAD_SEED"])
            self.assertEqual(pair["pair_id"], invocation["quality_pair_id"])
            self.assertEqual("candidate", invocation["quality_role"])
            self.assertEqual(
                pair["candidate_artifact_id"], invocation["quality_artifact_id"]
            )

    def test_dense_quality_invocation_uses_frozen_reference(self) -> None:
        run = _pilot_run()
        dense = run["quality_dense_reference"]
        worker = run["workers"][1]
        pair = _episode(run, "C02")["quality_pairs"][0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            derived = root / "derived"
            materialized = materialize_candidate_artifacts(
                dense, derived, repo_root=REPO_ROOT
            )
            invocation = build_episode_invocation(
                repo_root=REPO_ROOT,
                experiment_root=root,
                plan_id="plan-123",
                plan_sha256=PLAN_SHA256,
                run_sha256=RUN_SHA256,
                run=run,
                episode=dense,
                worker=worker,
                materialized_root=derived,
                materialization_receipt=materialized,
                runtime_receipt=_runtime_receipt(dense),
                lease_files={worker["gpu_uuid"]: root / "lease-1.json"},
                plan_source=PLAN_SOURCE,
                quality_pair=pair,
                require_clean_harness=False,
            )
            self.assertEqual("dense", invocation["quality_role"])
            self.assertEqual(pair["dense_artifact_id"], invocation["quality_artifact_id"])
            self.assertEqual("0", invocation["env"]["SANA_ENABLE_COMPILE"])
            self.assertEqual("0", invocation["env"]["SANA_EASYCACHE_THRESH"])
            self.assertIn("/DENSE/quality-v1/", invocation["output_path"])

    def test_probe_invocation_uses_guard_lease_and_candidate_runtime(self) -> None:
        run = _pilot_run()
        episode = _episode(run, "K15")
        worker = run["workers"][0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            derived = root / "derived"
            materialized = materialize_candidate_artifacts(
                episode, derived, repo_root=REPO_ROOT
            )
            invocation = build_episode_invocation(
                repo_root=REPO_ROOT,
                experiment_root=root,
                plan_id="plan-123",
                plan_sha256=PLAN_SHA256,
                run_sha256=RUN_SHA256,
                run=run,
                episode=episode,
                worker=worker,
                materialized_root=derived,
                materialization_receipt=materialized,
                runtime_receipt=_runtime_receipt(episode),
                lease_files={worker["gpu_uuid"]: root / "lease-0.json"},
                plan_source=PLAN_SOURCE,
                require_clean_harness=False,
            )
            self.assertEqual("gpu_preflight_probe", invocation["kind"])
            self.assertIn("--lease-file", invocation["argv"])
            self.assertIn("--guard-dir", invocation["argv"])
            self.assertIn("--runtime-python-root", invocation["argv"])
            self.assertTrue(invocation["output_path"].endswith("probe-result.json"))

    def test_k22_expected_failure_routes_to_a_benchmark_receipt(self) -> None:
        run = _pilot_run()
        episode = _episode(run, "K22")
        worker = run["workers"][0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            derived = root / "derived"
            materialized = materialize_candidate_artifacts(
                episode, derived, repo_root=REPO_ROOT
            )
            invocation = build_episode_invocation(
                repo_root=REPO_ROOT,
                experiment_root=root,
                plan_id="plan-123",
                plan_sha256=PLAN_SHA256,
                run_sha256=RUN_SHA256,
                run=run,
                episode=episode,
                worker=worker,
                materialized_root=derived,
                materialization_receipt=materialized,
                runtime_receipt=_runtime_receipt(episode),
                lease_files={worker["gpu_uuid"]: root / "lease-0.json"},
                plan_source=PLAN_SOURCE,
                require_clean_harness=False,
            )
        self.assertEqual("expected_fail_closed_generation", invocation["kind"])
        self.assertTrue(invocation["output_path"].endswith("benchmark.json"))
        self.assertEqual(
            episode["expected_failure_contract"],
            invocation["expected_failure_contract"],
        )

    def test_prompt_and_runtime_mismatches_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InvocationError):
                materialize_formal_prompt(Path(directory), "bad", "two\nlines")
            run = _pilot_run()
            episode = _episode(run, "K01")
            materialized = materialize_candidate_artifacts(
                episode, Path(directory) / "derived", repo_root=REPO_ROOT
            )
            bad_runtime = _runtime_receipt(episode)
            bad_runtime["runtime_ref"] = "0" * 40
            worker = run["workers"][0]
            with self.assertRaisesRegex(InvocationError, "runtime checkout receipt"):
                build_episode_invocation(
                    repo_root=REPO_ROOT,
                    experiment_root=Path(directory),
                    plan_id="plan-123",
                    plan_sha256=PLAN_SHA256,
                    run_sha256=RUN_SHA256,
                    run=run,
                    episode=episode,
                    worker=worker,
                    materialized_root=Path(directory) / "derived",
                    materialization_receipt=materialized,
                    runtime_receipt=bad_runtime,
                    lease_files={worker["gpu_uuid"]: Path(directory) / "lease.json"},
                    plan_source=PLAN_SOURCE,
                    require_clean_harness=False,
                )

    def test_invocation_fails_closed_when_runtime_receipt_reverification_rejects_it(self) -> None:
        run = _pilot_run()
        episode = _episode(run, "K01")
        worker = run["workers"][0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            derived = root / "derived"
            materialized = materialize_candidate_artifacts(
                episode, derived, repo_root=REPO_ROOT
            )
            self.runtime_verifier.stop()
            with mock.patch(
                "rolloutbench.invocation.verify_runtime_receipt",
                side_effect=RuntimeError("stale forged worktree"),
            ), self.assertRaisesRegex(InvocationError, "runtime checkout receipt"):
                build_episode_invocation(
                    repo_root=REPO_ROOT,
                    experiment_root=root,
                    plan_id="plan-123",
                    plan_sha256=PLAN_SHA256,
                    run_sha256=RUN_SHA256,
                    run=run,
                    episode=episode,
                    worker=worker,
                    materialized_root=derived,
                    materialization_receipt=materialized,
                    runtime_receipt=_runtime_receipt(episode),
                    lease_files={worker["gpu_uuid"]: root / "lease.json"},
                    plan_source=PLAN_SOURCE,
                    require_clean_harness=False,
                )

    def test_quality_pair_cannot_reuse_an_id_with_tampered_prompt(self) -> None:
        run = _pilot_run()
        episode = _episode(run, "C02")
        worker = run["workers"][1]
        pair = copy.deepcopy(episode["quality_pairs"][0])
        pair["prompt"] = "tampered prompt"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            derived = root / "derived"
            materialized = materialize_candidate_artifacts(
                episode, derived, repo_root=REPO_ROOT
            )
            with self.assertRaisesRegex(InvocationError, "not declared"):
                build_episode_invocation(
                    repo_root=REPO_ROOT,
                    experiment_root=root,
                    plan_id="plan-123",
                    plan_sha256=PLAN_SHA256,
                    run_sha256=RUN_SHA256,
                    run=run,
                    episode=episode,
                    worker=worker,
                    materialized_root=derived,
                    materialization_receipt=materialized,
                    runtime_receipt=_runtime_receipt(episode),
                    lease_files={worker["gpu_uuid"]: root / "lease-1.json"},
                    plan_source=PLAN_SOURCE,
                    quality_pair=pair,
                    require_clean_harness=False,
                )


if __name__ == "__main__":
    unittest.main()
