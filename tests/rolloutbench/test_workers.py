from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from rolloutbench.workers import (
    build_compatibility_key,
    evaluate_confirmation_reuse,
    validate_reset_proof,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_DIR = REPO_ROOT / "benchmarks" / "sana_video_2b_h100_v0"


def _key(**overrides: object) -> str:
    values = {
        "runtime_file_hashes": {"b.py": "b" * 64, "a.py": "a" * 64},
        "model_revision": "model-ref",
        "init_environment": {"torch": "2.11", "python": "3.12"},
        "dtype": "bfloat16",
        "backend": "cuda",
        "workload_fingerprint": {"frames": 81, "steps": 50},
        "gpu_arch": "sm90",
        "reset_api_version": "1",
    }
    values.update(overrides)
    return build_compatibility_key(**values)


def _proof(key: str) -> dict:
    return {
        "compatibility_key": key,
        "controller_history_cleared": True,
        "rng_reset": True,
        "scheduler_state_cleared": True,
        "prompt_state_cleared": True,
        "output_state_cleared": True,
        "compile_cache_contract_verified": True,
        "candidate_cache_contract_verified": True,
        "fresh_process_structural_equivalence": True,
        "fresh_process_telemetry_equivalence": True,
    }


class WorkerContractTests(unittest.TestCase):
    def test_compatibility_key_is_order_stable_and_covers_every_field(self) -> None:
        expected = _key()
        self.assertEqual(
            expected,
            _key(
                runtime_file_hashes={"a.py": "a" * 64, "b.py": "b" * 64},
                init_environment={"python": "3.12", "torch": "2.11"},
                workload_fingerprint={"steps": 50, "frames": 81},
            ),
        )
        mutations = {
            "runtime_file_hashes": {"a.py": "c" * 64, "b.py": "b" * 64},
            "model_revision": "other",
            "init_environment": {"torch": "2.12", "python": "3.12"},
            "dtype": "float16",
            "backend": "inductor",
            "workload_fingerprint": {"frames": 81, "steps": 49},
            "gpu_arch": "sm100",
            "reset_api_version": "2",
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                self.assertNotEqual(expected, _key(**{field: value}))

    def test_reset_proof_fails_closed_on_missing_false_or_wrong_key(self) -> None:
        key = _key()
        self.assertEqual("persistent", validate_reset_proof(key, _proof(key))["worker_mode"])
        missing = _proof(key)
        missing.pop("rng_reset")
        self.assertEqual("one_shot", validate_reset_proof(key, missing)["worker_mode"])
        false = _proof(key)
        false["prompt_state_cleared"] = False
        self.assertEqual("one_shot", validate_reset_proof(key, false)["worker_mode"])
        self.assertEqual("one_shot", validate_reset_proof(key, _proof("wrong"))["worker_mode"])

    def test_k01_k02_compile_lineage_is_separate_from_model_state(self) -> None:
        episodes = [
            json.loads(line)
            for line in (SUITE_DIR / "episodes.jsonl").read_text().splitlines()
        ]
        k02 = next(row for row in episodes if row["episode_id"] == "K02")
        key = _key()
        receipt = {
            "source_episode_id": "K01",
            "artifact": "torch_compile_cache",
            "compatibility_key": key,
            "sha256": "d" * 64,
        }

        one_shot = evaluate_confirmation_reuse(
            k02,
            compile_artifact_receipt=receipt,
            expected_compatibility_key=key,
        )
        self.assertTrue(one_shot["compile_artifact_reuse"]["allowed"])
        self.assertFalse(one_shot["persistent_model_state_reuse"]["allowed"])
        self.assertTrue(one_shot["confirmation_independence_preserved"])

        self.assertFalse(one_shot["persistent_model_state_reuse"]["allowed"])
        self.assertEqual(
            "confirmation_requires_fresh_process",
            one_shot["persistent_model_state_reuse"]["reason"],
        )
        self.assertTrue(one_shot["confirmation_independence_preserved"])

        incompatible_receipt = copy.deepcopy(receipt)
        incompatible_receipt["compatibility_key"] = "wrong"
        incompatible = evaluate_confirmation_reuse(
            k02,
            compile_artifact_receipt=incompatible_receipt,
            expected_compatibility_key=key,
        )
        self.assertFalse(incompatible["compile_artifact_reuse"]["allowed"])
        self.assertIn("compatibility key mismatch", incompatible["errors"])

        tampered = copy.deepcopy(k02)
        tampered["reuse"]["inputs"][0]["episode_id"] = "K00"
        rejected = evaluate_confirmation_reuse(
            tampered,
            compile_artifact_receipt=receipt,
            expected_compatibility_key=key,
        )
        self.assertFalse(rejected["compile_artifact_reuse"]["allowed"])


if __name__ == "__main__":
    unittest.main()
