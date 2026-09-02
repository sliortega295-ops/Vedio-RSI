from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rolloutbench.quality_contract import VBENCH_REF
from rolloutbench.validators import build_quality_plan
from rolloutbench.vbench_runner import (
    VBenchContractError,
    build_vbench_pair_plan,
    parse_vbench_pair_results,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_DIR = REPO_ROOT / "benchmarks" / "sana_video_2b_h100_v0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VBenchPairContractTests(unittest.TestCase):
    def setUp(self) -> None:
        protocol = json.loads(
            (SUITE_DIR / "quality_protocol.json").read_text(encoding="utf-8")
        )
        self.pair = build_quality_plan(protocol, ["C02"])[0]
        self.protocol = protocol

    def _fixture(self, root: Path) -> dict[str, object]:
        source = root / "VBench"
        (source / "vbench").mkdir(parents=True)
        (source / "evaluate.py").write_text("# pinned fixture\n", encoding="utf-8")
        (source / "vbench" / "VBench_full_info.json").write_text(
            "[]\n", encoding="utf-8"
        )
        cache = root / "cache"
        cache.mkdir()
        python = root / "python"
        python.write_text("", encoding="utf-8")
        dense = root / "dense.mp4"
        candidate = root / "candidate.mp4"
        dense.write_bytes(b"dense-video")
        candidate.write_bytes(b"candidate-video")
        output = root / "quality-results" / self.pair["pair_id"]
        return {
            "dense_video_path": dense,
            "candidate_video_path": candidate,
            "dense_receipt": {
                "artifact_id": self.pair["dense_artifact_id"],
                "video_path": str(dense),
                "sha256": _sha256(dense),
            },
            "candidate_receipt": {
                "artifact_id": self.pair["candidate_artifact_id"],
                "video_path": str(candidate),
                "sha256": _sha256(candidate),
            },
            "vbench_source_path": source,
            "vbench_source_ref": VBENCH_REF,
            "vbench_cache_path": cache,
            "python_bin": python,
            "output_path": output,
            "gpu_uuid": "GPU-83ed65f8-62e5-2a01-3471-8bfc752971d3",
            "source_verification": "NON_FORMAL_TEST_ONLY",
        }

    @staticmethod
    def _receipt_fingerprint(receipt: dict[str, object]) -> str:
        identity = {key: value for key, value in receipt.items() if key != "receipt_fingerprint"}
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def _execution_receipt(
        self,
        plan: dict[str, object],
        dense_path: Path,
        candidate_path: Path,
    ) -> dict[str, object]:
        invocations = plan["invocations"]
        videos = plan["videos"]
        receipt: dict[str, object] = {
            "schema_version": 1,
            "record_type": "vbench_execution_receipt",
            "status": "COMPLETED",
            "formality": "FORMAL",
            "plan_fingerprint": plan["plan_fingerprint"],
            "quality_protocol_fingerprint": plan["quality_protocol_fingerprint"],
            "vbench_source": {
                "path": plan["vbench_source_path"],
                "ref": plan["vbench_source_ref"],
                "verification": "FORMAL",
            },
            "invocations": {
                role: {
                    "command_fingerprint": invocations[role]["command_fingerprint"],
                    "video_path": videos[role]["path"],
                    "video_sha256": videos[role]["sha256"],
                    "result_path": str(result_path),
                    "result_sha256": _sha256(result_path),
                }
                for role, result_path in (("dense", dense_path), ("candidate", candidate_path))
            },
        }
        receipt["receipt_fingerprint"] = self._receipt_fingerprint(receipt)
        return receipt

    def test_plan_is_deterministic_pinned_and_requests_only_pair_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            first = build_vbench_pair_plan(self.pair, quality_protocol=self.protocol, **fixture)
            second = build_vbench_pair_plan(self.pair, quality_protocol=self.protocol, **fixture)

        self.assertEqual(first, second)
        self.assertEqual("NOT_RUN", first["execution_status"])
        self.assertEqual("NON_FORMAL_TEST_ONLY", first["formality"])
        self.assertFalse(first["performance_claim"])
        self.assertEqual(VBENCH_REF, first["vbench_source_ref"])
        self.assertEqual(self.pair["metrics"], first["metrics"])
        self.assertEqual(
            fixture["gpu_uuid"],
            first["invocations"]["dense"]["env"]["CUDA_VISIBLE_DEVICES"],
        )
        self.assertEqual({"dense", "candidate"}, set(first["invocations"]))
        for role, invocation in first["invocations"].items():
            with self.subTest(role=role):
                argv = invocation["argv"]
                dimension_index = argv.index("--dimension")
                mode_index = argv.index("--mode")
                self.assertEqual(
                    self.pair["metrics"], argv[dimension_index + 1 : mode_index]
                )
                self.assertEqual("custom_input", argv[mode_index + 1])
                self.assertEqual(
                    first["vbench_cache_path"],
                    invocation["env"]["VBENCH_CACHE_DIR"],
                )
                self.assertEqual(
                    str(Path(first["vbench_cache_path"]) / "torch_home"),
                    invocation["env"]["TORCH_HOME"],
                )
                self.assertEqual(
                    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                    invocation["env"]["PATH"],
                )
                self.assertEqual("1", invocation["env"]["PYTHONDONTWRITEBYTECODE"])
                self.assertIn(role, invocation["output_path"])

    def test_incomplete_pair_and_video_receipt_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            incomplete = dict(self.pair)
            incomplete.pop("source_sha256")
            with self.assertRaisesRegex(VBenchContractError, "incomplete"):
                build_vbench_pair_plan(incomplete, quality_protocol=self.protocol, **fixture)

            wrong_digest = dict(fixture)
            wrong_digest["dense_receipt"] = {
                **fixture["dense_receipt"],
                "sha256": "0" * 64,
            }
            with self.assertRaisesRegex(VBenchContractError, "SHA-256"):
                build_vbench_pair_plan(self.pair, quality_protocol=self.protocol, **wrong_digest)

            swapped_artifact = dict(fixture)
            swapped_artifact["candidate_receipt"] = {
                **fixture["candidate_receipt"],
                "artifact_id": self.pair["dense_artifact_id"],
            }
            with self.assertRaisesRegex(VBenchContractError, "artifact_id"):
                build_vbench_pair_plan(self.pair, quality_protocol=self.protocol, **swapped_artifact)

            Path(fixture["candidate_video_path"]).unlink()
            with self.assertRaisesRegex(VBenchContractError, "regular file"):
                build_vbench_pair_plan(self.pair, quality_protocol=self.protocol, **fixture)

    def test_formal_plan_rejects_a_non_git_vbench_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            fixture["source_verification"] = "FORMAL"
            with self.assertRaisesRegex(VBenchContractError, "Git checkout"):
                build_vbench_pair_plan(self.pair, quality_protocol=self.protocol, **fixture)

    def test_duplicate_or_empty_metrics_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            for metrics in ([], [self.pair["metrics"][0]] * 2):
                with self.subTest(metrics=metrics):
                    pair = {**self.pair, "metrics": metrics}
                    with self.assertRaisesRegex(VBenchContractError, "metrics"):
                        build_vbench_pair_plan(pair, quality_protocol=self.protocol, **fixture)

    def test_pair_must_exactly_match_the_canonical_frozen_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            for field, replacement in (
                ("prompt", "tampered prompt"),
                ("source_path", "prompts/other.txt"),
                ("source_sha256", "0" * 64),
                ("selection_sha256", "1" * 64),
                ("selected_line_number_one_based", 999),
            ):
                with self.subTest(field=field):
                    with self.assertRaisesRegex(VBenchContractError, "canonical frozen protocol"):
                        build_vbench_pair_plan(
                            {**self.pair, field: replacement},
                            quality_protocol=self.protocol,
                            **fixture,
                        )

    def test_official_eval_results_shape_emits_validator_score_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            fixture["source_verification"] = "FORMAL"
            with patch("rolloutbench.vbench_runner._validate_source_checkout"):
                plan = build_vbench_pair_plan(
                    self.pair, quality_protocol=self.protocol, **fixture
                )
            dense_path = root / "dense_eval_results.json"
            candidate_path = root / "candidate_eval_results.json"
            dense = {
                metric: [0.9 + index / 100, [{"video_results": 0.9}]]
                for index, metric in enumerate(self.pair["metrics"])
            }
            candidate = {
                metric: [0.8 + index / 100, [{"video_results": 0.8}]]
                for index, metric in enumerate(self.pair["metrics"])
            }
            dense_path.write_text(json.dumps(dense), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            execution_receipt_path = root / "execution-receipt.json"
            execution_receipt_path.write_text(
                json.dumps(self._execution_receipt(plan, dense_path, candidate_path)),
                encoding="utf-8",
            )

            result = parse_vbench_pair_results(
                self.pair,
                quality_protocol=self.protocol,
                plan=plan,
                execution_receipt_path=execution_receipt_path,
            )

        self.assertEqual("PARSED", result["status"])
        self.assertFalse(result["performance_claim"])
        self.assertEqual(
            [
                {
                    "pair_id": self.pair["pair_id"],
                    "metric": metric,
                    "dense_score": dense[metric][0],
                    "candidate_score": candidate[metric][0],
                }
                for metric in self.pair["metrics"]
            ],
            result["score_rows"],
        )

    def test_result_parser_rejects_missing_extra_nan_bool_and_wrong_shape(
        self,
    ) -> None:
        valid = {
            metric: [0.9, [{"video_results": 0.9}]]
            for metric in self.pair["metrics"]
        }
        mutations = {
            "missing": {
                key: value
                for key, value in valid.items()
                if key != self.pair["metrics"][0]
            },
            "extra": {**valid, "dynamic_degree": [0.9, []]},
            "non-finite": {**valid, self.pair["metrics"][0]: [math.nan, []]},
            "boolean": {**valid, self.pair["metrics"][0]: [True, []]},
            "official.*shape": {**valid, self.pair["metrics"][0]: {"overall": 0.9}},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            fixture["source_verification"] = "FORMAL"
            with patch("rolloutbench.vbench_runner._validate_source_checkout"):
                plan = build_vbench_pair_plan(
                    self.pair, quality_protocol=self.protocol, **fixture
                )
            dense_path = root / "dense.json"
            candidate_path = root / "candidate.json"
            dense_path.write_text(json.dumps(valid), encoding="utf-8")
            for expected, payload in mutations.items():
                with self.subTest(expected=expected):
                    candidate_path.write_text(json.dumps(payload), encoding="utf-8")
                    receipt_path = root / "execution-receipt.json"
                    receipt_path.write_text(
                        json.dumps(self._execution_receipt(plan, dense_path, candidate_path)),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(VBenchContractError, expected):
                        parse_vbench_pair_results(
                            self.pair,
                            quality_protocol=self.protocol,
                            plan=plan,
                            execution_receipt_path=receipt_path,
                        )

    def test_parser_rejects_nonformal_plan_and_tampered_execution_receipt(self) -> None:
        valid = {metric: [0.9, [{"video_results": 0.9}]] for metric in self.pair["metrics"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            informal_plan = build_vbench_pair_plan(
                self.pair, quality_protocol=self.protocol, **fixture
            )
            dense_path = root / "dense.json"
            candidate_path = root / "candidate.json"
            dense_path.write_text(json.dumps(valid), encoding="utf-8")
            candidate_path.write_text(json.dumps(valid), encoding="utf-8")
            receipt_path = root / "execution-receipt.json"
            receipt_path.write_text(
                json.dumps(self._execution_receipt(informal_plan, dense_path, candidate_path)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(VBenchContractError, "formal"):
                parse_vbench_pair_results(
                    self.pair,
                    quality_protocol=self.protocol,
                    plan=informal_plan,
                    execution_receipt_path=receipt_path,
                )

            fixture["source_verification"] = "FORMAL"
            with patch("rolloutbench.vbench_runner._validate_source_checkout"):
                formal_plan = build_vbench_pair_plan(
                    self.pair, quality_protocol=self.protocol, **fixture
                )
            receipt = self._execution_receipt(formal_plan, dense_path, candidate_path)
            receipt["invocations"]["candidate"]["video_sha256"] = "0" * 64
            receipt["receipt_fingerprint"] = self._receipt_fingerprint(receipt)
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(VBenchContractError, "video binding"):
                parse_vbench_pair_results(
                    self.pair,
                    quality_protocol=self.protocol,
                    plan=formal_plan,
                    execution_receipt_path=receipt_path,
                )


if __name__ == "__main__":
    unittest.main()
