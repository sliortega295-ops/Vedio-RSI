from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from rolloutbench.freeze import CACHE_REF, INTEGRATION_REF, KERNEL_REF, freeze_suite
from rolloutbench.quality_contract import K22_FAILURE_CONTRACT
from rolloutbench.schema import SuiteValidationError, validate_suite_directory


REPO_ROOT = Path(__file__).resolve().parents[2]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class FreezeSuiteTests(unittest.TestCase):
    def test_freeze_is_deterministic_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = Path(first_dir)
            second = Path(second_dir)
            freeze_suite(first, repo_root=REPO_ROOT)
            freeze_suite(second, repo_root=REPO_ROOT)

            expected_files = {
                "suite.json",
                "episodes.jsonl",
                "artifacts.json",
                "quality_protocol.json",
            }
            self.assertEqual(expected_files, {path.name for path in first.iterdir()})
            for name in expected_files:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

            report = validate_suite_directory(first)
            self.assertEqual(35, report.total_episodes)
            self.assertEqual({"validated": 29, "not_run": 4, "failed": 2}, report.historical_status_counts)
            self.assertEqual({"kernel": 23, "cache": 12}, report.component_counts)

    def test_episodes_are_bound_to_authoritative_git_lines(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            output = Path(output_dir)
            freeze_suite(output, repo_root=REPO_ROOT)
            episodes = _jsonl(output / "episodes.jsonl")

            refs = {"kernel": KERNEL_REF, "cache": CACHE_REF}
            source_lines: dict[str, list[bytes]] = {}
            for component, ref in refs.items():
                raw = subprocess.check_output(
                    ["git", "show", f"{ref}:TRAJECTORY.jsonl"], cwd=REPO_ROOT
                )
                source_lines[component] = raw.splitlines()

            self.assertEqual([f"K{round_id:02d}" for round_id in range(1, 24)], [e["episode_id"] for e in episodes[:23]])
            self.assertEqual([f"C{round_id:02d}" for round_id in range(1, 13)], [e["episode_id"] for e in episodes[23:]])
            for episode in episodes:
                line = source_lines[episode["component"]][episode["round"] - 1]
                self.assertEqual(hashlib.sha256(line).hexdigest(), episode["source"]["line_sha256"])

    def test_suite_pins_authority_and_artifact_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            output = Path(output_dir)
            freeze_suite(output, repo_root=REPO_ROOT)
            suite = _json(output / "suite.json")
            artifacts = _json(output / "artifacts.json")

            self.assertEqual(KERNEL_REF, suite["authority"]["kernel_ref"])
            self.assertEqual(CACHE_REF, suite["authority"]["cache_ref"])
            self.assertEqual(INTEGRATION_REF, suite["authority"]["integration_ref"])
            self.assertEqual(
                {
                    "git_available",
                    "remote_only_verified",
                    "local_source_verified",
                    "missing",
                    "regenerate",
                },
                {item["availability"] for item in artifacts["artifacts"]},
            )
            self.assertFalse(artifacts["large_artifacts_local"])

            for name in ("episodes.jsonl", "artifacts.json", "quality_protocol.json"):
                expected = hashlib.sha256((output / name).read_bytes()).hexdigest()
                self.assertEqual(expected, suite["file_hashes"][name]["sha256"])
                self.assertEqual("raw_file_bytes", suite["file_hashes"][name]["hash_scope"])
            self.assertEqual(suite["file_hashes"]["episodes.jsonl"]["sha256"], suite["episodes_sha256"])
            self.assertEqual(suite["file_hashes"]["artifacts.json"]["sha256"], suite["artifacts_sha256"])
            self.assertEqual(
                suite["file_hashes"]["quality_protocol.json"]["sha256"],
                suite["quality_protocol_sha256"],
            )

            self.assertEqual("db5f398b13ca086d09a50ce156c20527773841b1", suite["model"]["revision"])
            self.assertEqual("d2c6407cc9b9133f3fff49fe4b561f14980d3f8b", suite["authority"]["historical_harness_ref"])
            self.assertEqual(100, suite["workload"]["logical_dit_calls"])
            self.assertEqual("2.11.0+cu128", suite["environment"]["torch"])

    def test_quality_protocol_has_predeclared_candidates_and_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            output = Path(output_dir)
            freeze_suite(output, repo_root=REPO_ROOT)
            protocol = _json(output / "quality_protocol.json")

            self.assertEqual(
                ["C02", "C03", "C04", "C06", "C07", "C09", "C10", "C11", "C12"],
                protocol["formal_cache_candidates"],
            )
            self.assertEqual(
                {"C01": "provenance_failed", "C05": "calibration_only", "C08": "calibration_only"},
                protocol["excluded_cache_candidates"],
            )
            self.assertEqual([42, 12345], protocol["seeds"])
            self.assertEqual(8, protocol["matched_pairs_per_candidate"])
            self.assertEqual(0.005, protocol["acceptance"]["max_mean_relative_drop"])
            self.assertEqual(0.02, protocol["acceptance"]["max_single_dimension_drop"])
            self.assertEqual("secondary_ranking_only", protocol["lpips"]["role"])
            self.assertEqual(
                "rerun_for_each_candidate_pair",
                protocol["dense_measurement_reuse"]["vbench_scoring"],
            )
            selected = {
                row["suite"]: (row["selected_line_number_one_based"], row["prompt"], row["selection_sha256"])
                for row in protocol["prompt_selection"]["prompt_suites"]
            }
            self.assertEqual(
                (3, "a person washing the dishes", "0388d9179df4da12015f44777e6c56016d42bb83915d5e099e240703e0a1ab3f"),
                selected["subject_consistency"],
            )
            self.assertEqual(
                (13, "bedroom", "024d5d57f16e1c00291a5477cf75798fb9215703a64a1650b381a9b24a0725f9"),
                selected["scene"],
            )
            self.assertEqual(
                (2, "a toilet, frozen in time", "04175c2dacc00b9a608873b11c815adb59ec4ed5e44b7238c08879536564e2d5"),
                selected["temporal_flickering"],
            )
            self.assertEqual(
                (
                    73,
                    "A cute fluffy panda eating Chinese food in a restaurant",
                    "02369a8b9987de43da8a8a246f421a20211baea835a61e9d9cfc77c4d8c928a2",
                ),
                selected["overall_consistency"],
            )

    def test_validator_rejects_duplicate_episode_ids(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            output = Path(output_dir)
            freeze_suite(output, repo_root=REPO_ROOT)
            episodes = _jsonl(output / "episodes.jsonl")
            episodes[-1]["episode_id"] = episodes[0]["episode_id"]
            (output / "episodes.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in episodes),
                encoding="utf-8",
            )
            suite = _json(output / "suite.json")
            updated_sha = hashlib.sha256((output / "episodes.jsonl").read_bytes()).hexdigest()
            suite["file_hashes"]["episodes.jsonl"]["sha256"] = updated_sha
            suite["episodes_sha256"] = updated_sha
            (output / "suite.json").write_text(
                json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(SuiteValidationError, "duplicate episode_id"):
                validate_suite_directory(output)

    def test_validator_rejects_mutated_runtime_and_frontier_contracts(self) -> None:
        mutators = {
            "workload contract": lambda suite: suite["workload"].update(denoising_steps=49),
            "frontier contract": lambda suite: suite["frontier_contracts"]["legacy_oracle"].update(cache="C11"),
        }
        for expected_error, mutate in mutators.items():
            with self.subTest(expected_error=expected_error), tempfile.TemporaryDirectory() as output_dir:
                output = Path(output_dir)
                freeze_suite(output, repo_root=REPO_ROOT)
                suite = _json(output / "suite.json")
                mutate(suite)
                (output / "suite.json").write_text(
                    json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(SuiteValidationError, expected_error):
                    validate_suite_directory(output, repo_root=REPO_ROOT)

    def test_fifo_dependencies_validation_and_git_blob_closure(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            output = Path(output_dir)
            freeze_suite(output, repo_root=REPO_ROOT)
            episodes = _jsonl(output / "episodes.jsonl")
            by_id = {row["episode_id"]: row for row in episodes}

            expected_fifo = []
            for round_number in range(1, 13):
                expected_fifo.extend([f"K{round_number:02d}", f"C{round_number:02d}"])
            expected_fifo.extend(f"K{round_number:02d}" for round_number in range(13, 24))
            actual_fifo = [
                row["episode_id"] for row in sorted(episodes, key=lambda row: row["global_fifo_index"])
            ]
            self.assertEqual(expected_fifo, actual_fifo)
            self.assertEqual([], by_id["K01"]["depends_on"])
            self.assertEqual(["K01"], by_id["K02"]["depends_on"])
            self.assertEqual(["C01"], by_id["C02"]["depends_on"])
            self.assertEqual(
                [{"artifact": "torch_compile_cache", "episode_id": "K01", "required": True}],
                by_id["K02"]["reuse"]["inputs"],
            )

            for episode_id in ("K15", "K18", "K21", "K23"):
                episode = by_id[episode_id]
                self.assertEqual(1, episode["resources"]["gpu_count"])
                self.assertEqual("after_decide", episode["validation"]["earliest_legal_exit"])
                self.assertIn("microbenchmark", episode["validation"]["stages"])
                probe = episode["candidate"]["probe"]
                for item in (probe["source"], probe["result"]):
                    self.assertEqual(item["authority_reported_sha256"], item["blob_sha256"])

            for episode in episodes:
                config = episode["candidate"]["config"]
                if config is None:
                    continue
                self.assertIsNotNone(config["blob_sha256"])
                if episode["episode_id"] == "C01":
                    self.assertIsNone(config["authority_reported_sha256"])
                else:
                    self.assertEqual(config["authority_reported_sha256"], config["blob_sha256"])
                self.assertFalse(episode["golden"]["scheduler_visible"])

            self.assertEqual(
                dict(K22_FAILURE_CONTRACT),
                by_id["K22"]["replay"]["failure_contract"],
            )

            for episode_id in [f"C{number:02d}" for number in range(1, 13)]:
                stages = by_id[episode_id]["validation"]["stages"]
                for required in ("generate", "collect", "decide"):
                    self.assertIn(required, stages)
                if by_id[episode_id]["quality_eligibility"] == "formal":
                    self.assertIn("quality_v1", stages)
                else:
                    self.assertIn("legacy_sanity", stages)
                    self.assertNotIn("quality_v1", stages)

    def test_validator_fails_closed_on_hash_dag_stage_and_probe_mutations(self) -> None:
        mutators = {
            "file hash mismatch": lambda rows, quality: quality.update({"status": "tampered"}),
            "global_fifo_index": lambda rows, quality: rows[-1].update(global_fifo_index=0),
            "depends_on": lambda rows, quality: rows[1].update(depends_on=[]),
            "Cache validation": lambda rows, quality: rows[23]["validation"].update(stages=["preflight", "decide"]),
            "config Git hash": lambda rows, quality: rows[0]["candidate"]["config"].update(blob_sha256="0" * 64),
            "probe source Git hash": lambda rows, quality: rows[14]["candidate"]["probe"]["source"].update(blob_sha256="0" * 64),
            "quality candidate partition": lambda rows, quality: quality["formal_cache_candidates"].remove("C12"),
            "4 prompts x 2 seeds": lambda rows, quality: quality["prompt_selection"]["prompt_suites"].pop(),
            "quality acceptance contract": lambda rows, quality: quality["acceptance"].update(max_mean_relative_drop=0.05),
            "prompt selection digest": lambda rows, quality: quality["prompt_selection"]["prompt_suites"][0].update(selection_sha256="0" * 64),
        }
        for expected_error, mutate in mutators.items():
            with self.subTest(expected_error=expected_error), tempfile.TemporaryDirectory() as output_dir:
                output = Path(output_dir)
                freeze_suite(output, repo_root=REPO_ROOT)
                rows = _jsonl(output / "episodes.jsonl")
                quality = _json(output / "quality_protocol.json")
                mutate(rows, quality)
                (output / "episodes.jsonl").write_bytes(
                    b"".join(
                        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
                    )
                )
                (output / "quality_protocol.json").write_text(
                    json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                if expected_error != "file hash mismatch":
                    suite = _json(output / "suite.json")
                    updated_sha = hashlib.sha256((output / "episodes.jsonl").read_bytes()).hexdigest()
                    suite["file_hashes"]["episodes.jsonl"]["sha256"] = updated_sha
                    suite["episodes_sha256"] = updated_sha
                    quality_sha = hashlib.sha256((output / "quality_protocol.json").read_bytes()).hexdigest()
                    suite["file_hashes"]["quality_protocol.json"]["sha256"] = quality_sha
                    suite["quality_protocol_sha256"] = quality_sha
                    (output / "suite.json").write_text(
                        json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                    )
                with self.assertRaisesRegex(SuiteValidationError, expected_error):
                    validate_suite_directory(output, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
