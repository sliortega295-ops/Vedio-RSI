from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rolloutbench.acceptance import (
    EvidenceVerificationError,
    run_cpu_acceptance,
    verify_cpu_acceptance_pack,
)
from rolloutbench.cli import main


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_DIR = REPO_ROOT / "benchmarks" / "sana_video_2b_h100_v0"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class CpuAcceptanceTests(unittest.TestCase):
    def test_pack_records_four_system_oracle_and_ledger_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "acceptance"
            manifest = run_cpu_acceptance(
                SUITE_DIR, output, repo_root=REPO_ROOT, require_clean=False
            )

            self.assertTrue(manifest["status"].startswith("PASS"))
            self.assertEqual("PASS", manifest["cpu_contract_status"])
            self.assertFalse(manifest["performance_claim"])
            self.assertEqual("NOT_RUN", manifest["gpu_execution_status"])
            self.assertEqual("NOT_RUN", manifest["quality_execution_status"])
            self.assertEqual(40, len(manifest["source_revision"]))
            self.assertEqual(
                {"suite.json", "episodes.jsonl", "artifacts.json", "quality_protocol.json"},
                set(manifest["suite_file_sha256"]),
            )
            self.assertEqual(
                {"serial1": 1, "fifo2": 2, "optroll1": 1, "optroll2": 2},
                manifest["simulated_gpu_slot_concurrency_contract"],
            )
            for system in ("serial1", "fifo2", "optroll1", "optroll2"):
                simulation = _json(output / "simulations" / f"{system}.json")
                self.assertEqual(35, simulation["summary"]["sealed_decisions"])
                self.assertFalse(simulation["summary"]["performance_claim"])

            self.assertEqual(
                "PASS", _json(output / "historical" / "oracle-acceptance.json")["status"]
            )
            k20 = _json(output / "recovery" / "K20-generation-interrupt.json")
            self.assertEqual("PASS", k20["status"])
            self.assertEqual("retry", k20["resume_planner_action"])
            self.assertEqual(1, k20["physical_decision_records"])
            self.assertEqual("interrupted", k20["attempt_1_state"])
            self.assertEqual("completed", k20["attempt_2_state"])
            c12 = _json(output / "recovery" / "C12-before-decision-seal.json")
            self.assertEqual("PASS", c12["status"])
            self.assertEqual("reuse_completed", c12["resume_planner_action"])
            self.assertTrue(c12["completed_stage_preserved_after_reconstruct"])
            self.assertEqual(1, c12["physical_decision_records"])

            disk_manifest = _json(output / "MANIFEST.json")
            self.assertEqual(manifest, disk_manifest)
            for relative_path, expected_sha in disk_manifest[
                "artifact_sha256_by_path"
            ].items():
                actual = hashlib.sha256((output / relative_path).read_bytes()).hexdigest()
                self.assertEqual(expected_sha, actual)
            verification = verify_cpu_acceptance_pack(
                output, SUITE_DIR, repo_root=REPO_ROOT, allow_dirty=True
            )
            self.assertEqual("PASS", verification["status"])

    def test_output_directory_is_no_overwrite_and_cli_matches_api(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "acceptance"
            self.assertEqual(
                0,
                main(
                    [
                        "cpu-acceptance",
                        "--suite",
                        str(SUITE_DIR),
                        "--output-dir",
                        str(output),
                        "--repo-root",
                        str(REPO_ROOT),
                        "--allow-dirty",
                    ]
                ),
            )
            self.assertTrue(_json(output / "MANIFEST.json")["status"].startswith("PASS"))
            with self.assertRaises(FileExistsError):
                run_cpu_acceptance(
                    SUITE_DIR, output, repo_root=REPO_ROOT, require_clean=False
                )

    def test_verifier_rejects_tamper_and_unexpected_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tampered = root / "tampered"
            run_cpu_acceptance(
                SUITE_DIR, tampered, repo_root=REPO_ROOT, require_clean=False
            )
            target = tampered / "simulations" / "serial1.json"
            target.write_bytes(target.read_bytes() + b" ")
            with self.assertRaises(EvidenceVerificationError):
                verify_cpu_acceptance_pack(
                    tampered, SUITE_DIR, repo_root=REPO_ROOT, allow_dirty=True
                )

            extra = root / "extra"
            run_cpu_acceptance(
                SUITE_DIR, extra, repo_root=REPO_ROOT, require_clean=False
            )
            (extra / "unexpected.txt").write_text("unexpected", encoding="utf-8")
            with self.assertRaises(EvidenceVerificationError):
                verify_cpu_acceptance_pack(
                    extra, SUITE_DIR, repo_root=REPO_ROOT, allow_dirty=True
                )


if __name__ == "__main__":
    unittest.main()
