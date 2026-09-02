from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rolloutbench.decisions import DecisionError, finalize_run_decisions
from rolloutbench.events import EventLedger
from rolloutbench.pilot_runner import RunContext
from rolloutbench.quality_contract import K22_FAILURE_CONTRACT


class FreshDecisionTests(unittest.TestCase):
    def _context(self, root: Path) -> RunContext:
        return RunContext(
            plan_id="plan",
            plan_sha256="a" * 64,
            run_sha256="b" * 64,
            preparation_sha256="c" * 64,
            run={
                "run_id": "pilot-serial1-repeat-01",
                "episodes": [
                    {
                        "episode_id": "K20",
                        "component": "kernel",
                        "quality_eligibility": "not_applicable_lossless",
                    },
                    {
                        "episode_id": "C01",
                        "component": "cache",
                        "quality_eligibility": "provenance_failed",
                    },
                ],
            },
            preparation={"experiment_root": str(root / "experiment")},
            plan_path=root / "plan.json",
            preparation_path=root / "preparation.json",
        )

    def test_all_planned_episodes_receive_fresh_non_oracle_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = self._context(root)
            ledger = EventLedger(root / "events.jsonl")
            evidence = {
                "K20": {
                    "execution_status": "VALIDATED",
                    "evidence_kind": "generation",
                    "ranking_eligible": True,
                    "generation_s": 40.5,
                    "process_wall_s": 80.0,
                    "ranking_latency_s": 80.0,
                    "ranking_latency_contract": "one_shot",
                    "receipt_path": "/evidence/K20.json",
                    "receipt_sha256": "1" * 64,
                },
                "C01": {
                    "execution_status": "VALIDATED",
                    "evidence_kind": "generation",
                    "ranking_eligible": True,
                    "generation_s": 50.0,
                    "process_wall_s": 90.0,
                    "ranking_latency_s": 90.0,
                    "ranking_latency_contract": "one_shot",
                    "receipt_path": "/evidence/C01.json",
                    "receipt_sha256": "2" * 64,
                },
            }
            suite = {
                "frontier_contracts": {
                    "legacy_oracle": {"kernel": "K20", "cache": "C12"}
                }
            }
            result = finalize_run_decisions(
                context, ledger, suite, evidence
            )
            replay = finalize_run_decisions(context, ledger, suite, evidence)

        self.assertEqual(2, result["decision_count"])
        self.assertEqual("exact_validated", result["decisions"]["K20"]["outcome"])
        self.assertEqual(
            "excluded_provenance_failed",
            result["decisions"]["C01"]["outcome"],
        )
        self.assertEqual("K20", result["frontier"]["frontier"]["kernel"])
        self.assertEqual(
            "fresh_single_run_ranking_latency_s_one_shot_process_wall",
            result["frontier"]["selection_basis"],
        )
        self.assertEqual(result["frontier_event_id"], replay["frontier_event_id"])

    def test_preexisting_non_formal_decision_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = self._context(root)
            ledger = EventLedger(root / "events.jsonl")
            ledger.seal_decision("K20", {"outcome": "historical_retain"})
            evidence = {
                episode_id: {
                    "execution_status": "VALIDATED",
                    "evidence_kind": "generation",
                    "ranking_eligible": True,
                    "generation_s": 1.0,
                    "process_wall_s": 2.0,
                    "ranking_latency_s": 2.0,
                    "ranking_latency_contract": "one_shot",
                    "receipt_path": f"/evidence/{episode_id}.json",
                    "receipt_sha256": digit * 64,
                }
                for episode_id, digit in (("K20", "1"), ("C01", "2"))
            }
            with self.assertRaisesRegex(DecisionError, "pre-existing"):
                finalize_run_decisions(
                    context,
                    ledger,
                    {"frontier_contracts": {"legacy_oracle": {}}},
                    evidence,
                )

    def test_k22_decision_reuses_shared_validator_with_frozen_runtime_source(self):
        from rolloutbench.decisions import _expected_failure_evidence

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "benchmark.json"
            output.write_text("{}")
            worktree = root / "worktree"
            expected_source = {
                "harness_archival_parent": "d" * 40,
                "runtime_authority_sha": "a" * 40,
                "runtime_compat_sha": "c" * 40,
                "runtime_root": str(worktree / "external" / "sol_runtime"),
                "required_runtime_paths": ["python/runtime.py"],
                "critical_file_sha256": {"python/runtime.py": "f" * 64},
            }
            context = RunContext(
                **{
                    **self._context(root).__dict__,
                    "preparation": {
                        "experiment_root": str(root / "experiment"),
                        "runtime_receipts": {
                            "K22": {
                                "worktree_path": str(worktree),
                                "required_runtime_paths": expected_source[
                                    "required_runtime_paths"
                                ],
                                "critical_runtime_file_sha256": expected_source[
                                    "critical_file_sha256"
                                ],
                            }
                        },
                    },
                }
            )
            suite = {
                "authority": {
                    "historical_harness_ref": expected_source[
                        "harness_archival_parent"
                    ]
                },
                "model": {
                    "runtime_authority_ref": expected_source[
                        "runtime_authority_sha"
                    ],
                    "runtime_compat_ref": expected_source["runtime_compat_sha"],
                },
            }
            with patch(
                "rolloutbench.decisions.validate_k22_failure_artifacts",
                return_value={
                    "benchmark": {"process_wall_s": 2.0},
                    "child_returncode": 4,
                },
            ) as validator:
                result = _expected_failure_evidence(
                    context,
                    {
                        "episode_id": "K22",
                        "expected_failure_contract": dict(K22_FAILURE_CONTRACT),
                    },
                    {"output_path": str(output)},
                    suite,
                )
            validator.assert_called_once_with(
                output, expected_source=expected_source
            )
            self.assertEqual("EXPECTED_FAILURE_VALIDATED", result["execution_status"])


if __name__ == "__main__":
    unittest.main()
