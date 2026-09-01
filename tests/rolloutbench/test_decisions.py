from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rolloutbench.decisions import DecisionError, finalize_run_decisions
from rolloutbench.events import EventLedger
from rolloutbench.pilot_runner import RunContext


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


if __name__ == "__main__":
    unittest.main()
