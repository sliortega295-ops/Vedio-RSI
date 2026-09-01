from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from rolloutbench.cli import main
from rolloutbench.validators import (
    HistoricalOracleReplay,
    build_quality_plan,
    compare_historical_oracle,
    evaluate_quality_candidate,
    select_quality_frontier,
    validate_cache_receipt,
    validate_kernel_structure,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_DIR = REPO_ROOT / "benchmarks" / "sana_video_2b_h100_v0"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _episodes() -> list[dict]:
    return [json.loads(line) for line in (SUITE_DIR / "episodes.jsonl").read_text().splitlines()]


def _suite() -> dict:
    return _json(SUITE_DIR / "suite.json")


def _score_rows(plan: list[dict], drops: dict[str, float] | None = None) -> list[dict]:
    drops = drops or {}
    rows = []
    for pair in plan:
        for metric in pair["metrics"]:
            dense = 1.0
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "metric": metric,
                    "dense_score": dense,
                    "candidate_score": dense * (1.0 - drops.get(metric, 0.0)),
                }
            )
    return rows


class QualityValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = _json(SUITE_DIR / "quality_protocol.json")
        self.formal = self.protocol["formal_cache_candidates"]

    def test_plan_has_exactly_nine_candidates_times_eight_pairs(self) -> None:
        plan = build_quality_plan(self.protocol, self.formal)
        self.assertEqual(72, len(plan))
        self.assertEqual(8, len([row for row in plan if row["candidate_id"] == "C02"]))
        self.assertEqual(
            {42, 12345},
            {row["seed"] for row in plan},
        )
        first = plan[0]
        self.assertEqual("a person washing the dishes", first["prompt"])
        self.assertEqual(
            "0388d9179df4da12015f44777e6c56016d42bb83915d5e099e240703e0a1ab3f",
            first["selection_sha256"],
        )
        self.assertIn("dense_artifact_id", first)
        self.assertIn("candidate_artifact_id", first)
        with self.assertRaisesRegex(ValueError, "formal"):
            build_quality_plan(self.protocol, ["C01"])

    def test_acceptance_boundaries_are_inclusive(self) -> None:
        plan = build_quality_plan(self.protocol, ["C02"])
        dimensions = self.protocol["dimensions"]
        drops = {dimension: 0.0025 for dimension in dimensions}
        drops[dimensions[0]] = 0.02
        result = evaluate_quality_candidate(self.protocol, plan, _score_rows(plan, drops))
        self.assertEqual("PASS", result["status"])
        self.assertAlmostEqual(0.005, result["mean_relative_drop"])
        self.assertAlmostEqual(0.02, result["max_relative_drop"])

    def test_missing_nan_and_zero_dense_fail_closed(self) -> None:
        plan = build_quality_plan(self.protocol, ["C02"])
        valid = _score_rows(plan)
        mutations = {
            "missing": valid[:-1],
            "nan": [*valid[:-1], {**valid[-1], "candidate_score": math.nan}],
            "zero": [{**valid[0], "dense_score": 0.0}, *valid[1:]],
        }
        for expected, scores in mutations.items():
            with self.subTest(expected=expected):
                result = evaluate_quality_candidate(self.protocol, plan, scores)
                self.assertEqual("FAIL_CLOSED", result["status"])
                self.assertTrue(any(expected in item for item in result["errors"]))

    def test_lpips_is_secondary_and_missing_does_not_change_hard_pass(self) -> None:
        plan = build_quality_plan(self.protocol, ["C02"])
        scores = _score_rows(plan)
        missing = evaluate_quality_candidate(self.protocol, plan, scores)
        self.assertEqual("PASS", missing["status"])
        self.assertEqual("MISSING", missing["lpips"]["status"])
        complete_lpips = {pair["pair_id"]: [0.1] * 81 for pair in plan}
        complete = evaluate_quality_candidate(
            self.protocol, plan, scores, lpips_frame_values=complete_lpips
        )
        self.assertEqual("PASS", complete["status"])
        self.assertEqual("COMPLETE", complete["lpips"]["status"])
        self.assertEqual(648, complete["lpips"]["value_count"])
        invalid_lpips = {pair["pair_id"]: [0.1] * 81 for pair in plan}
        invalid_lpips[plan[0]["pair_id"]][0] = math.nan
        invalid = evaluate_quality_candidate(
            self.protocol, plan, scores, lpips_frame_values=invalid_lpips
        )
        self.assertEqual("PASS", invalid["status"])
        self.assertEqual("INVALID", invalid["lpips"]["status"])

    def test_quality_winner_uses_new_median_latency_and_is_not_hardcoded(self) -> None:
        passing = [
            {
                "candidate_id": candidate,
                "eligibility": "formal",
                "status": "PASS" if candidate in {"C02", "C12"} else "FAIL",
                "pass": candidate in {"C02", "C12"},
                "protocol_id": self.protocol["protocol_id"],
            }
            for candidate in self.formal
        ]
        selected = select_quality_frontier(
            self.protocol,
            passing,
            {"C02": [10.0, 11.0, 12.0], "C12": [20.0, 21.0, 22.0]},
        )
        self.assertEqual("C02", selected["winner"])
        self.assertEqual("SELECTED", selected["status"])
        not_run = select_quality_frontier(self.protocol, [], {})
        self.assertIsNone(not_run["winner"])
        self.assertEqual("NOT_RUN", not_run["status"])

    def test_quality_frontier_fails_closed_on_partial_or_duplicate_results(self) -> None:
        result = {
            "candidate_id": "C02",
            "eligibility": "formal",
            "status": "PASS",
            "pass": True,
            "protocol_id": self.protocol["protocol_id"],
        }
        partial = select_quality_frontier(self.protocol, [result], {"C02": [10.0]})
        self.assertEqual("FAIL_CLOSED", partial["status"])
        self.assertIn("missing formal results", " ".join(partial["errors"]))

        duplicated = select_quality_frontier(
            self.protocol,
            [result, copy.deepcopy(result)],
            {"C02": [10.0]},
        )
        self.assertEqual("FAIL_CLOSED", duplicated["status"])
        self.assertIn("duplicate", " ".join(duplicated["errors"]))

    def test_quality_frontier_requires_three_latencies_for_every_passing_candidate(self) -> None:
        results = [
            {
                "candidate_id": candidate,
                "eligibility": "formal",
                "status": "PASS" if candidate in {"C02", "C12"} else "FAIL",
                "pass": candidate in {"C02", "C12"},
                "protocol_id": self.protocol["protocol_id"],
            }
            for candidate in self.formal
        ]
        incomplete = select_quality_frontier(
            self.protocol,
            results,
            {"C02": [10.0, 11.0, 12.0], "C12": [20.0, 21.0]},
        )
        self.assertEqual("FAIL_CLOSED", incomplete["status"])
        self.assertIn("C12", " ".join(incomplete["errors"]))

    def test_typed_structural_receipts_fail_closed(self) -> None:
        exact = validate_kernel_structure(
            _suite()["workload"],
            {
                "denoising_steps": 50,
                "cfg_branches": 2,
                "logical_dit_calls": 100,
                "transformer_blocks": 20,
                "skipped_operations": 0,
            }
        )
        self.assertTrue(exact["pass"])
        self.assertFalse(
            validate_kernel_structure(
                _suite()["workload"],
                {**exact["observed"], "logical_dit_calls": 99},
            )[
                "pass"
            ]
        )

        formal = next(row for row in _episodes() if row["episode_id"] == "C02")
        receipt = {
            "episode_id": "C02",
            "completed_stages": ["generate", "collect", "quality_v1", "decide"],
            "stage_receipts": {
                stage: {"output_sha256": "1" * 64}
                for stage in ("generate", "collect", "quality_v1", "decide")
            },
        }
        self.assertTrue(validate_cache_receipt(formal, receipt)["pass"])
        incomplete = copy.deepcopy(receipt)
        incomplete["stage_receipts"].pop("quality_v1")
        self.assertFalse(validate_cache_receipt(formal, incomplete)["pass"])


class HistoricalAcceptanceTests(unittest.TestCase):
    def test_historical_replay_accepts_all_35_and_k20_c12(self) -> None:
        episodes = _episodes()
        frontier = _suite()["frontier_contracts"]["legacy_oracle"]
        replay = HistoricalOracleReplay().replay(episodes, frontier)
        self.assertTrue(replay["synthetic_historical_oracle_replay"])
        self.assertFalse(replay["performance_claim"])
        report = compare_historical_oracle(
            episodes, replay["decisions"], replay["frontier"], frontier
        )
        self.assertEqual("PASS", report["status"])
        self.assertEqual(35, report["episode_agreement"]["matched"])
        self.assertEqual({"kernel": "K20", "cache": "C12"}, replay["frontier"])

    def test_historical_acceptance_detects_decision_and_frontier_tamper(self) -> None:
        episodes = _episodes()
        frontier = _suite()["frontier_contracts"]["legacy_oracle"]
        replay = HistoricalOracleReplay().replay(episodes, frontier)
        replay["decisions"][0]["outcome"] = "tampered"
        replay["frontier"]["cache"] = "C11"
        report = compare_historical_oracle(
            episodes, replay["decisions"], replay["frontier"], frontier
        )
        self.assertEqual("FAIL", report["status"])
        self.assertFalse(report["episode_agreement"]["agrees"])
        self.assertFalse(report["frontier_agreement"]["agrees"])

    def test_cli_quality_plan_and_acceptance_check_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "quality-1.json"
            second = root / "quality-2.json"
            args = ["quality-plan", "--suite", str(SUITE_DIR)]
            self.assertEqual(0, main(args + ["--output", str(first)]))
            self.assertEqual(0, main(args + ["--output", str(second)]))
            self.assertEqual(first.read_bytes(), second.read_bytes())
            plan_payload = _json(first)
            formal = _json(SUITE_DIR / "quality_protocol.json")["formal_cache_candidates"]
            self.assertEqual(72, plan_payload["matched_pair_count"])
            self.assertEqual(9, plan_payload["candidate_count"])
            self.assertEqual(formal, plan_payload["candidate_ids"])
            self.assertEqual(64, len(plan_payload["quality_protocol_sha256"]))
            self.assertEqual(64, len(plan_payload["pairs_sha256"]))
            self.assertEqual("NOT_RUN", plan_payload["execution_status"])

            simulation = root / "oracle.json"
            simulation.write_text(
                json.dumps(
                    HistoricalOracleReplay().replay(
                        _episodes(), _suite()["frontier_contracts"]["legacy_oracle"]
                    )
                ),
                encoding="utf-8",
            )
            acceptance = root / "acceptance.json"
            self.assertEqual(
                0,
                main(
                    [
                        "acceptance-check",
                        "--suite",
                        str(SUITE_DIR),
                        "--simulation",
                        str(simulation),
                        "--output",
                        str(acceptance),
                    ]
                ),
            )
            self.assertEqual("PASS", _json(acceptance)["status"])


if __name__ == "__main__":
    unittest.main()
