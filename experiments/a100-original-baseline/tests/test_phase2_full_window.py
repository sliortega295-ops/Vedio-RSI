"""CPU-only synthetic checks for interval-weighted Phase 2 GPU accounting."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_phase2_full_window.py"
SPEC = importlib.util.spec_from_file_location("phase2_full_window_test", SCRIPT)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)
UUIDS = [f"GPU-test-{index}" for index in range(8)]


def sample(second: int, *, util: float = 0.0, app: bool = False) -> dict:
    return {"monotonic_ns": second * analysis.NS,
            "sample_overhead_s": 0.01,
            "gpus": [{"uuid": uuid, "gpu_util_pct": util if index == 0 else 0,
                      "memory_used_mib": 1000 if app and index == 0 else 0}
                     for index, uuid in enumerate(UUIDS)],
            "gpu_apps": ([{"gpu_uuid": UUIDS[0], "pid": 101}] if app else [])}


class Phase2FullWindowTest(unittest.TestCase):
    def budget(self) -> dict:
        return {"first_gpu_monotonic_ns": 0, "limit_s": 10,
                "first_gpu_utc": "2026-09-14T00:00:00+00:00"}

    def events(self, *, extra_same_gpu: bool = False) -> list[dict]:
        events = [{"event": "gpu_job_start", "run_dir": "/project/r1", "prompt_index": 1,
                   "pid": 101, "gpu_uuid": UUIDS[0], "monotonic_ns": 2 * analysis.NS},
                  {"event": "gpu_job_end", "run_dir": "/project/r1", "prompt_index": 1,
                   "pid": 101, "gpu_uuid": UUIDS[0], "monotonic_ns": 7 * analysis.NS}]
        if extra_same_gpu:
            events += [{"event": "gpu_job_start", "run_dir": "/project/r2", "prompt_index": 1,
                        "pid": 102, "gpu_uuid": UUIDS[0], "monotonic_ns": 4 * analysis.NS},
                       {"event": "gpu_job_end", "run_dir": "/project/r2", "prompt_index": 1,
                        "pid": 102, "gpu_uuid": UUIDS[0], "monotonic_ns": 6 * analysis.NS}]
        return events

    def samples(self) -> list[dict]:
        # Missing t=4 makes the [3,5) interval unknown, not two fabricated
        # 1-second observations. The other covered intervals are exactly 8 s.
        return [sample(t, util=100 if 2 <= t < 7 else 0, app=2 <= t < 7)
                for t in (0, 1, 2, 3, 5, 6, 7, 8, 9, 10)]

    def test_gap_is_unobserved_not_imputed(self) -> None:
        result = analysis.analyze(self.budget(), {"gpu_uuids": UUIDS},
                                  self.events(), self.samples(), cap_at_budget=True)
        self.assertEqual(result["wall_s"], 10)
        self.assertEqual(result["eight_gpu_capacity_s"], 80)
        self.assertEqual(result["allocated_distinct_uuid_gpu_s"], 5)
        self.assertEqual(result["seconds_by_concurrent_distinct_allocated_uuids"],
                         {"0": 5, "1": 5})
        self.assertEqual(result["observer"]["covered_s"], 8)
        self.assertEqual(result["observer"]["unobserved_s"], 2)
        self.assertEqual(result["observer"]["long_gap_count"], 1)
        self.assertEqual(result["unobserved_gpu_capacity_s"], 16)
        self.assertEqual(result["interval_hold_util_equivalent_gpu_s_lower_bound_estimate"], 3)
        self.assertEqual(result["interval_hold_util_equivalent_gpu_s_upper_bound_estimate"], 19)
        self.assertEqual(result["per_gpu_uuid"][UUIDS[0]]["util_equivalent_s_estimate"], 3)
        self.assertEqual(result["per_gpu_uuid"][UUIDS[0]]["app_present_s_estimate"], 3)
        self.assertEqual(result["per_gpu_uuid"][UUIDS[0]]["allocated_util_equivalent_s_estimate"], 3)
        self.assertTrue(all(result["per_gpu_uuid"][uuid]["util_equivalent_s_estimate"] == 0
                            for uuid in UUIDS[1:]))

    def test_overlapping_processes_on_one_uuid_do_not_exceed_card_capacity(self) -> None:
        result = analysis.analyze(self.budget(), {"gpu_uuids": UUIDS},
                                  self.events(extra_same_gpu=True), self.samples(),
                                  cap_at_budget=True)
        self.assertEqual(result["allocated_distinct_uuid_gpu_s"], 5)
        self.assertEqual(sum(job["process_gpu_s"] for job in result["job_intervals"]), 7)
        self.assertEqual(result["seconds_by_concurrent_distinct_allocated_uuids"],
                         {"0": 5, "1": 5})

    def test_exact_window_must_be_bracketed_and_uuid_pinned(self) -> None:
        with self.assertRaisesRegex(ValueError, "bracket"):
            analysis.analyze(self.budget(), {"gpu_uuids": UUIDS}, self.events(),
                             self.samples()[:-1], cap_at_budget=True)
        with self.assertRaisesRegex(ValueError, "eight assigned"):
            analysis.analyze(self.budget(), {"gpu_uuids": UUIDS[:-1]}, self.events(),
                             self.samples(), cap_at_budget=True)

    def test_duplicate_or_orphan_gpu_events_rejected(self) -> None:
        events = self.events()
        with self.assertRaisesRegex(ValueError, "duplicate GPU event"):
            analysis.analyze(self.budget(), {"gpu_uuids": UUIDS},
                             events + [dict(events[0])], self.samples(), cap_at_budget=True)
        with self.assertRaisesRegex(ValueError, "orphan GPU end"):
            analysis.analyze(self.budget(), {"gpu_uuids": UUIDS},
                             [events[1]], self.samples(), cap_at_budget=True)


if __name__ == "__main__":
    unittest.main()
