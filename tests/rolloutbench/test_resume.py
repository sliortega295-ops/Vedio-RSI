from __future__ import annotations

import unittest

from rolloutbench.events import LedgerState
from rolloutbench.resume import ResumePlanError, plan_stage_resume


def _state(stage_states: dict[tuple[str, str, int], str]) -> LedgerState:
    return LedgerState((), {}, stage_states, set())


class ResumePlannerTests(unittest.TestCase):
    def test_start_retry_and_reuse_are_derived_from_durable_state(self) -> None:
        self.assertEqual(
            {"action": "start", "attempt": 1, "reason": "no_prior_attempt"},
            plan_stage_resume(_state({}), episode_id="K20", stage="generate"),
        )
        self.assertEqual(
            "retry",
            plan_stage_resume(
                _state({("K20", "generate", 1): "interrupted"}),
                episode_id="K20",
                stage="generate",
            )["action"],
        )
        self.assertEqual(
            "reuse_completed",
            plan_stage_resume(
                _state({("C12", "quality_v1", 1): "completed"}),
                episode_id="C12",
                stage="quality_v1",
            )["action"],
        )

    def test_conflicting_attempts_fail_closed(self) -> None:
        contradictory = _state(
            {
                ("C12", "quality_v1", 1): "completed",
                ("C12", "quality_v1", 2): "interrupted",
            }
        )
        with self.assertRaises(ResumePlanError):
            plan_stage_resume(
                contradictory, episode_id="C12", stage="quality_v1"
            )


if __name__ == "__main__":
    unittest.main()
