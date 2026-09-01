from __future__ import annotations

from typing import Any

from .events import LedgerState


class ResumePlanError(RuntimeError):
    """Raised when durable stage history cannot produce one safe resume action."""


def plan_stage_resume(
    state: LedgerState, *, episode_id: str, stage: str
) -> dict[str, Any]:
    """Choose reuse, retry, or first start from durable ledger state only."""

    attempts = {
        attempt: status
        for (recorded_episode, recorded_stage, attempt), status in state.stage_states.items()
        if recorded_episode == episode_id and recorded_stage == stage
    }
    if any(not isinstance(attempt, int) or attempt < 1 for attempt in attempts):
        raise ResumePlanError("stage attempts must be positive integers")
    completed = [attempt for attempt, status in attempts.items() if status == "completed"]
    if completed:
        if len(completed) != 1 or any(attempt > completed[0] for attempt in attempts):
            raise ResumePlanError("stage history continues after durable completion")
        return {
            "action": "reuse_completed",
            "attempt": max(completed),
            "reason": "durable_stage_completion_exists",
        }
    if attempts:
        return {
            "action": "retry",
            "attempt": max(attempts) + 1,
            "reason": "prior_attempt_did_not_complete",
        }
    return {
        "action": "start",
        "attempt": 1,
        "reason": "no_prior_attempt",
    }
