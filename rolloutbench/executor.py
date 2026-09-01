from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FakeExecution:
    episode_id: str
    stages: tuple[str, ...]
    decision: dict[str, Any]
    frontier_candidate: bool


class FakeExecutor:
    """Contract-only executor for deterministic CPU simulations.

    It intentionally has no historical oracle input. Its outputs prove scheduling,
    validation routing, and exactly-once plumbing—not candidate performance.
    """

    def execute(self, episode: dict[str, Any]) -> FakeExecution:
        if "golden" in episode:
            raise ValueError("FakeExecutor accepts only the scheduler public view")
        episode_id = str(episode["episode_id"])
        stages = list(episode["validation"]["stages"])
        replay = episode.get("replay", {})
        fault = replay.get("fault_injection") or replay.get("failure_contract")
        if fault:
            stages = [stage for stage in stages if stage in {"acquire_gpu", "preflight", "decide"}]
            outcome = "recorded_failure"
            frontier_candidate = False
        elif episode["validation"]["contract"] == "gpu_preflight_exact":
            outcome = "preflight_rejected"
            frontier_candidate = False
        elif episode["quality_eligibility"] in {"provenance_failed", "calibration_only"}:
            outcome = "calibration_recorded"
            frontier_candidate = False
        else:
            outcome = "contract_validated"
            frontier_candidate = True
        return FakeExecution(
            episode_id=episode_id,
            stages=tuple(stages),
            decision={"outcome": outcome, "contract": episode["validation"]["contract"]},
            frontier_candidate=frontier_candidate,
        )
