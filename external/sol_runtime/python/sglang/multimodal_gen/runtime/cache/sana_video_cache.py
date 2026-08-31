# SPDX-License-Identifier: Apache-2.0
"""SANA-Video cache policies used by the Sol-Agent cache search.

The controller is deliberately model-local and OFF-safe.  It can skip only the
transformer block stack; the caller still executes patch/timestep/text setup,
output projection, unpatchify, scheduler updates, and every denoising step.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

import torch
import torch.nn.functional as F


FAMILIES = {"off", "easycache", "teacache", "taylorseer"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw in (None, "") else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw in (None, "") else float(raw)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw in {"1", "true", "True", "yes", "on"}:
        return True
    if raw in {"0", "false", "False", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be boolean, got {raw!r}")


@dataclass(frozen=True)
class SanaVideoCacheConfig:
    family: str = "off"
    threshold: float = 0.0
    warmup_steps: int = 3
    start_step: int = 0
    end_step: int = 49
    subsample: int = 8
    max_continuous_hits: int = 2
    taylor_order: int = 1
    taylor_damping: float = 1.0
    debug: bool = False

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"cache family must be one of {sorted(FAMILIES)}, got {self.family!r}")
        if self.threshold < 0:
            raise ValueError("cache threshold must be non-negative")
        if self.warmup_steps < 0 or self.start_step < 0:
            raise ValueError("cache warmup/start must be non-negative")
        if self.end_step <= self.start_step:
            raise ValueError("cache end_step must be greater than start_step")
        if self.subsample < 1:
            raise ValueError("cache subsample must be at least 1")
        if self.max_continuous_hits < 1:
            raise ValueError("cache max_continuous_hits must be at least 1")
        if self.taylor_order not in {1, 2}:
            raise ValueError("TaylorSeer order must be 1 or 2")
        if not 0.0 <= self.taylor_damping <= 1.5:
            raise ValueError("TaylorSeer damping must be in [0, 1.5]")

    @property
    def enabled(self) -> bool:
        return self.family != "off" and self.threshold > 0.0

    @classmethod
    def from_env(cls) -> "SanaVideoCacheConfig":
        legacy_threshold = _env_float("SGLANG_SANA_EASYCACHE_THRESH", 0.0)
        family = os.environ.get("SGLANG_SANA_CACHE_FAMILY", "").strip().lower()
        if not family:
            family = "easycache" if legacy_threshold > 0.0 else "off"
        threshold = _env_float("SGLANG_SANA_CACHE_THRESHOLD", legacy_threshold)
        return cls(
            family=family,
            threshold=threshold,
            warmup_steps=_env_int(
                "SGLANG_SANA_CACHE_WARMUP",
                _env_int("SGLANG_SANA_EASYCACHE_WARMUP", 3),
            ),
            start_step=_env_int("SGLANG_SANA_CACHE_START", 0),
            end_step=_env_int("SGLANG_SANA_CACHE_END", 49),
            subsample=_env_int(
                "SGLANG_SANA_CACHE_SUBSAMPLE",
                _env_int("SGLANG_SANA_EASYCACHE_SUBSAMPLE", 8),
            ),
            max_continuous_hits=_env_int("SGLANG_SANA_CACHE_MAX_HITS", 2),
            taylor_order=_env_int("SGLANG_SANA_TAYLOR_ORDER", 1),
            taylor_damping=_env_float("SGLANG_SANA_TAYLOR_DAMPING", 1.0),
            debug=_env_flag(
                "SGLANG_SANA_CACHE_DEBUG",
                bool(os.environ.get("SGLANG_SANA_EASYCACHE_DEBUG", "")),
            ),
        )


@dataclass(frozen=True)
class SanaVideoCacheDecision:
    step: int
    run_blocks: bool
    reason: str
    signal_distance: float | None
    accumulated_distance: float
    continuous_hits: int


@dataclass(frozen=True)
class SanaVideoCacheCall:
    """Exact request-local scheduler/CFG identity for one model forward."""

    step: int
    branch: int
    do_classifier_free_guidance: bool

    @classmethod
    def from_runtime(
        cls,
        *,
        step: int,
        is_cfg_negative: bool,
        do_classifier_free_guidance: bool,
    ) -> "SanaVideoCacheCall":
        if step < 0:
            raise ValueError("cache call step must be non-negative")
        return cls(
            step=step,
            branch=1 if is_cfg_negative else 0,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

    @property
    def starts_generation(self) -> bool:
        # Both a warmup request and the measured request begin here; this is
        # intentionally independent of the scalar scheduler timestep.
        return self.step == 0 and self.branch == 0

    @property
    def owns_decision(self) -> bool:
        return self.branch == 0

    @property
    def is_last_branch(self) -> bool:
        return self.branch == 1 if self.do_classifier_free_guidance else True


class SanaVideoCacheController:
    """Family-neutral decision, residual replay, and Taylor forecast state."""

    def __init__(self, config: SanaVideoCacheConfig) -> None:
        self.config = config
        self.reset()

    @classmethod
    def from_env(cls) -> "SanaVideoCacheController":
        return cls(SanaVideoCacheConfig.from_env())

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def reset(self) -> None:
        self.previous_input: torch.Tensor | None = None
        self.previous_output: torch.Tensor | None = None
        self.previous_output_norm: float | None = None
        self.transformation_rate: float | None = None
        self.previous_modulated_signal: torch.Tensor | None = None
        self.pending_signal: torch.Tensor | None = None
        self.accumulated_distance = 0.0
        self.continuous_hits = 0
        # Serial classifier-free guidance runs the conditional and
        # unconditional branches through the same model instance.  They may
        # share one adaptive recompute schedule, but their block residuals are
        # conditioning-dependent and must never be interchanged.
        self.residual_history: dict[int, list[tuple[int, torch.Tensor]]] = {}
        self.total_decisions = 0
        self.computes = 0
        self.hits = 0
        self.computed_steps: list[int] = []
        self.skipped_steps: list[int] = []
        self.reasons: dict[str, int] = {}

    def begin_call(self, call: SanaVideoCacheCall) -> bool:
        """Reset request-local state at an explicit conditional step zero."""
        if call.starts_generation:
            self.reset()
            return True
        return False

    def _sample(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[:, :: self.config.subsample].detach().float()

    def _modulated_signal(
        self,
        hidden_states: torch.Tensor,
        timestep_mod: torch.Tensor,
        scale_shift_table: torch.Tensor,
    ) -> torch.Tensor:
        sampled = self._sample(hidden_states)
        batch_size = sampled.shape[0]
        params = (
            scale_shift_table.detach().float()[None, None]
            + timestep_mod.detach().float().reshape(batch_size, timestep_mod.shape[1], 6, -1)
        )
        shift_msa, scale_msa = params[:, :, 0], params[:, :, 1]
        normalized = F.layer_norm(sampled, (sampled.shape[-1],))
        return (normalized * (1.0 + scale_msa) + shift_msa).detach()

    def _record(self, decision: SanaVideoCacheDecision) -> SanaVideoCacheDecision:
        self.total_decisions += 1
        self.reasons[decision.reason] = self.reasons.get(decision.reason, 0) + 1
        if decision.run_blocks:
            self.computes += 1
            self.computed_steps.append(decision.step)
        else:
            self.hits += 1
            self.skipped_steps.append(decision.step)
        return decision

    def _decision(
        self,
        step: int,
        run_blocks: bool,
        reason: str,
        distance: float | None = None,
    ) -> SanaVideoCacheDecision:
        return self._record(
            SanaVideoCacheDecision(
                step=step,
                run_blocks=run_blocks,
                reason=reason,
                signal_distance=distance,
                accumulated_distance=self.accumulated_distance,
                continuous_hits=self.continuous_hits,
            )
        )

    def decide(
        self,
        step: int,
        hidden_states: torch.Tensor,
        timestep_mod: torch.Tensor,
        scale_shift_table: torch.Tensor,
    ) -> SanaVideoCacheDecision:
        if not self.enabled:
            return self._decision(step, True, "disabled")

        family = self.config.family
        if family in {"teacache", "taylorseer"}:
            current_signal = self._modulated_signal(
                hidden_states, timestep_mod, scale_shift_table
            )
            previous_signal = self.previous_modulated_signal
            self.previous_modulated_signal = current_signal
            self.pending_signal = current_signal
        else:
            current_signal = None
            previous_signal = None
            self.pending_signal = None

        if step < max(self.config.start_step, self.config.warmup_steps):
            self.accumulated_distance = 0.0
            self.continuous_hits = 0
            return self._decision(step, True, "warmup_or_start")
        if step >= self.config.end_step:
            self.accumulated_distance = 0.0
            self.continuous_hits = 0
            return self._decision(step, True, "end_boundary")
        if not self.residual_history.get(0):
            return self._decision(step, True, "missing_residual")
        if self.continuous_hits >= self.config.max_continuous_hits:
            self.accumulated_distance = 0.0
            self.continuous_hits = 0
            return self._decision(step, True, "continuous_hit_cap")

        if family == "easycache":
            if (
                self.previous_input is None
                or self.previous_output_norm is None
                or self.transformation_rate is None
            ):
                return self._decision(step, True, "easycache_calibration")
            current_input = self._sample(hidden_states)
            input_change = (current_input - self.previous_input).abs().mean()
            distance = float(
                (
                    self.transformation_rate
                    * input_change
                    / max(self.previous_output_norm, 1e-6)
                ).item()
            )
        else:
            if previous_signal is None or current_signal is None:
                return self._decision(step, True, "signal_calibration")
            numerator = (current_signal - previous_signal).abs().mean()
            denominator = previous_signal.abs().mean().clamp_min(1e-6)
            distance = float((numerator / denominator).item())

        self.accumulated_distance += max(distance, 0.0)
        if self.accumulated_distance >= self.config.threshold:
            self.accumulated_distance = 0.0
            self.continuous_hits = 0
            return self._decision(step, True, "threshold_recompute", distance)

        self.continuous_hits += 1
        return self._decision(step, False, "cache_hit", distance)

    def after_compute(
        self,
        *,
        branch: int,
        step: int,
        hidden_before: torch.Tensor,
        hidden_after: torch.Tensor,
    ) -> None:
        if branch == 0 and self.config.family == "easycache":
            current_input = self._sample(hidden_before)
            current_output = self._sample(hidden_after)
            if self.previous_input is not None and self.previous_output is not None:
                input_change = (current_input - self.previous_input).abs().mean()
                output_change = (current_output - self.previous_output).abs().mean()
                if float(input_change.item()) > 1e-12:
                    self.transformation_rate = float((output_change / input_change).item())
            self.previous_input = current_input
            self.previous_output = current_output
            self.previous_output_norm = float(current_output.abs().mean().item())

        residual = (hidden_after - hidden_before).detach()
        history = self.residual_history.setdefault(branch, [])
        # Replacing an equal-step entry makes this robust to a repeated call
        # without manufacturing a zero-interval Taylor history point.
        if history and history[-1][0] == step:
            history[-1] = (step, residual)
        else:
            history.append((step, residual))
        history_limit = 3 if self.config.family == "taylorseer" else 1
        self.residual_history[branch] = history[-history_limit:]

        self.accumulated_distance = 0.0
        self.continuous_hits = 0

    def _forecast_residual(self, step: int, branch: int) -> torch.Tensor:
        history = self.residual_history.get(branch, [])
        if not history:
            raise RuntimeError(
                f"cache hit requested without residual history for CFG branch {branch}"
            )
        latest_step, latest = history[-1]
        if self.config.family != "taylorseer" or len(history) < 2:
            return latest

        previous_step, previous = history[-2]
        interval = max(latest_step - previous_step, 1)
        first_difference = (latest - previous) / interval
        dt = max(step - latest_step, 0)
        forecast_delta = dt * first_difference

        if self.config.taylor_order == 2 and len(history) >= 3:
            oldest_step, oldest = history[-3]
            previous_interval = max(previous_step - oldest_step, 1)
            previous_difference = (previous - oldest) / previous_interval
            second_divided_difference = (
                (first_difference - previous_difference)
                / max(latest_step - oldest_step, 1)
            )
            forecast_delta = forecast_delta + (
                dt * max(step - previous_step, 0) * second_divided_difference
            )

        return latest + self.config.taylor_damping * forecast_delta

    def reuse(
        self, step: int, hidden_states: torch.Tensor, *, branch: int = 0
    ) -> torch.Tensor:
        return hidden_states + self._forecast_residual(step, branch).to(
            device=hidden_states.device, dtype=hidden_states.dtype
        )

    def stats_summary(self) -> dict[str, Any]:
        return {
            "family": self.config.family,
            "threshold": self.config.threshold,
            "warmup_steps": self.config.warmup_steps,
            "start_step": self.config.start_step,
            "end_step": self.config.end_step,
            "subsample": self.config.subsample,
            "max_continuous_hits": self.config.max_continuous_hits,
            "taylor_order": self.config.taylor_order,
            "taylor_damping": self.config.taylor_damping,
            "total_decisions": self.total_decisions,
            "computes": self.computes,
            "hits": self.hits,
            "hit_rate": 0.0 if self.total_decisions == 0 else self.hits / self.total_decisions,
            "computed_steps": self.computed_steps,
            "skipped_steps": self.skipped_steps,
            "reasons": self.reasons,
            "residual_history_steps": {
                str(branch): [step for step, _ in history]
                for branch, history in sorted(self.residual_history.items())
            },
        }

    def format_decision(self, decision: SanaVideoCacheDecision) -> str:
        payload = {
            "family": self.config.family,
            "step": decision.step,
            "run_blocks": decision.run_blocks,
            "reason": decision.reason,
            "signal_distance": decision.signal_distance,
            "accumulated_distance": decision.accumulated_distance,
            "continuous_hits": decision.continuous_hits,
        }
        return "SANA_CACHE_DECISION " + json.dumps(payload, sort_keys=True)

    def format_summary(self) -> str:
        return "SANA_CACHE_SUMMARY " + json.dumps(self.stats_summary(), sort_keys=True)


__all__ = [
    "FAMILIES",
    "SanaVideoCacheCall",
    "SanaVideoCacheConfig",
    "SanaVideoCacheController",
    "SanaVideoCacheDecision",
]
