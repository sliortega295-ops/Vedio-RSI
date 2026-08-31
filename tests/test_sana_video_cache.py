from __future__ import annotations

import unittest

import torch

from sglang.multimodal_gen.runtime.cache.sana_video_cache import (
    SanaVideoCacheCall,
    SanaVideoCacheConfig,
    SanaVideoCacheController,
)


def tensors(value: float = 0.0):
    hidden = torch.full((1, 8, 4), value)
    timestep = torch.zeros((1, 1, 24))
    table = torch.zeros((6, 4))
    return hidden, timestep, table


def compute_step(controller, step, before, after):
    controller.after_compute(
        branch=0, step=step, hidden_before=before, hidden_after=after
    )
    controller.after_compute(
        branch=1, step=step, hidden_before=before, hidden_after=after
    )


class SanaVideoCacheControllerTests(unittest.TestCase):
    def test_off_identity_is_disabled(self):
        controller = SanaVideoCacheController(SanaVideoCacheConfig())
        hidden, timestep, table = tensors()
        decision = controller.decide(0, hidden, timestep, table)
        self.assertFalse(controller.enabled)
        self.assertTrue(decision.run_blocks)
        self.assertEqual(decision.reason, "disabled")

    def test_easycache_calibrates_then_hits(self):
        config = SanaVideoCacheConfig(
            family="easycache", threshold=10.0, warmup_steps=1, end_step=6
        )
        controller = SanaVideoCacheController(config)
        hidden0, timestep, table = tensors(0.0)
        decision0 = controller.decide(0, hidden0, timestep, table)
        self.assertTrue(decision0.run_blocks)
        compute_step(controller, 0, hidden0, hidden0 + 1.0)

        hidden1, _, _ = tensors(0.1)
        decision1 = controller.decide(1, hidden1, timestep, table)
        self.assertTrue(decision1.run_blocks)
        compute_step(controller, 1, hidden1, hidden1 + 1.1)

        hidden2, _, _ = tensors(0.11)
        decision2 = controller.decide(2, hidden2, timestep, table)
        self.assertFalse(decision2.run_blocks)
        self.assertEqual(decision2.reason, "cache_hit")

    def test_teacache_honors_continuous_hit_cap(self):
        config = SanaVideoCacheConfig(
            family="teacache",
            threshold=10.0,
            warmup_steps=1,
            end_step=6,
            max_continuous_hits=2,
        )
        controller = SanaVideoCacheController(config)
        hidden, timestep, table = tensors(0.2)
        self.assertTrue(controller.decide(0, hidden, timestep, table).run_blocks)
        compute_step(controller, 0, hidden, hidden + 1.0)
        self.assertFalse(controller.decide(1, hidden, timestep, table).run_blocks)
        self.assertFalse(controller.decide(2, hidden, timestep, table).run_blocks)
        capped = controller.decide(3, hidden, timestep, table)
        self.assertTrue(capped.run_blocks)
        self.assertEqual(capped.reason, "continuous_hit_cap")

    def test_taylorseer_first_order_forecasts_residual(self):
        config = SanaVideoCacheConfig(
            family="taylorseer",
            threshold=10.0,
            warmup_steps=2,
            end_step=6,
            taylor_order=1,
        )
        controller = SanaVideoCacheController(config)
        hidden, timestep, table = tensors(0.2)
        for step, residual in [(0, 1.0), (1, 2.0)]:
            self.assertTrue(controller.decide(step, hidden, timestep, table).run_blocks)
            compute_step(controller, step, hidden, hidden + residual)
        decision = controller.decide(2, hidden, timestep, table)
        self.assertFalse(decision.run_blocks)
        forecast = controller.reuse(2, hidden, branch=0)
        self.assertTrue(torch.allclose(forecast, hidden + 3.0))

    def test_taylorseer_second_order_uses_divided_differences(self):
        config = SanaVideoCacheConfig(
            family="taylorseer",
            threshold=10.0,
            warmup_steps=3,
            end_step=6,
            taylor_order=2,
        )
        controller = SanaVideoCacheController(config)
        hidden, timestep, table = tensors(0.2)
        for step, residual in [(0, 0.0), (1, 1.0), (2, 4.0)]:
            self.assertTrue(controller.decide(step, hidden, timestep, table).run_blocks)
            compute_step(controller, step, hidden, hidden + residual)
        decision = controller.decide(3, hidden, timestep, table)
        self.assertFalse(decision.run_blocks)
        forecast = controller.reuse(3, hidden, branch=0)
        self.assertTrue(torch.allclose(forecast, hidden + 9.0))

    def test_serial_cfg_replays_branch_specific_residuals(self):
        config = SanaVideoCacheConfig(
            family="easycache", threshold=10.0, warmup_steps=1, end_step=6
        )
        controller = SanaVideoCacheController(config)
        hidden, timestep, table = tensors(0.0)
        self.assertTrue(controller.decide(0, hidden, timestep, table).run_blocks)
        controller.after_compute(
            branch=0, step=0, hidden_before=hidden, hidden_after=hidden + 1.0
        )
        controller.after_compute(
            branch=1, step=0, hidden_before=hidden, hidden_after=hidden + 3.0
        )
        self.assertTrue(torch.allclose(controller.reuse(1, hidden, branch=0), hidden + 1.0))
        self.assertTrue(torch.allclose(controller.reuse(1, hidden, branch=1), hidden + 3.0))

    def test_repeated_branch_step_replaces_taylor_history(self):
        config = SanaVideoCacheConfig(
            family="taylorseer", threshold=10.0, warmup_steps=2, end_step=6
        )
        controller = SanaVideoCacheController(config)
        hidden, _, _ = tensors(0.0)
        controller.after_compute(
            branch=0, step=0, hidden_before=hidden, hidden_after=hidden + 1.0
        )
        controller.after_compute(
            branch=0, step=0, hidden_before=hidden, hidden_after=hidden + 2.0
        )
        self.assertEqual(len(controller.residual_history[0]), 1)
        self.assertTrue(
            torch.allclose(controller.reuse(1, hidden, branch=0), hidden + 2.0)
        )

    def test_formal_step_zero_resets_equal_timestep_warmup_state(self):
        config = SanaVideoCacheConfig(
            family="easycache", threshold=10.0, warmup_steps=1, end_step=6
        )
        controller = SanaVideoCacheController(config)
        hidden, timestep, table = tensors(0.0)
        warmup = SanaVideoCacheCall.from_runtime(
            step=0,
            is_cfg_negative=False,
            do_classifier_free_guidance=False,
        )
        self.assertTrue(controller.begin_call(warmup))
        controller.decide(0, hidden, timestep, table)
        controller.after_compute(
            branch=0, step=0, hidden_before=hidden, hidden_after=hidden + 9.0
        )
        self.assertEqual(controller.total_decisions, 1)
        self.assertIn(0, controller.residual_history)

        formal_cond = SanaVideoCacheCall.from_runtime(
            step=0,
            is_cfg_negative=False,
            do_classifier_free_guidance=True,
        )
        formal_uncond = SanaVideoCacheCall.from_runtime(
            step=0,
            is_cfg_negative=True,
            do_classifier_free_guidance=True,
        )
        self.assertTrue(controller.begin_call(formal_cond))
        self.assertFalse(controller.begin_call(formal_uncond))
        self.assertEqual(controller.total_decisions, 0)
        self.assertEqual(controller.residual_history, {})
        self.assertTrue(formal_cond.owns_decision)
        self.assertFalse(formal_cond.is_last_branch)
        self.assertFalse(formal_uncond.owns_decision)
        self.assertTrue(formal_uncond.is_last_branch)

    def test_exactly_one_decision_per_formal_scheduler_step(self):
        config = SanaVideoCacheConfig(
            family="teacache",
            threshold=10.0,
            warmup_steps=3,
            end_step=49,
            max_continuous_hits=2,
        )
        controller = SanaVideoCacheController(config)
        hidden, timestep, table = tensors(0.2)
        for step in range(50):
            for is_negative in (False, True):
                call = SanaVideoCacheCall.from_runtime(
                    step=step,
                    is_cfg_negative=is_negative,
                    do_classifier_free_guidance=True,
                )
                controller.begin_call(call)
                if call.owns_decision:
                    decision = controller.decide(step, hidden, timestep, table)
                if decision.run_blocks:
                    controller.after_compute(
                        branch=call.branch,
                        step=step,
                        hidden_before=hidden,
                        hidden_after=hidden + call.branch + 1.0,
                    )
                else:
                    controller.reuse(step, hidden, branch=call.branch)

        self.assertEqual(controller.total_decisions, 50)
        self.assertEqual(controller.computes + controller.hits, 50)
        self.assertEqual(
            sorted(controller.computed_steps + controller.skipped_steps),
            list(range(50)),
        )
        self.assertEqual(set(controller.residual_history), {0, 1})


if __name__ == "__main__":
    unittest.main()
