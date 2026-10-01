"""CPU-only exact Master12 image and original-reverify environment gates."""

from __future__ import annotations

import unittest
from unittest import mock

from orchestration import site_reverify_transport
from tools.symposium import site_codex_goal_session as goal


class Master12AdapterTests(unittest.TestCase):
    def test_exact_master12_images_cover_two_points_five_prompts(self) -> None:
        rows = goal.image_inputs("sana2ba100-master-0002", 12)
        self.assertEqual(len(rows), 20)
        self.assertEqual([row["round"] for row in rows], ["r05"] * 10 + ["r08"] * 10)
        self.assertEqual([row["prompt_index"] for row in rows],
                         [prompt for _round in ("r05", "r08")
                          for prompt in range(1, 6) for _kind in ("overview", "detail")])

    def test_changed_helper_source_fails_closed(self) -> None:
        with mock.patch.object(goal, "PHASE3_MASTER_CACHE_SOURCE_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "helper source SHA mismatch"):
                goal.image_inputs("sana2ba100-master-0002", 12)

    def test_reverify_env_only_exact_master12(self) -> None:
        self.assertEqual(goal.master12_reverify_env("sana2ba100-master-0002", 11), ({}, {}))
        self.assertEqual(goal.master12_reverify_env("sana2ba100-cache_ca-0002", 12), ({}, {}))

    def test_remote_worker_must_be_trusted_before_master12(self) -> None:
        with mock.patch.object(site_reverify_transport, "assert_approved", return_value={}), \
             mock.patch.object(site_reverify_transport, "trusted_remote_worker",
                               side_effect=RuntimeError("remote SHA missing")):
            with self.assertRaisesRegex(RuntimeError, "remote SHA missing"):
                goal.master12_reverify_env("sana2ba100-master-0002", 12)

    def test_success_env_is_fixed_shim_after_exact_checks(self) -> None:
        with mock.patch.object(site_reverify_transport, "assert_approved", return_value={}), \
             mock.patch.object(site_reverify_transport, "trusted_remote_worker",
                               return_value={"worker_sha256": "a" * 64}):
            env, receipt = goal.master12_reverify_env("sana2ba100-master-0002", 12)
        self.assertEqual(env, {"PLAN_EVAL_PYTHON": str(goal.PHASE3_REVERIFY_SHIM)})
        self.assertEqual(len(receipt["exact_run_names"]), 2)

    def test_live_post_reverify_state_rejects_unapproved_third_call(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "predecessor SHA mismatch"):
            goal.master12_reverify_env("sana2ba100-master-0002", 13)


if __name__ == "__main__":
    unittest.main()
