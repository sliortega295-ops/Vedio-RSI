"""Exact attempt-11 goal adapter tests; no Codex process or GPU is started."""

from __future__ import annotations

import unittest
from unittest import mock

from tools.symposium import site_codex_goal_session as adapter
from tools.symposium import site_phase3_multi_image_r06_r07_r08 as triple


class Attempt11AdapterTests(unittest.TestCase):
    def test_real_approved_route_is_thirty_images(self) -> None:
        images = adapter.image_inputs(triple.SESSION, triple.ATTEMPT)
        self.assertEqual(len(images), 30)
        self.assertEqual([row["round"] for row in images],
                         ["r06"] * 10 + ["r07"] * 10 + ["r08"] * 10)
        self.assertEqual({row["attachment_manifest_sha256"] for row in images},
                         {triple.sha(triple.ATTACHMENT)})

    def test_wrong_session_or_attempt_never_uses_triple_route(self) -> None:
        self.assertEqual(adapter.image_inputs(triple.SESSION, 12), [])
        self.assertEqual(adapter.image_inputs("sana2ba100-master-0002", 11), [])

    def test_helper_hash_failure_prevents_validator_call(self) -> None:
        with mock.patch.object(adapter, "PHASE3_TRIPLE_SOURCE_SHA256", "0" * 64), \
             mock.patch.object(triple, "inputs",
                               side_effect=AssertionError("validator was reached")):
            with self.assertRaisesRegex(RuntimeError, "source SHA mismatch"):
                adapter.image_inputs(triple.SESSION, triple.ATTEMPT)

    def test_validator_failure_never_falls_back(self) -> None:
        with mock.patch.object(triple, "inputs", side_effect=RuntimeError("fixed PNG changed")), \
             mock.patch.object(adapter, "phase3_cache_image_inputs",
                               side_effect=AssertionError("unsafe legacy fallback")):
            with self.assertRaisesRegex(RuntimeError, "fixed PNG changed"):
                adapter.image_inputs(triple.SESSION, triple.ATTEMPT)


if __name__ == "__main__":
    unittest.main()
