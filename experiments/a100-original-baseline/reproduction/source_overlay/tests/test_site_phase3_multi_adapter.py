"""Exact attempt-10 integration checks for the host image transport adapter."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest import mock

from tools.symposium import site_codex_goal_session as adapter
from tools.symposium import site_phase3_multi_image as multi


class Phase3MultiAdapterTests(unittest.TestCase):
    def test_attempt10_prepared_dry_run_is_twenty_images(self) -> None:
        images = adapter.image_inputs(multi.SESSION, multi.ATTEMPT,
                                      allow_prepared=True)
        self.assertEqual(len(images), 20)
        self.assertEqual([image["round"] for image in images],
                         ["r03"] * 10 + ["r05"] * 10)
        self.assertEqual({image["render_manifest_sha256"] for image in images},
                         {multi.RENDER_MANIFEST_SHA256})

    def test_prepared_attempt10_cannot_start_real_agent(self) -> None:
        scratch = Path("/tmp/lyy-experiments/SolAgent/phase3-image-transport-tests")
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="adapter10-prepared-", dir=scratch) as temp:
            directory = Path(temp)
            attachment = directory / multi.ATTACHMENT.name
            data = json.loads(multi.ATTACHMENT.read_text())
            data["status"] = "prepared"
            attachment.write_text(json.dumps(data, sort_keys=True) + "\n")
            with mock.patch.object(multi, "ATTACHMENTS", directory), \
                 mock.patch.object(multi, "ATTACHMENT", attachment):
                with self.assertRaisesRegex(RuntimeError, "approval mismatch"):
                    adapter.image_inputs(multi.SESSION, multi.ATTEMPT)

    def test_wrong_session_and_attempt_never_take_multi_route(self) -> None:
        self.assertEqual(adapter.image_inputs(multi.SESSION, 12), [])
        self.assertEqual(adapter.image_inputs("sana2ba100-master-0002", 10), [])

    def test_bad_module_source_hash_rejects_before_imported_validator(self) -> None:
        with mock.patch.object(adapter, "PHASE3_MULTI_SOURCE_SHA256", "0" * 64):
            with mock.patch.object(multi, "inputs",
                                   side_effect=AssertionError("validator reached")):
                with self.assertRaisesRegex(RuntimeError, "source SHA mismatch"):
                    adapter.image_inputs(multi.SESSION, multi.ATTEMPT,
                                         allow_prepared=True)

    def test_validator_failure_never_falls_back_to_v1(self) -> None:
        with mock.patch.object(multi, "inputs",
                               side_effect=RuntimeError("fixed source frame changed")):
            with mock.patch.object(adapter, "phase3_cache_image_inputs",
                                   side_effect=AssertionError("unsafe v1 fallback")):
                with self.assertRaisesRegex(RuntimeError, "fixed source frame changed"):
                    adapter.image_inputs(multi.SESSION, multi.ATTEMPT,
                                         allow_prepared=True)


if __name__ == "__main__":
    unittest.main()
