"""CPU-only exact-session tests for prepared R03/R05 image transport."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.symposium import site_phase3_multi_image as multi


class Phase3MultiImageTests(unittest.TestCase):
    def test_wrong_session_or_attempt_gets_no_images(self) -> None:
        self.assertEqual(multi.inputs("sana2ba100-master-0002", 10), [])
        self.assertEqual(multi.inputs(multi.SESSION, 9), [])
        self.assertEqual(multi.inputs(multi.SESSION, 11), [])

    def test_prepared_manifest_is_dry_run_only(self) -> None:
        scratch = Path("/tmp/lyy-experiments/SolAgent/phase3-image-transport-tests")
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="attempt10-prepared-", dir=scratch) as temp:
            directory = Path(temp)
            attachment = directory / multi.ATTACHMENT.name
            data = json.loads(multi.ATTACHMENT.read_text())
            data["status"] = "prepared"
            attachment.write_text(json.dumps(data, sort_keys=True) + "\n")
            with mock.patch.object(multi, "ATTACHMENTS", directory), \
                 mock.patch.object(multi, "ATTACHMENT", attachment):
                with self.assertRaisesRegex(RuntimeError, "approval mismatch"):
                    multi.inputs(multi.SESSION, multi.ATTEMPT)
        images = multi.inputs(multi.SESSION, multi.ATTEMPT, allow_prepared=True)
        self.assertEqual(len(images), 20)
        self.assertEqual([row["round"] for row in images], ["r03"] * 10 + ["r05"] * 10)
        self.assertEqual(sum(Path(row["path"]).stat().st_size for row in images), 48_381_103)

    def test_approved_fixture_returns_exact_twenty(self) -> None:
        scratch = Path("/tmp/lyy-experiments/SolAgent/phase3-image-transport-tests")
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="approved-fixture-", dir=scratch) as temp:
            directory = Path(temp)
            attachment = directory / multi.ATTACHMENT.name
            data = json.loads(multi.ATTACHMENT.read_text())
            data["status"] = "approved"
            attachment.write_text(json.dumps(data, sort_keys=True) + "\n")
            with mock.patch.object(multi, "ATTACHMENTS", directory), \
                 mock.patch.object(multi, "ATTACHMENT", attachment):
                images = multi.inputs(multi.SESSION, multi.ATTEMPT)
        self.assertEqual(len(images), 20)
        self.assertEqual({row["render_manifest_sha256"] for row in images},
                         {multi.RENDER_MANIFEST_SHA256})

    def test_changed_render_manifest_hash_rejected(self) -> None:
        original = multi.sha
        target = multi.RENDER / "render_manifest.json"
        with mock.patch.object(multi, "sha",
                               side_effect=lambda p: "0" * 64 if p == target else original(p)):
            with self.assertRaisesRegex(RuntimeError, "render manifest SHA mismatch"):
                multi.inputs(multi.SESSION, multi.ATTEMPT, allow_prepared=True)

    def test_changed_image_hash_rejected(self) -> None:
        original = multi.sha
        target = multi.RENDER / "r05-p02-detail.png"
        with mock.patch.object(multi, "sha",
                               side_effect=lambda p: "0" * 64 if p == target else original(p)):
            with self.assertRaisesRegex(RuntimeError, "PNG identity/SHA mismatch"):
                multi.inputs(multi.SESSION, multi.ATTEMPT, allow_prepared=True)

    def test_changed_candidate_frame_hash_rejected(self) -> None:
        original = multi.sha
        target = multi.SOURCE_RUNS["r05"]["frames"] / "f_001.png"
        with mock.patch.object(multi, "sha",
                               side_effect=lambda p: "0" * 64 if p == target else original(p)):
            with self.assertRaisesRegex(RuntimeError, "source frame changed"):
                multi.inputs(multi.SESSION, multi.ATTEMPT, allow_prepared=True)

    def test_changed_original_judge_receipt_rejected(self) -> None:
        original = multi.sha
        target = multi.SOURCE_RUNS["r05"]["receipt"]
        with mock.patch.object(multi, "sha",
                               side_effect=lambda p: "0" * 64 if p == target else original(p)):
            with self.assertRaisesRegex(RuntimeError, "judge receipt changed"):
                multi.inputs(multi.SESSION, multi.ATTEMPT, allow_prepared=True)


if __name__ == "__main__":
    unittest.main()
