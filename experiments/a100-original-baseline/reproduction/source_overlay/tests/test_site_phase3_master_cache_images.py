"""Scratch-only tests for the not-deployed Master independent image route."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from tools.symposium import site_phase3_master_cache_images as master
from tools.symposium import site_phase3_multi_image_r06_r07_r08 as triple


class MasterCacheImageTests(unittest.TestCase):
    @contextmanager
    def fixture(self, *, status: str = "prepared", round_id: str = "r07",
                point_run: str | None = None, delivery_sha: str | None = None):
        scratch = Path("/tmp/lyy-experiments/SolAgent/phase3-master-image-tests")
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="master-cache-", dir=scratch) as temp:
            directory = Path(temp)
            delivered = point_run or master.KNOWN_RUNS[round_id]
            delivery = directory / "DELIVERY.json"
            delivery.write_text(json.dumps({"schema_version": 2, "status": "complete",
                                            "component": "cache",
                                            "model_id": "sana_video_2b_a100",
                                            "frontier_points": [{"run_dir": delivered}]}) + "\n")
            images = [row for row in triple.inputs(triple.SESSION, triple.ATTEMPT)
                      if row["round"] == round_id]
            attachment = directory / master.ATTACHMENTS_BY_ATTEMPT[12].name
            attachment.write_text(json.dumps({
                "schema_version": 1, "status": status, "session": master.SESSION,
                "attempt": 12,
                "cache_delivery_sha256": delivery_sha or master.sha(delivery),
                "points": [{"round": round_id, "run_dir": delivered}],
                "image_count": 10,
                "image_bytes": sum(Path(row["path"]).stat().st_size for row in images),
            }, sort_keys=True) + "\n")
            with mock.patch.object(master, "CACHE", directory), \
                 mock.patch.object(master, "DELIVERY", delivery), \
                 mock.patch.object(master, "ATTACHMENTS", directory), \
                 mock.patch.object(master, "ATTACHMENTS_BY_ATTEMPT", {12: attachment}):
                yield

    def test_wrong_session_and_attempt_return_no_images(self) -> None:
        self.assertEqual(master.inputs("sana2ba100-cache_ca-0002", 12), [])
        self.assertEqual(master.inputs(master.SESSION, 11), [])

    def test_real_manifest_is_now_exactly_approved(self) -> None:
        rows = master.inputs(master.SESSION, 12)
        self.assertEqual(len(rows), 20)
        self.assertEqual({row["round"] for row in rows}, {"r05", "r08"})

    def test_attempt13_real_manifest_is_now_exactly_approved(self) -> None:
        rows = master.inputs(master.SESSION, 13)
        self.assertEqual(len(rows), 20)
        self.assertEqual({row["round"] for row in rows}, {"r05", "r08"})

    def test_prepared_fixture_is_dry_run_only(self) -> None:
        with self.fixture():
            with self.assertRaisesRegex(RuntimeError, "approval mismatch"):
                master.inputs(master.SESSION, 12)
            images = master.inputs(master.SESSION, 12, allow_prepared=True)
        self.assertEqual(len(images), 10)
        self.assertTrue(all(row["round"] == "r07" for row in images))

    def test_approved_fixture_returns_all_five_prompts(self) -> None:
        with self.fixture(status="approved"):
            images = master.inputs(master.SESSION, 12)
        self.assertEqual([(row["prompt_index"], row["kind"]) for row in images],
                         [(prompt, kind) for prompt in range(1, 6)
                          for kind in ("overview", "detail")])
        self.assertEqual(len({row["attachment_manifest_sha256"] for row in images}), 1)

    def test_changed_delivery_sha_rejected(self) -> None:
        with self.fixture(status="approved", delivery_sha="0" * 64):
            with self.assertRaisesRegex(RuntimeError, "DELIVERY SHA changed"):
                master.inputs(master.SESSION, 12)

    def test_unknown_delivered_run_rejected(self) -> None:
        with self.fixture(status="approved", point_run="runs/unapproved-other-run"):
            with self.assertRaisesRegex(RuntimeError, "exact full delivered frontier"):
                master.inputs(master.SESSION, 12)

    def test_helper_source_change_rejected(self) -> None:
        with self.fixture(status="approved"), \
             mock.patch.object(master, "TRIPLE_SOURCE_SHA256", "0" * 64):
            with self.assertRaisesRegex(RuntimeError, "helper SHA/import changed"):
                master.inputs(master.SESSION, 12)


if __name__ == "__main__":
    unittest.main()
