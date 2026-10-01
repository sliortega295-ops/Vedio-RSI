"""CPU-only fail-closed tests for the inactive R06/R07/R08 attempt-11 route."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.symposium import site_phase3_multi_image_r06_r07_r08 as route


class Attempt11ImageRouteTests(unittest.TestCase):
    def test_wrong_session_or_attempt_has_no_images(self) -> None:
        self.assertEqual(route.inputs("sana2ba100-master-0002", 11), [])
        self.assertEqual(route.inputs(route.SESSION, 10), [])
        self.assertEqual(route.inputs(route.SESSION, 12), [])

    def test_prepared_fixture_rejected_by_real_path(self) -> None:
        scratch = Path("/tmp/lyy-experiments/SolAgent/phase3-image-transport-tests")
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="attempt11-prepared-", dir=scratch) as temp:
            directory = Path(temp)
            attachment = directory / route.ATTACHMENT.name
            data = json.loads(route.ATTACHMENT.read_text())
            data["status"] = "prepared"
            attachment.write_text(json.dumps(data, sort_keys=True) + "\n")
            with mock.patch.object(route, "ATTACHMENTS", directory), \
                 mock.patch.object(route, "ATTACHMENT", attachment):
                with self.assertRaisesRegex(RuntimeError, "approval mismatch"):
                    route.inputs(route.SESSION, route.ATTEMPT)

    def test_non_override_dry_run_is_exact_thirty(self) -> None:
        images = route.inputs(route.SESSION, route.ATTEMPT, allow_prepared=True)
        self.assertEqual(len(images), 30)
        self.assertEqual(sum(Path(row["path"]).stat().st_size for row in images),
                         route.IMAGE_BYTES)
        self.assertEqual([(row["round"], row["prompt_index"], row["kind"])
                          for row in images],
                         [(round_id, prompt, kind) for round_id in route.ROUNDS
                          for prompt in range(1, 6) for kind in ("overview", "detail")])
        self.assertEqual(len({row["attachment_manifest_sha256"] for row in images}), 1)

    def test_approved_real_manifest_uses_exact_thirty(self) -> None:
        images = route.inputs(route.SESSION, route.ATTEMPT)
        self.assertEqual(len(images), 30)

    def bad_sha(self, target: Path) -> None:
        original = route.sha

        def swapped(path: Path) -> str:
            return "0" * 64 if path == target else original(path)

        with mock.patch.object(route, "sha", side_effect=swapped):
            route.inputs(route.SESSION, route.ATTEMPT, allow_prepared=True)

    def test_prior_helper_source_changed_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "helper source SHA/import mismatch"):
            self.bad_sha(route.PRIOR_SOURCE)

    def test_r08_render_manifest_changed_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "render manifest SHA mismatch"):
            self.bad_sha(route.RENDER / "render_manifest.json")

    def test_r08_renderer_source_changed_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "renderer SHA mismatch"):
            self.bad_sha(route.RENDER_SCRIPT)

    def test_r08_judge_receipt_changed_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "receipt changed"):
            self.bad_sha(route.R08_RECEIPT)

    def test_r08_png_changed_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "PNG identity/SHA mismatch"):
            self.bad_sha(route.RENDER / "r08-p02-detail.png")

    def test_r08_source_frame_changed_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ordered-frame digest changed"):
            self.bad_sha(route.R08_FRAMES / "f_001.png")


class R08PixelVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        scripts = str(route.PROJECT / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        import phase3_render_r08_visuals as rendered
        cls.rendered = rendered

    def test_actual_verifier_accepts_untouched_pixels(self) -> None:
        result = self.rendered.verify()
        self.assertEqual((result["image_count"], result["verified_tiles"]), (10, 194))

    def bad_render_sha(self, target: Path) -> None:
        core = self.rendered.render
        original = core.sha256

        def swapped(path: Path) -> str:
            return "0" * 64 if path == target else original(path)

        with mock.patch.object(core, "sha256", side_effect=swapped):
            self.rendered.verify()

    def test_actual_pixel_verifier_rejects_bad_png(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "rendered image SHA/path/size mismatch"):
            self.bad_render_sha(route.RENDER / "r08-p02-detail.png")

    def test_actual_pixel_verifier_rejects_bad_source(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ordered-frame SHA256 digest changed"):
            self.bad_render_sha(route.R08_FRAMES / "f_001.png")


if __name__ == "__main__":
    unittest.main()
