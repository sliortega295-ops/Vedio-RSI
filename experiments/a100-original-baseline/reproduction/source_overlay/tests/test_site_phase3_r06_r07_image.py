"""CPU-only fail-closed tests for a prepared, inactive R06/R07 image route."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from tools.symposium import site_phase3_multi_image_r06_r07 as route


class R06R07ImageRouteTests(unittest.TestCase):
    def test_wrong_session_or_attempt_has_no_images(self) -> None:
        self.assertEqual(route.inputs("unrelated", route.ATTEMPT), [])
        self.assertEqual(route.inputs(route.SESSION, route.ATTEMPT + 1), [])

    def test_prepared_manifest_cannot_enter_real_agent(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "approval mismatch"):
            route.inputs(route.SESSION, route.ATTEMPT)

    def test_prepared_static_dry_run_is_exact(self) -> None:
        images = route.inputs(route.SESSION, route.ATTEMPT, allow_prepared=True)
        self.assertEqual(len(images), 20)
        self.assertEqual(sum(Path(item["path"]).stat().st_size for item in images),
                         route.IMAGE_BYTES)
        self.assertEqual([(item["round"], item["prompt_index"], item["kind"])
                          for item in images],
                         [(round_id, prompt, kind) for round_id in route.ROUNDS
                          for prompt in range(1, 6) for kind in ("overview", "detail")])

    def bad_sha(self, target: Path) -> None:
        original = route.sha

        def swapped(path: Path) -> str:
            return "0" * 64 if path == target else original(path)

        with mock.patch.object(route, "sha", side_effect=swapped):
            route.inputs(route.SESSION, route.ATTEMPT, allow_prepared=True)

    def test_bad_render_manifest_rejected_by_actual_validator(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "render manifest SHA mismatch"):
            self.bad_sha(route.RENDER / "render_manifest.json")

    def test_bad_renderer_source_rejected_by_actual_validator(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "renderer SHA mismatch"):
            self.bad_sha(route.RENDER_SCRIPT)

    def test_bad_judge_receipt_rejected_by_actual_validator(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "judge receipt changed"):
            self.bad_sha(route.SOURCE_RUNS["r07"]["receipt"])

    def test_bad_png_rejected_by_actual_validator(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "PNG identity/SHA mismatch"):
            self.bad_sha(route.RENDER / "r06-p02-detail.png")

    def test_bad_source_frame_rejected_by_actual_validator(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "source frame changed"):
            self.bad_sha(route.SOURCE_RUNS["dense"]["frames"] / "f_001.png")


if __name__ == "__main__":
    unittest.main()
