from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from rolloutbench.lpips_cli import compute_lpips


class _Frame:
    shape = (480, 832, 3)


class _Runtime:
    def __init__(self, *, frame_count: int = 81, invalid_at: int | None = None):
        self.frames = [_Frame() for _ in range(frame_count)]
        self.invalid_at = invalid_at
        self.calls = 0

    def read_video(self, _path: Path):
        return self.frames

    def score(self, _dense, _candidate) -> float:
        index = self.calls
        self.calls += 1
        return math.nan if index == self.invalid_at else index / 1000.0


class LPIPSCLITests(unittest.TestCase):
    def test_all_81_frames_and_both_video_hashes_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense = root / "dense.mp4"
            candidate = root / "candidate.mp4"
            output = root / "result.json"
            dense.write_bytes(b"dense")
            candidate.write_bytes(b"candidate")
            payload = compute_lpips(
                dense.resolve(),
                candidate.resolve(),
                output.resolve(),
                pair_id="C02:scene:seed-42",
                runtime=_Runtime(),
            )
            self.assertEqual(81, payload["frame_count"])
            self.assertEqual(81, len(payload["values"]))
            self.assertEqual(
                hashlib.sha256(b"dense").hexdigest(),
                payload["dense_video"]["sha256"],
            )
            self.assertEqual(payload, json.loads(output.read_text()))
            self.assertRegex(payload["result_fingerprint"], r"^[0-9a-f]{64}$")

    def test_wrong_frame_count_or_nonfinite_value_fails_without_output(self) -> None:
        for runtime in (_Runtime(frame_count=80), _Runtime(invalid_at=7)):
            with self.subTest(runtime=runtime.__dict__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dense = root / "dense.mp4"
                candidate = root / "candidate.mp4"
                output = root / "result.json"
                dense.write_bytes(b"dense")
                candidate.write_bytes(b"candidate")
                with self.assertRaises(ValueError):
                    compute_lpips(
                        dense.resolve(),
                        candidate.resolve(),
                        output.resolve(),
                        pair_id="C02:scene:seed-42",
                        runtime=runtime,
                    )
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
