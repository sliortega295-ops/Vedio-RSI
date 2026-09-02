from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Protocol, Sequence


EXPECTED_FRAMES = 81
EXPECTED_HEIGHT = 480
EXPECTED_WIDTH = 832


class LPIPSRuntimeProtocol(Protocol):
    def read_video(self, path: Path) -> Sequence[Any]: ...

    def score(self, dense_frame: Any, candidate_frame: Any) -> float: ...


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video(path_value: Path | str, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} video must be an absolute regular file")
    return path


def _atomic_write_once(path: Path, content: bytes) -> None:
    if not path.is_absolute():
        raise ValueError("LPIPS output path must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite conflicting LPIPS result: {path}")
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _shape(frame: Any) -> tuple[int, ...]:
    value = getattr(frame, "shape", None)
    if value is None:
        raise ValueError("decoded LPIPS frame has no shape")
    return tuple(int(item) for item in value)


def _torch_frame_input(frame: Any) -> Any:
    """Return a NumPy-compatible value for Torch, including Decord NDArrays."""

    asnumpy = getattr(frame, "asnumpy", None)
    return asnumpy() if callable(asnumpy) else frame


class PyiqaLPIPSRuntime:
    """Offline pyiqa LPIPS-v0.1/AlexNet runtime, loaded only in the child process."""

    def __init__(self) -> None:
        import decord
        import pyiqa
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("formal LPIPS requires one visible CUDA device")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("formal LPIPS requires exactly one CUDA-visible device")
        self._decord = decord
        self._torch = torch
        self._metric = pyiqa.create_metric("lpips", device="cuda")

    def read_video(self, path: Path) -> Sequence[Any]:
        return self._decord.VideoReader(str(path), ctx=self._decord.cpu(0))

    def score(self, dense_frame: Any, candidate_frame: Any) -> float:
        torch = self._torch
        dense = (
            torch.as_tensor(_torch_frame_input(dense_frame), device="cuda")
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .div_(255.0)
        )
        candidate = (
            torch.as_tensor(_torch_frame_input(candidate_frame), device="cuda")
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .div_(255.0)
        )
        with torch.inference_mode():
            return float(self._metric(candidate, dense).item())


def compute_lpips(
    dense_video: Path | str,
    candidate_video: Path | str,
    output_path: Path | str,
    *,
    pair_id: str,
    runtime: LPIPSRuntimeProtocol | None = None,
) -> dict[str, Any]:
    """Measure all 81 aligned decoded frames and durably bind both videos."""

    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("LPIPS pair_id must be nonempty")
    dense_path = _video(dense_video, "dense")
    candidate_path = _video(candidate_video, "candidate")
    backend = runtime if runtime is not None else PyiqaLPIPSRuntime()
    dense_frames = backend.read_video(dense_path)
    candidate_frames = backend.read_video(candidate_path)
    if len(dense_frames) != EXPECTED_FRAMES or len(candidate_frames) != EXPECTED_FRAMES:
        raise ValueError("LPIPS inputs must each decode to exactly 81 frames")

    values: list[float] = []
    for index in range(EXPECTED_FRAMES):
        dense_frame = dense_frames[index]
        candidate_frame = candidate_frames[index]
        expected_shape = (EXPECTED_HEIGHT, EXPECTED_WIDTH, 3)
        if _shape(dense_frame) != expected_shape or _shape(candidate_frame) != expected_shape:
            raise ValueError(
                f"LPIPS frame {index} must have shape {expected_shape}"
            )
        value = float(backend.score(dense_frame, candidate_frame))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"LPIPS frame {index} produced an invalid value")
        values.append(value)

    identity = {
        "schema_version": 1,
        "status": "COMPLETED",
        "metric": "lpips_v0.1_alex",
        "pair_id": pair_id,
        "alignment": "all_corresponding_decoded_frames",
        "frame_count": EXPECTED_FRAMES,
        "frame_shape_hwc": [EXPECTED_HEIGHT, EXPECTED_WIDTH, 3],
        "dense_video": {"path": str(dense_path), "sha256": _sha256(dense_path)},
        "candidate_video": {
            "path": str(candidate_path),
            "sha256": _sha256(candidate_path),
        },
        "values": values,
        "mean": sum(values) / len(values),
        "performance_claim": False,
    }
    payload = {
        **identity,
        "result_fingerprint": hashlib.sha256(_canonical(identity)).hexdigest(),
    }
    _atomic_write_once(
        Path(output_path),
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n",
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rolloutbench-lpips")
    parser.add_argument("--dense-video", type=Path, required=True)
    parser.add_argument("--candidate-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = compute_lpips(
        args.dense_video,
        args.candidate_video,
        args.output,
        pair_id=args.pair_id,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "pair_id": payload["pair_id"],
                "frame_count": payload["frame_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
