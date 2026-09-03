from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .quality_contract import K22_FAILURE_CONTRACT


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SOURCE_SHA_KEYS = (
    "harness_archival_parent",
    "runtime_authority_sha",
    "runtime_compat_sha",
)
_GENERATION_MARKER = "==== STAGE 4: generate ===="
_GENERATE_FAIL_SENTINEL = "GENERATE_FAIL: no result returned (res=None)"
_CLEANUP_POSTLUDE = (
    re.compile(
        r"^\[\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] Generator was garbage collected "
        r"without being shut down\. Attempting to shut down the local server and "
        r"client\.$"
    ),
    re.compile(
        r"^/.+/multiprocessing/resource_tracker\.py:\d+: UserWarning: "
        r"resource_tracker: There appear to be \d+ leaked semaphore objects "
        r"to clean up at shutdown$"
    ),
    re.compile(r"^warnings\.warn\('resource_tracker: There appear to be %d '\)?$"),
)


class K22EvidenceError(ValueError):
    """Raised when K22 did not fail for its one frozen layout reason."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise K22EvidenceError(f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise K22EvidenceError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise K22EvidenceError(f"{label} must be an object")
    return value


def _validate_runtime_source(
    source: Any, expected_source: Mapping[str, Any]
) -> None:
    if not isinstance(source, Mapping) or dict(source) != dict(expected_source):
        raise K22EvidenceError("K22 runtime source does not match frozen evidence")
    if any(not _SHA1.fullmatch(str(source.get(key, ""))) for key in _SOURCE_SHA_KEYS):
        raise K22EvidenceError("K22 runtime source revisions are invalid")
    runtime_root = Path(str(source.get("runtime_root", "")))
    paths = source.get("required_runtime_paths")
    hashes = source.get("critical_file_sha256")
    if (
        not runtime_root.is_absolute()
        or not runtime_root.is_dir()
        or runtime_root.is_symlink()
        or not isinstance(paths, list)
        or not paths
        or not isinstance(hashes, Mapping)
        or set(hashes) != set(paths)
    ):
        raise K22EvidenceError("K22 runtime source inventory is invalid")
    for relative in paths:
        path = Path(relative) if isinstance(relative, str) else Path("..")
        target = runtime_root / path
        if (
            path.is_absolute()
            or ".." in path.parts
            or not target.is_file()
            or target.is_symlink()
            or hashes.get(relative) != _sha256(target)
        ):
            raise K22EvidenceError("K22 critical runtime evidence drifted")


def validate_k22_failure_artifacts(
    output_path: Path | str, *, expected_source: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one real K22 failure without accepting unrelated fatal errors."""

    output = Path(output_path)
    if output.name != "benchmark.json" or (output.parent / "out.mp4").exists():
        raise K22EvidenceError("K22 output artifact layout is invalid")
    benchmark = _regular_json(output, "K22 benchmark")
    run_config = _regular_json(output.parent / "run_config.json", "K22 run configuration")
    run_log = output.parent / "run.log"
    if not run_log.is_file() or run_log.is_symlink():
        raise K22EvidenceError("K22 run log is missing or unsafe")

    raw_log = run_log.read_bytes()
    try:
        transcript = _ANSI_ESCAPE.sub("", raw_log.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise K22EvidenceError("K22 run log is not UTF-8") from exc
    if transcript.count(_GENERATION_MARKER) != 1:
        raise K22EvidenceError("K22 generation boundary is missing or ambiguous")
    generation_log = transcript.split(_GENERATION_MARKER, 1)[1]
    marker = K22_FAILURE_CONTRACT["expected_log_marker"]
    sentinel_count = generation_log.count(_GENERATE_FAIL_SENTINEL)
    before_sentinel = generation_log.rsplit(_GENERATE_FAIL_SENTINEL, 1)[0]
    after_sentinel = generation_log.rsplit(_GENERATE_FAIL_SENTINEL, 1)[-1]
    terminal_lines = [
        line.strip() for line in before_sentinel.splitlines() if line.strip()
    ]
    post_sentinel_lines = [
        line.strip() for line in after_sentinel.splitlines() if line.strip()
    ]
    cleanup_only = all(
        any(pattern.fullmatch(line) for pattern in _CLEANUP_POSTLUDE)
        for line in post_sentinel_lines
    )
    terminal_exception = terminal_lines[-1] if terminal_lines else None
    expected_terminal_exception = f"Exception: Error executing request None: {marker}"
    if (
        sentinel_count != 1
        or terminal_exception != expected_terminal_exception
        or not cleanup_only
    ):
        raise K22EvidenceError("K22 generation terminated for an unexpected reason")

    timeline = benchmark.get("phase_timings", {}).get("marker_timeline")
    timeline_markers = (
        [row.get("marker") for row in timeline if isinstance(row, Mapping)]
        if isinstance(timeline, list)
        else []
    )
    if (
        timeline_markers.count("generation_started") != 1
        or "generation_completed" in timeline_markers
        or "runtime_completed" in timeline_markers
    ):
        raise K22EvidenceError("K22 phase timeline is not a generation failure")

    marker_count = raw_log.count(marker.encode("utf-8"))
    expected_failure = {
        "episode_id": K22_FAILURE_CONTRACT["episode_id"],
        "failure_code": K22_FAILURE_CONTRACT["failure_code"],
        "stage": K22_FAILURE_CONTRACT["stage"],
        "expected_log_marker": marker,
        "observed_marker_count": marker_count,
        "marker_matched": True,
        "generate_fail_sentinel": _GENERATE_FAIL_SENTINEL,
        "generate_fail_sentinel_count": 1,
        "post_sentinel_line_count": len(post_sentinel_lines),
        "post_sentinel_cleanup_only": True,
        "terminal_exception": expected_terminal_exception,
        "terminal_exception_matched": True,
        "child_returncode": K22_FAILURE_CONTRACT["child_returncode"],
        "config_id": K22_FAILURE_CONTRACT["config_id"],
        "config_sha256": K22_FAILURE_CONTRACT["config_sha256"],
        "runtime_ref": K22_FAILURE_CONTRACT["runtime_ref"],
        "run_log": {"path": str(run_log), "sha256": _sha256(run_log)},
    }
    child_returncode = benchmark.get("returncode")
    if (
        benchmark.get("schema_version") != 1
        or benchmark.get("status") != "FAILED"
        or type(child_returncode) is not int
        or child_returncode != K22_FAILURE_CONTRACT["child_returncode"]
        or benchmark.get("generation_s") is not None
        or benchmark.get("total_s") is not None
        or benchmark.get("residual_compute_apps") != []
        or marker_count < 1
        or benchmark.get("failure") != expected_failure
        or run_config.get("schema_version") != 1
        or run_config.get("config_id") != K22_FAILURE_CONTRACT["config_id"]
        or run_config.get("expected_failure_contract")
        != {
            key: K22_FAILURE_CONTRACT[key]
            for key in (
                "episode_id",
                "failure_code",
                "expected_log_marker",
                "config_sha256",
                "runtime_ref",
            )
        }
    ):
        raise K22EvidenceError("K22 failure receipt does not match its contract")
    _validate_runtime_source(run_config.get("source"), expected_source)
    return {
        "benchmark": benchmark,
        "child_returncode": child_returncode,
        "marker_count": marker_count,
        "run_log": run_log,
        "run_config": run_config,
    }
