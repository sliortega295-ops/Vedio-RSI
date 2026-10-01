#!/usr/bin/env python3
"""Narrow PLAN_EVAL_PYTHON transport for archived Cache verify_delivery.

This is not a general Python executable. It rejects everything except the
archived verifier's one exact lossy reassess argv shape and sends no shell.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


COORD = Path("/home/lyy/Experiments/SolAgent/20260913-a100-original-baseline/repo")
if str(COORD) not in sys.path:
    sys.path.insert(0, str(COORD))
from orchestration import site_job_broker, site_reverify_transport  # noqa: E402


PLAN_EVAL = COORD / "search/plan_eval.py"
PLAN_EVAL_SHA256 = "a06d9e8f845cd269cd80f3b085c5c03675129f145801ed77e9fe1e5d48ad4840"
DENSE_FRAMES = COORD / "runs/20260913-091006-sana_video_2b_a100_dense/outputs/frames"


def parse_exact(argv: list[str]) -> Path:
    if (len(argv) != 10 or argv[:5] != ["search/plan_eval.py", "--model",
                                       "sana_video_2b_a100", "--no-gemini", "--assess"]
            or argv[6] != "--out" or argv[8] != "--baseline-frames"
            or argv[9] != str(DENSE_FRAMES)):
        raise ValueError("site reverify shim rejects non-archived plan_eval arguments")
    run = Path(argv[5])
    if (not run.is_absolute() or run.is_symlink() or run != run.resolve()
            or run.parent != site_reverify_transport.RUNS or not run.is_dir()
            or argv[7] != str(run / "reverify_verdict.json")):
        raise ValueError("site reverify shim requires the exact Cache run/output path")
    if (not PLAN_EVAL.is_file() or PLAN_EVAL.is_symlink()
            or hashlib.sha256(PLAN_EVAL.read_bytes()).hexdigest() != PLAN_EVAL_SHA256):
        raise RuntimeError("archived plan_eval source SHA changed")
    return run


def main() -> int:
    try:
        run = parse_exact(sys.argv[1:])
        result = site_job_broker.request("reassess_quality", run)
        if result.get("status") != "original_remote_reverify_complete":
            raise RuntimeError("trusted reverify route returned a non-complete status")
        print(result.get("original_plan_eval_stdout", ""), end="")
        return int(result["original_plan_eval_exit_code"])
    except Exception as exc:
        # Never print credentials, untrusted payloads, or an Agent-supplied
        # command. The full typed broker failure receipt remains project-local.
        print(json.dumps({"site_reverify_status": "failed_closed",
                          "error_type": type(exc).__name__, "reason": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
