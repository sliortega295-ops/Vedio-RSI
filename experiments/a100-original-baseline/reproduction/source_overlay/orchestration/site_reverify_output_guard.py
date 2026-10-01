"""Staged-only pre-call guard for archived Cache verifier's fixed output path.

The archived verifier loads reverify_verdict.json even when PLAN_EVAL_PYTHON
fails. A previously successful numeric file at that path could therefore be
mistaken for a fresh result. This module is NOT imported by the live broker or
Agent shim. It must be explicitly reviewed against an actual Cache DELIVERY
and approved predecessor before being called by the site coordinator.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from orchestration import site_reverify_transport


RUNS = site_reverify_transport.RUNS


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(run: Path, *, expected_predecessor_sha256: str | None) -> dict:
    """Quarantine an approved stale verdict before any archived reverify call.

    The prior file is moved within the same real run, not deleted. The caller
    must separately pin this receipt and require the fixed output path absent
    before allowing the verifier. This is intentionally not a runnable CLI.
    """
    if (run.is_symlink() or not run.is_dir() or run != run.resolve()
            or run.parent != RUNS or RUNS.is_symlink()):
        raise ValueError("reverify guard requires an exact real Cache run")
    output = run / "reverify_verdict.json"
    if output.is_symlink():
        raise RuntimeError("reverify guard refuses a symlinked verdict")
    present = output.is_file()
    if present != (expected_predecessor_sha256 is not None):
        raise RuntimeError("reverify guard predecessor presence mismatch")
    if present and sha(output) != expected_predecessor_sha256:
        raise RuntimeError("reverify guard predecessor SHA mismatch")
    if expected_predecessor_sha256 is not None and (
            len(expected_predecessor_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_predecessor_sha256)):
        raise ValueError("reverify guard predecessor SHA is not canonical")
    parent = run / "phase3_reverify_preflight"
    if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
        raise RuntimeError("reverify guard archive parent is unsafe")
    parent.mkdir(exist_ok=True)
    attempt = (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
               + "_" + uuid.uuid4().hex)
    archive = parent / attempt
    archive.mkdir(exist_ok=False)
    old = archive / "prior_reverify_verdict.json"
    if present:
        os.replace(output, old)
        if sha(old) != expected_predecessor_sha256:
            raise RuntimeError("reverify guard archived predecessor SHA mismatch")
    if output.exists() or output.is_symlink():
        raise RuntimeError("reverify guard fixed verdict path remains occupied")
    receipt = {
        "schema_version": 1,
        "status": "predecessor_quarantined" if present else "predecessor_absent",
        "run_dir": str(run),
        "fixed_output": str(output),
        "fixed_output_absent": True,
        "predecessor_sha256": expected_predecessor_sha256,
        "archive": str(old) if present else None,
        "archived_sha256": sha(old) if present else None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path = archive / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    with receipt_path.open("rb") as stream:
        os.fsync(stream.fileno())
    return {**receipt, "receipt_path": str(receipt_path), "receipt_sha256": sha(receipt_path)}
