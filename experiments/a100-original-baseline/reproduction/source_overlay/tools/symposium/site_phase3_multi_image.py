"""Prepared-only exact R03/R05 image transport for a future Cache attempt.

This is a host-side transport check, not a visual judge. The active goal
session adapter does not import this module until a separately reviewed patch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT = Path("/home/lyy/Experiments/SolAgent/20260913-a100-original-baseline")
COORD = PROJECT / "repo"
RENDER = PROJECT / "artifacts/phase3/cache_image_transport/render_r03_r05_v1"
RENDER_MANIFEST_SHA256 = "a582270a33b97d2c8f5067ac3427ceee30de9dd9cec8dab7e42478323c0b390d"
RENDER_SCRIPT = PROJECT / "scripts/phase3_render_r03_r05_visuals.py"
RENDER_SCRIPT_SHA256 = "07ec7b34bd9fec9cef3152d10977d71aab2153c8bcf3dfdb07b09879f6e7233d"
ATTACHMENTS = PROJECT / "artifacts/phase3/cache_image_transport/attachments"
ATTACHMENT = ATTACHMENTS / "sana2ba100-cache_ca-0002-attempt-10-multi.json"
SESSION = "sana2ba100-cache_ca-0002"
ATTEMPT = 10
ROUNDS = ("r03", "r05")
IMAGE_COUNT = 20
IMAGE_BYTES = 48_381_103
SOURCE_RUNS = {
    "dense": {
        "frames": COORD / "runs/20260913-091006-sana_video_2b_a100_dense/outputs/frames",
        "digest": "728ddacbaa5c9656cf7952e60c835192de70f64a5db8c903f192fd148ed4c964",
    },
    "r03": {
        "frames": (COORD / "output/experiments/sana2ba100-cache_ca-0002/worktree/runs"
                   / "20260914-045221-cache_ca_r03_teacache_t0016/outputs/frames"),
        "digest": "d833d5685e177d565937409d90416d90e733d89b68f3b678617c303712a80f52",
        "receipt": (COORD / "output/experiments/sana2ba100-cache_ca-0002/worktree/runs"
                    / "20260914-045221-cache_ca_r03_teacache_t0016/phase3_quality_attempts"
                    / "20260914T053014.679948+0000_2942501/receipt.json"),
        "receipt_sha256": "26709c701d366e795236b6294187b0dbaaaa4d18168fb6645cdf1da5706dd6c4",
    },
    "r05": {
        "frames": (COORD / "output/experiments/sana2ba100-cache_ca-0002/worktree/runs"
                   / "20260914-054324-cache_ca_r05_taylorseer_t0180_w10_o1/outputs/frames"),
        "digest": "36b9df309d8b3ab3ca47c3e82b42bf3c2da002af493336ac01702d83b6f1e3d0",
        "receipt": (COORD / "output/experiments/sana2ba100-cache_ca-0002/worktree/runs"
                    / "20260914-054324-cache_ca_r05_taylorseer_t0180_w10_o1/phase3_quality_attempts"
                    / "20260914T062035.537017+0000_2975905/receipt.json"),
        "receipt_sha256": "636f7ff9b8d382045dfd41862c96ec9f79412dbf4e939f12a61a1948670d63d7",
    },
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, parent: Path | None = None) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"phase3 multi-image source missing or symlinked: {path}")
    if parent is not None and path.resolve().parent != parent.resolve():
        raise RuntimeError(f"phase3 multi-image path escaped its fixed parent: {path}")


def inputs(name: str, attempt: int, *, allow_prepared: bool = False) -> list[dict]:
    """Return exactly 20 verified files only for approved seq0002 Cache attempt 10."""
    if name != SESSION or attempt != ATTEMPT:
        return []
    if ATTACHMENTS.is_symlink() or RENDER.is_symlink():
        raise RuntimeError("phase3 multi-image directory is symlinked")
    regular(ATTACHMENT, ATTACHMENTS)
    attachment_sha = sha(ATTACHMENT)
    attachment = json.loads(ATTACHMENT.read_text())
    if (set(attachment) != {"schema_version", "status", "session", "attempt", "rounds",
                            "render_manifest_sha256", "image_count", "image_bytes"}
            or attachment["schema_version"] != 2
            or attachment["session"] != SESSION or attachment["attempt"] != ATTEMPT
            or attachment["rounds"] != list(ROUNDS)
            or attachment["render_manifest_sha256"] != RENDER_MANIFEST_SHA256
            or attachment["image_count"] != IMAGE_COUNT
            or attachment["image_bytes"] != IMAGE_BYTES
            or attachment["status"] not in (("approved", "prepared") if allow_prepared
                                          else ("approved",))):
        raise RuntimeError("phase3 multi-image attachment identity/approval mismatch")
    regular(RENDER_SCRIPT)
    if sha(RENDER_SCRIPT) != RENDER_SCRIPT_SHA256:
        raise RuntimeError("phase3 multi-image renderer script SHA mismatch")
    manifest_path = RENDER / "render_manifest.json"
    regular(manifest_path, RENDER)
    if sha(manifest_path) != RENDER_MANIFEST_SHA256:
        raise RuntimeError("phase3 multi-image render manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text())
    expected_order = [(round_id, prompt, kind) for round_id in ROUNDS
                      for prompt in range(1, 6) for kind in ("overview", "detail")]
    rows = manifest.get("images", [])
    if (manifest.get("schema_version") != 1
            or manifest.get("status") != "RENDERED_NOT_ATTACHED_TO_AGENT"
            or manifest.get("target_session_attempt") is not None
            or manifest.get("renderer_script_sha256") != RENDER_SCRIPT_SHA256
            or not isinstance(rows, list) or len(rows) != IMAGE_COUNT
            or [(row.get("round"), row.get("prompt_index"), row.get("kind"))
                for row in rows] != expected_order):
        raise RuntimeError("phase3 multi-image manifest semantic/order mismatch")
    source_runs = manifest.get("source_runs", {})
    if set(source_runs) != set(SOURCE_RUNS):
        raise RuntimeError("phase3 multi-image source round set mismatch")
    for key, spec in SOURCE_RUNS.items():
        recorded = source_runs[key]
        if (recorded.get("frames_digest") != spec["digest"]
                or recorded.get("run") != str(spec["frames"].parent.parent.relative_to(COORD))
                or (key != "dense" and recorded.get("original_judge_receipt_sha256")
                    != spec["receipt_sha256"])):
            raise RuntimeError(f"phase3 multi-image source identity mismatch: {key}")
        if key != "dense":
            receipt = spec["receipt"]
            regular(receipt)
            if sha(receipt) != spec["receipt_sha256"]:
                raise RuntimeError(f"phase3 multi-image judge receipt changed: {key}")
    seen_sources: set[Path] = set()
    images = []
    total = 0
    for row in rows:
        image = RENDER / row["file"]
        regular(image, RENDER)
        if (image.name != f"{row['round']}-p{row['prompt_index']:02d}-{row['kind']}.png"
                or image.stat().st_size != row["size_bytes"]
                or image.stat().st_size > 8 * 1024 * 1024
                or sha(image) != row["sha256"]):
            raise RuntimeError(f"phase3 multi-image PNG identity/SHA mismatch: {image}")
        for tile in row["tiles"]:
            side = tile["side"]
            key = row["round"] if side == "candidate" else "dense"
            if side not in ("candidate", "dense"):
                raise RuntimeError("phase3 multi-image tile side is invalid")
            source = COORD / tile["source"]
            if (source.is_symlink() or source.resolve().parent != SOURCE_RUNS[key]["frames"]
                    or not source.is_file() or sha(source) != tile["source_sha256"]):
                raise RuntimeError(f"phase3 multi-image source frame changed: {source}")
            seen_sources.add(source)
        total += row["size_bytes"]
        images.append({"path": str(image), "sha256": row["sha256"],
                       "round": row["round"], "prompt_index": row["prompt_index"],
                       "kind": row["kind"], "render_manifest_sha256": RENDER_MANIFEST_SHA256,
                       "attachment_manifest_sha256": attachment_sha})
    if total != IMAGE_BYTES or len(seen_sources) != 240:
        raise RuntimeError("phase3 multi-image total bytes or 80-frame coverage mismatch")
    return images
