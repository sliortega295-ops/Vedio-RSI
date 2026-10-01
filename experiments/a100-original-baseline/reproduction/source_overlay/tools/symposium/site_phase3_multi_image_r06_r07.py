"""Prepared-only exact R06/R07 image transport for a future Cache attempt.

This is a host-side attachment check, not a visual judge. No active session
adapter imports it; separate review/approval is required before any Agent use.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT = Path("/home/lyy/Experiments/SolAgent/20260913-a100-original-baseline")
COORD = PROJECT / "repo"
RENDER = PROJECT / "artifacts/phase3/cache_image_transport/render_r06_r07_v1"
RENDER_MANIFEST_SHA256 = "09921001565a907ed6c44b1680bba64713929cce776051eb18d99d911ccd3427"
RENDER_SCRIPT = PROJECT / "scripts/phase3_render_r06_r07_visuals.py"
RENDER_SCRIPT_SHA256 = "84a8400830a8fbdc9bfb5856f1bfbe7ca1ba05d8413ea47cebc8d7d61f86160e"
ATTACHMENTS = PROJECT / "artifacts/phase3/cache_image_transport/attachments"
ATTACHMENT = ATTACHMENTS / "sana2ba100-cache_ca-0002-attempt-11-multi.json"
SESSION = "sana2ba100-cache_ca-0002"
ATTEMPT = 11
ROUNDS = ("r06", "r07")
IMAGE_COUNT = 20
IMAGE_BYTES = 45_103_871
SOURCE_RUNS = {
    "dense": {
        "frames": COORD / "runs/20260913-091006-sana_video_2b_a100_dense/outputs/frames",
        "digest": "728ddacbaa5c9656cf7952e60c835192de70f64a5db8c903f192fd148ed4c964",
    },
    "r06": {
        "frames": (COORD / "output/experiments/sana2ba100-cache_ca-0002/worktree/runs"
                   / "20260914-063907-cache_ca_r06_easycache_t010_w30/outputs/frames"),
        "digest": "4c9f96ea9e749688ea335e705731b3b41c7460313902edae800824fdd670717c",
        "receipt": (COORD / "output/experiments/sana2ba100-cache_ca-0002/worktree/runs"
                    / "20260914-063907-cache_ca_r06_easycache_t010_w30/phase3_quality_attempts"
                    / "20260914T071512.599598+0000_3001916/receipt.json"),
        "receipt_sha256": "516e807b5ad862eaec3da2d04be4297c2c79bc62e3eb76286a6382aa8f22aaba",
    },
    "r07": {
        "frames": (COORD / "output/experiments/sana2ba100-cache_ca-0002/worktree/runs"
                   / "20260914-072228-cache_ca_r07_teacache_t00087/outputs/frames"),
        "digest": "46faed6d9430a46aad6c7986cd882c55673d7b07e0fd81c018204e3b91cc4b54",
        "receipt": (COORD / "output/experiments/sana2ba100-cache_ca-0002/worktree/runs"
                    / "20260914-072228-cache_ca_r07_teacache_t00087/phase3_quality_attempts"
                    / "20260914T080119.530175+0000_3026250/receipt.json"),
        "receipt_sha256": "f275f3740a99a2367d1a2d7dae04e6e19b3c9c5408fea196072cfbc33c521356",
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
        raise RuntimeError(f"phase3 R06/R07 source missing or symlinked: {path}")
    if parent is not None and path.resolve().parent != parent.resolve():
        raise RuntimeError(f"phase3 R06/R07 path escaped its fixed parent: {path}")


def inputs(name: str, attempt: int, *, allow_prepared: bool = False) -> list[dict]:
    """Return only the 20 checked files for this exact, approved later attempt."""
    if name != SESSION or attempt != ATTEMPT:
        return []
    if ATTACHMENTS.is_symlink() or RENDER.is_symlink():
        raise RuntimeError("phase3 R06/R07 directory is symlinked")
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
        raise RuntimeError("phase3 R06/R07 attachment identity/approval mismatch")
    regular(RENDER_SCRIPT)
    if sha(RENDER_SCRIPT) != RENDER_SCRIPT_SHA256:
        raise RuntimeError("phase3 R06/R07 renderer SHA mismatch")
    manifest_path = RENDER / "render_manifest.json"
    regular(manifest_path, RENDER)
    if sha(manifest_path) != RENDER_MANIFEST_SHA256:
        raise RuntimeError("phase3 R06/R07 render manifest SHA mismatch")
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
        raise RuntimeError("phase3 R06/R07 manifest semantic/order mismatch")
    source_runs = manifest.get("source_runs", {})
    if set(source_runs) != set(SOURCE_RUNS):
        raise RuntimeError("phase3 R06/R07 source round set mismatch")
    for key, spec in SOURCE_RUNS.items():
        recorded = source_runs[key]
        if (recorded.get("frames_digest") != spec["digest"]
                or recorded.get("run") != str(spec["frames"].parent.parent.relative_to(COORD))
                or (key != "dense" and recorded.get("original_judge_receipt_sha256")
                    != spec["receipt_sha256"])):
            raise RuntimeError(f"phase3 R06/R07 source identity mismatch: {key}")
        if key != "dense":
            regular(spec["receipt"])
            if sha(spec["receipt"]) != spec["receipt_sha256"]:
                raise RuntimeError(f"phase3 R06/R07 judge receipt changed: {key}")
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
            raise RuntimeError(f"phase3 R06/R07 PNG identity/SHA mismatch: {image}")
        for tile in row["tiles"]:
            side = tile["side"]
            key = row["round"] if side == "candidate" else "dense"
            if side not in ("candidate", "dense"):
                raise RuntimeError("phase3 R06/R07 tile side is invalid")
            source = COORD / tile["source"]
            if (source.is_symlink() or source.resolve().parent != SOURCE_RUNS[key]["frames"]
                    or not source.is_file() or sha(source) != tile["source_sha256"]):
                raise RuntimeError(f"phase3 R06/R07 source frame changed: {source}")
            seen_sources.add(source)
        total += row["size_bytes"]
        images.append({"path": str(image), "sha256": row["sha256"],
                       "round": row["round"], "prompt_index": row["prompt_index"],
                       "kind": row["kind"], "render_manifest_sha256": RENDER_MANIFEST_SHA256,
                       "attachment_manifest_sha256": attachment_sha})
    if total != IMAGE_BYTES or len(seen_sources) != 240:
        raise RuntimeError("phase3 R06/R07 total bytes or 80-frame coverage mismatch")
    return images
