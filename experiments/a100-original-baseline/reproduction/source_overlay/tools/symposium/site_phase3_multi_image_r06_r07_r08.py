"""Prepared-only, exact Cache attempt-11 R06/R07/R08 image transport.

This host-side site adapter verifies already-rendered candidate/dense pixels and
their sources. It supplies no quality verdict and is not imported by the live
goal broker until separately reviewed and approved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.symposium import site_phase3_multi_image_r06_r07 as prior


PROJECT = Path("/home/lyy/Experiments/SolAgent/20260913-a100-original-baseline")
COORD = PROJECT / "repo"
ATTACHMENTS = PROJECT / "artifacts/phase3/cache_image_transport/attachments"
ATTACHMENT = ATTACHMENTS / "sana2ba100-cache_ca-0002-attempt-11-r06-r07-r08.json"
RENDER = PROJECT / "artifacts/phase3/cache_image_transport/render_r08_v1"
RENDER_SHA256 = "020336e94a679a50caa60ff68ad9070be0481b5c92d649ba1dc872cc640e6d38"
RENDER_SCRIPT = PROJECT / "scripts/phase3_render_r08_visuals.py"
RENDER_SCRIPT_SHA256 = "168850e4945e024b78be23a96c7724578134837c52af3b28f785d4ca838c07ad"
PRIOR_SOURCE = COORD / "tools/symposium/site_phase3_multi_image_r06_r07.py"
PRIOR_SOURCE_SHA256 = "5ba1f67f4bda6cc4000ee7c4d5277ef4df95cc27f3f5b5dc393a8531c50d6de7"
SESSION = "sana2ba100-cache_ca-0002"
ATTEMPT = 11
ROUNDS = ("r06", "r07", "r08")
IMAGE_COUNT = 30
IMAGE_BYTES = 68_830_999
DENSE_FRAMES = COORD / "runs/20260913-091006-sana_video_2b_a100_dense/outputs/frames"
R08_FRAMES = (COORD / "output/experiments/sana2ba100-cache_ca-0002/worktree/runs"
              / "20260914-083326-cache_ca_r08_taylorseer_t0180_w30_o1/outputs/frames")
DENSE_DIGEST = "728ddacbaa5c9656cf7952e60c835192de70f64a5db8c903f192fd148ed4c964"
R08_DIGEST = "586be1f2c9c97bbb2f87ebd43e8d7bae9e2889b7db6680f6bc62ab9b85bab88c"
R08_RECEIPT = (R08_FRAMES.parent.parent / "phase3_quality_attempts"
               / "20260914T091442.857077+0000_3058273/receipt.json")
R08_RECEIPT_SHA256 = "e83d0c0b31e4b5d770154f6633b1e7212291c5df23945d2ca7c114d0e30caa19"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, parent: Path | None = None) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"phase3 attempt11 source missing or symlinked: {path}")
    if parent is not None and path.resolve().parent != parent.resolve():
        raise RuntimeError(f"phase3 attempt11 path escaped fixed parent: {path}")


def frame_hashes(directory: Path, expected_digest: str) -> list[str]:
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError(f"phase3 attempt11 frame directory missing or symlinked: {directory}")
    files = sorted(directory.glob("f_*.png"))
    if [path.name for path in files] != [f"f_{index:03d}.png" for index in range(1, 81)]:
        raise RuntimeError(f"phase3 attempt11 frame coverage changed: {directory}")
    hashes = []
    for path in files:
        regular(path, directory)
        hashes.append(sha(path))
    actual = hashlib.sha256("".join(value + "\n" for value in hashes).encode()).hexdigest()
    if actual != expected_digest:
        raise RuntimeError(f"phase3 attempt11 ordered-frame digest changed: {directory}")
    return hashes


def inputs(name: str, attempt: int, *, allow_prepared: bool = False) -> list[dict]:
    """Return 30 images only after all exact sources and approval are checked."""
    if name != SESSION or attempt != ATTEMPT:
        return []
    if ATTACHMENTS.is_symlink() or RENDER.is_symlink():
        raise RuntimeError("phase3 attempt11 attachment/render directory is symlinked")
    regular(ATTACHMENT, ATTACHMENTS)
    attachment_sha = sha(ATTACHMENT)
    attachment = json.loads(ATTACHMENT.read_text())
    if (set(attachment) != {"schema_version", "status", "session", "attempt", "rounds",
                            "render_manifest_sha256", "image_count", "image_bytes"}
            or attachment["schema_version"] != 3
            or attachment["session"] != SESSION or attachment["attempt"] != ATTEMPT
            or attachment["rounds"] != list(ROUNDS)
            or attachment["render_manifest_sha256"] != {
                "r06_r07": prior.RENDER_MANIFEST_SHA256, "r08": RENDER_SHA256}
            or attachment["image_count"] != IMAGE_COUNT
            or attachment["image_bytes"] != IMAGE_BYTES
            or attachment["status"] not in (("prepared", "approved") if allow_prepared
                                              else ("approved",))):
        raise RuntimeError("phase3 attempt11 attachment identity/approval mismatch")
    regular(PRIOR_SOURCE, COORD / "tools/symposium")
    if Path(prior.__file__).resolve() != PRIOR_SOURCE or sha(PRIOR_SOURCE) != PRIOR_SOURCE_SHA256:
        raise RuntimeError("phase3 attempt11 R06/R07 helper source SHA/import mismatch")
    earlier = prior.inputs(name, attempt, allow_prepared=True)
    if len(earlier) != 20 or [row["round"] for row in earlier] != ["r06"] * 10 + ["r07"] * 10:
        raise RuntimeError("phase3 attempt11 earlier-round image set mismatch")
    regular(RENDER_SCRIPT, PROJECT / "scripts")
    if sha(RENDER_SCRIPT) != RENDER_SCRIPT_SHA256:
        raise RuntimeError("phase3 attempt11 R08 renderer SHA mismatch")
    regular(R08_RECEIPT, R08_RECEIPT.parent)
    if sha(R08_RECEIPT) != R08_RECEIPT_SHA256:
        raise RuntimeError("phase3 attempt11 R08 original judge receipt changed")
    manifest_path = RENDER / "render_manifest.json"
    regular(manifest_path, RENDER)
    if sha(manifest_path) != RENDER_SHA256:
        raise RuntimeError("phase3 attempt11 R08 render manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text())
    expected_order = [("r08", prompt, kind) for prompt in range(1, 6)
                      for kind in ("overview", "detail")]
    rows = manifest.get("images", [])
    if (manifest.get("schema_version") != 1
            or manifest.get("status") != "RENDERED_NOT_ATTACHED_TO_AGENT"
            or manifest.get("target_session_attempt") is not None
            or manifest.get("renderer_script_sha256") != RENDER_SCRIPT_SHA256
            or not isinstance(rows, list) or len(rows) != 10
            or [(row.get("round"), row.get("prompt_index"), row.get("kind"))
                for row in rows] != expected_order
            or manifest.get("source_runs") != {
                "dense": {"frames_digest": DENSE_DIGEST,
                          "run": str(DENSE_FRAMES.parent.parent.relative_to(COORD))},
                "r08": {"frames_digest": R08_DIGEST,
                        "original_judge_receipt_sha256": R08_RECEIPT_SHA256,
                        "run": str(R08_FRAMES.parent.parent.relative_to(COORD))}}):
        raise RuntimeError("phase3 attempt11 R08 render source/order mismatch")
    frame_sets = {"dense": (DENSE_FRAMES, frame_hashes(DENSE_FRAMES, DENSE_DIGEST)),
                  "r08": (R08_FRAMES, frame_hashes(R08_FRAMES, R08_DIGEST))}
    images = []
    seen_sources: set[Path] = set()
    for row in rows:
        image = RENDER / row["file"]
        regular(image, RENDER)
        if (image.name != f"r08-p{row['prompt_index']:02d}-{row['kind']}.png"
                or image.stat().st_size != row["size_bytes"]
                or image.stat().st_size > 8 * 1024 * 1024
                or sha(image) != row["sha256"]):
            raise RuntimeError(f"phase3 attempt11 R08 PNG identity/SHA mismatch: {image}")
        expected_indices = list(range((row["prompt_index"] - 1) * 16 + 1,
                                      row["prompt_index"] * 16 + 1))
        tiles = row.get("tiles", [])
        if row["kind"] == "overview" and (
                len(tiles) != 32 or sorted({tile.get("frame_index") for tile in tiles})
                != expected_indices):
            raise RuntimeError("phase3 attempt11 R08 overview time coverage changed")
        if row["kind"] == "detail" and not (6 <= len(tiles) <= 8):
            raise RuntimeError("phase3 attempt11 R08 detail coverage changed")
        for tile in tiles:
            side = tile.get("side")
            if side not in ("candidate", "dense"):
                raise RuntimeError("phase3 attempt11 R08 tile side invalid")
            index = tile.get("frame_index")
            if index not in expected_indices:
                raise RuntimeError("phase3 attempt11 R08 tile prompt/index mismatch")
            key = "r08" if side == "candidate" else "dense"
            directory, hashes = frame_sets[key]
            source = directory / f"f_{index:03d}.png"
            if (tile.get("source") != str(source.relative_to(COORD))
                    or tile.get("source_sha256") != hashes[index - 1]):
                raise RuntimeError(f"phase3 attempt11 R08 source frame changed: {source}")
            seen_sources.add(source)
        images.append({"path": str(image), "sha256": row["sha256"],
                       "round": "r08", "prompt_index": row["prompt_index"],
                       "kind": row["kind"], "render_manifest_sha256": RENDER_SHA256,
                       "attachment_manifest_sha256": attachment_sha})
    if len(seen_sources) != 160:
        raise RuntimeError("phase3 attempt11 R08 80-frame source coverage incomplete")
    result = [{**row, "attachment_manifest_sha256": attachment_sha} for row in earlier] + images
    if len(result) != IMAGE_COUNT or sum(Path(row["path"]).stat().st_size for row in result) != IMAGE_BYTES:
        raise RuntimeError("phase3 attempt11 total image count/bytes mismatch")
    return result
