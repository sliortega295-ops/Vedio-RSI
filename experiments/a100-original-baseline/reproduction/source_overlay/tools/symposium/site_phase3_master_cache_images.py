"""STAGED, not imported: exact Master Cache-frontier independent image route.

No Master attachment manifest exists yet. A real Cache DELIVERY must first name
every frontier point, after which the coordinator may review and approve one
hash-pinned image manifest for one future Master attempt. This supplies pixels,
not an outer visual opinion or an objective quality verdict.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.symposium import site_phase3_multi_image as duo
from tools.symposium import site_phase3_multi_image_r06_r07_r08 as triple


PROJECT = Path("/home/lyy/Experiments/SolAgent/20260913-a100-original-baseline")
COORD = PROJECT / "repo"
CACHE = COORD / "output/experiments/sana2ba100-cache_ca-0002/worktree"
DELIVERY = CACHE / "DELIVERY.json"
ATTACHMENTS = PROJECT / "artifacts/phase3/cache_image_transport/attachments"
ATTACHMENTS_BY_ATTEMPT = {
    12: ATTACHMENTS / "sana2ba100-master-0002-attempt-12-cache.json",
    13: ATTACHMENTS / "sana2ba100-master-0002-attempt-13-cache.json",
}
SESSION = "sana2ba100-master-0002"
DUO_SOURCE = COORD / "tools/symposium/site_phase3_multi_image.py"
DUO_SOURCE_SHA256 = "d6f145a19b45c9f7a7295daa89d49d2e96b5c3bca7141c8a85136f19ecb036bd"
TRIPLE_SOURCE = COORD / "tools/symposium/site_phase3_multi_image_r06_r07_r08.py"
TRIPLE_SOURCE_SHA256 = "05f5c6a99b0b0f06b565ce7f1984a1e07a3eea6618c1c21a9e682f03a0c2e61d"
CACHE_SESSION = "sana2ba100-cache_ca-0002"
KNOWN_RUNS = {
    "r03": "runs/20260914-045221-cache_ca_r03_teacache_t0016",
    "r05": "runs/20260914-054324-cache_ca_r05_taylorseer_t0180_w10_o1",
    "r06": "runs/20260914-063907-cache_ca_r06_easycache_t010_w30",
    "r07": "runs/20260914-072228-cache_ca_r07_teacache_t00087",
    "r08": "runs/20260914-083326-cache_ca_r08_taylorseer_t0180_w30_o1",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, parent: Path | None = None) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Master image source missing/symlinked: {path}")
    if parent is not None and path.resolve().parent != parent.resolve():
        raise RuntimeError(f"Master image source escaped fixed parent: {path}")


def inputs(name: str, attempt: int, *, allow_prepared: bool = False) -> list[dict]:
    if name != SESSION or attempt not in ATTACHMENTS_BY_ATTEMPT:
        return []
    attachment_path = ATTACHMENTS_BY_ATTEMPT[attempt]
    if ATTACHMENTS.is_symlink() or CACHE.is_symlink():
        raise RuntimeError("Master image attachment/Cache worktree directory is symlinked")
    regular(attachment_path, ATTACHMENTS)
    attachment_sha = sha(attachment_path)
    attachment = json.loads(attachment_path.read_text())
    if (set(attachment) != {"schema_version", "status", "session", "attempt",
                            "cache_delivery_sha256", "points", "image_count", "image_bytes"}
            or attachment["schema_version"] != 1
            or attachment["session"] != SESSION or attachment["attempt"] != attempt
            or attachment["status"] not in (("prepared", "approved") if allow_prepared
                                              else ("approved",))):
        raise RuntimeError("Master image attachment identity/approval mismatch")
    regular(DELIVERY, CACHE)
    if sha(DELIVERY) != attachment["cache_delivery_sha256"]:
        raise RuntimeError("Master image Cache DELIVERY SHA changed")
    delivery = json.loads(DELIVERY.read_text())
    if (delivery.get("schema_version") != 2 or delivery.get("status") != "complete"
            or delivery.get("component") != "cache"
            or delivery.get("model_id") != "sana_video_2b_a100"):
        raise RuntimeError("Master image Cache DELIVERY identity invalid")
    delivered = delivery.get("frontier_points")
    if not isinstance(delivered, list) or not delivered:
        raise RuntimeError("Master image Cache DELIVERY frontier missing")
    run_dirs = [point.get("run_dir") if isinstance(point, dict) else None for point in delivered]
    points = attachment["points"]
    if (not isinstance(points, list) or not points
            or any(not isinstance(point, dict) or set(point) != {"round", "run_dir"}
                   for point in points)
            or [point["run_dir"] for point in points] != run_dirs
            or len(set(run_dirs)) != len(run_dirs)
            or any(KNOWN_RUNS.get(point["round"]) != point["run_dir"] for point in points)):
        raise RuntimeError("Master image point list is not the exact full delivered frontier")
    for path, expected, module in ((DUO_SOURCE, DUO_SOURCE_SHA256, duo),
                                   (TRIPLE_SOURCE, TRIPLE_SOURCE_SHA256, triple)):
        regular(path, COORD / "tools/symposium")
        if Path(module.__file__).resolve() != path or sha(path) != expected:
            raise RuntimeError("Master image source helper SHA/import changed")
    needed = {point["round"] for point in points}
    sources: dict[str, list[dict]] = {}
    if needed.intersection(("r03", "r05")):
        rows = duo.inputs(CACHE_SESSION, 10)
        sources.update({round_id: [row for row in rows if row["round"] == round_id]
                        for round_id in ("r03", "r05")})
    if needed.intersection(("r06", "r07", "r08")):
        rows = triple.inputs(CACHE_SESSION, 11)
        sources.update({round_id: [row for row in rows if row["round"] == round_id]
                        for round_id in ("r06", "r07", "r08")})
    images = []
    for point in points:
        rows = sources.get(point["round"], [])
        if (len(rows) != 10
                or [(row["prompt_index"], row["kind"]) for row in rows]
                != [(prompt, kind) for prompt in range(1, 6)
                    for kind in ("overview", "detail")]):
            raise RuntimeError("Master image delivered point has incomplete five-prompt coverage")
        for row in rows:
            images.append({**row,
                           "source_attachment_manifest_sha256": row["attachment_manifest_sha256"],
                           "attachment_manifest_sha256": attachment_sha,
                           "delivered_run_dir": point["run_dir"]})
    total = sum(Path(row["path"]).stat().st_size for row in images)
    if len(images) != attachment["image_count"] or total != attachment["image_bytes"]:
        raise RuntimeError("Master image attachment count/bytes mismatch")
    return images
