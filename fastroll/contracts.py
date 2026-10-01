"""Strict plans, immutable source identities and artifact receipts (stdlib only)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROLES = {"primary", "generate", "reference_generate", "reference_score", "score", "validate", "build"}
CACHE_ROLES = {"reference_generate", "reference_score"}
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")
TERMINAL = {"succeeded", "failed", "cancelled", "blocked"}


class ContractError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ContractError(x)))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(canonical(value) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def within(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ContractError(f"path escapes its allowed root: {path}")
    return resolved


def pin_source(root: Path) -> dict:
    """Pin every tracked file; reject untracked source rather than silently omit it."""
    root = root.resolve(strict=True)
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ContractError("source snapshot must have a clean worktree, including untracked files")
    files = git("ls-files", "-z").split("\0")
    pins = {}
    for name in filter(None, files):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ContractError("symlinks/submodules require a separate resolved source snapshot")
        pins[name] = sha256(path)
    if not pins:
        raise ContractError("source snapshot is empty")
    return {"root": str(root), "commit": git("rev-parse", "HEAD"), "files": pins}


def verify_source(source: dict) -> None:
    root = Path(source["root"])
    for name, expected in source["files"].items():
        path = within(root / name, root)
        if path.is_symlink() or sha256(path) != expected:
            raise ContractError(f"source changed: {name}")
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True)
    tracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"], text=True)
    if head != source["commit"] or dirty or set(filter(None, tracked.split("\0"))) != set(source["files"]):
        raise ContractError("source commit, file inventory, or worktree changed")


def validate_plan(raw: dict) -> dict:
    p = json.loads(canonical(raw))
    if set(p) != {"schema_version", "batch_id", "tasks"} or p["schema_version"] != 1:
        raise ContractError("plan requires schema_version=1, batch_id and tasks only")
    if not isinstance(p["batch_id"], str) or not ID.fullmatch(p["batch_id"]):
        raise ContractError("invalid batch_id")
    tasks = p["tasks"]
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 10000:
        raise ContractError("tasks must contain 1..10000 entries")
    seen = set()
    for t in tasks:
        allowed = {"id", "role", "resource", "session", "payload", "depends_on", "inputs", "priority", "estimate_s", "timeout_s", "retries", "mode", "reuse", "min_free_mib", "device"}
        if not isinstance(t, dict) or set(t) - allowed:
            raise ContractError("unknown task field")
        for k in ("id", "role", "resource", "session", "payload"):
            if k not in t:
                raise ContractError(f"missing task field: {k}")
        if not isinstance(t["id"], str) or not ID.fullmatch(t["id"]) or t["id"] in seen:
            raise ContractError("invalid or duplicate task id")
        seen.add(t["id"])
        if t["role"] not in ROLES or t["resource"] not in {"cpu", "gpu"}:
            raise ContractError("invalid role/resource")
        defaults = dict(depends_on=[], inputs={}, priority=0, estimate_s=60.0, timeout_s=1800.0, retries=0, mode="oneshot", reuse=False, min_free_mib=0, device=None)
        for k, v in defaults.items():
            t.setdefault(k, v)
        for k in ("estimate_s", "timeout_s"):
            if type(t[k]) not in (int, float) or not math.isfinite(t[k]) or t[k] <= 0:
                raise ContractError(f"invalid {k}")
        for k in ("priority", "retries", "min_free_mib"):
            if type(t[k]) is not int or (k != "priority" and t[k] < 0):
                raise ContractError(f"invalid {k}")
        if t["retries"] > 3 or t["mode"] not in {"oneshot", "resident"} or type(t["reuse"]) is not bool:
            raise ContractError("invalid retry/mode/reuse")
        if t["reuse"] and t["role"] not in CACHE_ROLES:
            raise ContractError("only reference generation/scoring can be reused; never candidate latency or verdicts")
        if t["role"] == "primary" and t["mode"] != "oneshot":
            raise ContractError("primary timing must remain isolated one-shot")
        if not isinstance(t["payload"], dict) or not isinstance(t["inputs"], dict):
            raise ContractError("payload/inputs must be objects")
        if not isinstance(t["depends_on"], list) or any(not isinstance(x, str) for x in t["depends_on"]) or len(set(t["depends_on"])) != len(t["depends_on"]):
            raise ContractError("invalid dependencies")
        for name, binding in t["inputs"].items():
            if not ID.fullmatch(name) or not isinstance(binding, dict) or set(binding) != {"task", "artifact"}:
                raise ContractError("input must bind a named predecessor artifact")
            if binding["task"] not in t["depends_on"]:
                raise ContractError("input task must be a direct dependency")
        s = t["session"]
        if not isinstance(s, dict) or set(s) - {"backend", "config", "source", "python", "assets"}:
            raise ContractError("invalid session")
        if s.get("backend") not in {"demo", "command", "python", "sana", "compare"}:
            raise ContractError("unknown backend")
        s.setdefault("config", {})
        if not isinstance(s["config"], dict):
            raise ContractError("session config must be an object")
        if s["backend"] == "command" and t["mode"] != "oneshot":
            raise ContractError("command adapters cannot pretend to be resident")
        if s["backend"] == "compare" and (t["resource"] != "cpu" or t["mode"] != "oneshot"):
            raise ContractError("compare runs one-shot on CPU")
        if s["backend"] in {"command", "python", "sana"}:
            src = s.get("source", {})
            if set(src) != {"root", "commit", "files"} or not Path(src["root"]).is_absolute() or not re.fullmatch(r"[0-9a-f]{40}", src["commit"]):
                raise ContractError("real adapters require an immutable pinned Git source")
            if not src["files"] or any(not isinstance(k, str) or not SHA.fullmatch(v) for k, v in src["files"].items()):
                raise ContractError("invalid source pins")
        s.setdefault("assets", [])
        if not isinstance(s["assets"], list):
            raise ContractError("session assets must be snapshot manifests")
        for asset in s["assets"]:
            if not isinstance(asset, dict) or set(asset) != {"root", "files"} or not Path(asset["root"]).is_absolute() or not asset["files"]:
                raise ContractError("invalid asset manifest")
            if any(not SHA.fullmatch(v) for v in asset["files"].values()):
                raise ContractError("invalid asset digest")
        if s["backend"] == "sana" and not any(str(Path(a["root"]).resolve()) == str(Path(s["config"].get("model_path", "")).resolve()) for a in s["assets"]):
            raise ContractError("SANA model_path must be covered by a complete asset manifest")
        if t["device"] is not None and (t["resource"] != "gpu" or not isinstance(t["device"], str)):
            raise ContractError("device affinity requires a GPU UUID")
    graph = {t["id"]: set(t["depends_on"]) for t in tasks}
    if any(ds - seen for ds in graph.values()):
        raise ContractError("missing dependency")
    done = set()
    while len(done) < len(graph):
        ready = {k for k, ds in graph.items() if k not in done and ds <= done}
        if not ready:
            raise ContractError("dependency cycle")
        done |= ready
    return p


def artifact_receipt(path: Path, root: Path) -> dict:
    path = within(path, root)
    if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
        raise ContractError("artifact must be a nonempty regular file")
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def verify_result(result: dict, root: Path) -> None:
    canonical(result)
    if not isinstance(result.get("metrics"), dict) or not isinstance(result.get("artifacts"), dict):
        raise ContractError("result requires metrics and artifacts")
    for a in result["artifacts"].values():
        if artifact_receipt(Path(a["path"]), root) != a:
            raise ContractError("artifact receipt changed")

# Source/model snapshots are trusted operator inputs, not mutable Agent state.
# Cache file hashes only while size, inode, mtime AND ctime remain unchanged.
# This is for a trusted local filesystem, not protection against its administrator.
_ASSET_HASHES: dict[str, tuple] = {}


def pin_assets(root: Path) -> dict:
    root = root.resolve(strict=True)
    files = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}
    if not files:
        raise ContractError("asset snapshot is empty")
    return {"root": str(root), "files": files}


def verify_assets(manifests: list[dict]) -> None:
    for manifest in manifests:
        root = Path(manifest["root"]).resolve(strict=True)
        files = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}
        if set(files) != set(manifest["files"]):
            raise ContractError("asset inventory changed")
        for name, path in files.items():
            expected = manifest["files"][name]
            stat = path.stat()
            identity = (expected, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            key = str(path)
            if _ASSET_HASHES.get(key) != identity:
                if sha256(path) != expected:
                    raise ContractError(f"asset changed: {name}")
                _ASSET_HASHES[key] = identity
