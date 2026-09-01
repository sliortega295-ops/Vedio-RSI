from __future__ import annotations

import fcntl
import functools
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .runtime_manifest import COMMON_RUNTIME_FILES, build_runtime_manifest

CRITICAL_RUNTIME_FILES = COMMON_RUNTIME_FILES
_RUNTIME_ROOT = Path("external/sol_runtime")


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def _registered_worktrees(repository: Path) -> set[Path]:
    output = _git(repository, "worktree", "list", "--porcelain")
    return {
        Path(line.removeprefix("worktree ")).resolve()
        for line in output.splitlines()
        if line.startswith("worktree ")
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@functools.lru_cache(maxsize=128)
def _critical_hashes(
    checkout: str, runtime_tree_oid: str, required_paths: tuple[str, ...]
) -> tuple[tuple[str, str], ...]:
    del runtime_tree_oid  # immutable Git tree identity is the cache invalidator
    target = Path(checkout)
    hashes: list[tuple[str, str]] = []
    for relative in required_paths:
        path = target / _RUNTIME_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"critical runtime file is missing from checkout: {relative}")
        hashes.append((relative, _sha256(path)))
    return tuple(hashes)


def _validate_existing_checkout(
    repository: Path,
    target: Path,
    commit: str,
    expected_tree_oid: str,
    required_paths: tuple[str, ...],
) -> tuple[str, dict[str, str]]:
    if target not in _registered_worktrees(repository):
        raise RuntimeError(f"existing target is not a registered Git worktree: {target}")
    status = _git(target, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(f"existing candidate worktree is not clean: {target}")

    symbolic_ref = subprocess.run(
        ["git", "-C", str(target), "symbolic-ref", "-q", "HEAD"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if symbolic_ref.returncode == 0:
        raise RuntimeError(f"existing candidate worktree is not detached: {target}")
    if symbolic_ref.returncode != 1:
        raise RuntimeError(f"could not determine detached HEAD state: {target}")
    if _git(target, "rev-parse", "--verify", "HEAD^{commit}") != commit:
        raise RuntimeError(f"existing candidate worktree HEAD does not match requested commit: {target}")

    runtime_tree_oid = _git(target, "rev-parse", f"HEAD:{_RUNTIME_ROOT.as_posix()}")
    if runtime_tree_oid != expected_tree_oid:
        raise RuntimeError("candidate runtime tree disagrees with the authority manifest")
    hashes = dict(
        _critical_hashes(str(target), runtime_tree_oid, required_paths)
    )
    return runtime_tree_oid, hashes


def verify_runtime_receipt(
    repository_root: Path | str,
    receipt: Mapping[str, Any],
    expected_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate a receipt's live worktree against its planned runtime manifest."""

    repository = Path(repository_root).resolve()
    runtime_ref = expected_manifest.get("git_ref")
    ref_role = expected_manifest.get("ref_role")
    runtime_tree_oid = expected_manifest.get("runtime_tree_oid")
    paths_value = expected_manifest.get("required_runtime_paths")
    if (
        not isinstance(runtime_ref, str)
        or not isinstance(ref_role, str)
        or not isinstance(runtime_tree_oid, str)
        or not isinstance(paths_value, list)
        or any(not isinstance(path, str) for path in paths_value)
    ):
        raise RuntimeError("planned runtime manifest is incomplete")
    required_paths = tuple(paths_value)
    if (
        receipt.get("status") != "READY"
        or receipt.get("runtime_ref") != runtime_ref
        or receipt.get("ref_role") != ref_role
        or receipt.get("runtime_tree_oid") != runtime_tree_oid
        or receipt.get("required_runtime_paths") != paths_value
    ):
        raise RuntimeError("runtime receipt disagrees with the planned runtime manifest")
    worktree_path = receipt.get("worktree_path")
    if not isinstance(worktree_path, str) or not worktree_path:
        raise RuntimeError("runtime receipt has no worktree path")
    target = Path(worktree_path).resolve()
    actual_tree_oid, actual_hashes = _validate_existing_checkout(
        repository, target, runtime_ref, runtime_tree_oid, required_paths
    )
    if receipt.get("critical_runtime_file_sha256") != actual_hashes:
        raise RuntimeError("runtime receipt critical file hashes are stale or corrupt")
    return {
        "schema_version": 1,
        "status": "READY",
        "runtime_ref": runtime_ref,
        "ref_role": ref_role,
        "worktree_path": str(target),
        "runtime_tree_oid": actual_tree_oid,
        "required_runtime_paths": paths_value,
        "critical_runtime_file_sha256": actual_hashes,
    }


def prepare_runtime_checkout(
    episode: Mapping[str, Any], repository_root: Path | str, worktree_root: Path | str
) -> dict[str, Any]:
    """Create or validate a clean detached candidate worktree under a root lock."""

    repository = Path(repository_root).resolve()
    root = Path(worktree_root).resolve()
    if "golden" in episode:
        raise ValueError("runtime checkout accepts a public episode without golden")
    candidate = episode.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("candidate descriptor is missing")
    manifest = build_runtime_manifest(repository, candidate)
    planned = episode.get("runtime_checkout")
    if isinstance(planned, Mapping):
        comparable = {
            key: planned.get(key)
            for key in (
                "git_ref",
                "ref_role",
                "runtime_tree_oid",
                "required_runtime_paths",
            )
        }
        if comparable != manifest:
            raise RuntimeError("planned runtime manifest disagrees with Git authority")
    commit = str(manifest["git_ref"])
    ref_role = str(manifest["ref_role"])
    required_paths = tuple(str(path) for path in manifest["required_runtime_paths"])
    target = root / commit
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".runtime_checkout.lock").open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if not target.exists():
                _git(repository, "worktree", "add", "--detach", str(target), commit)
            runtime_tree_oid, hashes = _validate_existing_checkout(
                repository,
                target,
                commit,
                str(manifest["runtime_tree_oid"]),
                required_paths,
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return {
        "schema_version": 1,
        "status": "READY",
        "runtime_ref": commit,
        "ref_role": ref_role,
        "worktree_path": str(target),
        "runtime_tree_oid": runtime_tree_oid,
        "required_runtime_paths": list(required_paths),
        "critical_runtime_file_sha256": hashes,
    }
