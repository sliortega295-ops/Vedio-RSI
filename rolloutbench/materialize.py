from __future__ import annotations

import fcntl
import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_AUTHORITY_REF_RE = re.compile(r"[0-9a-f]{7,64}")


class AuthorityMaterializationError(RuntimeError):
    """Raised when a public authority object cannot be materialized exactly."""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _required_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AuthorityMaterializationError(f"{label} is missing")
    return value


def _safe_source_path(value: Any) -> str:
    path = _required_string(value, label="authority path")
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise AuthorityMaterializationError("authority path must be a safe relative path")
    return path


def _read_authority_blob(repository: Path, authority_ref: str, path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{authority_ref}:{path}"],
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuthorityMaterializationError(
            f"authority object is unavailable: {authority_ref}:{path}"
        ) from exc


def _validate_descriptor(item: Any, *, label: str) -> tuple[str, str]:
    if not isinstance(item, Mapping):
        raise AuthorityMaterializationError(f"{label} descriptor is missing")
    path = _safe_source_path(item.get("path"))
    digest = _required_string(item.get("blob_sha256"), label=f"{label} blob SHA-256")
    if not _SHA256_RE.fullmatch(digest):
        raise AuthorityMaterializationError(f"{label} blob SHA-256 is invalid")
    if item.get("hash_scope") != "raw_git_blob_bytes":
        raise AuthorityMaterializationError(f"{label} hash scope is invalid")
    reported = item.get("authority_reported_sha256")
    if reported is not None and reported != digest:
        raise AuthorityMaterializationError(f"{label} declared SHA-256 values disagree")
    return path, digest


def _atomic_write_exact(target: Path, content: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / f".{target.name}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if target.exists():
                if not target.is_file() or target.read_bytes() != content:
                    raise AuthorityMaterializationError(f"refusing overwrite of {target}")
                return
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=target.parent, prefix=f".{target.name}.", delete=False
                ) as handle:
                    temporary = Path(handle.name)
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
                temporary = None
                directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _contained_target(root: Path, relative: Path) -> Path:
    cursor = root
    for part in relative.parts[:-1]:
        cursor /= part
        if cursor.is_symlink():
            raise AuthorityMaterializationError(
                f"derived artifact parent is a symlink: {cursor}"
            )
        if cursor.exists() and not cursor.is_dir():
            raise AuthorityMaterializationError(
                f"derived artifact parent is not a directory: {cursor}"
            )
    target = root / relative
    if target.is_symlink():
        raise AuthorityMaterializationError(
            f"derived artifact target is a symlink: {target}"
        )
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise AuthorityMaterializationError(
            f"derived artifact escapes its root: {target}"
        ) from exc
    return target


def materialize_candidate_artifacts(
    episode: Mapping[str, Any],
    derived_root: Path | str,
    *,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Materialize public candidate config and probe source from frozen authority blobs."""

    if "golden" in episode:
        raise AuthorityMaterializationError("materializer accepts only a public descriptor")
    episode_id = _required_string(episode.get("episode_id"), label="episode_id")
    candidate = episode.get("candidate")
    if not isinstance(candidate, Mapping):
        raise AuthorityMaterializationError("candidate descriptor is missing")
    authority_ref = _required_string(candidate.get("authority_ref"), label="authority_ref").lower()
    if not _AUTHORITY_REF_RE.fullmatch(authority_ref):
        raise AuthorityMaterializationError("authority_ref is invalid")

    objects: list[tuple[str, str, str]] = []
    config = candidate.get("config")
    if config is not None:
        path, digest = _validate_descriptor(config, label="config")
        objects.append(("config", path, digest))
    probe = candidate.get("probe")
    if probe is not None:
        if not isinstance(probe, Mapping):
            raise AuthorityMaterializationError("probe descriptor is invalid")
        path, digest = _validate_descriptor(probe.get("source"), label="probe source")
        objects.append(("probe_source", path, digest))

    repository = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    root = Path(derived_root).resolve()
    artifacts: list[dict[str, Any]] = []
    for kind, source_path, declared_digest in objects:
        content = _read_authority_blob(repository, authority_ref, source_path)
        actual_digest = _sha256(content)
        if actual_digest != declared_digest:
            raise AuthorityMaterializationError(
                f"SHA-256 mismatch for {authority_ref}:{source_path}"
            )
        relative_path = Path(episode_id) / kind / source_path
        _atomic_write_exact(_contained_target(root, relative_path), content)
        artifacts.append(
            {
                "kind": kind,
                "source_path": source_path,
                "relative_path": relative_path.as_posix(),
                "sha256": actual_digest,
                "size_bytes": len(content),
            }
        )
    if not artifacts:
        raise AuthorityMaterializationError("candidate descriptor has no materializable authority object")
    return {
        "schema_version": 1,
        "episode_id": episode_id,
        "authority_ref": authority_ref,
        "artifacts": artifacts,
    }
