#!/usr/bin/env python3
"""Length-framed, SHA-checked project-local transfer over the site's PTY SSH.

The gateway requires a PTY, which makes rsync's binary protocol unusable. This
helper transports base64 tar archives and rejects unsafe archive members before
extracting any file. It has no network or GPU behavior of its own.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path, PurePosixPath

LOCAL_PROJECT = Path("/home/lyy/Experiments/SolAgent/20260913-a100-original-baseline")
REMOTE_PROJECT = Path("/share/project/shuchenweng/yongyan_liu/Experiments/SolAgent/20260913-a100-original-baseline")


def project() -> Path:
    return REMOTE_PROJECT if REMOTE_PROJECT.is_dir() else LOCAL_PROJECT


def allowed(path: Path) -> Path:
    root = project().resolve()
    target = path.resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError(f"transfer target outside exact experiment project: {target}")
    return target


def pack(source: Path, excludes: list[str]) -> bytes:
    source = allowed(source)
    if not source.is_dir():
        raise ValueError(f"source is not a directory: {source}")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(source.rglob("*")):
            rel = path.relative_to(source)
            if any(part in rel.parts or rel == Path(part) for part in excludes):
                continue
            if path.is_symlink():
                # Never export a link into the receiving project. Unrelated
                # archived model profiles contain links not used by this run.
                continue
            if path.is_file() or path.is_dir():
                archive.add(path, arcname=rel.as_posix(), recursive=False)
    return buffer.getvalue()


def unpack(archive_bytes: bytes, dest: Path) -> dict:
    dest = allowed(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or not name.parts or any(part in ("", "..") for part in name.parts):
                raise ValueError(f"unsafe archive member name: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"archive member is not a regular file/directory: {member.name}")
            path = dest / name
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"archive member escapes destination: {member.name}")
            for parent in (path, *path.parents):
                if parent == root.parent:
                    break
                if parent.is_symlink():
                    raise ValueError(f"archive member crosses a symlink: {member.name}")
        archive.extractall(dest, members=members)
    return {"member_count": len(members), "dest": str(dest)}


def recv(dest: Path, encoded_length: int, expected_sha256: str) -> dict:
    if encoded_length <= 0 or encoded_length > 2_000_000_000:
        raise ValueError("invalid transfer length")
    encoded = bytearray()
    while len(encoded) < encoded_length:
        chunk = sys.stdin.buffer.read(encoded_length - len(encoded))
        if not chunk:
            raise EOFError("PTY closed before the framed transfer completed")
        encoded.extend(chunk)
    archive = base64.b64decode(encoded)
    observed = hashlib.sha256(archive).hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"transfer SHA256 mismatch: {observed} != {expected_sha256}")
    result = unpack(archive, dest)
    return {**result, "archive_bytes": len(archive), "archive_sha256": observed,
            "status": "received_verified"}


def send(source: Path, excludes: list[str]) -> None:
    archive = pack(source, excludes)
    encoded = base64.b64encode(archive)
    digest = hashlib.sha256(archive).hexdigest()
    sys.stdout.buffer.write(f"SOL_ARCHIVE {len(encoded)} {digest}\n".encode())
    sys.stdout.buffer.write(encoded)
    sys.stdout.buffer.write(b"\nSOL_ARCHIVE_END\n")
    sys.stdout.buffer.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("recv", "send"))
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--encoded-length", type=int)
    parser.add_argument("--sha256")
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()
    if args.command == "recv":
        if args.dest is None or args.encoded_length is None or args.sha256 is None:
            parser.error("recv requires --dest, --encoded-length and --sha256")
        print(json.dumps(recv(args.dest, args.encoded_length, args.sha256), sort_keys=True))
    else:
        if args.source is None:
            parser.error("send requires --source")
        send(args.source, args.exclude)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
