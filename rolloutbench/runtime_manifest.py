from __future__ import annotations

import functools
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping


COMMON_RUNTIME_FILES = (
    "scripts/sana/sana_video_sglang_run.py",
    "python/sglang/multimodal_gen/registry_sana.py",
    "python/sglang/multimodal_gen/configs/models/dits/sana_video.py",
    "python/sglang/multimodal_gen/configs/pipeline_configs/sana_video.py",
    "python/sglang/multimodal_gen/configs/sample/sana_video.py",
    "python/sglang/multimodal_gen/runtime/models/dits/sana_video.py",
    "python/sglang/multimodal_gen/runtime/pipelines/sana_video.py",
)
OPTIONAL_HISTORICAL_RUNTIME_FILES = (
    "python/sglang/jit_kernel/diffusion/triton/sana_rope.py",
    "python/sglang/multimodal_gen/runtime/cache/sana_video_cache.py",
)
KNOWN_RUNTIME_RECEIPT_FILES = COMMON_RUNTIME_FILES + OPTIONAL_HISTORICAL_RUNTIME_FILES
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")


class RuntimeManifestError(ValueError):
    """Raised when a historical candidate cannot name one exact runtime tree."""


def resolve_candidate_runtime_ref(candidate: Mapping[str, Any]) -> tuple[str, str]:
    commit = candidate.get("candidate_commit")
    if commit == "not_created_preflight_rejection":
        if not isinstance(candidate.get("probe"), Mapping):
            raise RuntimeManifestError(
                "missing candidate commit is valid only for a preflight probe"
            )
        commit = candidate.get("parent_sha")
        role = "parent_for_preflight_rejection"
    else:
        role = "candidate_commit"
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        raise RuntimeManifestError(
            "runtime ref must be a lowercase 40-character hexadecimal commit"
        )
    return commit, role


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeManifestError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


@functools.lru_cache(maxsize=128)
def _manifest_for_ref(repository_text: str, runtime_ref: str) -> tuple[str, tuple[str, ...]]:
    repository = Path(repository_text)
    try:
        resolved = _git(
            repository, "rev-parse", "--verify", f"{runtime_ref}^{{commit}}"
        )
    except RuntimeManifestError as exc:
        raise RuntimeManifestError(
            f"runtime ref does not resolve to a commit: {runtime_ref}"
        ) from exc
    if resolved != runtime_ref:
        raise RuntimeManifestError(f"runtime ref did not resolve exactly: {runtime_ref}")
    runtime_tree_oid = _git(
        repository, "rev-parse", f"{runtime_ref}:external/sol_runtime"
    )
    names = set(
        _git(
            repository,
            "ls-tree",
            "-r",
            "--name-only",
            runtime_ref,
            "--",
            "external/sol_runtime",
        ).splitlines()
    )
    present = tuple(
        relative
        for relative in KNOWN_RUNTIME_RECEIPT_FILES
        if f"external/sol_runtime/{relative}" in names
    )
    missing_common = sorted(set(COMMON_RUNTIME_FILES) - set(present))
    if missing_common:
        raise RuntimeManifestError(
            f"historical runtime is missing common files: {missing_common}"
        )
    return runtime_tree_oid, present


def build_runtime_manifest(
    repository_root: Path | str, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    repository = Path(repository_root).resolve()
    runtime_ref, ref_role = resolve_candidate_runtime_ref(candidate)
    runtime_tree_oid, required_paths = _manifest_for_ref(str(repository), runtime_ref)
    return {
        "git_ref": runtime_ref,
        "ref_role": ref_role,
        "runtime_tree_oid": runtime_tree_oid,
        "required_runtime_paths": list(required_paths),
    }
