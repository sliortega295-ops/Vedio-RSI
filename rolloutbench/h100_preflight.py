from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit

from .schema import validate_suite_directory


DEFAULT_PROFILE = Path("benchmarks/sana_video_2b_h100_v0/h100_profile.json")
_DINO_REPOSITORY_URL = "https://github.com/facebookresearch/dino.git"
_FULL_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_REMOTE_QUERY_TIMEOUT_S = 600
_MODEL_ENV_PATHS = {
    "python_bin": ("SANA_PYTHON_BIN", "file"),
    "dependency_overlay": ("SANA_DEPENDENCY_OVERLAY", "dir"),
    "kernel_staging": ("SANA_KERNEL_STAGING", "dir"),
    "model_path": ("SANA_MODEL_PATH", "dir"),
}


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[list[str], str], CommandResult]


def _write_receipt_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.name}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if path.exists():
                if not path.is_file() or path.read_bytes() != content:
                    raise FileExistsError(
                        f"refusing to overwrite a conflicting preflight receipt: {path}"
                    )
                return
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=path.parent, prefix=f".{path.name}.", delete=False
                ) as handle:
                    temporary = Path(handle.name)
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                temporary = None
                directory_fd = os.open(
                    path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _repo_file(repository: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty repository-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must stay inside the repository")
    path = repository / relative
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    return path


def _under(path: str, root: str, label: str) -> None:
    candidate = PurePosixPath(path)
    parent = PurePosixPath(root)
    if not candidate.is_absolute() or not candidate.is_relative_to(parent):
        raise ValueError(f"{label} must be an absolute path under {root}")


def build_preflight_spec(
    profile_path: Path | str = DEFAULT_PROFILE,
    *,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Bind remote-only profile data to the authoritative model and suite files."""

    repository = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    profile_file = Path(profile_path)
    if not profile_file.is_absolute():
        profile_file = repository / profile_file
    profile = _load_json_object(profile_file)
    if profile.get("schema_version") != 1 or profile.get("ssh_host") != "BAAI":
        raise ValueError("H100 profile must be schema v1 and use the BAAI SSH host")
    model_file = _repo_file(repository, profile.get("model_profile"), "model_profile")
    suite_file = _repo_file(repository, profile.get("suite"), "suite")
    validate_suite_directory(suite_file.parent, repo_root=repository)
    try:
        model = tomllib.loads(model_file.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"cannot read model profile {model_file}: {exc}") from exc
    suite = _load_json_object(suite_file)
    official = model.get("official_config", {})
    env = model.get("env", {})
    expected_environment = suite.get("environment")
    if not isinstance(expected_environment, dict) or set(expected_environment) != {
        "python", "torch", "cuda", "triton", "gpu"
    }:
        raise ValueError("frozen suite environment contract is incomplete")
    if official.get("revision") != suite.get("model", {}).get("revision"):
        raise ValueError("model profile revision does not match the frozen suite")
    if env.get("SANA_MODEL_REVISION") != official.get("revision"):
        raise ValueError("model environment revision does not match official_config")
    if PurePosixPath(str(env.get("SANA_MODEL_PATH", ""))).name != official.get("revision"):
        raise ValueError("model snapshot path does not end in the frozen revision")

    path_names = (
        "persistent_path", "remote_benchmark_root", "remote_repo_path",
        "vbench_source_path", "vbench_cache_path",
    )
    path_values = {name: profile.get(name) for name in path_names}
    if not all(isinstance(value, str) and value for value in path_values.values()):
        raise ValueError("remote H100 paths must be nonempty strings")
    persistent = path_values["persistent_path"]
    for name in path_names[1:]:
        _under(path_values[name], persistent, name)

    dino_source = profile.get("dino_source")
    if not isinstance(dino_source, dict) or set(dino_source) != {
        "repository_url", "path", "git_ref"
    }:
        raise ValueError("DINO source contract must declare repository_url, path, and git_ref")
    dino_path = dino_source.get("path")
    dino_ref = dino_source.get("git_ref")
    if dino_source.get("repository_url") != _DINO_REPOSITORY_URL:
        raise ValueError("DINO source must name the official facebookresearch repository")
    if not isinstance(dino_path, str) or not dino_path:
        raise ValueError("DINO source path must be nonempty")
    _under(dino_path, path_values["remote_benchmark_root"], "dino_source.path")
    if not isinstance(dino_ref, str) or not _FULL_GIT_SHA_RE.fullmatch(dino_ref):
        raise ValueError("DINO source git_ref must be a full lowercase commit SHA")

    runtime_paths: list[dict[str, str]] = []
    for path_id, (env_name, path_type) in _MODEL_ENV_PATHS.items():
        value = env.get(env_name)
        if not isinstance(value, str) or not value.startswith("/"):
            raise ValueError(f"model profile lacks absolute {env_name}")
        runtime_paths.append({"id": path_id, "path": value, "type": path_type})
    remote_repo = path_values["remote_repo_path"]
    submodule = model.get("submodule")
    if (
        not isinstance(submodule, str)
        or not submodule
        or Path(submodule).is_absolute()
        or ".." in Path(submodule).parts
    ):
        raise ValueError("model profile submodule must be repository-relative")
    for path_id, model_key in (
        ("run_script", "run_script"),
        ("eval_profile", "eval_profile"),
        ("model_contract", "model_contract"),
    ):
        relative = model.get(model_key)
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"model profile {model_key} must be repository-relative")
        remote_relative = (
            PurePosixPath(submodule) / relative
            if path_id == "run_script"
            else PurePosixPath(relative)
        )
        runtime_paths.append({
            "id": path_id,
            "path": str(PurePosixPath(remote_repo) / remote_relative),
            "type": "file",
        })
    prompt_relative = env.get("SANA_PROMPT_FILE")
    if not isinstance(prompt_relative, str) or Path(prompt_relative).is_absolute() or ".." in Path(prompt_relative).parts:
        raise ValueError("SANA_PROMPT_FILE must be repository-relative")
    runtime_paths.append({
        "id": "prompt_file",
        "path": str(PurePosixPath(remote_repo) / prompt_relative),
        "type": "file",
    })

    gpu_targets = profile.get("gpu_targets")
    if not isinstance(gpu_targets, list) or len(gpu_targets) != 2:
        raise ValueError("H100 profile must declare exactly two target GPUs")
    normalized_gpus: list[dict[str, Any]] = []
    for item in gpu_targets:
        if not isinstance(item, dict) or type(item.get("index")) is not int:
            raise ValueError("GPU target is malformed")
        uuid = item.get("uuid")
        if item["index"] not in {6, 7} or not isinstance(uuid, str) or not uuid.startswith("GPU-"):
            raise ValueError("GPU target must be the frozen index 6 or 7 UUID")
        normalized_gpus.append({"index": item["index"], "uuid": uuid})
    if {item["index"] for item in normalized_gpus} != {6, 7} or len(
        {item["uuid"] for item in normalized_gpus}
    ) != 2:
        raise ValueError("GPU targets must uniquely cover indices 6 and 7")

    quality_path = suite_file.parent / "quality_protocol.json"
    quality = _load_json_object(quality_path)
    artifacts_path = suite_file.parent / "artifacts.json"
    artifacts = _load_json_object(artifacts_path)
    historical_artifacts = [
        row
        for row in artifacts.get("artifacts", [])
        if isinstance(row, dict) and row.get("artifact_id") == "historical_remote_run_outputs"
    ]
    if len(historical_artifacts) != 1 or not isinstance(
        historical_artifacts[0].get("storage_type"), str
    ):
        raise ValueError("frozen artifacts lack the persistent storage type")
    vbench_ref = quality.get("vbench", {}).get("git_ref")
    if not isinstance(vbench_ref, str) or len(vbench_ref) != 40:
        raise ValueError("quality protocol lacks the pinned VBench revision")
    weights = profile.get("vbench_weights")
    if not isinstance(weights, list) or len(weights) != 8:
        raise ValueError("H100 profile must declare exactly eight VBench weight assets")
    normalized_weights: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    vbench_cache = path_values["vbench_cache_path"]
    for item in weights:
        if not isinstance(item, dict) or set(item) != {
            "id", "relative_path", "sha256", "size_bytes", "source_url"
        }:
            raise ValueError("VBench weight entry is malformed")
        asset_id = item.get("id")
        relative = item.get("relative_path")
        digest = item.get("sha256")
        size_bytes = item.get("size_bytes")
        source_url = item.get("source_url")
        parsed_source = urlsplit(source_url) if isinstance(source_url, str) else None
        relative_path = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            not isinstance(asset_id, str) or not asset_id or asset_id in seen_ids
            or relative_path is None or relative_path.is_absolute() or ".." in relative_path.parts
            or not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest)
            or type(size_bytes) is not int or size_bytes <= 0
            or parsed_source is None or parsed_source.scheme != "https"
            or not parsed_source.hostname
        ):
            raise ValueError(
                "VBench weight IDs, paths, SHA-256, sizes, and source URLs must be exact and safe"
            )
        seen_ids.add(asset_id)
        normalized_weights.append({
            "id": asset_id,
            "path": str(PurePosixPath(vbench_cache) / relative_path),
            "sha256": digest,
            "size_bytes": size_bytes,
            "source_url": source_url,
        })

    return {
        "schema_version": 1,
        "ssh_host": "BAAI",
        "profile_path": str(profile_file),
        "profile_sha256": _sha256(profile_file),
        "model_profile_path": str(model_file),
        "model_profile_sha256": _sha256(model_file),
        "suite_path": str(suite_file),
        "suite_sha256": _sha256(suite_file),
        "quality_protocol_sha256": _sha256(quality_path),
        "artifacts_sha256": _sha256(artifacts_path),
        "persistent_path": persistent,
        "expected_storage_type": historical_artifacts[0]["storage_type"],
        "remote_benchmark_root": path_values["remote_benchmark_root"],
        "remote_repo_path": remote_repo,
        "expected_environment": {
            key: str(expected_environment[key]) for key in ("python", "torch", "cuda", "triton")
        },
        "expected_gpu_name": str(expected_environment["gpu"]),
        "target_gpus": sorted(normalized_gpus, key=lambda item: item["index"]),
        "runtime_paths": runtime_paths,
        "python_bin": str(env["SANA_PYTHON_BIN"]),
        "model_path": str(env["SANA_MODEL_PATH"]),
        "model_revision": str(official["revision"]),
        "model_class_name": str(official["pipeline_class_name"]),
        "vbench_source_path": path_values["vbench_source_path"],
        "vbench_cache_path": vbench_cache,
        "vbench_git_ref": vbench_ref,
        "vbench_weights": normalized_weights,
        "dino_source": {
            "repository_url": _DINO_REPOSITORY_URL,
            "path": dino_path,
            "git_ref": dino_ref,
            "required_file": "hubconf.py",
        },
    }


def _remote_script(spec: Mapping[str, Any]) -> str:
    encoded = base64.b64encode(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    return f'''import base64
import datetime
import hashlib
import json
import os
import socket
import subprocess

SPEC = json.loads(base64.b64decode({encoded!r}).decode("utf-8"))

def run(argv):
    try:
        result = subprocess.run(argv, text=True, capture_output=True, check=False)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as exc:
        return 127, "", type(exc).__name__ + ": " + str(exc)

def path_receipt(item):
    path = item["path"]
    expected_type = item["type"]
    exists = os.path.exists(path)
    matches = os.path.isfile(path) if expected_type == "file" else os.path.isdir(path)
    return {{"path": path, "expected_type": expected_type, "exists": exists, "type_matches": matches}}

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

storage_path = SPEC["persistent_path"]
mount_rc, mount_out, mount_err = run(["findmnt", "-n", "-T", storage_path, "-o", "TARGET,SOURCE,FSTYPE"])
mount_parts = mount_out.split(None, 2) if mount_rc == 0 else []
disk_rc, disk_out, disk_err = run(["df", "-Pk", storage_path])
disk_lines = [line for line in disk_out.splitlines() if line.strip()]
disk_parts = disk_lines[-1].split() if disk_rc == 0 and len(disk_lines) >= 2 else []

gpu_rc, gpu_out, gpu_err = run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"])
gpus = []
gpu_parse_errors = []
target_indices = {{item["index"] for item in SPEC["target_gpus"]}}
if gpu_rc == 0:
    for line in gpu_out.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            gpu_parse_errors.append("wrong GPU column count")
            continue
        try:
            index, memory = int(parts[0]), int(parts[3])
        except ValueError:
            gpu_parse_errors.append("invalid GPU numeric field")
            continue
        if index in target_indices:
            gpus.append({{"index": index, "uuid": parts[1], "name": parts[2], "memory_total_mib": memory}})

apps_rc, apps_out, apps_err = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"])
apps = []
apps_parse_errors = []
if apps_rc == 0:
    for line in apps_out.splitlines():
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            apps_parse_errors.append("wrong compute-app column count")
            continue
        try:
            apps.append({{"gpu_uuid": parts[0], "pid": int(parts[1]), "process_name": parts[2], "used_memory_mib": int(parts[3])}})
        except ValueError:
            apps_parse_errors.append("invalid compute-app numeric field")
            continue

runtime_code = "import json,platform; import torch,triton; print(json.dumps({{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'triton':triton.__version__}},sort_keys=True))"
runtime_rc, runtime_out, runtime_err = run([SPEC["python_bin"], "-c", runtime_code])
try:
    runtime_values = json.loads(runtime_out) if runtime_rc == 0 else {{}}
except Exception:
    runtime_values = {{}}

model_index_path = os.path.join(SPEC["model_path"], "model_index.json")
model_index_exists = os.path.isfile(model_index_path)
model_index_valid, model_class = False, None
if model_index_exists:
    try:
        with open(model_index_path, "r", encoding="utf-8") as handle:
            model_index = json.load(handle)
        model_class = model_index.get("_class_name")
        model_index_valid = isinstance(model_index, dict)
    except Exception:
        pass

vbench_source = SPEC["vbench_source_path"]
git_rc, git_out, git_err = run(["git", "-C", vbench_source, "rev-parse", "HEAD"]) if os.path.isdir(vbench_source) else (1, "", "source missing")
weights = []
for item in SPEC["vbench_weights"]:
    path = item["path"]
    exists, is_file = os.path.exists(path), os.path.isfile(path)
    weights.append({{
        "id": item["id"], "path": path, "exists": exists, "is_file": is_file,
        "readable": os.access(path, os.R_OK) if exists else False,
        "size_bytes": os.path.getsize(path) if is_file else 0,
        "sha256": file_sha256(path) if is_file else None,
    }})

dino_spec = SPEC["dino_source"]
dino_path = dino_spec["path"]
dino_is_dir = os.path.isdir(dino_path)
if dino_is_dir:
    dino_head_rc, dino_head_out, dino_head_err = run(
        ["git", "--no-optional-locks", "-C", dino_path, "rev-parse", "--verify", "HEAD^{{commit}}"]
    )
    dino_symbolic_rc, dino_symbolic_out, dino_symbolic_err = run(
        ["git", "--no-optional-locks", "-C", dino_path, "symbolic-ref", "-q", "HEAD"]
    )
    dino_status_rc, dino_status_out, dino_status_err = run(
        ["git", "--no-optional-locks", "-C", dino_path, "status", "--porcelain", "--untracked-files=all"]
    )
else:
    dino_head_rc, dino_head_out, dino_head_err = 1, "", "source missing"
    dino_symbolic_rc, dino_symbolic_out, dino_symbolic_err = 1, "", "source missing"
    dino_status_rc, dino_status_out, dino_status_err = 1, "", "source missing"
dino_hubconf = os.path.join(dino_path, dino_spec["required_file"])
dino_hubconf_exists = os.path.exists(dino_hubconf)
dino_hubconf_is_file = os.path.isfile(dino_hubconf)

payload = {{
    "schema_version": 1,
    "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "hostname": socket.gethostname(),
    "persistent_storage": {{
        "path": storage_path, "exists": os.path.exists(storage_path),
        "mount": {{
            "ok": mount_rc == 0 and len(mount_parts) == 3,
            "target": mount_parts[0] if len(mount_parts) == 3 else None,
            "source": mount_parts[1] if len(mount_parts) == 3 else None,
            "fstype": mount_parts[2] if len(mount_parts) == 3 else None,
            "error": mount_err if mount_rc != 0 else None,
        }},
        "disk": {{
            "ok": disk_rc == 0 and len(disk_parts) >= 6,
            "total_kib": int(disk_parts[1]) if len(disk_parts) >= 6 and disk_parts[1].isdigit() else None,
            "available_kib": int(disk_parts[3]) if len(disk_parts) >= 6 and disk_parts[3].isdigit() else None,
            "capacity": disk_parts[4] if len(disk_parts) >= 6 else None,
            "error": disk_err if disk_rc != 0 else None,
        }},
    }},
    "gpus": gpus, "compute_apps": apps,
    "nvidia_smi_ok": gpu_rc == 0 and apps_rc == 0 and not gpu_parse_errors and not apps_parse_errors,
    "nvidia_smi_error": "; ".join(
        item for item in (gpu_err, apps_err, *gpu_parse_errors, *apps_parse_errors) if item
    ) or None,
    "runtime": {{"ok": runtime_rc == 0 and isinstance(runtime_values, dict), **runtime_values, "error": runtime_err if runtime_rc != 0 else None}},
    "runtime_paths": {{item["id"]: path_receipt(item) for item in SPEC["runtime_paths"]}},
    "model": {{
        "path": SPEC["model_path"], "model_index_path": model_index_path,
        "model_index_exists": model_index_exists, "model_index_valid": model_index_valid,
        "class_name": model_class,
    }},
    "remote_benchmark_root": {{
        "path": SPEC["remote_benchmark_root"],
        "exists": os.path.exists(SPEC["remote_benchmark_root"]),
        "is_dir": os.path.isdir(SPEC["remote_benchmark_root"]),
    }},
    "vbench": {{
        "source_path": vbench_source, "source_exists": os.path.exists(vbench_source),
        "source_is_dir": os.path.isdir(vbench_source), "git_ok": git_rc == 0,
        "git_ref": git_out if git_rc == 0 else None,
        "git_error": git_err if git_rc != 0 else None,
        "cache_path": SPEC["vbench_cache_path"], "weights": weights,
        "dino_source": {{
            "path": dino_path,
            "exists": os.path.exists(dino_path),
            "is_dir": dino_is_dir,
            "git_ok": (
                dino_head_rc == 0
                and dino_symbolic_rc in (0, 1)
                and dino_status_rc == 0
            ),
            "head": dino_head_out if dino_head_rc == 0 else None,
            "detached": dino_symbolic_rc == 1 and not dino_symbolic_out,
            "clean": dino_status_rc == 0 and not dino_status_out,
            "git_error": "; ".join(
                item
                for item in (dino_head_err, dino_symbolic_err, dino_status_err)
                if item and item != "source missing"
            ) or ("source missing" if not dino_is_dir else None),
            "hubconf": {{
                "path": dino_hubconf,
                "exists": dino_hubconf_exists,
                "is_file": dino_hubconf_is_file,
                "readable": os.access(dino_hubconf, os.R_OK) if dino_hubconf_exists else False,
                "size_bytes": os.path.getsize(dino_hubconf) if dino_hubconf_is_file else 0,
            }},
        }},
    }},
}}
print(json.dumps(payload, sort_keys=True))
'''


def _default_command_runner(argv: list[str], stdin: str) -> CommandResult:
    result = subprocess.run(
        argv,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
        timeout=_REMOTE_QUERY_TIMEOUT_S,
    )
    return CommandResult(result.returncode, result.stdout, result.stderr)


def _error_receipt(spec: Mapping[str, Any], error: str) -> dict[str, Any]:
    return {
        "schema_version": 1, "query_status": "ERROR", "runtime_ready": False,
        "two_gpu_idle_point_in_time": False, "quality_ready": False, "pilot_ready": False,
        "ssh_host": spec["ssh_host"], "profile_sha256": spec["profile_sha256"],
        "suite_sha256": spec["suite_sha256"],
        "artifacts_sha256": spec["artifacts_sha256"],
        "gpu_idle_scope": {
            "observed_at_utc": None, "ownership_verified": False, "ownership_claim": False,
            "note": "GPU idleness is a point-in-time compute-app observation, not an ownership grant.",
        },
        "checks": {"query": {"pass": False, "errors": [error]}},
        "observation": None, "read_only": True, "gpu_execution": False,
        "vbench_execution": False,
    }


def _mapping(value: Any, label: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{label} is missing or is not an object")
        return {}
    return value


def _evaluate(spec: Mapping[str, Any], observation: Any) -> dict[str, Any]:
    parse_errors: list[str] = []
    observed = _mapping(observation, "remote observation", parse_errors)
    if observed.get("schema_version") != 1:
        parse_errors.append("remote observation schema_version mismatch")
    timestamp, hostname = observed.get("observed_at_utc"), observed.get("hostname")
    if not isinstance(timestamp, str) or not timestamp:
        parse_errors.append("remote observation lacks observed_at_utc")
    if not isinstance(hostname, str) or not hostname:
        parse_errors.append("remote observation lacks hostname")

    runtime_errors: list[str] = []
    storage = _mapping(observed.get("persistent_storage"), "persistent_storage", runtime_errors)
    mount = _mapping(storage.get("mount"), "persistent_storage.mount", runtime_errors)
    disk = _mapping(storage.get("disk"), "persistent_storage.disk", runtime_errors)
    if storage.get("path") != spec["persistent_path"] or storage.get("exists") is not True:
        runtime_errors.append("persistent storage path is absent or mismatched")
    if mount.get("ok") is not True or not all(
        isinstance(mount.get(key), str) and mount.get(key) for key in ("target", "source", "fstype")
    ):
        runtime_errors.append("persistent mount receipt is incomplete")
    elif spec["expected_storage_type"] not in mount["source"]:
        runtime_errors.append(
            f'persistent mount source does not identify {spec["expected_storage_type"]}'
        )
    if disk.get("ok") is not True or type(disk.get("available_kib")) is not int or disk.get("available_kib", 0) <= 0:
        runtime_errors.append("persistent disk receipt is incomplete or has no free space")
    runtime = _mapping(observed.get("runtime"), "runtime", runtime_errors)
    if runtime.get("ok") is not True:
        runtime_errors.append("runtime Python probe failed")
    for key, expected in spec["expected_environment"].items():
        if runtime.get(key) != expected:
            runtime_errors.append(f"{key} mismatch: expected {expected}, observed {runtime.get(key)}")
    path_rows = _mapping(observed.get("runtime_paths"), "runtime_paths", runtime_errors)
    for expected in spec["runtime_paths"]:
        row = _mapping(path_rows.get(expected["id"]), f'runtime_paths.{expected["id"]}', runtime_errors)
        if (
            row.get("path") != expected["path"] or row.get("expected_type") != expected["type"]
            or row.get("exists") is not True or row.get("type_matches") is not True
        ):
            runtime_errors.append(f'runtime path {expected["id"]} is missing or mismatched')
    model = _mapping(observed.get("model"), "model", runtime_errors)
    if (
        model.get("path") != spec["model_path"] or model.get("model_index_exists") is not True
        or model.get("model_index_valid") is not True
        or model.get("class_name") != spec["model_class_name"]
    ):
        runtime_errors.append("model path/model_index pipeline class mismatch")
    runtime_ready = not parse_errors and not runtime_errors

    gpu_errors: list[str] = []
    if observed.get("nvidia_smi_ok") is not True:
        gpu_errors.append("nvidia-smi GPU or compute-app query failed")
    gpu_rows = observed.get("gpus")
    if not isinstance(gpu_rows, list):
        gpu_errors.append("GPU rows are missing")
        gpu_rows = []
    by_index: dict[int, Mapping[str, Any]] = {}
    for row_value in gpu_rows:
        row = _mapping(row_value, "GPU row", gpu_errors)
        index = row.get("index")
        if type(index) is not int or index in by_index:
            gpu_errors.append("GPU row has invalid or duplicate index")
            continue
        by_index[index] = row
    for expected in spec["target_gpus"]:
        row = by_index.get(expected["index"], {})
        if row.get("uuid") != expected["uuid"]:
            gpu_errors.append(f'GPU {expected["index"]} UUID mismatch')
        if "H100" not in str(row.get("name", "")):
            gpu_errors.append(f'GPU {expected["index"]} is not reported as H100')
        if type(row.get("memory_total_mib")) is not int or row.get("memory_total_mib", 0) <= 0:
            gpu_errors.append(f'GPU {expected["index"]} memory receipt is invalid')
    if set(by_index) != {item["index"] for item in spec["target_gpus"]}:
        gpu_errors.append("GPU query did not return exactly the two target indices")
    apps = observed.get("compute_apps")
    if not isinstance(apps, list):
        gpu_errors.append("compute-app rows are missing")
        apps = []
    target_uuids = {item["uuid"] for item in spec["target_gpus"]}
    busy_apps: list[Mapping[str, Any]] = []
    for app_value in apps:
        app = _mapping(app_value, "compute-app row", gpu_errors)
        if not isinstance(app.get("gpu_uuid"), str) or type(app.get("pid")) is not int:
            gpu_errors.append("compute-app row is malformed")
            continue
        if app["gpu_uuid"] in target_uuids:
            busy_apps.append(app)
    if busy_apps:
        gpu_errors.append("one or both target GPUs have active compute applications")
    idle = not parse_errors and not gpu_errors

    quality_errors: list[str] = []
    vbench = _mapping(observed.get("vbench"), "vbench", quality_errors)
    if (
        vbench.get("source_path") != spec["vbench_source_path"]
        or vbench.get("source_exists") is not True or vbench.get("source_is_dir") is not True
        or vbench.get("git_ok") is not True or vbench.get("git_ref") != spec["vbench_git_ref"]
        or vbench.get("cache_path") != spec["vbench_cache_path"]
    ):
        quality_errors.append("VBench source path or pinned Git revision mismatch")
    weight_rows = vbench.get("weights")
    if not isinstance(weight_rows, list):
        quality_errors.append("VBench weight rows are missing")
        weight_rows = []
    weights_by_id: dict[str, Mapping[str, Any]] = {}
    for row_value in weight_rows:
        row = _mapping(row_value, "VBench weight row", quality_errors)
        asset_id = row.get("id")
        if not isinstance(asset_id, str) or asset_id in weights_by_id:
            quality_errors.append("VBench weight row has invalid or duplicate ID")
            continue
        weights_by_id[asset_id] = row
    for expected in spec["vbench_weights"]:
        row = weights_by_id.get(expected["id"], {})
        if (
            row.get("path") != expected["path"] or row.get("exists") is not True
            or row.get("is_file") is not True or row.get("readable") is not True
            or row.get("size_bytes") != expected["size_bytes"]
            or row.get("sha256") != expected["sha256"]
        ):
            quality_errors.append(f'VBench weight {expected["id"]} is missing or invalid')
    if set(weights_by_id) != {item["id"] for item in spec["vbench_weights"]}:
        quality_errors.append("VBench weight query did not return the exact frozen asset set")
    dino = _mapping(vbench.get("dino_source"), "vbench.dino_source", quality_errors)
    hubconf = _mapping(dino.get("hubconf"), "vbench.dino_source.hubconf", quality_errors)
    expected_dino = spec["dino_source"]
    expected_hubconf = str(
        PurePosixPath(expected_dino["path"]) / expected_dino["required_file"]
    )
    if (
        dino.get("path") != expected_dino["path"]
        or dino.get("exists") is not True
        or dino.get("is_dir") is not True
        or dino.get("git_ok") is not True
        or dino.get("head") != expected_dino["git_ref"]
        or dino.get("detached") is not True
        or dino.get("clean") is not True
        or hubconf.get("path") != expected_hubconf
        or hubconf.get("exists") is not True
        or hubconf.get("is_file") is not True
        or hubconf.get("readable") is not True
        or type(hubconf.get("size_bytes")) is not int
        or hubconf.get("size_bytes", 0) <= 0
    ):
        quality_errors.append(
            "DINO source must be the exact detached clean frozen HEAD with readable hubconf.py"
        )
    quality_ready = not parse_errors and not quality_errors

    root_errors: list[str] = []
    remote_root = _mapping(observed.get("remote_benchmark_root"), "remote_benchmark_root", root_errors)
    if (
        remote_root.get("path") != spec["remote_benchmark_root"]
        or remote_root.get("exists") is not True or remote_root.get("is_dir") is not True
    ):
        root_errors.append("remote benchmark root is missing or mismatched")
    pilot_ready = runtime_ready and idle and quality_ready and not root_errors
    return {
        "schema_version": 1, "query_status": "PASS" if not parse_errors else "ERROR",
        "runtime_ready": runtime_ready, "two_gpu_idle_point_in_time": idle,
        "quality_ready": quality_ready, "pilot_ready": pilot_ready,
        "ssh_host": spec["ssh_host"], "hostname": hostname if isinstance(hostname, str) else None,
        "profile_sha256": spec["profile_sha256"],
        "model_profile_sha256": spec["model_profile_sha256"],
        "suite_sha256": spec["suite_sha256"],
        "quality_protocol_sha256": spec["quality_protocol_sha256"],
        "artifacts_sha256": spec["artifacts_sha256"],
        "gpu_idle_scope": {
            "observed_at_utc": timestamp if isinstance(timestamp, str) else None,
            "ownership_verified": False, "ownership_claim": False,
            "note": "GPU idleness is a point-in-time compute-app observation, not an ownership grant.",
            "busy_compute_apps": list(busy_apps),
        },
        "checks": {
            "parse": {"pass": not parse_errors, "errors": parse_errors},
            "runtime": {"pass": runtime_ready, "errors": runtime_errors},
            "gpu_point_in_time": {"pass": idle, "errors": gpu_errors},
            "quality": {"pass": quality_ready, "errors": quality_errors},
            "remote_root": {"pass": not root_errors, "errors": root_errors},
        },
        "observation": dict(observed), "read_only": True,
        "gpu_execution": False, "vbench_execution": False,
    }


def run_h100_preflight(
    profile_path: Path | str = DEFAULT_PROFILE,
    *,
    repo_root: Path | str | None = None,
    output_path: Path | str | None = None,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    spec = build_preflight_spec(profile_path, repo_root=repo_root)
    command = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
        spec["ssh_host"], "python3", "-",
    ]
    execute = runner or _default_command_runner
    try:
        result = execute(command, _remote_script(spec))
        if result.returncode != 0:
            detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no stderr"
            receipt = _error_receipt(
                spec, f"SSH query failed with return code {result.returncode}: {detail}"
            )
        else:
            try:
                observation = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                receipt = _error_receipt(spec, f"remote output is not one JSON object: {exc}")
            else:
                receipt = _evaluate(spec, observation)
    except Exception as exc:
        receipt = _error_receipt(spec, f"SSH query raised {type(exc).__name__}: {exc}")
    if output_path is not None:
        target = Path(output_path)
        content = (
            json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        _write_receipt_once(
            target,
            content,
        )
    return receipt
