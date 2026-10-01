#!/usr/bin/env python3
"""Restricted host-side queue for run-bundle and CPU judge operations.

Codex workspace-write cannot resolve the private SSH gateway. The host daemon
executes ONLY typed operations on prepared, hash-locked run bundles
within this new project. It accepts no command strings or target hosts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import tomllib
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

COORD = Path("/home/lyy/Experiments/SolAgent/20260913-a100-original-baseline/repo")
PROJECT = COORD.parent
# Import host-side transport/template code before accepting any Agent-owned
# worktree path. Never prepend an Executor worktree to this daemon's sys.path:
# otherwise a worktree could replace the privileged SSH transport module.
if str(COORD) not in sys.path:
    sys.path.insert(0, str(COORD))
from scripts.launch_config import render_launch_script
from orchestration import site_remote
if (Path(sys.modules["scripts.launch_config"].__file__).resolve()
        != COORD / "scripts/launch_config.py"
        or Path(site_remote.__file__).resolve() != COORD / "orchestration/site_remote.py"):
    raise RuntimeError("site job broker did not load coordinator-owned transport/template modules")
QUEUE = COORD / ".symposium/site-job-broker"
REQUESTS = QUEUE / "requests"
RESPONSES = QUEUE / "responses"
STATE = QUEUE / "daemon.json"
HEARTBEAT = QUEUE / "heartbeat.json"
AUDIT = PROJECT / "telemetry/site_job_broker_requests.jsonl"
# Only the fresh, non-contaminated official Executor worktrees may submit.
# Each sandbox writes its OWN queue; the host daemon reads these exact paths.
ALLOWED_EXPERIMENT_UIDS = ("sana2ba100-kernel_aw-0002", "sana2ba100-cache_ca-0002")
EXPERIMENT_WORKTREES = {
    uid: COORD / "output/experiments" / uid / "worktree" for uid in ALLOWED_EXPERIMENT_UIDS
}
EXPERIMENT_QUEUES = {
    uid: worktree / ".symposium/site-job-broker"
    for uid, worktree in EXPERIMENT_WORKTREES.items()
}
CPU_PROBES = {
    "site-transport-broker-cpu-01": "356424d1a1cc08a1cab510c6c0acadaf79cf39b520868882d1321500d6d14761",
    "site-transport-broker-cpu-02": "357d33e6f79d77ced74065a6aff4a2b69bbe2f18e0f0e8541bec67d136ef0423",
    "site-transport-broker-cpu-03": "554cc9e847c31218ab5c57596a73088a593f1b17767002c57f15cbd495bcf520",
    "site-transport-broker-cpu-04": "200cd3b6a005bd0c9cab24bef75b155e8ffdfe4274c00c023a79c6e0216379c8",
}
PYTHON = "/tmp/distillschedule-helios-official-torch210/bin/python"
MODEL = "/share/project/shuchenweng/yongyan_liu/Experiments/SolRolloutBench/20260912-a100-eight/models/sana-video-2b/db5f398b13ca086d09a50ce156c20527773841b1"
OVERLAY = "/share/project/shuchenweng/yongyan_liu/Experiments/SolRolloutBench/20260912-a100-eight/envs/sglang-overlay"
RUN_SCRIPT = "models/sana_video_2b_a100/baseline/scripts/run_sana_video_2b_gpu.sh"
REQUIRED_ENV = {"SOL_SITE_REMOTE": "1", "PYTHON_BIN": PYTHON,
                "SOL_A100_PYTHON": PYTHON, "SOL_A100_MODEL_PATH": MODEL,
                "SOL_A100_DEP_OVERLAY": OVERLAY,
                "SOL_A100_PROMPTS_FILE": "models/sana_video_2b_a100/prompts/validation_first5.txt",
                "SOL_A100_SEED": "42", "SOL_A100_EXPECTED_PROMPTS": "5",
                "SOL_A100_SGL_KERNEL_BACKEND": "torch_fallback",
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1", "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1"}
SAFE_EXTRA_ENV = {"HF_HUB_DISABLE_TELEMETRY", "PYTHONUNBUFFERED",
                  "TOKENIZERS_PARALLELISM", "SOL_A100_COMPILE"}
OFFICIAL = {"model": "sana_video_2b", "model_revision": "db5f398b13ca086d09a50ce156c20527773841b1",
            "width": 832, "height": 480, "frames": 81, "fps": 16, "steps": 50,
            "guidance_scale": 6.0, "num_gpus": 1, "prompt_count": 5, "seed": 42,
            "transformer_precision": "bf16", "vae_precision": "fp32"}
# Pinned from the remote first-GPU receipt in this project, SHA256
# f5502224c63b9bf21b6e93bc33921bc8f66b9281780a83d7b22214fc5684b262.
# Reserve the longest observed five-prompt candidate wall (~19.6 min) plus
# about one minute for transfer/clock uncertainty. This is a site budget
# safety check, not an additional candidate-quality or method gate.
FIRST_GPU_DEADLINE_UNIX_NS = 1789312179836267316
GPU_SUBMIT_RESERVE_NS = 21 * 60 * 10**9
# This project-local control-plane grant is independent of an Executor's
# AUTO_VIDEO_DISABLE_SINGLE_FLIGHT_GUARD environment variable. The archived
# Master explicitly granted only these two exact overlapping run bundles while
# the unresolved R4 worker remained live (goal corrections at 12:10/12:25 UTC).
# A later candidate needs a NEW audited Master grant and a host-side exact
# run/manifest allowlist update; an Executor cannot self-authorize overlap.
R4_RUN = EXPERIMENT_WORKTREES["sana2ba100-kernel_aw-0002"] / "runs/20260913-114414-kernel_compile_bf16_autotune_qkv"
R4_MASTER_GRANTED_OVERLAP_MANIFESTS = {
    "20260913-121315-kernel_compile_bf16_default_qkv":
        "bde77247ebf7dd07115a82b712e2efb243da36047e85c0b26df5378c01dc355f",
    "20260913-123046-kernel_compile_bf16_default_ffn_inverted_linear":
        "efdcd32c6205a8fba4ac8767119d4cf255bf4e7fcdfe5f066aa43c2b877b9db4",
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def digest(data: dict) -> str:
    return hashlib.sha256(canonical(data)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_gpu_submit_budget(run_dir: Path | None = None, now_ns: int | None = None) -> dict:
    """Reject Phase 1 after its fixed cap; gate Phase 2 on separate remote state."""
    if run_dir is not None:
        manifest_path = run_dir / "manifest.resolved.toml"
        with manifest_path.open("rb") as stream:
            manifest = tomllib.load(stream)
        phase = manifest.get("resolved_profile", {}).get("env", {}).get("SOL_A100_BUDGET_PHASE")
        if phase == "phase2":
            auth_path = PROJECT / "state/phase2_authorization.json"
            if not auth_path.is_file() or auth_path.is_symlink():
                raise RuntimeError("phase2 GPU authorization is absent on host")
            auth = read(auth_path)
            if (auth.get("schema_version") != 1 or auth.get("phase") != "phase2"
                    or auth.get("status") != "armed"
                    or auth.get("remote_project") != str(site_remote.REMOTE_PROJECT)
                    or auth.get("remote_host") != "job-c6cfcde5-db4b-4527-8eb8-7518319a0277-master-0"
                    or auth.get("max_window_s") != 6 * 3600
                    or auth.get("submit_reserve_s") != GPU_SUBMIT_RESERVE_NS // 10**9
                    or not isinstance(auth.get("excluded_run_names"), list)
                    or run_dir.name in auth["excluded_run_names"]):
                raise RuntimeError("phase2 host authorization identity or run exclusion differs")
            remote = site_remote.phase2_budget_status_direct()
            if (remote.get("phase") != "phase2" or remote.get("authorized") is not True
                    or remote.get("authorization_sha256") != file_sha(auth_path)
                    or remote.get("new_submit_open") is not True):
                raise TimeoutError("phase2 remote budget/authorization mismatch or submit cutoff reached")
            return {**remote, "host_authorization_sha256": file_sha(auth_path)}
        if phase is not None:
            raise ValueError("unknown site GPU budget phase in resolved manifest")
    observed_ns = time.time_ns() if now_ns is None else now_ns
    latest_ns = FIRST_GPU_DEADLINE_UNIX_NS - GPU_SUBMIT_RESERVE_NS
    if observed_ns >= latest_ns:
        raise ValueError(
            "first-GPU six-hour budget: new GPU submit closed at "
            f"{datetime.fromtimestamp(latest_ns / 1e9, timezone.utc).isoformat()} "
            f"(deadline={FIRST_GPU_DEADLINE_UNIX_NS}, reserve_s=1260)"
        )
    return {"phase": "phase1", "checked_at_unix_ns": observed_ns, "latest_submit_unix_ns": latest_ns,
            "deadline_unix_ns": FIRST_GPU_DEADLINE_UNIX_NS,
            "reserve_s": GPU_SUBMIT_RESERVE_NS // 10**9}


def kernel_runs_started_by_host_broker() -> list[Path]:
    """Return exact Kernel runs with successful host-owned remote starts.

    Executor metadata may be edited (as R7 demonstrated), so it is not an
    authoritative inventory of live GPU work. Only the host broker's own
    append-only response audit supplies the previous-start identity here.
    """
    if not AUDIT.is_file():
        raise RuntimeError("host submit audit missing; cannot check Kernel single-flight")
    remote_parent = site_remote.remote_path(R4_RUN.parent)
    seen: set[Path] = set()
    with AUDIT.open() as stream:
        for line in stream:
            record = json.loads(line)  # malformed audit fails closed
            if record.get("ok") is not True:
                continue
            remote_run = (record.get("result") or {}).get("remote_run_dir")
            if not isinstance(remote_run, str):
                continue
            path = Path(remote_run)
            if path.parent == remote_parent and path.name not in CPU_PROBES:
                seen.add(R4_RUN.parent / path.name)
    return sorted(seen)


def assert_kernel_single_flight_grant(run: Path) -> dict:
    """Fail closed if any earlier host-started Kernel GPU run is still live."""
    checks = []
    active = []
    for previous in kernel_runs_started_by_host_broker():
        if previous == run:
            continue
        state = site_remote._status_direct(previous)
        if type(state.get("finished")) is not bool:
            raise RuntimeError(f"remote Kernel status lacks a finished boolean: {previous.name}")
        checks.append({"run": previous.name, "finished": state["finished"],
                       "checked_at_utc": state.get("checked_at_utc")})
        if not state["finished"]:
            active.append(previous)
    if active:
        allowed = R4_MASTER_GRANTED_OVERLAP_MANIFESTS.get(run.name)
        if (active != [R4_RUN] or allowed is None
                or allowed != file_sha(run / "manifest.resolved.toml")):
            raise ValueError(
                "active Kernel single-flight: this exact run lacks a host-pinned "
                "archived Master overlap grant for the current active run set "
                f"{[prior.name for prior in active]}"
            )
    return {"checked_at_utc": utc(), "prior_host_starts": checks,
            "active_prior_runs": [prior.name for prior in active],
            "grant": "exact_r4_master_grant" if active else "no_active_kernel_run"}


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def client_queue() -> Path:
    caller_root = Path(__file__).resolve().parents[1]
    if caller_root == COORD:
        return QUEUE
    for uid, worktree in EXPERIMENT_WORKTREES.items():
        if caller_root == worktree and worktree.is_dir() and not worktree.is_symlink():
            return EXPERIMENT_QUEUES[uid]
    raise ValueError("site job broker caller is not the coordinator or a fresh allowed Executor worktree")


def valid_run(run_dir: Path, launch: Path | None, command: str) -> tuple[Path, Path]:
    run = run_dir.resolve()
    if not run.is_dir() or run_dir.is_symlink() or not run.is_relative_to(COORD):
        raise ValueError("site job run dir outside the new coordinator project")
    rel = run.relative_to(COORD).parts
    shape = (len(rel) == 2 and rel[0] == "runs") or (
        len(rel) == 6 and rel[:2] == ("output", "experiments")
        and rel[2] in ALLOWED_EXPERIMENT_UIDS
        and rel[3:5] == ("worktree", "runs"))
    if not shape:
        raise ValueError(f"site job run dir is not an official run-bundle path: {run}")
    script = (launch or run / "launch.sh").resolve()
    if script != run / "launch.sh" or not script.is_file() or script.is_symlink():
        raise ValueError("site job launch script must be the run's launch.sh")
    metadata_path = run / "metadata.json"
    manifest_path = run / "manifest.resolved.toml"
    if (not metadata_path.is_file() or not manifest_path.is_file()
            or metadata_path.is_symlink() or manifest_path.is_symlink()):
        raise ValueError("site job requires prepared metadata and resolved manifest")
    metadata = read(metadata_path)
    with manifest_path.open("rb") as stream:
        manifest = tomllib.load(stream)
    if metadata.get("run_dir") != str(run) or manifest.get("resolved", {}).get("run_dir") != str(run):
        raise ValueError("prepared run path disagrees with metadata/manifest")
    if metadata.get("mode") != "sbatch" or manifest.get("resolved", {}).get("mode") != "sbatch":
        raise ValueError("site job must preserve archived sbatch launch mode")
    if command == "submit" and metadata.get("status") != "prepared":
        raise ValueError("site job submit requires metadata status=prepared")
    if run.name in CPU_PROBES:
        expected_probe_parent = (EXPERIMENT_WORKTREES["sana2ba100-cache_ca-0002"] / "runs"
                                 if run.name in ("site-transport-broker-cpu-03", "site-transport-broker-cpu-04")
                                 else COORD / "runs")
        if (metadata.get("site_cpu_probe") is not True
                or manifest.get("site_cpu_probe", {}).get("enabled") is not True
                or run.parent != expected_probe_parent
                or file_sha(script) != CPU_PROBES[run.name]):
            raise ValueError("CPU-only probe launch hash/identity differs")
    else:
        profile = manifest.get("resolved_profile", {})
        env = profile.get("env", {})
        runtime = (run.parent.parent if len(rel) == 6 else COORD) / "models/sana_video_2b_a100/baseline"
        expected_script = runtime / "scripts/run_sana_video_2b_gpu.sh"
        if (manifest.get("model_profile") != "sana_video_2b_a100"
                or profile.get("model_profile") != "sana_video_2b_a100"
                or profile.get("submodule") != "models/sana_video_2b_a100/baseline"
                or profile.get("run_script") != "scripts/run_sana_video_2b_gpu.sh"
                or manifest.get("resolved", {}).get("run_script") != str(expected_script)
                or metadata.get("runtime_root") != str(runtime)
                or metadata.get("launch_script") != str(script)
                or metadata.get("config_id") != manifest.get("id")
                or profile.get("official_config") != OFFICIAL
                or metadata.get("official_config") != OFFICIAL):
            raise ValueError("site job is not a prepared SANA-Video 2B A100 bundle")
        if any(env.get(key) != value for key, value in REQUIRED_ENV.items()):
            raise ValueError("site job pinned model/python/offline env differs")
        if env.get("SOL_A100_BUDGET_PHASE") not in (None, "phase2"):
            raise ValueError("site job has an unknown GPU budget phase")
        if (env.get("AUTOVIDEO_REPO_ROOT") != str(runtime.parents[2])
                or env.get("AUTOVIDEO_RUNTIME_ROOT") != str(runtime)
                or env.get("AUTOVIDEO_CONFIG_ID") != manifest.get("id")):
            raise ValueError("site job root/config identity env differs")
        allowed_env = set(REQUIRED_ENV) | SAFE_EXTRA_ENV | {
            "AUTOVIDEO_REPO_ROOT", "AUTOVIDEO_RUNTIME_ROOT", "AUTOVIDEO_CONFIG_ID",
            "SOL_A100_BUDGET_PHASE"}
        unknown = [key for key in env if key not in allowed_env and not re.fullmatch(r"SGLANG_SANA_[A-Z0-9_]+", key)]
        if unknown or any(value == "<redacted>" for value in env.values()):
            raise ValueError(f"site job has unsafe or redacted environment variables: {unknown}")
        expected_launch = render_launch_script(run, runtime, expected_script, env, run / "outputs")
        if script.read_text() != expected_launch:
            raise ValueError("site job launch.sh differs from the archived launcher template")
    return run, script


def stamps(run: Path, script: Path) -> dict:
    return {"launch_sha256": file_sha(script),
            "manifest_sha256": file_sha(run / "manifest.resolved.toml"),
            "metadata_sha256": file_sha(run / "metadata.json")}


def dispatch(request: dict, queue_root: Path) -> dict:
    command = request["command"]
    if command not in ("submit", "status", "sync", "assess_quality", "reassess_quality"):
        raise ValueError("operation outside job broker whitelist")
    run, script = valid_run(Path(request["run_dir"]),
                            Path(request["launch_script"]) if request["launch_script"] else None,
                            command)
    if queue_root == QUEUE:
        if command == "reassess_quality":
            if run.parent != EXPERIMENT_WORKTREES["sana2ba100-cache_ca-0002"] / "runs":
                raise ValueError("coordinator reverify requires the exact seq0002 Cache run parent")
        elif run.parent != COORD / "runs":
            raise ValueError("coordinator queue may only submit coordinator runs")
    else:
        if command == "reassess_quality":
            raise ValueError("independent reverify cannot originate in an Executor queue")
        allowed_run_parent = queue_root.parent.parent / "runs"
        if run.parent != allowed_run_parent:
            raise ValueError("Executor queue may only submit its own worktree runs")
    if stamps(run, script) != {key: request[key] for key in ("launch_sha256", "manifest_sha256", "metadata_sha256")}:
        raise ValueError("prepared run bundle changed after request SHA256 lock")
    if command == "assess_quality":
        if queue_root != EXPERIMENT_QUEUES["sana2ba100-cache_ca-0002"]:
            raise ValueError("CPU quality route requires the exact seq0002 Cache queue")
        from orchestration import site_quality_transport
        return site_quality_transport.assess(run)
    if command == "reassess_quality":
        from orchestration import site_reverify_transport
        return site_reverify_transport.assess(run)
    if command == "submit":
        single_flight = None
        if run.name not in CPU_PROBES:
            assert_gpu_submit_budget(run)
            if run.parent == R4_RUN.parent:
                # Check *all* prior GPU jobs against the host's own start
                # audit, not Executor-edited metadata or a single R4 path.
                # Otherwise R8 could overlap R7 if R4 alone finishes first.
                # Trusted remote status/SSH failure prevents any transfer.
                single_flight = assert_kernel_single_flight_grant(run)
        result = site_remote._submit_direct(run, script)
        if single_flight is not None:
            result["site_single_flight_preflight"] = single_flight
        return result
    if command == "status":
        return site_remote._status_direct(run)
    return site_remote._sync_run_direct(run)


def heartbeat(stop: threading.Event, state: dict) -> None:
    while not stop.is_set():
        write(HEARTBEAT, {"daemon_pid": state["pid"], "started_at_utc": state["started_at_utc"],
                          "utc": utc(), "unix_ns": time.time_ns()})
        stop.wait(1.0)


def serve() -> int:
    queue_roots = (QUEUE, *EXPERIMENT_QUEUES.values())
    for root in queue_roots:
        if root.is_symlink() or (root.exists() and root.resolve() != root):
            raise RuntimeError(f"site job queue path is not a real owned directory: {root}")
        (root / "requests").mkdir(parents=True, exist_ok=True)
        (root / "responses").mkdir(parents=True, exist_ok=True)
        if any((root / part).is_symlink() for part in ("requests", "responses")):
            raise RuntimeError(f"site job queue subdirectory is a symlink: {root}")
    if str(COORD) not in sys.path:
        sys.path.insert(0, str(COORD))
    os.environ.pop("SOL_SITE_JOB_BROKER", None)
    state = {"schema_version": 1, "pid": os.getpid(), "host": socket.gethostname(),
             "started_at_utc": utc(), "started_monotonic_ns": time.monotonic_ns(),
             "allowed_operations": ["submit", "status", "sync", "assess_quality",
                                    "reassess_quality"]}
    write(STATE, state)
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop, state), daemon=True).start()
    while True:
        pending = ((root, path) for root in queue_roots
                   for path in sorted((root / "requests").glob("*.json")))
        for root, path in pending:
            claimed = path.with_suffix(".processing")
            try:
                path.rename(claimed)
            except FileNotFoundError:
                continue
            request = {}
            try:
                if claimed.is_symlink():
                    raise ValueError("site job request cannot be a symlink")
                request = read(claimed)
                if set(request) != {"request_id", "command", "run_dir", "launch_script",
                                    "launch_sha256", "manifest_sha256", "metadata_sha256",
                                    "sent_at_utc", "caller_pid"}:
                    raise ValueError("unexpected site job request fields")
                if not re.fullmatch(r"[0-9a-f]{32}", str(request.get("request_id", ""))):
                    raise ValueError("invalid site job request ID")
                result = dispatch(request, root)
                response = {"ok": True, "result": result}
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()[-1600:]}
            response.update({"schema_version": 1, "request_id": request.get("request_id", path.stem),
                             "request_sha256": digest(request), "daemon_pid": state["pid"],
                             "daemon_started_at_utc": state["started_at_utc"],
                             "responded_at_utc": utc(), "responded_monotonic_ns": time.monotonic_ns()})
            response["response_sha256"] = digest(response)
            write(root / "responses" / f"{response['request_id']}.json", response)
            AUDIT.parent.mkdir(parents=True, exist_ok=True)
            with AUDIT.open("a") as stream:
                stream.write(json.dumps(response, sort_keys=True) + "\n")
        time.sleep(0.2)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        proc = Path(f"/proc/{pid}")
        state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
        command = (proc / "cmdline").read_bytes().replace(b"\x00", b" ").decode()
        return state != "Z" and str(COORD / "orchestration/site_job_broker.py") in command and " serve" in command
    except OSError:
        return False


def start_daemon() -> dict:
    state = read(STATE)
    if state and process_alive(int(state["pid"])):
        return {"status": "already_running", **state}
    QUEUE.mkdir(parents=True, exist_ok=True)
    log = (QUEUE / "daemon.log").open("a")
    child = subprocess.Popen([sys.executable, __file__, "serve"], cwd=COORD,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                             start_new_session=True)
    log.close()
    for _ in range(50):
        state = read(STATE)
        if state.get("pid") == child.pid:
            return {"status": "started", **state}
        time.sleep(0.1)
    raise RuntimeError("site job broker did not publish startup receipt")


def request(command: str, run_dir: Path, launch_script: Path | None = None) -> dict:
    run, script = valid_run(run_dir, launch_script, command)
    selected_queue = client_queue()
    state, beat = read(STATE), read(HEARTBEAT)
    if (not state or beat.get("daemon_pid") != state.get("pid")
            or beat.get("started_at_utc") != state.get("started_at_utc")
            or time.time_ns() - int(beat.get("unix_ns", 0)) > 5 * 10**9):
        raise RuntimeError("site job broker host daemon heartbeat absent or stale")
    request_id = uuid.uuid4().hex
    payload = {"request_id": request_id, "command": command, "run_dir": str(run),
               "launch_script": str(script), **stamps(run, script),
               "sent_at_utc": utc(), "caller_pid": os.getpid()}
    selected_requests = selected_queue / "requests"
    selected_responses = selected_queue / "responses"
    selected_requests.mkdir(parents=True, exist_ok=True)
    selected_responses.mkdir(parents=True, exist_ok=True)
    temp = selected_requests / f".{request_id}.tmp"
    write(temp, payload)
    temp.replace(selected_requests / f"{request_id}.json")
    response_path = selected_responses / f"{request_id}.json"
    deadline = time.monotonic() + (1800 if command in ("assess_quality", "reassess_quality") else
                                   900 if command == "submit" else 120)
    while time.monotonic() < deadline:
        if response_path.exists():
            response = read(response_path)
            observed_hash = response.pop("response_sha256", None)
            if (observed_hash != digest(response)
                    or response.get("request_sha256") != digest(payload)):
                raise RuntimeError("site job broker request/response hash mismatch")
            response["response_sha256"] = observed_hash
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error")))
            return {**response["result"], "site_job_broker_receipt": response}
        time.sleep(0.1)
    raise TimeoutError("site job broker response timeout")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("daemon-start", "serve"))
    args = parser.parse_args()
    result = start_daemon() if args.command == "daemon-start" else serve()
    if isinstance(result, dict):
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
