#!/usr/bin/env python3
"""Host-side Codex exec broker preserving Symposium goal start/status lifecycle.

Site adaptation: the A100 node has no Codex CLI/auth or tmux autorun. This
broker uses the existing host login, detached Codex exec processes, and the
same goal.md/context.json, worktree, name, and JSON status contract. It does
not propose candidates, alter prompts, or run GPU work itself.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

COORD_ROOT = Path("/home/lyy/Experiments/SolAgent/20260913-a100-original-baseline/repo")
PROJECT = COORD_ROOT.parent
STATES = COORD_ROOT / ".symposium/site-goal-sessions"
EVENTS = PROJECT / "telemetry/local_events.jsonl"
QUEUE = COORD_ROOT / ".symposium/site-broker"
REQUESTS = QUEUE / "requests"
RESPONSES = QUEUE / "responses"
DAEMON_STATE = QUEUE / "daemon.json"
HEARTBEAT = QUEUE / "heartbeat.json"
AUDIT = PROJECT / "telemetry/broker_requests.jsonl"
VALID_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
FRESH_MASTER = COORD_ROOT / "output/orchestrated/sana2ba100-20260913-093521/master"
FRESH_EXECUTORS = {
    "sana2ba100-kernel_aw-0002": "kernel_aw",
    "sana2ba100-cache_ca-0002": "cache_ca",
}
PHASE2_IMAGE_DIR = PROJECT / "artifacts/phase2/r9_auth"
PHASE2_IMAGE_SHA256 = (
    ("pair-p1.png", "9a1811871f1fbc06604dd7ce0504da75ba9363ef206d5f797decc37bbda2a643"),
    ("pair-p2.png", "5a6a2b51e01ace81eaf7a4c1b767d92417455465ff077fdb6a9c0908b45d7145"),
    ("pair-p3.png", "7f9e90282f517ebf40d359837aa5ae1f160e3d65ed9e896f6cbd007bffde7e24"),
    ("pair-p4.png", "3ca7ce77623bc8c6e0284d914d3a0b73be8083e600e464daf9de3b672a761a6e"),
    ("pair-p5.png", "748e90d46714c5b45087481a4fa1d2f364922fb756838d07ccd40328db6c53e2"),
)
PHASE3_IMAGE_DIR = PROJECT / "artifacts/phase3/cache_image_transport/render_v1"
PHASE3_IMAGE_MANIFEST_SHA256 = "a78beed6a1d8af44c59edad5741bfc9da07a9931ecb64f989905f937ca0f2d06"
PHASE3_ATTACHMENTS = PROJECT / "artifacts/phase3/cache_image_transport/attachments"
PHASE3_VISUAL_SESSIONS = {"sana2ba100-master-0002", "sana2ba100-cache_ca-0002"}
PHASE3_MULTI_SOURCE = COORD_ROOT / "tools/symposium/site_phase3_multi_image.py"
PHASE3_MULTI_SOURCE_SHA256 = "d6f145a19b45c9f7a7295daa89d49d2e96b5c3bca7141c8a85136f19ecb036bd"
PHASE3_TRIPLE_SOURCE = COORD_ROOT / "tools/symposium/site_phase3_multi_image_r06_r07_r08.py"
PHASE3_TRIPLE_SOURCE_SHA256 = "05f5c6a99b0b0f06b565ce7f1984a1e07a3eea6618c1c21a9e682f03a0c2e61d"
PHASE3_MASTER_CACHE_SOURCE = COORD_ROOT / "tools/symposium/site_phase3_master_cache_images.py"
PHASE3_MASTER_CACHE_SOURCE_SHA256 = "6423e146747bbd24f2a381b880dc12b1dcf2c0f31acdfee5262cfb807a89479d"
PHASE3_REVERIFY_SHIM = COORD_ROOT / "orchestration/site_reverify_python.py"
PHASE3_REVERIFY_SHIM_SHA256 = "8ed13b96eac075da0e9a74265b80077ac92ab975899d266a70fd9c0a484bd317"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(event_name: str, **fields: object) -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    row = {"event": event_name, "utc": utc(), "host": socket.gethostname(),
           "monotonic_ns": time.monotonic_ns(), **fields}
    with EVENTS.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)


def owned(raw: str) -> Path:
    path = Path(raw).resolve()
    if not path.is_relative_to(COORD_ROOT):
        raise ValueError(f"broker path must be in the new coordinator repo: {path}")
    return path


def paths(args: argparse.Namespace) -> tuple[Path, Path, Path, str]:
    worktree = owned(args.worktree or str(COORD_ROOT))
    goal = owned(args.goal_dir)
    name = args.name or goal.name
    if worktree == COORD_ROOT:
        allowed = name == "sana2ba100-master-0002" and goal == FRESH_MASTER
    else:
        tech = FRESH_EXECUTORS.get(name)
        expected_worktree = COORD_ROOT / "output/experiments" / name / "worktree"
        allowed = tech is not None and worktree == expected_worktree and goal == worktree / "goals" / tech
    if not allowed or not worktree.is_dir() or not goal.is_dir() or worktree.is_symlink() or goal.is_symlink():
        raise ValueError("site Codex broker requires an exact fresh seq0002 name/worktree/goal identity")
    if not (goal / "goal.md").is_file() or not (goal / "context.json").is_file():
        raise ValueError("goal directory requires goal.md and context.json")
    if not VALID_NAME.fullmatch(name):
        raise ValueError(f"invalid session name: {name}")
    return worktree, goal, STATES / f"{name}.json", name


def read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        stat = (Path("/proc") / str(pid) / "stat").read_text()
        return stat.rsplit(") ", 1)[1].split()[0] != "Z"
    except (ProcessLookupError, OSError):
        return False


def phase2_master_image_inputs(name: str, attempt: int) -> list[dict[str, str]]:
    """Attach only the five hash-locked, Master-made R9/dense sheets on attempt 8.

    This is a site transport correction for the archived Master's visual gate,
    not an outer quality judgement or a new optimization prompt. Executors and
    every other Master attempt receive no image attachments.
    """
    if name != "sana2ba100-master-0002" or attempt != 8:
        return []
    manifest_path = PHASE2_IMAGE_DIR / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError("phase2 image manifest missing or symlinked")
    manifest = read(manifest_path)
    entries = manifest.get("image_order")
    if (manifest.get("session") != name or manifest.get("attempt") != attempt
            or manifest.get("prompts_sha256") !=
            "de249238cd19dd5b77de61eb69fd114228d4813a0335a1c794e366bbcd00e7e0"
            or manifest.get("seed") != 42 or not isinstance(entries, list)
            or len(entries) != len(PHASE2_IMAGE_SHA256)):
        raise RuntimeError("phase2 image manifest identity/workload mismatch")
    images = []
    for index, ((filename, expected_sha), entry) in enumerate(
            zip(PHASE2_IMAGE_SHA256, entries), start=1):
        if (entry.get("prompt_index") != index or entry.get("file") != filename
                or entry.get("sha256") != expected_sha):
            raise RuntimeError(f"phase2 image {index} manifest mismatch")
        path = PHASE2_IMAGE_DIR / filename
        if not path.is_file() or path.is_symlink() or path.resolve().parent != PHASE2_IMAGE_DIR.resolve():
            raise RuntimeError(f"phase2 image {index} path missing or unsafe")
        if path.stat().st_size > 8 * 1024 * 1024:
            raise RuntimeError(f"phase2 image {index} exceeds 8 MiB")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha:
            raise RuntimeError(f"phase2 image {index} content hash mismatch")
        images.append({"path": str(path), "sha256": digest, "prompt_index": index})
    return images


def phase3_cache_image_inputs(name: str, attempt: int,
                              *, allow_prepared: bool = False) -> list[dict[str, str]]:
    """Attach only a coordinator-approved, exact-session Cache/dense visual set.

    The immutable source/render manifest and images live outside the Agent's
    writable worktree. A prepared manifest is visible to dry-run but cannot be
    used by a real Codex invocation until separately marked approved.
    """
    if name not in PHASE3_VISUAL_SESSIONS or attempt < 8:
        return []
    path = PHASE3_ATTACHMENTS / f"{name}-attempt-{attempt:02d}.json"
    if not path.exists():
        return []
    if (PHASE3_ATTACHMENTS.is_symlink() or PHASE3_IMAGE_DIR.is_symlink()
            or not path.is_file() or path.is_symlink()
            or path.resolve().parent != PHASE3_ATTACHMENTS.resolve()):
        raise RuntimeError("phase3 attachment manifest path is missing or unsafe")
    attachment_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    attachment = read(path)
    if (attachment.get("schema_version") != 1
            or attachment.get("session") != name or attachment.get("attempt") != attempt
            or attachment.get("round") not in ("r01", "r02")
            or attachment.get("render_manifest_sha256") != PHASE3_IMAGE_MANIFEST_SHA256
            or attachment.get("status") not in (("approved", "prepared") if allow_prepared else ("approved",))):
        raise RuntimeError("phase3 attachment manifest identity/approval mismatch")
    source_manifest_path = PHASE3_IMAGE_DIR / "render_manifest.json"
    if (not source_manifest_path.is_file() or source_manifest_path.is_symlink()
            or hashlib.sha256(source_manifest_path.read_bytes()).hexdigest()
            != PHASE3_IMAGE_MANIFEST_SHA256):
        raise RuntimeError("phase3 source/render manifest SHA mismatch")
    source_manifest = read(source_manifest_path)
    if (source_manifest.get("schema_version") != 1
            or source_manifest.get("status") != "RENDERED_NOT_ATTACHED_TO_AGENT"
            or source_manifest.get("target_session_attempt") is not None):
        raise RuntimeError("phase3 render manifest semantic mismatch")
    expected = [row for row in source_manifest["images"]
                if row["round"] == attachment["round"]]
    if (len(expected) != 10
            or [(row["prompt_index"], row["kind"]) for row in expected]
            != [(prompt, kind) for prompt in range(1, 6)
                for kind in ("overview", "detail")]):
        raise RuntimeError("phase3 rendered image set is not five prompt overview/detail pairs")
    requested = attachment.get("image_order")
    if (not isinstance(requested, list) or requested !=
            [{"file": row["file"], "sha256": row["sha256"]} for row in expected]):
        raise RuntimeError("phase3 attachment image order/SHA differs from fixed render")
    checked_sources = set()
    for row in expected:
        for tile in row["tiles"]:
            source = COORD_ROOT / tile["source"]
            if source in checked_sources:
                continue
            if (not source.is_file() or source.is_symlink()
                    or not source.resolve().is_relative_to(COORD_ROOT)
                    or hashlib.sha256(source.read_bytes()).hexdigest() != tile["source_sha256"]):
                raise RuntimeError(f"phase3 rendered tile source SHA/path changed: {source}")
            checked_sources.add(source)
    images = []
    for row in expected:
        image = PHASE3_IMAGE_DIR / row["file"]
        if (not image.is_file() or image.is_symlink()
                or image.resolve().parent != PHASE3_IMAGE_DIR.resolve()
                or image.stat().st_size != row["size_bytes"]
                or image.stat().st_size > 8 * 1024 * 1024
                or hashlib.sha256(image.read_bytes()).hexdigest() != row["sha256"]):
            raise RuntimeError(f"phase3 image path/size/SHA mismatch: {image}")
        images.append({"path": str(image), "sha256": row["sha256"],
                       "prompt_index": row["prompt_index"], "kind": row["kind"],
                       "round": row["round"],
                       "attachment_manifest_sha256": attachment_sha})
    return images


def image_inputs(name: str, attempt: int,
                 *, allow_prepared: bool = False) -> list[dict[str, str]]:
    # Master attempts 12 and 13 each receive their own exact full Cache DELIVERY
    # frontier, independently from any Executor image input. No other Master
    # attempt inherits this site image transport.
    if name == "sana2ba100-master-0002" and attempt in (12, 13):
        if (not PHASE3_MASTER_CACHE_SOURCE.is_file()
                or PHASE3_MASTER_CACHE_SOURCE.is_symlink()
                or hashlib.sha256(PHASE3_MASTER_CACHE_SOURCE.read_bytes()).hexdigest()
                != PHASE3_MASTER_CACHE_SOURCE_SHA256):
            raise RuntimeError("Master Cache image helper source SHA mismatch")
        if str(COORD_ROOT) not in sys.path:
            sys.path.insert(0, str(COORD_ROOT))
        from tools.symposium import site_phase3_master_cache_images
        if Path(site_phase3_master_cache_images.__file__).resolve() != PHASE3_MASTER_CACHE_SOURCE:
            raise RuntimeError("Master Cache image helper import path mismatch")
        return site_phase3_master_cache_images.inputs(name, attempt,
                                                       allow_prepared=allow_prepared)
    # The exact Cache attempt 11 has a separate approved 30-image R06/R07/R08
    # route. A failed pin or approval check must never fall through to the old
    # prepared R06/R07-only manifest or the earlier v1 route.
    if name == "sana2ba100-cache_ca-0002" and attempt == 11:
        if (not PHASE3_TRIPLE_SOURCE.is_file() or PHASE3_TRIPLE_SOURCE.is_symlink()
                or hashlib.sha256(PHASE3_TRIPLE_SOURCE.read_bytes()).hexdigest()
                != PHASE3_TRIPLE_SOURCE_SHA256):
            raise RuntimeError("phase3 triple-image transport source SHA mismatch")
        if str(COORD_ROOT) not in sys.path:
            sys.path.insert(0, str(COORD_ROOT))
        from tools.symposium import site_phase3_multi_image_r06_r07_r08
        if Path(site_phase3_multi_image_r06_r07_r08.__file__).resolve() != PHASE3_TRIPLE_SOURCE:
            raise RuntimeError("phase3 triple-image transport import path mismatch")
        return site_phase3_multi_image_r06_r07_r08.inputs(
            name, attempt, allow_prepared=allow_prepared)
    # The exact Cache attempt 10 has its own separately reviewed, 20-image
    # R03/R05 transport. Never fall through to the older R01/R02 v1 manifest
    # if its source, approval, or frame pins fail.
    if name == "sana2ba100-cache_ca-0002" and attempt == 10:
        if (not PHASE3_MULTI_SOURCE.is_file() or PHASE3_MULTI_SOURCE.is_symlink()
                or hashlib.sha256(PHASE3_MULTI_SOURCE.read_bytes()).hexdigest()
                != PHASE3_MULTI_SOURCE_SHA256):
            raise RuntimeError("phase3 multi-image transport source SHA mismatch")
        # Direct `python tools/symposium/site_codex_goal_session.py ...` puts
        # only this script directory on sys.path. Resolve the reviewed helper
        # from the fixed coordinator checkout, never an Executor worktree.
        if str(COORD_ROOT) not in sys.path:
            sys.path.insert(0, str(COORD_ROOT))
        from tools.symposium import site_phase3_multi_image
        if Path(site_phase3_multi_image.__file__).resolve() != PHASE3_MULTI_SOURCE:
            raise RuntimeError("phase3 multi-image transport import path mismatch")
        return site_phase3_multi_image.inputs(name, attempt,
                                              allow_prepared=allow_prepared)
    phase2 = phase2_master_image_inputs(name, attempt)
    phase3 = phase3_cache_image_inputs(name, attempt, allow_prepared=allow_prepared)
    if phase2 and phase3:
        raise RuntimeError("multiple phase image transports cannot be combined")
    return phase2 or phase3


def master12_reverify_env(name: str, attempt: int) -> tuple[dict[str, str], dict]:
    if name != "sana2ba100-master-0002" or attempt not in (12, 13):
        return {}, {}
    if (not PHASE3_REVERIFY_SHIM.is_file() or PHASE3_REVERIFY_SHIM.is_symlink()
            or hashlib.sha256(PHASE3_REVERIFY_SHIM.read_bytes()).hexdigest()
            != PHASE3_REVERIFY_SHIM_SHA256):
        raise RuntimeError("Master12 fixed reverify shim source SHA mismatch")
    if str(COORD_ROOT) not in sys.path:
        sys.path.insert(0, str(COORD_ROOT))
    from orchestration import site_reverify_transport
    names = ("20260914-054324-cache_ca_r05_taylorseer_t0180_w10_o1",
             "20260914-083326-cache_ca_r08_taylorseer_t0180_w30_o1")
    if set(site_reverify_transport.APPROVED_POINTS) != set(names):
        raise RuntimeError("Master12 reverify approval set differs from delivered frontier")
    for run_name in names:
        site_reverify_transport.assert_approved(site_reverify_transport.RUNS / run_name)
    trusted = site_reverify_transport.trusted_remote_worker()
    return {"PLAN_EVAL_PYTHON": str(PHASE3_REVERIFY_SHIM)}, {
        "shim": str(PHASE3_REVERIFY_SHIM), "shim_sha256": PHASE3_REVERIFY_SHIM_SHA256,
        "trusted_remote_worker": trusted, "exact_run_names": names,
    }


def start(args: argparse.Namespace) -> dict:
    worktree, goal, state_path, name = paths(args)
    previous = read(state_path)
    prior_pid = previous.get("pid")
    if isinstance(prior_pid, int) and process_alive(prior_pid):
        if not args.force:
            raise RuntimeError(f"session already alive: {name} pid={prior_pid}")
        os.killpg(prior_pid, signal.SIGTERM)
        for _ in range(50):
            if not process_alive(prior_pid):
                break
            time.sleep(0.1)
        if process_alive(prior_pid):
            raise RuntimeError(f"prior session did not stop safely: {name} pid={prior_pid}")
    attempt = int(previous.get("attempt", 0)) + 1
    image_inputs(name, attempt)
    master12_reverify_env(name, attempt)
    worker = subprocess.Popen([sys.executable, __file__, "worker", str(goal),
                               "--name", name, "--worktree", str(worktree),
                               "--attempt", str(attempt)],
                              cwd=worktree, stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              start_new_session=True,
                              env={**os.environ, "SOL_SITE_CODEX_BROKER": "1",
                                   "SOL_SITE_REMOTE": "1", "SOL_SITE_JOB_BROKER": "1"})
    context = read(goal / "context.json")
    data = {"session": name, "goal_dir": str(goal), "worktree": str(worktree),
            "state_file": str(state_path), "pid": worker.pid, "attempt": attempt,
            "role": context.get("role", "implementation"), "status": "starting",
            "resource_state": "active", "executor": "site-codex-exec-broker",
            "sandbox": "workspace-write", "started_at_utc": utc(),
            "started_monotonic_ns": time.monotonic_ns(), "host": socket.gethostname()}
    write(state_path, data)
    emit("agent_session_start", name=name, role=data["role"], pid=worker.pid,
         attempt=attempt, worktree=str(worktree), goal_dir=str(goal))
    return {**data, "alive": process_alive(worker.pid)}


def worker(args: argparse.Namespace) -> int:
    worktree, goal, state_path, name = paths(args)
    attempt = args.attempt
    context = read(goal / "context.json")
    prefix = goal / f"agent_attempt_{attempt:02d}"
    raw_path = prefix.with_suffix(".jsonl")
    timed_path = goal / f"agent_attempt_{attempt:02d}.timed.jsonl"
    last_path = goal / f"agent_attempt_{attempt:02d}.last.md"
    receipt_path = goal / f"agent_attempt_{attempt:02d}.receipt.json"
    transport_images = image_inputs(name, attempt)
    reverify_env, reverify_transport = master12_reverify_env(name, attempt)
    command = [
        "codex", "exec", "-s", "workspace-write", "-C", str(worktree),
    ]
    if transport_images:
        command += ["--image", *(image["path"] for image in transport_images)]
    command += ["--json", "-o", str(last_path), "-"]
    start_ns = time.monotonic_ns()
    emit("agent_api_begin", name=name, role=context.get("role"), attempt=attempt,
         command=command, pid=os.getpid(), transport_images=transport_images,
         site_reverify_transport=reverify_transport)
    with raw_path.open("w") as raw, timed_path.open("w") as timed:
        proc = subprocess.Popen(command, cwd=worktree,
                                env={**os.environ, "SOL_SITE_CODEX_BROKER": "1",
                                     "SOL_SITE_REMOTE": "1", "SOL_SITE_JOB_BROKER": "1",
                                     **reverify_env},
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write((goal / "goal.md").read_text())
        proc.stdin.close()
        for line in proc.stdout:
            raw.write(line)
            raw.flush()
            stamp = {"utc": utc(), "host": socket.gethostname(),
                     "monotonic_ns": time.monotonic_ns(), "session": name,
                     "attempt": attempt, "line": line.rstrip("\n")}
            try:
                codex_event = json.loads(line)
            except json.JSONDecodeError:
                codex_event = None
            if codex_event:
                stamp["codex_type"] = codex_event.get("type")
                stamp["item_type"] = (codex_event.get("item") or {}).get("type")
            timed.write(json.dumps(stamp, sort_keys=True) + "\n")
            timed.flush()
        code = proc.wait()
    done_ns = time.monotonic_ns()
    receipt = {"schema_version": 1, "session": name, "role": context.get("role"),
               "attempt": attempt, "worker_pid": os.getpid(), "codex_pid": proc.pid,
               "exit_code": code, "started_monotonic_ns": start_ns,
               "finished_monotonic_ns": done_ns, "duration_s": (done_ns - start_ns) / 1e9,
               "finished_at_utc": utc(), "raw_events": str(raw_path),
               "timed_events": str(timed_path), "last_message": str(last_path),
               "command": command, "sandbox": "workspace-write",
               "transport_images": transport_images,
               "site_reverify_transport": reverify_transport}
    write(receipt_path, receipt)
    state = read(state_path)
    if state.get("attempt") == attempt:
        state.update({"status": "finished" if code == 0 else "failed",
                      "exit_code": code, "finished_at_utc": receipt["finished_at_utc"],
                      "receipt": str(receipt_path)})
        write(state_path, state)
    emit("agent_api_end", name=name, role=context.get("role"), attempt=attempt,
         exit_code=code, duration_s=receipt["duration_s"], pid=os.getpid())
    return code


def status(args: argparse.Namespace) -> dict:
    worktree, goal, state_path, name = paths(args)
    state = read(state_path)
    pid = state.get("pid")
    attempt = state.get("attempt", 0)
    receipt = read(goal / f"agent_attempt_{attempt:02d}.receipt.json") if attempt else {}
    return {"session": name, "goal_dir": str(goal), "state_file": str(state_path),
            "alive": bool(isinstance(pid, int) and process_alive(pid) and not receipt),
            "state": state, "receipt": receipt,
            "checked_at_utc": utc()}


def capture(args: argparse.Namespace) -> str:
    _, goal, _, _ = paths(args)
    state = status(args)["state"]
    path = goal / f"agent_attempt_{int(state.get('attempt', 0)):02d}.jsonl"
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-args.lines:])


def stop(args: argparse.Namespace) -> dict:
    _, _, state_path, name = paths(args)
    state = read(state_path)
    pid = state.get("pid")
    was_alive = isinstance(pid, int) and process_alive(pid)
    if was_alive:
        os.killpg(pid, signal.SIGTERM)
    state.update({"status": "stopped", "resource_state": "stopped", "stopped_at_utc": utc()})
    write(state_path, state)
    emit("agent_session_stop", name=name, pid=pid, alive_before=was_alive)
    return {"session": name, "alive_before": was_alive, "alive_after": process_alive(pid) if was_alive else False}


def canonical(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def digest(data: dict) -> str:
    return hashlib.sha256(canonical(data)).hexdigest()


def dispatch(args: argparse.Namespace) -> dict:
    if args.command in ("start", "resume"):
        if args.command == "resume":
            args.force = True
        return start(args)
    if args.command == "status":
        return status(args)
    if args.command == "stop":
        return stop(args)
    raise ValueError(f"broker command not allowed: {args.command}")


def serve() -> int:
    REQUESTS.mkdir(parents=True, exist_ok=True)
    RESPONSES.mkdir(parents=True, exist_ok=True)
    state = {"schema_version": 1, "pid": os.getpid(), "host": socket.gethostname(),
             "started_at_utc": utc(), "started_monotonic_ns": time.monotonic_ns(),
             "allowed_commands": ["start", "status", "stop", "resume"]}
    write(DAEMON_STATE, state)
    emit("broker_daemon_start", pid=os.getpid())
    last_heartbeat = 0.0
    while True:
        if time.monotonic() - last_heartbeat >= 1.0:
            write(HEARTBEAT, {"daemon_pid": os.getpid(), "utc": utc(),
                              "unix_ns": time.time_ns(),
                              "started_at_utc": state["started_at_utc"]})
            last_heartbeat = time.monotonic()
        for path in sorted(REQUESTS.glob("*.json")):
            claimed = path.with_suffix(".processing")
            try:
                path.rename(claimed)
            except FileNotFoundError:
                continue
            request = read(claimed)
            request_hash = digest(request)
            request_id = request.get("request_id")
            try:
                if not isinstance(request_id, str) or not re.fullmatch(r"[0-9a-f]{32}", request_id):
                    raise ValueError("invalid request ID")
                if request.get("command") not in state["allowed_commands"]:
                    raise ValueError("command outside fixed broker whitelist")
                if set(request) != {"request_id", "command", "goal_dir", "worktree", "name", "force", "sent_at_utc", "caller_pid"}:
                    raise ValueError("unexpected broker request fields")
                if type(request["force"]) is not bool:
                    raise ValueError("force must be boolean")
                args = argparse.Namespace(**request)
                result = dispatch(args)
                response = {"ok": True, "result": result}
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()[-1500:]}
            response.update({"schema_version": 1, "request_id": request_id,
                             "request_sha256": request_hash, "daemon_pid": os.getpid(),
                             "daemon_started_at_utc": state["started_at_utc"],
                             "responded_at_utc": utc(), "responded_monotonic_ns": time.monotonic_ns()})
            response["response_sha256"] = digest(response)
            response_path = RESPONSES / f"{request_id}.json"
            write(response_path, response)
            AUDIT.parent.mkdir(parents=True, exist_ok=True)
            with AUDIT.open("a") as stream:
                stream.write(json.dumps(response, sort_keys=True) + "\n")
            emit("broker_request_response", request_id=request_id,
                 command=request.get("command"), ok=response["ok"],
                 daemon_pid=os.getpid(), request_sha256=request_hash,
                 response_sha256=response["response_sha256"])
        time.sleep(0.2)


def start_daemon() -> dict:
    if DAEMON_STATE.exists():
        state = read(DAEMON_STATE)
        if process_alive(int(state.get("pid", -1))):
            return {"status": "already_running", **state}
    REQUESTS.mkdir(parents=True, exist_ok=True)
    RESPONSES.mkdir(parents=True, exist_ok=True)
    log = (QUEUE / "daemon.log").open("a")
    child = subprocess.Popen([sys.executable, __file__, "serve"], cwd=COORD_ROOT,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                             start_new_session=True)
    log.close()
    for _ in range(50):
        if DAEMON_STATE.exists():
            state = read(DAEMON_STATE)
            if state.get("pid") == child.pid:
                return {"status": "started", **state}
        time.sleep(0.1)
    raise RuntimeError("host broker daemon did not publish startup receipt")


def client(args: argparse.Namespace) -> dict:
    # Path/name checks are repeated by the daemon; this side only writes a
    # typed request, never an arbitrary shell command or executable path.
    paths(args)
    state = read(DAEMON_STATE)
    heartbeat = read(HEARTBEAT)
    if (not state or heartbeat.get("daemon_pid") != state.get("pid")
            or heartbeat.get("started_at_utc") != state.get("started_at_utc")
            or time.time_ns() - int(heartbeat.get("unix_ns", 0)) > 5 * 10**9):
        raise RuntimeError("host broker daemon heartbeat is absent or stale")
    request_id = uuid.uuid4().hex
    request = {"request_id": request_id, "command": args.command,
               "goal_dir": str(owned(args.goal_dir)),
               "worktree": str(owned(args.worktree or str(COORD_ROOT))),
               "name": args.name, "force": bool(args.force),
               "sent_at_utc": utc(), "caller_pid": os.getpid()}
    REQUESTS.mkdir(parents=True, exist_ok=True)
    RESPONSES.mkdir(parents=True, exist_ok=True)
    temp = REQUESTS / f".{request_id}.tmp"
    final = REQUESTS / f"{request_id}.json"
    write(temp, request)
    temp.replace(final)
    response_path = RESPONSES / f"{request_id}.json"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if response_path.exists():
            response = read(response_path)
            hash_value = response.pop("response_sha256", None)
            if hash_value != digest(response) or response.get("request_sha256") != digest(request):
                raise RuntimeError("broker response/request receipt SHA256 mismatch")
            response["response_sha256"] = hash_value
            if not response.get("ok"):
                raise RuntimeError(str(response.get("error")))
            return {**response["result"], "broker_receipt": response}
        time.sleep(0.1)
    raise TimeoutError("host broker response did not arrive")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("daemon-start")
    subs.add_parser("serve")
    dry = subs.add_parser("image-dry-run")
    dry.add_argument("--name", required=True)
    dry.add_argument("--attempt", required=True, type=int)
    dry.add_argument("--allow-prepared", action="store_true")
    for command in ("start", "worker", "status", "stop", "resume"):
        sub = subs.add_parser(command)
        sub.add_argument("goal_dir")
        sub.add_argument("--name")
        sub.add_argument("--worktree")
        sub.add_argument("--force", action="store_true")
        sub.add_argument("--attempt", type=int, default=0)
        sub.add_argument("--lines", type=int, default=80)
        sub.add_argument("--note", default="")
    args = parser.parse_args()
    if args.command == "daemon-start":
        result = start_daemon()
    elif args.command == "serve":
        return serve()
    elif args.command == "worker":
        return worker(args)
    elif args.command == "image-dry-run":
        images = image_inputs(args.name, args.attempt, allow_prepared=args.allow_prepared)
        if not images:
            raise RuntimeError("no image transport is configured for this exact session/attempt")
        result = {"session": args.name, "attempt": args.attempt,
                  "status": "prepared_not_approved" if args.allow_prepared else "approved",
                  "image_count": len(images), "images": images,
                  "total_image_bytes": sum(Path(image["path"]).stat().st_size for image in images),
                  "command_fragment": ["--image", *(image["path"] for image in images)]}
    else:
        result = client(args)
    print(json.dumps(result, sort_keys=True) if not isinstance(result, str) else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
