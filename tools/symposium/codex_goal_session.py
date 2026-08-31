#!/usr/bin/env python3
"""Manage durable Codex goal sessions.

The default ``exec-json`` backend starts ``codex exec --json`` directly and
persists its thread id, process identity, JSONL output, stderr, and lifecycle
state.  The historical tmux/``codex_auto_run.py`` transport remains available
with ``--backend autorun`` or ``SYMPOSIUM_CODEX_BACKEND=autorun``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
AUTORUN_SESSION_RE = re.compile(r"^Codex running in tmux session:\s*(\S+)\s*$", re.MULTILINE)
DEFAULT_AUTORUN_MODEL = "gpt-5.6-sol"
DEFAULT_AUTORUN_SANDBOX = "workspace-write"
DEFAULT_BACKEND = "exec-json"
VALID_BACKENDS = ("exec-json", "autorun")
THREAD_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def state_root(root: Path) -> Path:
    return root / ".symposium" / "scratch" / "codex-goal-sessions"


def require_tmux() -> str:
    tmux = shutil.which("tmux")
    if not tmux:
        raise SystemExit("tmux is required for managed Codex goal sessions.")
    return tmux


def sanitize(value: str) -> str:
    cleaned = VALID_NAME.sub("-", value.strip())
    return cleaned.strip("-") or "goal"


def goal_id(goal_dir: Path) -> str:
    return sanitize(goal_dir.name)


def session_name_for(goal_dir: Path, name: str | None = None) -> str:
    return sanitize(name) if name else f"autovideo-{goal_id(goal_dir)}"


def autorun_prefix_for(goal_dir: Path, name: str | None = None) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", session_name_for(goal_dir, name)).strip("-_")
    return cleaned[:48].rstrip("-_") or "autovideo"


def state_path(root: Path, goal_dir: Path, name: str | None = None) -> Path:
    return state_root(root) / f"{session_name_for(goal_dir, name)}.json"


def run_tmux(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [require_tmux(), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or proc.stdout.strip() or f"tmux failed: {args}")
    return proc


def tmux_alive(session: str) -> bool:
    return run_tmux(["has-session", "-t", session], check=False).returncode == 0


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Session state is not valid JSON: {path}: {exc}") from exc


def load_context(goal_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((goal_dir / "context.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write_state(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def resolve_goal_dir(root: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.exists():
        raise SystemExit(f"Goal directory does not exist: {path}")
    if not (path / "goal.md").exists() or not (path / "context.json").exists():
        raise SystemExit(f"Goal directory must contain goal.md and context.json: {path}")
    return path


def resolve_worktree(root: Path, raw: str | None) -> Path:
    if not raw:
        return root
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"Worktree directory does not exist: {path}")
    if not (path / "tools/symposium/start_codex_goal.sh").exists():
        raise SystemExit(f"Worktree does not look like autovideo: {path}")
    return path


def relative_to_root(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def infer_run_id(worktree: Path, context: dict[str, Any]) -> str:
    raw = context.get("run_id")
    if isinstance(raw, str) and raw:
        return raw
    for env_name in ("SYMPOSIUM_CURRENT_RUN_ID", "AUTO_VIDEO_RUN_ID", "RUN_ID"):
        raw = os.environ.get(env_name, "")
        if raw:
            return raw
    parts = worktree.resolve().parts
    for idx, part in enumerate(parts[:-2]):
        if part == "output" and parts[idx + 1] == "fanout_runs":
            return parts[idx + 2]
    return ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def saved_backend(state: dict[str, Any]) -> str | None:
    backend = state.get("backend")
    if backend in VALID_BACKENDS:
        return str(backend)
    if state.get("tmux_session") or state.get("executor") == "codex-autorun":
        return "autorun"
    return None


def requested_backend(args: argparse.Namespace, state: dict[str, Any] | None = None) -> str:
    explicit = getattr(args, "backend", None) or os.environ.get("SYMPOSIUM_CODEX_BACKEND")
    if explicit:
        if explicit not in VALID_BACKENDS:
            raise SystemExit(
                f"Unsupported Codex session backend {explicit!r}; choose one of {VALID_BACKENDS}"
            )
        return str(explicit)
    prior = saved_backend(state or {})
    return prior or DEFAULT_BACKEND


def backend_for_existing(args: argparse.Namespace, state: dict[str, Any]) -> str:
    prior = saved_backend(state)
    requested = getattr(args, "backend", None)
    if prior and requested and prior != requested:
        raise SystemExit(
            f"Session state uses backend {prior!r}, not requested backend {requested!r}"
        )
    return prior or requested_backend(args, state)


def proc_identity(pid: int) -> dict[str, int | str] | None:
    """Return Linux process identity fields that survive neither exit nor PID reuse."""
    if pid <= 1:
        return None
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        close = raw.rfind(")")
        if close < 0:
            return None
        fields = raw[close + 2 :].split()
        # fields[0] is /proc stat field 3 (state), fields[2] field 5 (pgrp),
        # and fields[19] field 22 (process start time in clock ticks).
        if len(fields) <= 19 or fields[0] == "Z":
            return None
        return {
            "pid": pid,
            "state": fields[0],
            "process_group_id": int(fields[2]),
            "pid_start_time_ticks": int(fields[19]),
        }
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError, ValueError):
        return None


def exec_identity_matches(state: dict[str, Any]) -> bool:
    try:
        pid = int(state.get("pid") or 0)
        expected_start = int(state.get("pid_start_time_ticks") or -1)
        expected_group = int(state.get("process_group_id") or -1)
    except (TypeError, ValueError):
        return False
    current = proc_identity(pid)
    return bool(
        current
        and current["pid_start_time_ticks"] == expected_start
        and current["process_group_id"] == expected_group
        and expected_group == pid
    )


def terminate_exec_process(state: dict[str, Any], timeout: float = 3.0) -> bool:
    """Terminate only the exact process group recorded in state.

    False means the PID/start-time/PGID tuple no longer identifies a live owned
    process, so no signal was sent.
    """
    if not exec_identity_matches(state):
        return False
    pgid = int(state["process_group_id"])
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        if not exec_identity_matches(state):
            return True
        time.sleep(0.03)
    if exec_identity_matches(state):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return True


def resolve_codex_binary(args: argparse.Namespace) -> str:
    raw = (
        getattr(args, "codex_binary", None)
        or os.environ.get("SYMPOSIUM_CODEX_BINARY")
        or os.environ.get("CODEX_AUTORUN_CODEX_BINARY")
        or "codex"
    )
    if os.path.sep in raw:
        path = Path(raw).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SystemExit(f"Codex binary is not executable: {path}")
        return str(path)
    found = shutil.which(raw)
    if not found:
        raise SystemExit(f"Codex binary not found on PATH: {raw}")
    return found


def codex_extra_args(args: argparse.Namespace) -> list[str]:
    result = list(getattr(args, "codex_arg", None) or [])
    raw = os.environ.get("SYMPOSIUM_CODEX_EXEC_ARGS", "")
    if raw:
        result.extend(shlex.split(raw))

    # Keep transport-owned identity and containment flags immutable.  Accept a
    # deliberately small extension surface instead of trying to blacklist every
    # spelling through which current or future Codex versions could override the
    # sandbox, working directory, model, JSONL stream, or config policy.
    safe_flags = {"--approve-for-me", "--strict-config"}
    safe_value_options = {"--enable", "--disable", "--profile", "-p", "--thread-source"}
    index = 0
    while index < len(result):
        item = result[index]
        if item in safe_flags:
            index += 1
            continue
        option, separator, value = item.partition("=")
        if separator and option in safe_value_options and value:
            index += 1
            continue
        if item in safe_value_options:
            if index + 1 >= len(result) or not result[index + 1]:
                raise SystemExit(f"Codex option {item!r} requires a value")
            index += 2
            continue
        raise SystemExit(
            f"Codex option {item!r} is not allowed by the managed exec-json transport"
        )
    return result


def runtime_paths(state_file: Path, attempt: int) -> tuple[Path, Path, Path]:
    runtime_dir = state_file.with_suffix(".d")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    stem = f"attempt-{attempt:04d}"
    stdout_path = runtime_dir / f"{stem}.stdout.jsonl"
    events_path = runtime_dir / f"{stem}.events.jsonl"
    stderr_path = runtime_dir / f"{stem}.stderr.log"
    if events_path.exists() or events_path.is_symlink():
        events_path.unlink()
    # Codex --json stdout is itself the canonical event JSONL.  Keep both
    # durable names without a lossy background tee or an untracked helper.
    events_path.symlink_to(stdout_path.name)
    return stdout_path, events_path, stderr_path


def thread_id_from_event(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("thread_id", "session_id", "threadId", "sessionId"):
            candidate = value.get(key)
            if isinstance(candidate, str) and THREAD_ID_RE.fullmatch(candidate):
                return candidate
        for nested in value.values():
            found = thread_id_from_event(nested)
            if found:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = thread_id_from_event(nested)
            if found:
                return found
    return None


def read_thread_id(stdout_path: Path) -> tuple[str | None, str | None]:
    try:
        raw = stdout_path.read_text(errors="replace")
    except FileNotFoundError:
        return None, None
    for line_no, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            return None, f"invalid Codex JSONL at line {line_no}: {exc}"
        found = thread_id_from_event(event)
        if found:
            return found, None
    return None, None


def tail_text(path: Path, lines: int) -> str:
    try:
        items = path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(items[-max(0, lines) :])


def resume_correction(goal_dir: Path, state: dict[str, Any]) -> tuple[str, str | None]:
    marker = goal_dir / "STOP_HOOK_RESUME.md"
    if not marker.is_file():
        return "", None
    text = marker.read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest == state.get("last_correction_sha256"):
        return "", digest
    return text, digest


def consume_resume_marker(goal_dir: Path) -> None:
    marker = goal_dir / "STOP_HOOK_RESUME.md"
    if marker.is_file():
        marker.replace(goal_dir / "STOP_HOOK_RESUME.last.md")


def run_goal_launcher(
    launcher: Path,
    goal_arg: str,
    worktree: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(launcher), goal_arg],
        cwd=worktree,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def start_autorun(args: argparse.Namespace) -> dict[str, Any]:
    root = project_root()
    worktree = resolve_worktree(root, args.worktree)
    goal_dir = resolve_goal_dir(worktree, args.goal_dir)
    context = load_context(goal_dir)
    run_id = infer_run_id(worktree, context)
    session_prefix = autorun_prefix_for(goal_dir, args.name)
    state_file = state_path(root, goal_dir, args.name)
    previous_state = load_state(state_file)
    previous_session = str(previous_state.get("session") or session_prefix)

    if tmux_alive(previous_session):
        if not args.force:
            raise SystemExit(
                f"Session already exists: {previous_session}. "
                "Use status/capture/send/attach or --force."
            )
        run_tmux(["kill-session", "-t", previous_session])

    launcher = worktree / "tools/symposium/start_codex_goal.sh"
    goal_arg = relative_to_root(worktree, goal_dir)
    goal_file = relative_to_root(worktree, goal_dir / "goal.md")
    env = os.environ.copy()
    env["TERM"] = "xterm-256color"
    env["SYMPOSIUM_AUTORUN_DETACH"] = "1"
    env["SYMPOSIUM_AUTORUN_SESSION_PREFIX"] = session_prefix
    env["SYMPOSIUM_SESSION_NAME"] = session_prefix
    env.setdefault("CODEX_AUTORUN_MODEL", DEFAULT_AUTORUN_MODEL)
    env.setdefault("CODEX_AUTORUN_SANDBOX", DEFAULT_AUTORUN_SANDBOX)
    if context.get("role") != "gate":
        env["SYMPOSIUM_EXECUTOR_SESSION_NAME"] = session_prefix
    if run_id:
        env["SYMPOSIUM_CURRENT_RUN_ID"] = run_id
        env["AUTO_VIDEO_RUN_ID"] = run_id
        env["RUN_ID"] = run_id

    proc = run_goal_launcher(launcher, goal_arg, worktree, env)
    if proc.returncode != 0:
        raise SystemExit(
            proc.stderr.strip()
            or proc.stdout.strip()
            or f"Codex autorun launcher failed: {launcher}"
        )
    match = AUTORUN_SESSION_RE.search(proc.stdout)
    if not match:
        raise SystemExit(
            "Codex autorun launcher did not report its tmux session. "
            f"stdout: {proc.stdout[-1000:]!r}"
        )
    session = match.group(1)
    time.sleep(args.startup_delay)
    follow_command = (
        f"codex-autorun TUI in {worktree} with initial prompt file {goal_file}; "
        f"model={env['CODEX_AUTORUN_MODEL']} sandbox={env['CODEX_AUTORUN_SANDBOX']}"
    )

    data = {
        "backend": "autorun",
        "session": session,
        "tmux_session": session,
        "session_prefix": session_prefix,
        "goal_dir": str(goal_dir),
        "goal_id": goal_id(goal_dir),
        "role": context.get("role", "implementation"),
        "dimension": context.get("dimension", "general"),
        "run_id": run_id,
        "worktree": str(worktree),
        "branch": context.get("root_branch"),
        "submodule_branch": context.get("submodule_branch"),
        "status": "starting",
        "resource_state": "active",
        "goal_follow_command": follow_command,
        "executor": "codex-autorun",
        "model": env["CODEX_AUTORUN_MODEL"],
        "sandbox": env["CODEX_AUTORUN_SANDBOX"],
        "launcher_stdout_tail": proc.stdout[-2000:],
        "launcher_stderr_tail": proc.stderr[-2000:],
        "last_capture_at_utc": None,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "state_file": str(state_file),
        "root": str(root),
    }
    write_state(state_file, data)
    return {**data, "alive": tmux_alive(session)}


def start_exec(args: argparse.Namespace) -> dict[str, Any]:
    root = project_root()
    worktree = resolve_worktree(root, args.worktree)
    goal_dir = resolve_goal_dir(worktree, args.goal_dir)
    context = load_context(goal_dir)
    run_id = infer_run_id(worktree, context)
    session_prefix = autorun_prefix_for(goal_dir, args.name)
    state_file = state_path(root, goal_dir, args.name)
    previous_state = load_state(state_file)

    if exec_identity_matches(previous_state):
        previous_session = previous_state.get("thread_id") or previous_state.get("session")
        if not args.force:
            raise SystemExit(
                f"Session already exists: {previous_session or session_prefix}. "
                "Use status/capture/send or --force."
            )
        terminate_exec_process(previous_state)

    prior_thread = previous_state.get("thread_id")
    if not isinstance(prior_thread, str) or not THREAD_ID_RE.fullmatch(prior_thread):
        candidate = previous_state.get("session")
        prior_thread = (
            candidate
            if isinstance(candidate, str) and THREAD_ID_RE.fullmatch(candidate)
            else None
        )
    launch_mode = "resume" if prior_thread else "initial"
    correction, correction_digest = resume_correction(goal_dir, previous_state)
    prompt = correction if prior_thread else (goal_dir / "goal.md").read_text()

    codex_binary = resolve_codex_binary(args)
    model = args.model or os.environ.get("CODEX_AUTORUN_MODEL", DEFAULT_AUTORUN_MODEL)
    sandbox = args.sandbox or os.environ.get(
        "CODEX_AUTORUN_SANDBOX", DEFAULT_AUTORUN_SANDBOX
    )
    if sandbox not in {"read-only", "workspace-write"}:
        raise SystemExit(
            "The exec-json transport permits only read-only or workspace-write sandbox; "
            f"got {sandbox!r}"
        )
    extra_args = codex_extra_args(args)
    if prior_thread:
        command = [codex_binary, "exec", "resume", "--json", "-m", model, *extra_args, prior_thread]
    else:
        command = [
            codex_binary,
            "exec",
            "--json",
            "--color",
            "never",
            "-C",
            str(worktree),
            "-m",
            model,
            "-s",
            sandbox,
            "-c",
            "check_for_update=false",
            *extra_args,
        ]
    if prompt:
        command.append("-")

    attempt = int(previous_state.get("attempt") or 0) + 1
    stdout_path, events_path, stderr_path = runtime_paths(state_file, attempt)
    env = os.environ.copy()
    env["TERM"] = "xterm-256color"
    env["SYMPOSIUM_SESSION_NAME"] = session_prefix
    if context.get("role") != "gate":
        env["SYMPOSIUM_EXECUTOR_SESSION_NAME"] = session_prefix
    if run_id:
        env["SYMPOSIUM_CURRENT_RUN_ID"] = run_id
        env["AUTO_VIDEO_RUN_ID"] = run_id
        env["RUN_ID"] = run_id

    with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
        proc = subprocess.Popen(
            command,
            cwd=worktree,
            env=env,
            text=True,
            stdin=subprocess.PIPE,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        if proc.stdin is not None:
            try:
                if prompt:
                    proc.stdin.write(prompt)
                    proc.stdin.flush()
            except BrokenPipeError:
                pass
            finally:
                proc.stdin.close()

    identity = proc_identity(proc.pid) or {
        "pid": proc.pid,
        "process_group_id": proc.pid,
        "pid_start_time_ticks": -1,
    }
    data: dict[str, Any] = {
        "backend": "exec-json",
        "session": prior_thread or session_prefix,
        "thread_id": prior_thread,
        "session_prefix": session_prefix,
        "goal_dir": str(goal_dir),
        "goal_id": goal_id(goal_dir),
        "role": context.get("role", "implementation"),
        "dimension": context.get("dimension", "general"),
        "run_id": run_id,
        "worktree": str(worktree),
        "branch": context.get("root_branch"),
        "submodule_branch": context.get("submodule_branch"),
        "status": "starting",
        "resource_state": "active",
        "executor": "codex-exec-json",
        "launch_mode": launch_mode,
        "attempt": attempt,
        "pid": identity["pid"],
        "pid_start_time_ticks": identity["pid_start_time_ticks"],
        "process_group_id": identity["process_group_id"],
        "codex_binary": codex_binary,
        "model": model,
        "sandbox": sandbox,
        "stdout_path": str(stdout_path),
        "jsonl_path": str(events_path),
        "stderr_path": str(stderr_path),
        "last_capture_at_utc": None,
        "started_at_utc": utc_now(),
        "state_file": str(state_file),
        "root": str(root),
        "goal_follow_command": (
            f"codex exec {launch_mode} in {worktree}; model={model} sandbox={sandbox}"
        ),
    }
    write_state(state_file, data)

    deadline = time.monotonic() + max(0.05, args.id_timeout)
    thread_id: str | None = None
    failure: str | None = None
    while time.monotonic() < deadline:
        thread_id, failure = read_thread_id(stdout_path)
        if thread_id or failure:
            break
        if proc.poll() is not None:
            failure = (
                f"codex exec exited with rc={proc.returncode} before reporting a thread/session id"
            )
            break
        time.sleep(0.03)
    if not thread_id and not failure:
        failure = (
            f"codex exec did not report a thread/session id within {args.id_timeout:g}s"
        )
    if thread_id and prior_thread and thread_id != prior_thread:
        failure = (
            f"codex exec resume reported a different thread id: {thread_id} != {prior_thread}"
        )
        thread_id = None
    if failure:
        terminate_exec_process(data)
        data.update(
            {
                "status": "launch_failed",
                "resource_state": "stopped",
                "launch_error": failure,
                "failed_at_utc": utc_now(),
            }
        )
        write_state(state_file, data)
        raise SystemExit(failure)

    assert thread_id is not None
    if args.startup_delay > 0:
        time.sleep(args.startup_delay)
    data.update(
        {
            "session": thread_id,
            "thread_id": thread_id,
            "status": "running",
            "launch_error": None,
        }
    )
    if correction_digest:
        data["last_correction_sha256"] = correction_digest
        data["last_correction_at_utc"] = utc_now()
        consume_resume_marker(goal_dir)
    write_state(state_file, data)
    return {**data, "alive": exec_identity_matches(data)}


def start(args: argparse.Namespace) -> dict[str, Any]:
    root = project_root()
    worktree = resolve_worktree(root, args.worktree)
    goal_dir = resolve_goal_dir(worktree, args.goal_dir)
    prior_state = load_state(state_path(root, goal_dir, args.name))
    backend = requested_backend(args, prior_state)
    prior_backend = saved_backend(prior_state)
    if prior_backend and prior_backend != backend:
        raise SystemExit(
            f"Existing session state uses backend {prior_backend!r}; "
            f"refusing implicit migration to {backend!r}"
        )
    if backend == "exec-json":
        return start_exec(args)
    return start_autorun(args)


def session_from_args(args: argparse.Namespace) -> tuple[Path, Path, str, dict[str, Any]]:
    root = project_root()
    worktree = resolve_worktree(root, getattr(args, "worktree", None))
    goal_dir = resolve_goal_dir(worktree, args.goal_dir)
    state_file = state_path(root, goal_dir, getattr(args, "name", None))
    state = load_state(state_file)
    session = state.get("session") or session_name_for(goal_dir, getattr(args, "name", None))
    return root, goal_dir, session, state


def status_autorun(args: argparse.Namespace) -> dict[str, Any]:
    root, goal_dir, session, state = session_from_args(args)
    alive = tmux_alive(session)
    pane = {}
    if alive:
        pane_proc = run_tmux(
            [
                "display-message",
                "-p",
                "-t",
                session,
                "#{session_name}\t#{pane_id}\t#{pane_pid}\t#{pane_current_command}\t#{pane_dead_status}",
            ],
            check=False,
        )
        if pane_proc.returncode == 0 and pane_proc.stdout.strip():
            parts = pane_proc.stdout.rstrip("\n").split("\t")
            if len(parts) >= 5:
                pane = {
                    "session_name": parts[0],
                    "pane_id": parts[1],
                    "pane_pid": parts[2],
                    "pane_current_command": parts[3],
                    "pane_dead_status": parts[4],
                }
    return {
        "alive": alive,
        "session": session,
        "goal_dir": str(goal_dir),
        "state": state,
        "pane": pane,
        "state_file": str(state_path(root, goal_dir, getattr(args, "name", None))),
    }


def capture_autorun(args: argparse.Namespace) -> str:
    root, goal_dir, session, state = session_from_args(args)
    if not tmux_alive(session):
        raise SystemExit(f"Session is not running: {session}")
    proc = run_tmux(
        [
            "capture-pane",
            "-p",
            "-J",
            "-t",
            session,
            "-S",
            f"-{args.lines}",
        ]
    )
    state.update(
        {
            "session": session,
            "goal_dir": str(goal_dir),
            "last_capture_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_state(state_path(root, goal_dir, getattr(args, "name", None)), state)
    return proc.stdout.rstrip("\n")


def send_autorun(args: argparse.Namespace) -> dict[str, Any]:
    root, goal_dir, session, state = session_from_args(args)
    if not tmux_alive(session):
        raise SystemExit(f"Session is not running: {session}")
    if args.text:
        run_tmux(["send-keys", "-t", session, "--", args.text])
    if args.enter:
        run_tmux(["send-keys", "-t", session, "Enter"])
    state.update(
        {
            "session": session,
            "goal_dir": str(goal_dir),
            "last_sent_at_utc": datetime.now(timezone.utc).isoformat(),
            "last_sent_text": args.text,
        }
    )
    write_state(state_path(root, goal_dir, getattr(args, "name", None)), state)
    return {"session": session, "sent": bool(args.text), "enter": args.enter}


def attach_autorun(args: argparse.Namespace) -> None:
    _, _, session, _ = session_from_args(args)
    if not tmux_alive(session):
        raise SystemExit(f"Session is not running: {session}")
    os.execvp(require_tmux(), ["tmux", "attach-session", "-t", session])


def stop_autorun(args: argparse.Namespace) -> dict[str, Any]:
    root, goal_dir, session, state = session_from_args(args)
    alive_before = tmux_alive(session)
    if alive_before:
        run_tmux(["kill-session", "-t", session])
    state.update(
        {
            "session": session,
            "goal_dir": str(goal_dir),
            "status": "stopped",
            "resource_state": "stopped",
            "stopped_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_state(state_path(root, goal_dir, getattr(args, "name", None)), state)
    return {"session": session, "alive_before": alive_before, "alive_after": tmux_alive(session)}


def release_autorun(args: argparse.Namespace) -> dict[str, Any]:
    root, goal_dir, session, state = session_from_args(args)
    alive_before = tmux_alive(session)
    if alive_before and not args.keep_session:
        run_tmux(["kill-session", "-t", session])
    state.update(
        {
            "session": session,
            "goal_dir": str(goal_dir),
            "status": "released",
            "resource_state": "released",
            "released_at_utc": datetime.now(timezone.utc).isoformat(),
            "release_note": args.note,
        }
    )
    write_state(state_path(root, goal_dir, getattr(args, "name", None)), state)
    return {
        "session": session,
        "alive_before": alive_before,
        "alive_after": tmux_alive(session),
        "resource_state": "released",
    }


def status_exec(args: argparse.Namespace) -> dict[str, Any]:
    root, goal_dir, session, state = session_from_args(args)
    return {
        "alive": exec_identity_matches(state),
        "backend": "exec-json",
        "session": session,
        "goal_dir": str(goal_dir),
        "state": state,
        "process": {
            "pid": state.get("pid"),
            "pid_start_time_ticks": state.get("pid_start_time_ticks"),
            "process_group_id": state.get("process_group_id"),
            "identity_matched": exec_identity_matches(state),
        },
        "pane": {},
        "state_file": str(state_path(root, goal_dir, getattr(args, "name", None))),
    }


def capture_exec(args: argparse.Namespace) -> str:
    root, goal_dir, session, state = session_from_args(args)
    if not exec_identity_matches(state):
        raise SystemExit(f"Session is not running: {session}")
    stdout = tail_text(Path(str(state.get("stdout_path") or "")), args.lines)
    stderr = tail_text(Path(str(state.get("stderr_path") or "")), args.lines)
    pieces = [stdout] if stdout else []
    if stderr:
        pieces.append("[stderr]\n" + stderr)
    state.update(
        {
            "session": session,
            "goal_dir": str(goal_dir),
            "last_capture_at_utc": utc_now(),
        }
    )
    write_state(state_path(root, goal_dir, getattr(args, "name", None)), state)
    return "\n".join(pieces)


def send_exec(args: argparse.Namespace) -> dict[str, Any]:
    root, goal_dir, session, state = session_from_args(args)
    if not exec_identity_matches(state):
        raise SystemExit(f"Session is not running: {session}")
    queue_stdout = ""
    queue_stderr = ""
    if args.text:
        codex_binary = str(state.get("codex_binary") or shutil.which("codex") or "codex")
        proc = subprocess.run(
            [codex_binary, "queue", "--thread", session, "--message", args.text],
            cwd=str(state.get("worktree") or project_root()),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        queue_stdout = proc.stdout
        queue_stderr = proc.stderr
        if proc.returncode != 0:
            raise SystemExit(
                proc.stderr.strip()
                or proc.stdout.strip()
                or f"codex queue failed with rc={proc.returncode}"
            )
    state.update(
        {
            "session": session,
            "goal_dir": str(goal_dir),
            "last_sent_at_utc": utc_now(),
            "last_sent_text": args.text,
            "last_queue_stdout_tail": queue_stdout[-2000:],
            "last_queue_stderr_tail": queue_stderr[-2000:],
            "queue_count": int(state.get("queue_count") or 0) + bool(args.text),
        }
    )
    write_state(state_path(root, goal_dir, getattr(args, "name", None)), state)
    return {"session": session, "sent": bool(args.text), "enter": args.enter}


def attach_exec(args: argparse.Namespace) -> None:
    _, _, session, state = session_from_args(args)
    if not exec_identity_matches(state):
        raise SystemExit(f"Session is not running: {session}")
    raise SystemExit(
        "The exec-json backend has no interactive pane; use capture/send/status instead."
    )


def stop_exec(args: argparse.Namespace) -> dict[str, Any]:
    root, goal_dir, session, state = session_from_args(args)
    alive_before = exec_identity_matches(state)
    identity_matched = terminate_exec_process(state) if alive_before else False
    state.update(
        {
            "session": session,
            "goal_dir": str(goal_dir),
            "status": "stopped",
            "resource_state": "stopped",
            "stopped_at_utc": utc_now(),
        }
    )
    write_state(state_path(root, goal_dir, getattr(args, "name", None)), state)
    return {
        "session": session,
        "alive_before": alive_before,
        "alive_after": exec_identity_matches(state),
        "identity_matched": identity_matched,
    }


def release_exec(args: argparse.Namespace) -> dict[str, Any]:
    root, goal_dir, session, state = session_from_args(args)
    alive_before = exec_identity_matches(state)
    identity_matched = alive_before
    if alive_before and not args.keep_session:
        identity_matched = terminate_exec_process(state)
    state.update(
        {
            "session": session,
            "goal_dir": str(goal_dir),
            "status": "released",
            "resource_state": "released",
            "released_at_utc": utc_now(),
            "release_note": args.note,
        }
    )
    write_state(state_path(root, goal_dir, getattr(args, "name", None)), state)
    return {
        "session": session,
        "alive_before": alive_before,
        "alive_after": exec_identity_matches(state),
        "identity_matched": identity_matched,
        "resource_state": "released",
    }


def status(args: argparse.Namespace) -> dict[str, Any]:
    _, _, _, state = session_from_args(args)
    if backend_for_existing(args, state) == "exec-json":
        return status_exec(args)
    return status_autorun(args)


def capture(args: argparse.Namespace) -> str:
    _, _, _, state = session_from_args(args)
    if backend_for_existing(args, state) == "exec-json":
        return capture_exec(args)
    return capture_autorun(args)


def send(args: argparse.Namespace) -> dict[str, Any]:
    _, _, _, state = session_from_args(args)
    if backend_for_existing(args, state) == "exec-json":
        return send_exec(args)
    return send_autorun(args)


def attach(args: argparse.Namespace) -> None:
    _, _, _, state = session_from_args(args)
    if backend_for_existing(args, state) == "exec-json":
        return attach_exec(args)
    return attach_autorun(args)


def stop(args: argparse.Namespace) -> dict[str, Any]:
    _, _, _, state = session_from_args(args)
    if backend_for_existing(args, state) == "exec-json":
        return stop_exec(args)
    return stop_autorun(args)


def release(args: argparse.Namespace) -> dict[str, Any]:
    _, _, _, state = session_from_args(args)
    if backend_for_existing(args, state) == "exec-json":
        return release_exec(args)
    return release_autorun(args)


def list_sessions(_: argparse.Namespace) -> list[dict[str, Any]]:
    root = project_root()
    items: list[dict[str, Any]] = []
    for path in sorted(state_root(root).glob("*.json")):
        state = load_state(path)
        session = state.get("session") or path.stem
        backend = saved_backend(state) or DEFAULT_BACKEND
        alive = exec_identity_matches(state) if backend == "exec-json" else tmux_alive(session)
        items.append(
            {
                "state_file": str(path),
                "session": session,
                "backend": backend,
                "alive": alive,
                **state,
            }
        )
    return items


def watch(args: argparse.Namespace) -> int:
    try:
        while True:
            os.system("clear")
            print(capture(args))
            print()
            print(f"[watching {session_from_args(args)[2]} every {args.interval:g}s; Ctrl-C to stop]")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def print_result(value: Any) -> None:
    if isinstance(value, str):
        print(value)
    else:
        print(json.dumps(value, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    start_parser = sub.add_parser("start", help="Start a detached Codex goal session")
    start_parser.add_argument("goal_dir")
    start_parser.add_argument("--name")
    start_parser.add_argument("--worktree", help="Autovideo worktree where the goal runs")
    start_parser.add_argument(
        "--backend",
        choices=VALID_BACKENDS,
        help=(
            "Session transport (default: exec-json; override with "
            "SYMPOSIUM_CODEX_BACKEND or select autorun for legacy tmux)"
        ),
    )
    start_parser.add_argument("--codex-binary", help="Codex CLI executable for exec-json")
    start_parser.add_argument("--model", help="Codex model (default: CODEX_AUTORUN_MODEL)")
    start_parser.add_argument(
        "--sandbox", choices=("read-only", "workspace-write"), help="Codex exec sandbox"
    )
    start_parser.add_argument(
        "--codex-arg",
        action="append",
        default=[],
        help="Additional safe codex exec/resume argument (repeatable)",
    )
    start_parser.add_argument("--force", action="store_true")
    start_parser.add_argument("--rows", type=int, default=40)
    start_parser.add_argument("--cols", type=int, default=120)
    start_parser.add_argument("--startup-delay", type=float, default=2.0)
    start_parser.add_argument(
        "--id-timeout",
        type=float,
        default=float(os.environ.get("SYMPOSIUM_CODEX_ID_TIMEOUT", "30")),
        help="Seconds to wait for a valid thread/session id in Codex JSONL",
    )
    start_parser.set_defaults(func=start)

    for command, help_text, func in (
        ("status", "Show session status", status),
        ("capture", "Capture recent terminal output", capture),
        ("send", "Send text or enter to the session", send),
        ("attach", "Attach to the interactive session", attach),
        ("stop", "Stop the session", stop),
        ("release", "Release session resources and mark state released", release),
        ("watch", "Continuously capture the session", watch),
    ):
        p = sub.add_parser(command, help=help_text)
        p.add_argument("goal_dir")
        p.add_argument("--name")
        p.add_argument("--worktree", help="Autovideo worktree that owns the goal")
        p.add_argument("--backend", choices=VALID_BACKENDS)
        if command in {"capture", "watch"}:
            p.add_argument("--lines", type=int, default=80)
        if command == "send":
            p.add_argument("--text", default="")
            p.add_argument("--enter", action="store_true")
        if command == "release":
            p.add_argument("--keep-session", action="store_true")
            p.add_argument("--note", default="")
        if command == "watch":
            p.add_argument("--interval", type=float, default=3.0)
        p.set_defaults(func=func)

    list_parser = sub.add_parser("list", help="List known sessions")
    list_parser.set_defaults(func=list_sessions)

    args = parser.parse_args()
    result = args.func(args)
    if result is not None:
        print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
