from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "tools" / "symposium" / "codex_goal_session.py"
THREAD_ID = "12345678-1234-4234-8234-123456789abc"


FAKE_CODEX = r'''#!/usr/bin/env python3
import json
import os
import re
import signal
import sys
import time

args = sys.argv[1:]
calls = os.environ["FAKE_CODEX_CALLS"]

def record(payload):
    with open(calls, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")

def option(name):
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError):
        return ""

if args and args[0] == "queue":
    record({"kind": "queue", "thread": option("--thread"), "message": option("--message")})
    raise SystemExit(0)

if not args or args[0] != "exec":
    record({"kind": "unexpected", "argv": args})
    raise SystemExit(2)

is_resume = "resume" in args
thread = os.environ.get("FAKE_CODEX_THREAD_ID", "12345678-1234-4234-8234-123456789abc")
if is_resume:
    resume_at = args.index("resume")
    for value in args[resume_at + 1:]:
        if re.fullmatch(r"[0-9a-fA-F-]{36}", value):
            thread = value
            break

prompt = sys.stdin.read()
record({"kind": "exec", "resume": is_resume, "thread": thread, "prompt": prompt, "argv": args})
mode = os.environ.get("FAKE_CODEX_MODE", "normal")
if mode == "malformed":
    print("{not-json", flush=True)
    time.sleep(0.2)
    raise SystemExit(3)
if mode == "missing-id":
    print(json.dumps({"type": "turn.started"}), flush=True)
    raise SystemExit(0)

print(json.dumps({"type": "thread.started", "thread_id": thread}), flush=True)

def stopped(signum, frame):
    record({"kind": "signal", "signal": signum, "thread": thread})
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stopped)
signal.signal(signal.SIGINT, stopped)
deadline = time.monotonic() + float(os.environ.get("FAKE_CODEX_EXIT_AFTER", "60"))
while time.monotonic() < deadline:
    time.sleep(0.02)
'''


class ExecJsonTransportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="symposium-exec-test-")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        symposium = self.repo / "tools" / "symposium"
        symposium.mkdir(parents=True)
        self.script = symposium / "codex_goal_session.py"
        shutil.copy2(SOURCE, self.script)
        launcher = symposium / "start_codex_goal.sh"
        launcher.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")
        launcher.chmod(0o755)
        self.fake = Path(self.temp.name) / "fake-codex"
        self.fake.write_text(FAKE_CODEX, encoding="utf-8")
        self.fake.chmod(0o755)
        self.calls = Path(self.temp.name) / "calls.jsonl"
        self.goals: list[Path] = []
        self.live_states: list[Path] = []

    def tearDown(self) -> None:
        for state_path in self.live_states:
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                pid = int(state.get("pid") or 0)
                pgid = int(state.get("process_group_id") or 0)
                if pid > 1:
                    os.kill(pid, 0)
                if pgid > 1:
                    os.killpg(pgid, signal.SIGKILL)
            except (FileNotFoundError, json.JSONDecodeError, ProcessLookupError, PermissionError, ValueError):
                pass

    def goal(self, name: str) -> Path:
        path = self.repo / "goals" / name
        path.mkdir(parents=True)
        (path / "goal.md").write_text(f"goal for {name}\n", encoding="utf-8")
        (path / "context.json").write_text(json.dumps({"role": "executor", "run_id": "test-run"}), encoding="utf-8")
        self.goals.append(path)
        return path

    def env(self, **updates: str) -> dict[str, str]:
        result = os.environ.copy()
        result.update({
            "FAKE_CODEX_CALLS": str(self.calls),
            "FAKE_CODEX_THREAD_ID": THREAD_ID,
        })
        result.update(updates)
        return result

    def cli(self, *args: object, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            [sys.executable, str(self.script), *(str(item) for item in args)],
            cwd=self.repo,
            env=env or self.env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        if check and proc.returncode != 0:
            self.fail(f"CLI failed rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}")
        return proc

    def start(self, goal: Path, name: str, *, env: dict[str, str] | None = None, backend: bool = True) -> dict[str, object]:
        args: list[object] = [
            "start", goal, "--name", name, "--worktree", self.repo,
            "--codex-binary", self.fake, "--id-timeout", "2", "--startup-delay", "0",
        ]
        if backend:
            args.extend(["--backend", "exec-json"])
        proc = self.cli(*args, env=env)
        payload = json.loads(proc.stdout)
        state_path = Path(str(payload["state_file"]))
        self.live_states.append(state_path)
        return payload

    def read_calls(self) -> list[dict[str, object]]:
        if not self.calls.exists():
            return []
        return [json.loads(line) for line in self.calls.read_text(encoding="utf-8").splitlines()]

    def wait_dead(self, goal: Path, name: str) -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = json.loads(self.cli("status", goal, "--name", name, "--worktree", self.repo).stdout)
            if not status["alive"]:
                return
            time.sleep(0.03)
        self.fail("fake codex did not exit")

    def test_default_exec_start_id_alive_capture_and_queue_once(self) -> None:
        goal = self.goal("basic")
        started = self.start(goal, "basic", backend=False)
        self.assertEqual(started["backend"], "exec-json")
        self.assertEqual(started["session"], THREAD_ID)
        self.assertTrue(started["alive"])
        status = json.loads(self.cli("status", goal, "--name", "basic", "--worktree", self.repo).stdout)
        self.assertTrue(status["alive"])
        self.assertEqual(status["session"], THREAD_ID)
        captured = self.cli("capture", goal, "--name", "basic", "--worktree", self.repo, "--lines", "20").stdout
        self.assertIn("thread.started", captured)
        launch = next(item for item in self.read_calls() if item["kind"] == "exec")
        argv = list(launch["argv"])
        self.assertEqual(argv[0], "exec")
        self.assertIn("--json", argv)
        self.assertIn("--color", argv)
        self.assertIn("-C", argv)
        self.assertEqual(argv[argv.index("-s") + 1], "workspace-write")
        self.assertEqual(argv[-1], "-")
        sent = json.loads(self.cli(
            "send", goal, "--name", "basic", "--worktree", self.repo,
            "--text", "one correction", "--enter",
        ).stdout)
        self.assertTrue(sent["sent"])
        queues = [item for item in self.read_calls() if item["kind"] == "queue"]
        self.assertEqual(queues, [{"kind": "queue", "message": "one correction", "thread": THREAD_ID}])

    def test_duplicate_start_does_not_launch_twice(self) -> None:
        goal = self.goal("duplicate")
        self.start(goal, "duplicate")
        duplicate = self.cli(
            "start", goal, "--name", "duplicate", "--worktree", self.repo,
            "--backend", "exec-json", "--codex-binary", self.fake,
            "--id-timeout", "1", "--startup-delay", "0",
            check=False,
        )
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertIn("already", duplicate.stderr.lower())
        self.assertEqual(len([item for item in self.read_calls() if item["kind"] == "exec"]), 1)

    def test_exit_resumes_same_thread_and_correction_is_injected_once(self) -> None:
        goal = self.goal("resume")
        self.start(goal, "resume", env=self.env(FAKE_CODEX_EXIT_AFTER="0.15"))
        self.wait_dead(goal, "resume")
        correction = "fix exactly once"
        (goal / "goal.md").write_text((goal / "goal.md").read_text(encoding="utf-8") + correction, encoding="utf-8")
        (goal / "STOP_HOOK_RESUME.md").write_text(correction, encoding="utf-8")
        resumed = self.start(goal, "resume")
        self.assertEqual(resumed["launch_mode"], "resume")
        self.assertEqual(resumed["session"], THREAD_ID)
        self.cli("stop", goal, "--name", "resume", "--worktree", self.repo)
        resumed_again = self.start(goal, "resume")
        self.assertEqual(resumed_again["session"], THREAD_ID)
        prompts = [str(item["prompt"]) for item in self.read_calls() if item["kind"] == "exec" and item["resume"]]
        self.assertEqual(sum(correction in prompt for prompt in prompts), 1)
        resume_calls = [item for item in self.read_calls() if item["kind"] == "exec" and item["resume"]]
        self.assertTrue(resume_calls)
        self.assertEqual(resume_calls[0]["argv"][:2], ["exec", "resume"])
        self.assertIn(THREAD_ID, resume_calls[0]["argv"])
        self.assertFalse((goal / "STOP_HOOK_RESUME.md").exists())
        self.assertTrue((goal / "STOP_HOOK_RESUME.last.md").exists())

    def test_transport_owned_options_cannot_be_overridden(self) -> None:
        forbidden = (
            "--dangerously-bypass-approvals-and-sandbox",
            "--sandbox=danger-full-access",
            "--config=approval_policy=never",
            "--model=another-model",
            "--cd=/tmp",
        )
        for index, value in enumerate(forbidden):
            with self.subTest(value=value):
                goal = self.goal(f"forbidden-{index}")
                proc = self.cli(
                    "start", goal, "--name", f"forbidden-{index}",
                    "--worktree", self.repo, "--backend", "exec-json",
                    "--codex-binary", self.fake, f"--codex-arg={value}",
                    "--id-timeout", "0.5", "--startup-delay", "0",
                    check=False,
                )
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("not allowed", proc.stderr.lower())

        self.assertFalse([item for item in self.read_calls() if item["kind"] == "exec"])

    def test_pid_reuse_guard_prevents_stop_and_release_from_killing(self) -> None:
        goal = self.goal("pid-guard")
        started = self.start(goal, "pid-guard")
        state_path = Path(str(started["state_file"]))
        state = json.loads(state_path.read_text(encoding="utf-8"))
        pid = int(state["pid"])
        state["pid_start_time_ticks"] = int(state["pid_start_time_ticks"]) + 1
        state_path.write_text(json.dumps(state), encoding="utf-8")
        stopped = json.loads(self.cli("stop", goal, "--name", "pid-guard", "--worktree", self.repo).stdout)
        self.assertFalse(stopped["identity_matched"])
        os.kill(pid, 0)
        released = json.loads(self.cli("release", goal, "--name", "pid-guard", "--worktree", self.repo).stdout)
        self.assertFalse(released["identity_matched"])
        os.kill(pid, 0)

    def test_normal_stop_and_release_terminate_only_owned_group(self) -> None:
        goal = self.goal("lifecycle")
        self.start(goal, "lifecycle")
        stopped = json.loads(self.cli("stop", goal, "--name", "lifecycle", "--worktree", self.repo).stdout)
        self.assertTrue(stopped["identity_matched"])
        self.assertFalse(stopped["alive_after"])
        self.start(goal, "lifecycle")
        released = json.loads(self.cli("release", goal, "--name", "lifecycle", "--worktree", self.repo, "--note", "done").stdout)
        self.assertTrue(released["identity_matched"])
        self.assertFalse(released["alive_after"])
        self.assertEqual(released["resource_state"], "released")

    def test_malformed_jsonl_and_missing_thread_id_fail_closed(self) -> None:
        for mode in ("malformed", "missing-id"):
            with self.subTest(mode=mode):
                goal = self.goal(mode)
                proc = self.cli(
                    "start", goal, "--name", mode, "--worktree", self.repo,
                    "--backend", "exec-json", "--codex-binary", self.fake,
                    "--id-timeout", "0.5", "--startup-delay", "0",
                    env=self.env(FAKE_CODEX_MODE=mode), check=False,
                )
                self.assertNotEqual(proc.returncode, 0)
                self.assertRegex(proc.stderr.lower(), r"json|thread|session|id")
                state_files = list((self.repo / ".symposium" / "scratch" / "codex-goal-sessions").glob(f"{mode}.json"))
                self.assertEqual(len(state_files), 1)
                state = json.loads(state_files[0].read_text(encoding="utf-8"))
                self.assertEqual(state["status"], "launch_failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
