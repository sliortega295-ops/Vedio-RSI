#!/usr/bin/env python3
"""One-node PTY SSH transport; site adaptation, not an optimizer or scheduler."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import select
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))
from orchestration import site_transfer

LOCAL_PROJECT = Path("/home/lyy/Experiments/SolAgent/20260913-a100-original-baseline")
REMOTE_PROJECT = Path("/share/project/shuchenweng/yongyan_liu/Experiments/SolAgent/20260913-a100-original-baseline")
TARGET = "job-c6cfcde5-db4b-4527-8eb8-7518319a0277-master-0.shuchenweng.baai-vision_vision.dx-calc1.job@ssh.platform-multi.baai.ac.cn"
SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ControlMaster=no", "-o", "ControlPath=none", "-p", "2222")
TRUSTED_SITE_JOB_SHA256 = "ef686fb4131e2846286ba474ba4a76e09413ee156f8a1b587f58e6f513b30f1d"
TRUSTED_OBSERVER_SHA256 = "4c541a43ac76d65416839b009fc07e3eeb9f83e5274c01a6931bc86fa237bf83"


def remote_path(local: Path) -> Path:
    path = local.resolve()
    if not path.is_relative_to(LOCAL_PROJECT):
        raise ValueError(f"local path outside this project: {path}")
    return REMOTE_PROJECT / path.relative_to(LOCAL_PROJECT)


def ssh(*argv: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["ssh", "-tt", *SSH_OPTS, TARGET, shlex.join(argv)],
                          text=True, capture_output=True, check=True, timeout=timeout)


def _pty_send(script: str, payload: bytes, timeout: int) -> str:
    command = ["ssh", "-tt", *SSH_OPTS, TARGET, shlex.join(["bash", "-lc", script])]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    assert proc.stdin is not None and proc.stdout is not None
    try:
        deadline = time.monotonic() + 30
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([proc.stdout], [], [], remaining)[0]:
                raise TimeoutError("remote PTY transfer did not become ready")
            ready = proc.stdout.readline().strip()
            if ready == b"SOL_READY":
                break
            if not ready and proc.poll() is not None:
                raise RuntimeError("remote PTY closed before transfer readiness")
        stdout, stderr = proc.communicate(input=payload, timeout=timeout)
        if proc.returncode:
            raise RuntimeError(f"remote transfer rc={proc.returncode}: {stderr[-1000:]!r} {stdout[-1000:]!r}")
        return stdout.decode(errors="replace")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def _json_receipt(output: str) -> dict:
    rows = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    if not rows:
        raise RuntimeError(f"JSON receipt missing: {output[-1000:]}")
    return rows[-1]


def bootstrap_transfer() -> dict:
    local = Path(__file__).with_name("site_transfer.py")
    remote = remote_path(local)
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    try:
        if ssh("sha256sum", str(remote), timeout=20).stdout.split()[0] == digest:
            return {"status": "already_verified", "sha256": digest, "remote": str(remote)}
    except (subprocess.CalledProcessError, IndexError):
        pass
    ssh("mkdir", "-p", str(remote.parent), timeout=30)
    encoded = base64.b64encode(local.read_bytes()) + b"\n"
    temp = remote.with_name(".site_transfer.py.bootstrap")
    script = ("set -e; stty -echo -icanon -ixon; printf 'SOL_READY\\n'; "
              f"head -c {len(encoded)} | base64 -d > {shlex.quote(str(temp))}; "
              f"test \"$(sha256sum {shlex.quote(str(temp))} | cut -d' ' -f1)\" = {digest}; "
              f"mv {shlex.quote(str(temp))} {shlex.quote(str(remote))}; echo SOL_BOOTSTRAP_OK")
    if "SOL_BOOTSTRAP_OK" not in _pty_send(script, encoded, timeout=60):
        raise RuntimeError("transfer helper bootstrap missing receipt")
    if ssh("sha256sum", str(remote), timeout=20).stdout.split()[0] != digest:
        raise RuntimeError("remote transfer helper hash mismatch")
    return {"status": "bootstrapped_verified", "sha256": digest, "remote": str(remote)}


def send_tree(source: Path, remote_dest: Path, excludes: list[str] | None = None) -> dict:
    if not remote_dest.is_relative_to(REMOTE_PROJECT) or remote_dest == REMOTE_PROJECT:
        raise ValueError("send destination outside the new remote project")
    bootstrap = bootstrap_transfer()
    archive = site_transfer.pack(source, excludes or [])
    digest = hashlib.sha256(archive).hexdigest()
    encoded = base64.encodebytes(archive)
    remote_tool = remote_path(Path(__file__).with_name("site_transfer.py"))
    script = ("set -e; stty -echo -icanon -ixon; printf 'SOL_READY\\n'; "
              f"python3 {shlex.quote(str(remote_tool))} recv --dest {shlex.quote(str(remote_dest))} "
              f"--encoded-length {len(encoded)} --sha256 {digest}")
    receipt = _json_receipt(_pty_send(script, encoded, timeout=900))
    if receipt.get("archive_sha256") != digest or receipt.get("status") != "received_verified":
        raise RuntimeError(f"remote archive verification failed: {receipt}")
    return {"local_source": str(source), "remote_dest": str(remote_dest),
            "archive_sha256": digest, "archive_bytes": len(archive),
            "remote_receipt": receipt, "bootstrap": bootstrap}


def pull_tree(remote_source: Path, local_dest: Path, excludes: list[str] | None = None) -> dict:
    if not remote_source.is_relative_to(REMOTE_PROJECT) or remote_source == REMOTE_PROJECT:
        raise ValueError("pull source outside the new remote project")
    bootstrap_transfer()
    remote_tool = remote_path(Path(__file__).with_name("site_transfer.py"))
    command = ["python3", str(remote_tool), "send", "--source", str(remote_source)]
    for item in excludes or []:
        command.extend(["--exclude", item])
    raw = subprocess.run(["ssh", "-tt", *SSH_OPTS, TARGET, shlex.join(command)],
                         capture_output=True, check=True, timeout=900).stdout
    match = re.search(rb"SOL_ARCHIVE (\d+) ([0-9a-f]{64})\r?\n", raw)
    if not match:
        raise RuntimeError(f"pull archive header missing: {raw[:500]!r}")
    size, digest = int(match.group(1)), match.group(2).decode()
    encoded = raw[match.end(): match.end() + size]
    if len(encoded) != size or b"SOL_ARCHIVE_END" not in raw[match.end() + size:]:
        raise RuntimeError("pull archive framing incomplete")
    archive = base64.b64decode(encoded, validate=True)
    observed = hashlib.sha256(archive).hexdigest()
    if observed != digest:
        raise RuntimeError(f"pull archive SHA256 mismatch: {observed} != {digest}")
    extracted = site_transfer.unpack(archive, local_dest)
    return {"remote_source": str(remote_source), "local_dest": str(local_dest),
            "archive_sha256": digest, "archive_bytes": len(archive), **extracted}


def sync_source() -> dict:
    source = LOCAL_PROJECT / "source/sol_runtime"
    remote = REMOTE_PROJECT / "source/sol_runtime"
    core = remote / "python/sglang/multimodal_gen/runtime/models/dits/sana_video.py"
    expected = "6a4c853e8f443734db72befdca0910b4cbf2ccf4a83792c7be03427a36522332"
    try:
        if ssh("sha256sum", str(core), timeout=20).stdout.split()[0] == expected:
            return {"status": "already_verified", "core_sha256": expected, "remote": str(remote)}
    except (subprocess.CalledProcessError, IndexError):
        pass
    receipt = send_tree(source, remote, excludes=[".git", "__pycache__"])
    observed = ssh("sha256sum", str(core), timeout=20).stdout.split()[0]
    if observed != expected:
        raise RuntimeError("remote SGLang clean source hash differs after sync")
    return {**receipt, "core_sha256": observed, "status": "sent_verified"}


def sync_code(repo: Path) -> dict:
    excludes = [".git", ".symposium", "runs", "output", "logs", "__pycache__", ".pytest_cache"]
    return send_tree(repo, remote_path(repo), excludes=excludes)


def _repo_for_run(run_dir: Path) -> Path:
    """Return the checkout that owns an official coordinator/executor run."""
    run = run_dir.resolve()
    repo = run.parents[1]
    coordinator = LOCAL_PROJECT / "repo"
    coordinator_run = repo == coordinator and run.parent == coordinator / "runs"
    allowed_executor_repos = {
        coordinator / "output/experiments/sana2ba100-kernel_aw-0002/worktree",
        coordinator / "output/experiments/sana2ba100-cache_ca-0002/worktree",
    }
    executor_run = repo in allowed_executor_repos and run.parent == repo / "runs"
    if not run.is_dir() or not (coordinator_run or executor_run):
        raise ValueError("site submission requires an official coordinator/executor run bundle")
    return repo


def trusted_remote_job_script() -> tuple[Path, str]:
    """Pin every privileged remote job operation to the coordinator shim."""
    local = LOCAL_PROJECT / "repo/orchestration/site_job.py"
    remote = REMOTE_PROJECT / "repo/orchestration/site_job.py"
    if not local.is_file() or local.is_symlink():
        raise RuntimeError("trusted local site-job shim is missing or a symlink")
    if hashlib.sha256(local.read_bytes()).hexdigest() != TRUSTED_SITE_JOB_SHA256:
        raise RuntimeError("trusted local site-job shim SHA256 mismatch")
    observed = ssh("sha256sum", str(remote), timeout=20).stdout.split()[0]
    if observed != TRUSTED_SITE_JOB_SHA256:
        raise RuntimeError("trusted remote site-job shim SHA256 mismatch")
    return remote, observed


def trusted_remote_observer_script() -> tuple[Path, str]:
    """Never execute an Executor-owned observer or infer freshness from local logs."""
    local = LOCAL_PROJECT / "repo/orchestration/site_observer.py"
    remote = REMOTE_PROJECT / "repo/orchestration/site_observer.py"
    if not local.is_file() or local.is_symlink():
        raise RuntimeError("trusted local site observer is missing or a symlink")
    if hashlib.sha256(local.read_bytes()).hexdigest() != TRUSTED_OBSERVER_SHA256:
        raise RuntimeError("trusted local site observer SHA256 mismatch")
    observed = ssh("sha256sum", str(remote), timeout=20).stdout.split()[0]
    if observed != TRUSTED_OBSERVER_SHA256:
        raise RuntimeError("trusted remote site observer SHA256 mismatch")
    return remote, observed


def phase2_budget_status_direct() -> dict:
    script, observed = trusted_remote_job_script()
    result = _json_receipt(ssh("python3", str(script), "budget-status", timeout=30).stdout)
    result["trusted_job_script_sha256"] = observed
    return result


def phase2_observer_audit_direct() -> dict:
    script, observed = trusted_remote_observer_script()
    result = _json_receipt(ssh("python3", str(script), "--phase", "phase2", "audit", timeout=30).stdout)
    result["trusted_observer_script_sha256"] = observed
    if (result.get("phase") != "phase2" or result.get("status") != "running"
            or result.get("sample_count", 0) < 2
            or not result.get("last_sample_uuid_complete")
            or result.get("last_sample_error") is not None
            or result.get("last_sample_age_s", 999) > 3.0):
        raise RuntimeError(f"phase2 observer is absent, stale, or incomplete: {result}")
    return result


def _submit_direct(run_dir: Path, launch_script: Path) -> dict:
    local_repo = _repo_for_run(run_dir)
    if launch_script.resolve() != run_dir.resolve() / "launch.sh" or not launch_script.is_file():
        raise ValueError("site submission requires the run bundle's launch.sh")
    from orchestration import site_job_broker
    budget_pre_transfer = (site_job_broker.assert_gpu_submit_budget(run_dir)
                           if run_dir.name not in site_job_broker.CPU_PROBES else None)
    observer_pre_transfer = (phase2_observer_audit_direct()
                             if budget_pre_transfer and budget_pre_transfer.get("phase") == "phase2" else None)
    source_sync = sync_source()
    code_sync = sync_code(local_repo)
    remote_repo = remote_path(local_repo)
    remote_run = remote_path(run_dir)
    script_text = launch_script.read_text()
    if str(local_repo) not in script_text:
        raise RuntimeError("launch script lacks its local root, cannot safely remap")
    remote_launch = run_dir / "remote-launch.sh"
    remote_launch.write_text(script_text.replace(str(local_repo), str(remote_repo)))
    remote_launch.chmod(0o755)
    run_sync = send_tree(run_dir, remote_run)
    # A Codex Executor may edit its own worktree. Never execute its copy of
    # the remote control-plane shim, even though its launch bundle is vetted.
    job_script, observed_job_sha = trusted_remote_job_script()
    budget_pre_start = (site_job_broker.assert_gpu_submit_budget(run_dir)
                        if run_dir.name not in site_job_broker.CPU_PROBES else None)
    observer_pre_start = (phase2_observer_audit_direct()
                          if budget_pre_start and budget_pre_start.get("phase") == "phase2" else None)
    phase_argv = (["--phase", "phase2"] if budget_pre_start and budget_pre_start.get("phase") == "phase2"
                  else [])
    receipt = _json_receipt(ssh("python3", str(job_script), "start", "--run-dir", str(remote_run),
                                "--launch", str(remote_run / "remote-launch.sh"),
                                *phase_argv, timeout=60).stdout)
    if receipt.get("status") != "started" or not isinstance(receipt.get("pid"), int):
        raise RuntimeError(f"remote start receipt invalid: {receipt}")
    return {"schema_version": 1, "transport": "ssh_pty_checked_tar_site_adapter",
            "remote_repo": str(remote_repo), "remote_run_dir": str(remote_run),
            "remote_job_pid": receipt["pid"], "remote_host": receipt.get("host"),
            "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
            "site_budget_pre_transfer": budget_pre_transfer,
            "site_budget_pre_start": budget_pre_start,
            "site_observer_pre_transfer": observer_pre_transfer,
            "site_observer_pre_start": observer_pre_start,
            "source_sync": source_sync, "code_sync": code_sync, "run_sync": run_sync,
            "trusted_job_script": {"remote": str(job_script), "sha256": observed_job_sha},
            "start": receipt}


def _status_direct(run_dir: Path) -> dict:
    remote_run = remote_path(run_dir)
    _repo_for_run(run_dir)
    job_script, observed_job_sha = trusted_remote_job_script()
    result = _json_receipt(ssh("python3", str(job_script),
                               "status", "--run-dir", str(remote_run), timeout=30).stdout)
    result["trusted_job_script"] = {"remote": str(job_script), "sha256": observed_job_sha}
    if run_dir.name in (
        "20260913-110805-kernel_compile_bf16_max_autotune",
        "20260913-114414-kernel_compile_bf16_autotune_qkv",
    ) and not result.get("finished"):
        # Read-only, exact-run transport diagnostic. The Executor still owns
        # the original method/budget decision; this is not a candidate gate.
        log = remote_run / "outputs/prompts/p01/run.log"
        probe = (
            "import json,sys; from datetime import datetime,timezone; from pathlib import Path; "
            "p=Path(sys.argv[1]); s=p.read_text(errors='replace') if p.is_file() else ''; "
            "now=datetime.now(timezone.utc); "
            "print(json.dumps({'observed_at_utc':now.isoformat(),"
            "'p01_log_bytes':p.stat().st_size if p.is_file() else 0,"
            "'p01_log_mtime_utc':datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat() if p.is_file() else None,"
            "'p01_log_age_s':max(0.0,now.timestamp()-p.stat().st_mtime) if p.is_file() else None,"
            "'p01_autotune_choice_timeout_count':s.count('Timed out benchmarking choice'),"
            "'p01_video_exists':(p.parent/'out.mp4').is_file(),"
            "'source_log':str(p)}))"
        )
        try:
            result["site_observer_preflight_signal"] = _json_receipt(
                ssh("python3", "-c", probe, str(log), timeout=20).stdout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as exc:
            result["site_observer_preflight_signal"] = {"status": "unavailable",
                                                         "error_type": type(exc).__name__}
    return result


def _sync_run_direct(run_dir: Path) -> dict:
    remote_run = remote_path(run_dir)
    state = _status_direct(run_dir)
    if state.get("finished"):
        pull_tree(remote_run / "outputs", run_dir / "outputs")
        pull_tree(remote_run, run_dir / "site_remote_receipts", excludes=["outputs"])
    return state


def submit(run_dir: Path, launch_script: Path) -> dict:
    if os.environ.get("SOL_SITE_JOB_BROKER") == "1":
        from orchestration import site_job_broker
        return site_job_broker.request("submit", run_dir, launch_script)
    return _submit_direct(run_dir, launch_script)


def status(run_dir: Path) -> dict:
    if os.environ.get("SOL_SITE_JOB_BROKER") == "1":
        from orchestration import site_job_broker
        return site_job_broker.request("status", run_dir)
    return _status_direct(run_dir)


def sync_run(run_dir: Path) -> dict:
    if os.environ.get("SOL_SITE_JOB_BROKER") == "1":
        from orchestration import site_job_broker
        return site_job_broker.request("sync", run_dir)
    return _sync_run_direct(run_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "sync", "sync-source"))
    parser.add_argument("run_dir", type=Path, nargs="?")
    args = parser.parse_args()
    if args.command == "sync-source":
        result = sync_source()
    elif args.run_dir is None:
        parser.error("status/sync requires run_dir")
    else:
        result = status(args.run_dir) if args.command == "status" else sync_run(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
