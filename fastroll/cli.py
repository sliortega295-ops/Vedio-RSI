"""Machine-readable tools for Codex: submit, start, status, wait, collect, cancel."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .contracts import ContractError, atomic_json, canonical, digest, pin_source, pin_assets, read_json, validate_plan
from .runtime import Controller
from .store import Store


def expand_candidate_matrix(spec: dict) -> dict:
    """Fan out independent samples without inventing candidate search dependencies.

Each sample is a sampling payload (explicit prompt/seed/shape). An optional
reference session creates shared reference-generation nodes; a paired evaluator
runs only after both videos exist and is NEVER cached as a candidate verdict.
"""
    samples, candidates = spec["samples"], spec["candidates"]
    if not samples or not candidates:
        raise ContractError("matrix requires candidates and samples")
    tasks = []
    references = {}
    for candidate in candidates:
        cid = candidate["id"]
        predecessor = candidate.get("depends_on", [])
        deps = [f"{p}-s{i}-score" if spec.get("paired_evaluator") else f"{p}-s{i}" for p in predecessor for i in range(len(samples))]
        # An optional isolated primary keeps performance ranking separate from
        # resident, parallel quality-sample generation.
        if candidate.get("primary", True):
            tasks.append({"id": cid + "-primary", "role": "primary", "resource": "gpu",
                          "session": candidate["session"], "payload": samples[0], "depends_on": deps,
                          "timeout_s": candidate.get("timeout_s", 1800)})
            deps = [cid + "-primary"]
        for i, sample in enumerate(samples):
            task_id = f"{cid}-s{i}"
            tasks.append({"id": task_id, "role": "generate", "resource": "gpu", "session": candidate["session"],
                          "payload": sample, "depends_on": deps, "mode": candidate.get("mode", "oneshot"),
                          "timeout_s": candidate.get("timeout_s", 1800)})
            if "reference_session" in spec:
                ref_id = "ref-" + digest({"session": spec["reference_session"], "sample": sample})[:20]
                if ref_id not in references:
                    references[ref_id] = True
                    tasks.append({"id": ref_id, "role": "reference_generate", "resource": "gpu",
                                  "session": spec["reference_session"], "payload": sample, "reuse": True,
                                  "mode": spec.get("reference_mode", "oneshot")})
                if "paired_evaluator" in spec:
                    evaluator = spec["paired_evaluator"]
                    tasks.append({"id": task_id + "-score", "role": "score", "resource": evaluator.get("resource", "gpu"),
                                  "session": evaluator["session"], "payload": evaluator.get("payload", {}),
                                  "mode": evaluator.get("mode", "oneshot"),
                                  "depends_on": [task_id, ref_id],
                                  "inputs": {"candidate": {"task": task_id, "artifact": "video"},
                                             "reference": {"task": ref_id, "artifact": "video"}}})
            elif "paired_evaluator" in spec:
                raise ContractError("paired evaluator requires reference_session")
    return validate_plan({"schema_version": 1, "batch_id": spec["batch_id"], "tasks": tasks})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FastRoll video optimization execution harness")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("submit", "status", "collect", "verify", "cancel", "wait", "sessions", "run", "start"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--workspace", type=Path, required=True)
        if name == "submit":
            cmd.add_argument("--plan", type=Path, required=True)
        if name == "cancel":
            cmd.add_argument("--batch")
        if name == "wait":
            cmd.add_argument("--timeout", type=float, default=3600)
        if name in {"run", "start"}:
            exclusive = cmd.add_mutually_exclusive_group(required=True)
            exclusive.add_argument("--authorization", type=Path)
            exclusive.add_argument("--simulate-gpus", type=int)
            cmd.add_argument("--cpu-workers", type=int, default=2)
            cmd.add_argument("--idle-exit-s", type=float, default=30)
            cmd.add_argument("--resident-idle-s", type=float, default=60)
    cmd = sub.add_parser("pin-source")
    cmd.add_argument("--root", type=Path, required=True)
    cmd = sub.add_parser("pin-assets")
    cmd.add_argument("--root", type=Path, required=True)
    cmd = sub.add_parser("expand")
    cmd.add_argument("--matrix", type=Path, required=True)
    cmd.add_argument("--output", type=Path, required=True)
    cmd = sub.add_parser("qualify-reset")
    cmd.add_argument("--session", type=Path, required=True)
    cmd.add_argument("--samples", type=Path, required=True)
    cmd.add_argument("--workspace", type=Path, required=True)
    group = cmd.add_mutually_exclusive_group(required=True)
    group.add_argument("--authorization", type=Path)
    group.add_argument("--simulate-gpus", type=int)
    cmd.add_argument("--timeout", type=float, default=1800)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "qualify-reset":
            from .qualification import qualify
            result = qualify(read_json(args.session), read_json(args.samples), args.workspace,
                             authorization=args.authorization, simulate_gpus=args.simulate_gpus, timeout_s=args.timeout)
        elif args.command == "pin-source":
            result = pin_source(args.root)
        elif args.command == "pin-assets":
            result = pin_assets(args.root)
        elif args.command == "expand":
            result = expand_candidate_matrix(read_json(args.matrix))
            atomic_json(args.output, result)
            result = {"plan": str(args.output.resolve()), "hash": digest(result), "tasks": len(result["tasks"])}
        else:
            store = Store(args.workspace)
            if args.command == "submit":
                result = {"batch_id": store.submit(read_json(args.plan)), "status": "QUEUED", "gpu_started": False}
            elif args.command == "sessions":
                sessions = {digest(r["spec"]["session"]): r["spec"]["session"] for r in store.rows()}
                result = {"sessions": sessions, "authorization_template": {
                    "authorized": False, "issuer": "RESOURCE_OWNER_MUST_APPROVE", "workspace": str(store.root),
                    "gpu_uuids": [], "source_roots": [], "approved_sessions": sorted(sessions),
                    "not_before": 0, "expires_at": 0, "lock_root": "/tmp/vedio-rsi-gpu-locks"}}
            elif args.command == "cancel":
                store.cancel(args.batch)
                result = store.report()
            elif args.command == "verify":
                result = store.verify()
            elif args.command == "collect":
                verification = store.verify()
                result = {"verification": verification, **store.report()}
            elif args.command == "status":
                result = store.report()
                result.pop("task_results")
                rows = {r["id"]: r for r in store.rows()}
                ready, waiting = [], {}
                for row in rows.values():
                    if row["state"] != "queued":
                        continue
                    missing = [d for d in row["spec"]["depends_on"] if rows[row["batch"] + "/" + d]["state"] != "succeeded"]
                    if missing:
                        waiting[row["id"]] = missing
                    else:
                        ready.append({"task": row["id"], "resource": row["spec"]["resource"], "device_constraint": row["spec"]["device"], "ready_at": row["ready_at"]})
                result.update(ready_count=len(ready), ready_preview=ready[:20], waiting_dependencies=dict(list(waiting.items())[:20]))
            elif args.command == "wait":
                deadline = time.monotonic() + args.timeout
                while True:
                    result = store.report()
                    if not result["tasks"]["running"] and not result["tasks"]["queued"]:
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("wait deadline reached; no tasks were cancelled")
                    time.sleep(0.5)
            elif args.command == "start":
                command = [sys.executable, "-m", "fastroll", "run", "--workspace", str(store.root),
                           "--cpu-workers", str(args.cpu_workers), "--idle-exit-s", str(args.idle_exit_s),
                           "--resident-idle-s", str(args.resident_idle_s)]
                command += ["--authorization", str(args.authorization.resolve())] if args.authorization else ["--simulate-gpus", str(args.simulate_gpus)]
                env = dict(os.environ)
                env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
                with (store.root / "controller.log").open("ab") as log:
                    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, env=env)
                result = {"controller_pid": proc.pid, "status": "SPAWNED_NOT_YET_READY", "log": str(store.root / "controller.log")}
            elif args.command == "run":
                controller = Controller(store, authorization=args.authorization, simulate_gpus=args.simulate_gpus,
                                        cpu_workers=args.cpu_workers, idle_exit_s=args.idle_exit_s,
                                        resident_idle_s=args.resident_idle_s)
                for sig in (signal.SIGINT, signal.SIGTERM):
                    signal.signal(sig, lambda *_: controller.stop.set())
                result = controller.drain()
            else:
                raise AssertionError(args.command)
        print(canonical(result).decode())
        if args.command in {"run", "wait", "collect"}:
            states = result.get("tasks", {})
            if any(states.get(k, 0) for k in ("failed", "cancelled", "blocked", "queued", "running")):
                return 2
        return 0
    except Exception as exc:
        print(canonical({"status": "ERROR", "error": type(exc).__name__ + ": " + str(exc)}).decode(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
