"""Work-conserving DAG dispatcher, sticky resident sessions and verified reuse."""
from __future__ import annotations

import concurrent.futures
import json
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .contracts import (CACHE_ROLES, TERMINAL, ContractError, artifact_receipt,
                        atomic_json, canonical, digest, read_json, sha256,
                        verify_result, verify_source, verify_assets)
from .resources import Authority, FileLock, GPUReservation, inventory
from .store import Store


def _session_env(session: dict, device: dict, directory: Path) -> dict[str, str]:
    # Do not propagate API tokens or Codex credentials to execution workers.
    keep = ("PATH", "LD_LIBRARY_PATH", "VIRTUAL_ENV", "CONDA_PREFIX", "LANG", "LC_ALL", "TMPDIR")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    home = directory / "home"
    home.mkdir(parents=True, exist_ok=True)
    env.update(HOME=str(home), PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTHONPATH=str(Path(__file__).resolve().parent.parent),
               CUDA_VISIBLE_DEVICES=str(device.get("index", "")) if device.get("real") else "",
               FASTROLL_GPU_UUID=device["uuid"], FASTROLL_GPU_INDEX=str(device.get("index", -1)),
               FASTROLL_SOURCE_ROOT=session.get("source", {}).get("root", str(directory)),
               TORCHINDUCTOR_CACHE_DIR=str(directory / "inductor"),
               TRITON_CACHE_DIR=str(directory / "triton"))
    extra = session["config"].get("env", {})
    if not isinstance(extra, dict):
        raise ContractError("adapter env must be an object")
    for key, value in extra.items():
        if not key.startswith(("SGLANG_", "SANA_", "TORCH_", "TORCHINDUCTOR_")) or key.endswith("CACHE_DIR"):
            raise ContractError(f"unsupported recipe environment override: {key}")
        env[key] = str(value)
    return env


class Session:
    def __init__(self, spec: dict, device: dict, root: Path):
        self.spec, self.device = spec, device
        self.key = digest(spec)
        self.directory = root / "sessions" / uuid.uuid4().hex
        self.directory.mkdir(parents=True)
        self.log = (self.directory / "worker.stderr.log").open("wb")
        env = _session_env(spec, device, self.directory)
        python = spec.get("python", sys.executable)
        if not Path(python).is_absolute() or not Path(python).is_file():
            self.log.close()
            raise ContractError("session python must be an absolute existing executable")
        try:
            self.process = subprocess.Popen(
                [python, "-m", "fastroll.worker"], cwd=self.directory, env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                start_new_session=True, bufsize=0)
        except BaseException:
            self.log.close()
            raise
        self.last_used = time.monotonic()

    def call(self, request: dict, timeout: float, cancelled: threading.Event) -> dict:
        p = self.process
        assert p.stdin is not None and p.stdout is not None
        p.stdin.write(canonical(request) + b"\n")
        p.stdin.flush()
        deadline = time.monotonic() + timeout
        buffer = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(p.stdout, selectors.EVENT_READ)
            while True:
                if cancelled.is_set():
                    raise RuntimeError("task cancelled or authority revoked")
                if time.monotonic() >= deadline:
                    raise TimeoutError("worker request deadline exceeded (including initialization)")
                if not selector.select(min(0.2, max(0.001, deadline - time.monotonic()))):
                    continue
                chunk = os.read(p.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError(f"worker exited {p.poll()}; see {self.directory}/worker.stderr.log")
                buffer.extend(chunk)
                if len(buffer) > 8 * 1024 * 1024:
                    raise ContractError("oversized worker protocol response")
                if b"\n" in buffer:
                    line, rest = bytes(buffer).split(b"\n", 1)
                    if rest.strip():
                        raise ContractError("worker emitted extra protocol data")
                    response = json.loads(line)
                    if response.get("request_id") != request["request_id"]:
                        raise ContractError("worker response identity mismatch")
                    if not response.get("ok"):
                        raise RuntimeError(response.get("error", "worker failure"))
                    self.last_used = time.monotonic()
                    return response

    def close(self):
        p = self.process
        # Always kill the process GROUP, even if its leader has already exited.
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait(timeout=5)
        if p.stdin:
            p.stdin.close()
        if p.stdout:
            p.stdout.close()
        self.log.close()


def _residency_check(task: dict, device: dict | None = None, simulation: bool = False):
    if task["mode"] != "resident" or task["session"]["backend"] in {"demo", "compare"}:
        return
    config = task["session"]["config"]
    certificate = config.get("reset_certificate")
    if not isinstance(certificate, dict):
        raise ContractError("real resident adapter requires a source-bound A-B-A reset certificate")
    session_without_certificate = json.loads(canonical(task["session"]))
    session_without_certificate["config"].pop("reset_certificate", None)
    path = Path(certificate.get("path", ""))
    if not path.is_absolute() or sha256(path) != certificate.get("sha256"):
        raise ContractError("reset certificate is missing or changed")
    value = read_json(path)
    if value.get("status") != "PASS" or value.get("session_hash") != digest(session_without_certificate) or value.get("protocol") != "fresh-vs-resident-A-B-A-v1" or not value.get("evidence"):
        raise ContractError("resident certificate does not bind this runtime/recipe")
    if device is not None:
        from .qualification import sampling_domain
        if value.get("simulation") != simulation or device["uuid"] not in value.get("devices", []) or value.get("sampling_domain") != sampling_domain(task["payload"]):
            raise ContractError("resident certificate device or sampling domain mismatch")
    from .qualification import verify_certificate
    verify_certificate(value, session_without_certificate)


class Controller:
    def __init__(self, store: Store, *, authorization: Path | None = None,
                 simulate_gpus: int | None = None, cpu_workers: int = 2,
                 idle_exit_s: float = 3, resident_idle_s: float = 60):
        if not 1 <= cpu_workers <= 64 or (simulate_gpus is not None and not 0 <= simulate_gpus <= 32):
            raise ContractError("invalid worker limits")
        self.store = store
        self.authority = Authority(authorization, store.root) if authorization else None
        if simulate_gpus is None and self.authority is None:
            raise PermissionError("run requires external authorization; demo requires --simulate-gpus")
        if simulate_gpus is not None and authorization:
            raise ContractError("simulation and real authorization cannot be mixed")
        self.simulation = simulate_gpus is not None
        self.simulate_gpus = simulate_gpus or 0
        self.cpu_workers = cpu_workers
        self.idle_exit_s = max(0, idle_exit_s)
        self.resident_idle_s = max(0, resident_idle_s)
        self.stop = threading.Event()
        self.sessions: dict[str, Session] = {}
        self.code_identity = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}

    def _inputs(self, row: dict, rows: dict) -> dict:
        resolved = {}
        for name, binding in row["spec"]["inputs"].items():
            parent = rows[row["batch"] + "/" + binding["task"]]
            verify_result(parent["result"], self.store.root)
            resolved[name] = parent["result"]["artifacts"][binding["artifact"]]
        return resolved

    def _cache_key(self, row: dict, device: dict, inputs: dict) -> str | None:
        t = row["spec"]
        if not t["reuse"]:
            return None
        return digest({"session": t["session"], "payload": t["payload"], "role": t["role"],
                       "inputs": {k: {"sha256": v["sha256"], "bytes": v["bytes"]} for k, v in inputs.items()},
                       "device": device["uuid"], "harness": self.code_identity})

    def _execute(self, row: dict, device: dict, inputs: dict, cancelled: threading.Event) -> dict:
        t = row["spec"]
        started = time.monotonic()
        slot = device["uuid"]
        out = self.store.root / "artifacts" / row["id"] / f"attempt-{row['attempts'] + 1:04d}"
        request_id = digest({"task": row["id"], "attempt": row["attempts"] + 1, "spec": t})
        session = self.sessions.get(slot)
        result = None
        try:
            out.mkdir(parents=True, exist_ok=False)
            if session is not None and (session.key != digest(t["session"]) or t["mode"] == "oneshot"):
                session.close()
                self.sessions.pop(slot, None)
                session = None
            if "source" in t["session"]:
                verify_source(t["session"]["source"])
            if not self.simulation:
                assert self.authority is not None
                self.authority.admit(t["session"])
            verify_assets(t["session"].get("assets", []))
            _residency_check(t, device, self.simulation)
            if session is None:
                if device.get("real") and inventory()[slot]["free_mib"] < t["min_free_mib"]:
                    raise RuntimeError("insufficient free GPU memory at admission")
                session = Session(t["session"], device, self.store.root)
                self.sessions[slot] = session
                self.store.annotate("session_started", row["id"], {"device": slot, "pid": session.process.pid, "session": session.key})
            request = {"request_id": request_id, "session": t["session"], "payload": t["payload"],
                       "inputs": inputs, "output_dir": str(out), "source_root": t["session"].get("source", {}).get("root", str(out))}
            atomic_json(out / "request.json", request)
            remaining = t["timeout_s"] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("admission consumed the task deadline")
            response = session.call(request, remaining, cancelled)
            if cancelled.is_set():
                raise RuntimeError("cancelled before result commit")
            raw = response["result"]
            if not isinstance(raw, dict) or not isinstance(raw.get("metrics"), dict) or not isinstance(raw.get("artifacts"), dict):
                raise ContractError("invalid adapter result contract")
            if t["role"] in {"generate", "reference_generate"} and "video" not in raw["artifacts"]:
                raise ContractError("generation requires a real video artifact")
            artifacts = {k: artifact_receipt(Path(p) if Path(p).is_absolute() else out / p, out) for k, p in raw["artifacts"].items()}
            result = {"metrics": raw["metrics"], "artifacts": artifacts,
                      "provenance": {"request_id": request_id, "session_hash": response["session_hash"], "device": slot,
                                     "startup_s": response["startup_s"], "reset_s": response["reset_s"], "execute_s": response["execute_s"],
                                     "mode": t["mode"], "simulation": self.simulation}}
            verify_result(result, self.store.root)
            atomic_json(out / "result.json", result)
            return {"ok": True, "result": result}
        except Exception as exc:
            if session:
                session.close()
                self.sessions.pop(slot, None)
            return {"ok": False, "error": type(exc).__name__ + ": " + str(exc)}
        finally:
            if t["mode"] == "oneshot" and slot in self.sessions:
                self.sessions.pop(slot).close()
            if result is not None and t["role"] == "primary":
                result["metrics"]["primary_wall_s"] = time.monotonic() - started
                atomic_json(out / "result.json", result)
            self.store.annotate("attempt_finished", row["id"], {"wall_s": time.monotonic() - started,
                                "resource": t["resource"], "device": slot, "attempt": row["attempts"] + 1})

    def drain(self) -> dict:
        lock = FileLock(self.store.root / "controller.lock")
        reservation = None
        active = {}
        pool = None
        started = time.monotonic()
        try:
            if self.authority:
                reservation = GPUReservation(self.authority)
                devices = [{**d, "resource": "gpu", "real": True} for d in reservation.devices]
            else:
                devices = [{"uuid": f"SIM-{n}", "resource": "gpu", "real": False, "total_mib": 10**9} for n in range(self.simulate_gpus)]
            devices += [{"uuid": f"CPU-{n}", "resource": "cpu", "real": False} for n in range(self.cpu_workers)]
            self.store.verify()
            self.store.recover()
            self.store.annotate("controller_started", "", {"simulation": self.simulation, "devices": [d["uuid"] for d in devices], "harness": self.code_identity})
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(devices))
            last_activity = time.monotonic()
            while not self.stop.is_set():
                if self.authority:
                    self.authority.check()
                if any(sha256(Path(__file__).parent / name) != h for name, h in self.code_identity.items()):
                    raise ContractError("harness code changed during execution")
                rows = {r["id"]: r for r in self.store.rows()}
                for future, info in list(active.items()):
                    row, device, cancelled, cache_key = info
                    if rows[row["id"]]["state"] == "cancelled":
                        cancelled.set()
                    if not future.done():
                        continue
                    outcome = future.result()
                    if outcome["ok"]:
                        committed = self.store.transition(row["id"], {"running"}, "succeeded", {"result": outcome["result"]})
                        if committed and cache_key:
                            self.store.cache_put(cache_key, outcome["result"], row["id"])
                    else:
                        retry = row["attempts"] < row["spec"]["retries"]
                        self.store.transition(row["id"], {"running"}, "queued" if retry else "failed",
                                              {"error": outcome["error"], "ready_at": time.time() + min(30, 2 ** row["attempts"])})
                    del active[future]
                    last_activity = time.monotonic()
                rows = {r["id"]: r for r in self.store.rows()}
                for row in rows.values():
                    if row["state"] == "queued" and any(rows[row["batch"] + "/" + dep]["state"] in {"failed", "cancelled", "blocked"} for dep in row["spec"]["depends_on"]):
                        self.store.transition(row["id"], {"queued"}, "blocked", {"error": "dependency did not succeed"})
                ready = [r for r in rows.values() if r["state"] == "queued" and r["ready_at"] <= time.time()
                         and all(rows[r["batch"] + "/" + d]["state"] == "succeeded" for d in r["spec"]["depends_on"])]
                busy = {info[1]["uuid"] for info in active.values()}
                inflight_keys = {info[3] for info in active.values() if info[3]}
                for device in devices:
                    slot = device["uuid"]
                    if slot in busy:
                        continue
                    current = self.sessions.get(slot)
                    eligible = [r for r in ready if r["spec"]["resource"] == device["resource"]
                                and (r["spec"]["device"] is None or r["spec"]["device"] == slot)
                                and r["spec"]["min_free_mib"] <= device.get("total_mib", 10**9)]
                    # Soft affinity, bounded aging: never dedicate a GPU to one technique.
                    eligible.sort(key=lambda r: (r["spec"]["priority"] + int((time.time() - r["submitted"]) / 30),
                                                 bool(current and current.key == digest(r["spec"]["session"])),
                                                 r["spec"]["estimate_s"]), reverse=True)
                    for row in eligible:
                        t = row["spec"]
                        try:
                            if self.simulation and t["session"]["backend"] not in {"demo", "compare"}:
                                raise PermissionError("simulation cannot execute real adapters or arbitrary code")
                            if self.authority:
                                self.authority.admit(t["session"])
                            if "source" in t["session"]:
                                verify_source(t["session"]["source"])
                            verify_assets(t["session"].get("assets", []))
                            inputs = self._inputs(row, rows)
                            key = self._cache_key(row, device, inputs)
                            if key in inflight_keys:
                                continue
                            hit = self.store.cache_get(key) if key else None
                            if hit:
                                result = {**hit["result"], "provenance": {**hit["result"]["provenance"], "reused_from": hit["producer"]}}
                                if self.store.transition(row["id"], {"queued"}, "succeeded", {"result": result}):
                                    self.store.annotate("cache_hit", row["id"], {"producer": hit["producer"], "key": key})
                                ready.remove(row)
                                last_activity = time.monotonic()
                                continue
                            if not self.store.transition(row["id"], {"queued"}, "running", {"device": slot}):
                                ready.remove(row)
                                continue
                            cancel = threading.Event()
                            future = pool.submit(self._execute, row, device, inputs, cancel)
                            active[future] = (row, device, cancel, key)
                            if key:
                                inflight_keys.add(key)
                            ready.remove(row)
                            last_activity = time.monotonic()
                            break
                        except Exception as exc:
                            self.store.transition(row["id"], {"queued"}, "failed", {"error": str(exc)})
                            ready.remove(row)
                    if slot not in {info[1]["uuid"] for info in active.values()}:
                        current = self.sessions.get(slot)
                        if current and time.monotonic() - current.last_used > self.resident_idle_s:
                            self.sessions.pop(slot).close()
                if not active and time.monotonic() - last_activity >= self.idle_exit_s:
                    if not any(r["state"] == "queued" and r["ready_at"] > time.time() for r in self.store.rows()):
                        break
                time.sleep(0.02)
        finally:
            self.stop.set()
            for _, _, cancel, _ in active.values():
                cancel.set()
            if pool:
                pool.shutdown(wait=True)
            for session in list(self.sessions.values()):
                session.close()
            self.sessions.clear()
            self.store.annotate("controller_stopped", "", {"wall_s": time.monotonic() - started,
                                "reserved_gpu_seconds": (time.monotonic() - started) * (len(reservation.devices) if reservation else self.simulate_gpus),
                                "simulation": self.simulation})
            if reservation:
                reservation.close()
            lock.close()
        return self.store.report()
