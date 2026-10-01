"""Explicit, authorized A-B-A qualification of a fixed resident adapter.

This is a functional qualification, not a speed benchmark. Video equivalence
uses decoded-frame hashes rather than container metadata. Certificates cannot
be transferred between physical GPUs or simulation and real execution.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

from .contracts import ContractError, atomic_json, digest, read_json, sha256, verify_source, verify_assets
from .resources import Authority, FileLock, GPUReservation
from .runtime import Session


def sampling_domain(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in {"prompt", "seed"}}


def qualify(session: dict, samples: list[dict], root: Path, *, authorization: Path | None,
            simulate_gpus: int | None, timeout_s: float = 1800) -> dict:
    if len(samples) != 2 or samples[0] == samples[1] or sampling_domain(samples[0]) != sampling_domain(samples[1]):
        raise ContractError("qualification needs two distinct prompt/seed samples with identical remaining parameters")
    if session.get("backend") == "command" or "reset_certificate" in session.get("config", {}):
        raise ContractError("qualify a resident-capable session without an existing certificate")
    session = json.loads(json.dumps(session))
    session.setdefault("config", {})
    session.setdefault("assets", [])
    simulation = simulate_gpus is not None
    if simulation and (authorization or session.get("backend") != "demo" or not 1 <= simulate_gpus <= 32):
        raise ContractError("simulation qualification is only for the demo backend")
    if not simulation and authorization is None:
        raise PermissionError("qualification requires its own explicit GPU authorization")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = FileLock(root / "controller.lock")
    authority, reservation = None, None
    live = []
    stop = threading.Event()
    evidence = []
    started = time.monotonic()
    try:
        if not simulation:
            authority = Authority(authorization, root)
            authority.admit(session)
            reservation = GPUReservation(authority)
            devices = [{**d, "real": True} for d in reservation.devices]
        else:
            devices = [{"uuid": f"SIM-{i}", "real": False} for i in range(simulate_gpus)]
        if not devices:
            raise ContractError("qualification requires at least one approved device")
        def watch_authority():
            while not stop.wait(0.2):
                try:
                    if authority:
                        authority.check()
                except Exception:
                    stop.set()
        monitor = threading.Thread(target=watch_authority, daemon=True)
        monitor.start()
        for device in devices:
            signatures = {}
            for mode in ("oneshot", "resident"):
                worker = None
                for index, payload in enumerate([samples[0], samples[1], samples[0]]):
                    if authority:
                        authority.admit(session)
                    if "source" in session:
                        verify_source(session["source"])
                    verify_assets(session.get("assets", []))
                    if worker is None:
                        worker = Session(session, device, root)
                        live.append(worker)
                    rid = digest({"session": session, "device": device["uuid"], "mode": mode, "index": index})
                    out = root / "qualification" / rid
                    out.mkdir(parents=True, exist_ok=False)
                    request = {"request_id": rid, "session": session, "payload": payload, "inputs": {},
                               "output_dir": str(out), "source_root": session.get("source", {}).get("root", str(out))}
                    response = worker.call(request, timeout_s, stop)
                    artifacts = response["result"]["artifacts"]
                    if not artifacts:
                        raise ContractError("qualification must compare actual output artifacts")
                    values = {}
                    for name, raw in artifacts.items():
                        path = Path(raw).resolve(strict=True)
                        if not path.is_relative_to(out) or not path.is_file():
                            raise ContractError("qualification artifact escaped its request")
                        if path.suffix.lower() == ".mp4":
                            value = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "hash", "-hash", "sha256", "-"], text=True, timeout=120).strip()
                        else:
                            value = sha256(path)
                        values[name] = value
                    signatures[(mode, index)] = values
                    receipt = out / "qualification-receipt.json"
                    atomic_json(receipt, {"device": device["uuid"], "mode": mode, "index": index,
                                          "request": request, "response": response, "decoded_artifact_hashes": values,
                                          "output_hashes": {name: {"path": str(Path(raw).resolve()), "sha256": sha256(Path(raw))} for name, raw in artifacts.items()}})
                    evidence.append({"path": str(receipt), "sha256": sha256(receipt)})
                    if mode == "oneshot":
                        worker.close()
                        live.remove(worker)
                        worker = None
                if worker:
                    worker.close()
                    live.remove(worker)
            if any(signatures[("oneshot", i)] != signatures[("resident", i)] for i in range(3)) or signatures[("resident", 0)] != signatures[("resident", 2)]:
                raise ContractError("fresh/resident or A-B-A equivalence failed; residency remains disabled")
        value = {"status": "PASS", "protocol": "fresh-vs-resident-A-B-A-v1", "session_hash": digest(session),
                 "simulation": simulation, "devices": [d["uuid"] for d in devices], "sampling_domain": sampling_domain(samples[0]),
                 "evidence": evidence, "wall_s": time.monotonic() - started,
                 "scope": "only this runtime, recipe, sampling domain and these GPUs; not general quality equivalence"}
        cert = root / "reset-certificate.json"
        if cert.exists():
            raise ContractError("use a new qualification directory; never overwrite evidence")
        atomic_json(cert, value)
        return {"certificate": {"path": str(cert), "sha256": sha256(cert)}, **value}
    finally:
        stop.set()
        for worker in live:
            worker.close()
        if reservation:
            reservation.close()
        lock.close()


def verify_certificate(value: dict, session: dict) -> None:
    """Replay six source-bound observations per certified device."""
    records = {}
    for item in value["evidence"]:
        receipt = Path(item["path"])
        if sha256(receipt) != item["sha256"]:
            raise ContractError("qualification receipt changed")
        row = read_json(receipt)
        key = (row["device"], row["mode"], row["index"])
        if key in records or row["device"] not in value["devices"] or row["mode"] not in {"oneshot", "resident"} or row["index"] not in {0, 1, 2}:
            raise ContractError("invalid or duplicate qualification observation")
        request, response = row["request"], row["response"]
        expected_id = digest({"session": session, "device": row["device"], "mode": row["mode"], "index": row["index"]})
        if request["session"] != session or request["request_id"] != expected_id or response.get("request_id") != expected_id or response.get("ok") is not True or response.get("session_hash") != digest(session):
            raise ContractError("qualification does not bind adapter execution")
        if sampling_domain(request["payload"]) != value["sampling_domain"]:
            raise ContractError("qualification sampling parameters changed")
        outputs = row["output_hashes"]
        if not outputs or set(outputs) != set(row["decoded_artifact_hashes"]) or set(outputs) != set(response["result"]["artifacts"]):
            raise ContractError("qualification artifact inventory mismatch")
        for name, output in outputs.items():
            if str(Path(response["result"]["artifacts"][name]).resolve()) != output["path"] or sha256(Path(output["path"])) != output["sha256"]:
                raise ContractError("qualification output artifact changed")
        records[key] = row
    if len(records) != 6 * len(value["devices"]):
        raise ContractError("incomplete qualification matrix")
    for device in value["devices"]:
        for index in range(3):
            fresh, resident = records[(device, "oneshot", index)], records[(device, "resident", index)]
            if fresh["request"]["payload"] != resident["request"]["payload"] or fresh["decoded_artifact_hashes"] != resident["decoded_artifact_hashes"]:
                raise ContractError("fresh/resident equivalence failed")
        a, b, again = (records[(device, "resident", i)] for i in range(3))
        if a["request"]["payload"] != again["request"]["payload"] or a["request"]["payload"] == b["request"]["payload"] or a["decoded_artifact_hashes"] != again["decoded_artifact_hashes"]:
            raise ContractError("A-B-A reset evidence failed")
