"""New-session worker; JSON protocol is isolated from library stdout/stderr."""
from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

from .adapters import build_adapter
from .contracts import ContractError, canonical, digest, verify_source, verify_assets


def _die(*_):
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    # The worker and all its adapter descendants share this new process group.
    os.killpg(os.getpgrp(), signal.SIGTERM)


def main():
    parent = os.getppid()
    signal.signal(signal.SIGTERM, _die)
    def watch_parent():
        while True:
            if os.getppid() != parent or parent == 1:
                _die()
            time.sleep(0.2)
    threading.Thread(target=watch_parent, daemon=True).start()
    # Duplicate the protocol FD, then redirect fd=1 as well as Python stdout.
    # Native CUDA/C++ logging must not corrupt JSON replies.
    wire = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    adapter = None
    identity = None
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("op") == "shutdown":
                break
            request_id = request["request_id"]
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    session = request["session"]
                    if "source" in session:
                        verify_source(session["source"])
                    verify_assets(session.get("assets", []))
                    current = digest(session)
                    if identity is not None and current != identity:
                        raise ContractError("worker cannot switch its immutable session")
                    startup_s = 0.0
                    if adapter is None:
                        t = time.monotonic()
                        adapter = build_adapter(session)
                        startup_s = time.monotonic() - t
                        identity = current
                    t = time.monotonic()
                    adapter.reset(request)
                    reset_s = time.monotonic() - t
                    t = time.monotonic()
                    result = adapter.run(request)
                    execute_s = time.monotonic() - t
                    verify_assets(session.get("assets", []))
                    if "source" in session:
                        verify_source(session["source"])
                response = {"request_id": request_id, "ok": True, "result": result,
                            "startup_s": startup_s, "reset_s": reset_s, "execute_s": execute_s,
                            "pid": os.getpid(), "session_hash": identity}
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                response = {"request_id": request_id, "ok": False, "error": type(exc).__name__ + ": " + str(exc)}
            wire.write(canonical(response).decode() + "\n")
            wire.flush()
            if not response["ok"]:
                break  # never reuse a potentially contaminated adapter after failure
    finally:
        if adapter is not None:
            with contextlib.redirect_stdout(sys.stderr):
                adapter.close()


if __name__ == "__main__":
    main()
