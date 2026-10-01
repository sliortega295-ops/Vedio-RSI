"""Trusted execution adapters. Heavy dependencies are imported only in workers.

A Python adapter exposes reset(request), run(request), close(). A result is
{metrics: object, artifacts: {name: path}}. It is evidence, not a prose verdict.
"""
from __future__ import annotations

import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .contracts import ContractError, atomic_json, digest, read_json, sha256, within


class DemoAdapter:
    """CPU-only synthetic test backend; never represents real video performance."""
    def __init__(self, config: dict):
        self.config = config
        self.count = 0

    def reset(self, request: dict):
        self.seed = int(request["payload"].get("seed", 0))

    def run(self, request: dict) -> dict:
        p = request["payload"]
        if p.get("native_stdout"):
            os.write(1, b"native backend log (not protocol)\n")
        time.sleep(min(float(p.get("sleep_s", 0)), 2))
        if p.get("fail"):
            raise RuntimeError("deliberate demo failure")
        self.count += 1
        output = Path(request["output_dir"]) / "demo.json"
        atomic_json(output, {"value": random.Random(self.seed).random(), "prompt": p.get("prompt", "demo")})
        return {"metrics": {"demo_only": True, "session_requests": self.count}, "artifacts": {"video": str(output)}}

    def close(self):
        pass


class CommandAdapter:
    def __init__(self, config: dict):
        self.config = config
        if not isinstance(config.get("argv"), list) or not config["argv"] or not all(isinstance(x, str) for x in config["argv"]):
            raise ContractError("command argv must be a nonempty string array (no shell parsing)")

    def reset(self, request: dict):
        pass

    def run(self, request: dict) -> dict:
        out = Path(request["output_dir"])
        req, result = out / "request.json", out / "adapter-result.json"
        atomic_json(req, request)
        replacements = {"{request}": str(req), "{result}": str(result), "{output_dir}": str(out)}
        argv = list(self.config["argv"])
        for key, value in replacements.items():
            argv = [x.replace(key, value) for x in argv]
        with (out / "command.stdout.log").open("wb") as stdout, (out / "command.stderr.log").open("wb") as stderr:
            proc = subprocess.run(argv, cwd=request["source_root"], stdout=stdout, stderr=stderr, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"adapter process returned {proc.returncode}; see command.stderr.log")
        value = read_json(result)
        if value.get("request_id") != request["request_id"]:
            raise ContractError("command result is stale or belongs to a different request")
        return value

    def close(self):
        pass


class CompareAdapter:
    """Threshold a predeclared set of metric JSON artifacts, not LLM self-reports."""
    def __init__(self, config: dict):
        self.config = config

    def reset(self, request: dict):
        pass

    def run(self, request: dict) -> dict:
        checks = request["payload"].get("checks", [])
        if not checks:
            raise ContractError("comparison requires nonempty predeclared checks")
        observations = []
        for check in checks:
            value = read_json(Path(request["inputs"][check["input"]]["path"]))
            for key in check["key"].split("."):
                value = value[key]
            if type(value) not in (int, float) or type(check["limit"]) not in (int, float):
                raise ContractError("numeric metric and threshold required")
            if check["op"] not in {"le", "ge"}:
                raise ContractError("comparison op must be le/ge")
            ok = value <= check["limit"] if check["op"] == "le" else value >= check["limit"]
            observations.append({**check, "observed": value, "passed": ok})
        result = {"metrics": {"quality_pass": all(x["passed"] for x in observations)}, "checks": observations}
        path = Path(request["output_dir"]) / "comparison.json"
        atomic_json(path, result)
        return {"metrics": result["metrics"], "artifacts": {"comparison": str(path)}}

    def close(self):
        pass


class SanaAdapter:
    """Resident fixed-recipe SGLang SANA adapter for the pinned Sol runtime API.

    No cross-recipe hot patching. Model/runtime/recipe changes create a new process.
    The operator must certify request-boundary reset before enabling residency.
    """
    def __init__(self, config: dict):
        self.config = config
        root = Path(os.environ["FASTROLL_SOURCE_ROOT"])
        model = Path(config["model_path"]).resolve(strict=True)
        if not model.is_dir():
            raise ContractError("model_path must be an existing local snapshot; no downloads")
        sys.path.insert(0, str(root / "python"))
        runner = root / "scripts/sana/sana_video_sglang_run.py"
        spec = importlib.util.spec_from_file_location("_fastroll_sana_entry", runner)
        if spec is None or spec.loader is None:
            raise ContractError("pinned SANA runner cannot be loaded")
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        entry._install_sana_minimal_package_shims()
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
        base = int(config.get("port_base", 16000)) + int(os.environ["FASTROLL_GPU_INDEX"]) * 32
        import socket
        sockets = []
        try:
            for port in (base, base + 1, base + 2, base + 3):
                sock = socket.socket()
                sockets.append(sock)
                sock.bind(("127.0.0.1", port))
        finally:
            for sock in sockets:
                sock.close()
        self.gen = DiffGenerator.from_pretrained(
            model_path=str(model), pipeline_class_name="SanaVideoPipeline", num_gpus=1,
            dit_cpu_offload=False, enable_torch_compile=bool(config.get("compile", False)),
            warmup=bool(config.get("warmup", True)), port=base, master_port=base + 1,
            scheduler_port=base + 2, nccl_port=base + 3, strict_ports=True,
        )

    def reset(self, request: dict):
        # SGLang receives an explicit seed and constructs the per-request sampling
        # state. No inference caches are shared across recipes. GPU reset tests
        # are additionally required for any source/recipe certified as resident.
        if type(request["payload"].get("seed")) is not int:
            raise ContractError("every SANA request requires an explicit seed")

    def run(self, request: dict) -> dict:
        import torch
        p = request["payload"]
        allowed = {"prompt", "negative_prompt", "seed", "num_frames", "num_inference_steps", "height", "width", "guidance_scale", "fps"}
        if set(p) - allowed or not isinstance(p.get("prompt"), str):
            raise ContractError("invalid SANA sampling payload")
        sampling = {**p, "output_file_name": "fastroll_" + request["request_id"], "save_output": True}
        torch.cuda.synchronize()
        started = time.monotonic()
        results = self.gen.generate(sampling_params_kwargs=sampling)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        items = results if isinstance(results, (list, tuple)) else [results]
        paths = [Path(x.output_file_path) for x in items if getattr(x, "output_file_path", None)]
        if len(paths) != 1 or not paths[0].is_file():
            raise ContractError("SANA returned no unique real video artifact")
        output = Path(request["output_dir"]) / "video.mp4"
        shutil.copy2(paths[0], output)
        probe = subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_read_frames", "-of", "json", str(output)], text=True)
        streams = json.loads(probe)["streams"]
        if len(streams) != 1 or any(int(streams[0][key]) != p[field] for key, field in (("width", "width"), ("height", "height"), ("nb_read_frames", "num_frames"))):
            raise ContractError("generated video violates shape/frame contract")
        return {"metrics": {"request_s": elapsed, "quality_validated": False}, "artifacts": {"video": str(output)}}

    def close(self):
        if hasattr(self, "gen"):
            self.gen.shutdown()


def build_adapter(session: dict):
    backend, config = session["backend"], session["config"]
    if backend == "python":
        root = Path(os.environ["FASTROLL_SOURCE_ROOT"])
        module = within(root / config["module"], root)
        spec = importlib.util.spec_from_file_location("_fastroll_site_adapter", module)
        if spec is None or spec.loader is None:
            raise ContractError("adapter module cannot be imported")
        loaded = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(root))
        spec.loader.exec_module(loaded)
        adapter = getattr(loaded, config.get("factory", "Adapter"))(config.get("options", {}))
    else:
        adapter = {"demo": DemoAdapter, "command": CommandAdapter, "compare": CompareAdapter, "sana": SanaAdapter}[backend](config)
    if any(not callable(getattr(adapter, name, None)) for name in ("reset", "run", "close")):
        raise ContractError("adapter must implement reset, run and close")
    return adapter
