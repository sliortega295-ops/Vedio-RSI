from __future__ import annotations

import ast
import importlib.util
import json
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rolloutbench.runplan import build_experiment_plan


REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER = (
    REPO_ROOT
    / "models"
    / "sana_video_2b_h100"
    / "baseline"
    / "port_isolated_exec.py"
)
SUITE_DIR = REPO_ROOT / "benchmarks" / "sana_video_2b_h100_v0"
GPU_UUIDS = (
    "GPU-83ed65f8-62e5-2a01-3471-8bfc752971d3",
    "GPU-847305ce-670b-91ee-e0a9-aa3b7833df23",
)


def _adapter_module():
    spec = importlib.util.spec_from_file_location("port_isolated_exec", ADAPTER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the port-isolation adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PortIsolatedExecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.adapter = _adapter_module()

    def _run(
        self,
        source: str,
        *,
        port: int = 29500,
        master_port: int = 28000,
        scheduler_port: int = 26000,
        nccl_port: int = 27000,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "historical_runner.py"
            target.write_text(source, encoding="utf-8")
            return subprocess.run(
                [
                    sys.executable,
                    str(ADAPTER),
                    "--target",
                    str(target),
                    "--port",
                    str(port),
                    "--master-port",
                    str(master_port),
                    "--scheduler-port",
                    str(scheduler_port),
                    "--nccl-port",
                    str(nccl_port),
                    "--strict-ports",
                    "--",
                    "--candidate-flag",
                    "preserved",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

    def test_injects_exact_ports_without_changing_target_cli_or_file_identity(self) -> None:
        result = self._run(
            """
import json
import sys
import types

class DiffGenerator:
    @classmethod
    def from_pretrained(cls, **kwargs):
        print("FAKE_CALL " + json.dumps(kwargs, sort_keys=True))
        return types.SimpleNamespace(
            server_args=types.SimpleNamespace(**kwargs),
            port_args=types.SimpleNamespace(
                master_port=kwargs["master_port"], nccl_port=kwargs["nccl_port"]
            ),
        )

if __name__ == "__main__":
    print("TARGET_IDENTITY " + json.dumps({"argv": sys.argv, "file": __file__}, sort_keys=True))
    DiffGenerator.from_pretrained(model_path="model", num_gpus=1)
""".lstrip()
        )
        self.assertEqual(0, result.returncode, result.stdout)
        marker = next(
            line.removeprefix("ROLLOUTBENCH_PORT_ISOLATION ")
            for line in result.stdout.splitlines()
            if line.startswith("ROLLOUTBENCH_PORT_ISOLATION ")
        )
        receipt = json.loads(marker)
        self.assertEqual(
            {
                "port": 29500,
                "master_port": 28000,
                "scheduler_port": 26000,
                "nccl_port": 27000,
            },
            receipt["ports"],
        )
        self.assertIs(receipt["strict_ports"], True)
        self.assertEqual("AVAILABLE", receipt["port_preflight"]["status"])
        self.assertEqual(1, receipt["injected_call_count"])
        call = json.loads(
            next(
                line.removeprefix("FAKE_CALL ")
                for line in result.stdout.splitlines()
                if line.startswith("FAKE_CALL ")
            )
        )
        self.assertEqual(29500, call["port"])
        self.assertEqual(28000, call["master_port"])
        self.assertEqual(26000, call["scheduler_port"])
        self.assertEqual(27000, call["nccl_port"])
        self.assertIs(call["strict_ports"], True)
        effective = json.loads(
            next(
                line.removeprefix("ROLLOUTBENCH_EFFECTIVE_PORTS ")
                for line in result.stdout.splitlines()
                if line.startswith("ROLLOUTBENCH_EFFECTIVE_PORTS ")
            )
        )
        self.assertEqual(call["master_port"], effective["port_args"]["master_port"])
        self.assertEqual(call["nccl_port"], effective["port_args"]["nccl_port"])
        identity = json.loads(
            next(
                line.removeprefix("TARGET_IDENTITY ")
                for line in result.stdout.splitlines()
                if line.startswith("TARGET_IDENTITY ")
            )
        )
        self.assertTrue(identity["file"].endswith("historical_runner.py"))
        self.assertEqual(
            [identity["file"], "--candidate-flag", "preserved"], identity["argv"]
        )

    def test_fails_closed_if_historical_runner_already_sets_a_managed_port(self) -> None:
        result = self._run(
            """
class DiffGenerator:
    @classmethod
    def from_pretrained(cls, **kwargs):
        return object()

DiffGenerator.from_pretrained(model_path="model", master_port=1)
""".lstrip()
        )
        self.assertNotEqual(0, result.returncode)
        self.assertIn("already sets managed port", result.stdout)

    def test_fails_closed_unless_exactly_one_generator_call_is_found(self) -> None:
        result = self._run("print('no generator call')\n")
        self.assertNotEqual(0, result.returncode)
        self.assertIn("expected exactly one DiffGenerator.from_pretrained call", result.stdout)

    def test_fails_closed_for_opaque_or_multiple_generator_calls(self) -> None:
        opaque = self._run(
            "DiffGenerator.from_pretrained(**{'model_path': 'model'})\n"
        )
        self.assertNotEqual(0, opaque.returncode)
        self.assertIn("opaque kwargs expansion", opaque.stdout)
        multiple = self._run(
            "DiffGenerator.from_pretrained(model_path='a')\n"
            "DiffGenerator.from_pretrained(model_path='b')\n"
        )
        self.assertNotEqual(0, multiple.returncode)
        self.assertIn("expected exactly one", multiple.stdout)

    def test_fails_closed_for_invalid_or_duplicate_managed_ports(self) -> None:
        source = "DiffGenerator.from_pretrained(model_path='model')\n"
        duplicate = self._run(source, port=28000)
        self.assertNotEqual(0, duplicate.returncode)
        self.assertIn("managed ports must be distinct", duplicate.stdout)
        invalid = self._run(source, nccl_port=70000)
        self.assertNotEqual(0, invalid.returncode)
        self.assertIn("managed ports must be decimal", invalid.stdout)

    def test_preserves_python_spawn_bootstrap_identity(self) -> None:
        result = self._run(
            """
import json
import multiprocessing as mp
import types

def child(queue):
    queue.put({"file": __file__, "name": __name__})

class DiffGenerator:
    @classmethod
    def from_pretrained(cls, **kwargs):
        context = mp.get_context("spawn")
        queue = context.Queue()
        process = context.Process(target=child, args=(queue,))
        process.start()
        child_identity = queue.get(timeout=10)
        process.join(timeout=10)
        print("SPAWN_RESULT " + json.dumps({
            "child": child_identity,
            "exitcode": process.exitcode,
            "ports": kwargs,
        }, sort_keys=True))
        return types.SimpleNamespace(
            server_args=types.SimpleNamespace(**kwargs),
            port_args=types.SimpleNamespace(
                master_port=kwargs["master_port"], nccl_port=kwargs["nccl_port"]
            ),
        )

if __name__ == "__main__":
    DiffGenerator.from_pretrained(model_path="model")
""".lstrip()
        )
        self.assertEqual(0, result.returncode, result.stdout)
        payload = json.loads(
            next(
                line.removeprefix("SPAWN_RESULT ")
                for line in result.stdout.splitlines()
                if line.startswith("SPAWN_RESULT ")
            )
        )
        self.assertEqual(0, payload["exitcode"])
        self.assertEqual("__mp_main__", payload["child"]["name"])
        self.assertTrue(payload["child"]["file"].endswith("historical_runner.py"))
        self.assertEqual(28000, payload["ports"]["master_port"])

    def test_fails_closed_when_any_managed_port_is_already_occupied(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.bind(("", 28000))
            occupied.listen(1)
            result = self._run(
                """
class DiffGenerator:
    @classmethod
    def from_pretrained(cls, **kwargs):
        raise AssertionError("target must not execute when a port is occupied")

DiffGenerator.from_pretrained(model_path="model")
""".lstrip()
            )
        self.assertNotEqual(0, result.returncode)
        self.assertIn("managed port 28000 is unavailable", result.stdout)

    def test_every_frozen_pilot_runtime_has_the_exact_instrumentable_shape(self) -> None:
        plan = build_experiment_plan(
            SUITE_DIR,
            scope="pilot",
            repetitions=5,
            gpu_uuids=GPU_UUIDS,
            repo_root=REPO_ROOT,
        )
        run = next(
            row
            for row in plan["runs"]
            if row["run_id"] == "pilot-serial1-repeat-01"
        )
        episodes = [run["quality_dense_reference"], *run["episodes"]]
        refs = {
            episode["runtime_checkout"]["git_ref"] for episode in episodes
        }
        ports = {
            "port": 29500,
            "master_port": 28000,
            "scheduler_port": 26000,
            "nccl_port": 27000,
            "strict_ports": True,
        }
        for ref in sorted(refs):
            with self.subTest(ref=ref):
                source = subprocess.check_output(
                    [
                        "git",
                        "show",
                        f"{ref}:external/sol_runtime/scripts/sana/sana_video_sglang_run.py",
                    ],
                    cwd=REPO_ROOT,
                )
                _code, count = self.adapter._instrument(
                    source, Path(f"{ref}/sana_video_sglang_run.py"), ports
                )
                self.assertEqual(1, count)
                server_args_source = subprocess.check_output(
                    [
                        "git",
                        "show",
                        f"{ref}:external/sol_runtime/python/sglang/"
                        "multimodal_gen/runtime/server_args.py",
                    ],
                    cwd=REPO_ROOT,
                )
                server_args_tree = ast.parse(server_args_source)
                server_args_fields = {
                    statement.target.id
                    for node in server_args_tree.body
                    if isinstance(node, ast.ClassDef)
                    and node.name == "ServerArgs"
                    for statement in node.body
                    if isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                }
                self.assertTrue(set(ports).issubset(server_args_fields))


if __name__ == "__main__":
    unittest.main()
