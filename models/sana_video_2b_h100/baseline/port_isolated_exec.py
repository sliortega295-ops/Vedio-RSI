#!/usr/bin/env python3
"""Execute one archived SANA runner with an auditable worker-local port set.

The historical candidate script remains byte-for-byte unchanged.  This adapter
parses it, injects the managed ServerArgs keyword arguments into its single
``DiffGenerator.from_pretrained`` call, and executes the resulting code with the
original script path and argv identity.  Any unexpected source shape fails
closed instead of silently running without isolation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import socket
import sys
from pathlib import Path
from typing import Sequence


_PORT_ARGUMENTS = ("port", "master_port", "scheduler_port", "nccl_port")
_MANAGED_SERVER_ARGS = (*_PORT_ARGUMENTS, "strict_ports")
_EFFECTIVE_VALIDATOR = "_rolloutbench_verify_effective_ports_v1"
_EXPECTED_SERVER_ARGS: dict[str, int | bool] | None = None


class PortIsolationError(RuntimeError):
    """Raised when an archived runner cannot be isolated exactly."""


class _PortInjector(ast.NodeTransformer):
    def __init__(self, server_args: dict[str, int | bool]) -> None:
        self.server_args = server_args
        self.injected_call_count = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "DiffGenerator"
        ):
            return node
        if any(keyword.arg is None for keyword in node.keywords):
            raise PortIsolationError(
                "DiffGenerator.from_pretrained uses an opaque kwargs expansion"
            )
        existing = {
            str(keyword.arg) for keyword in node.keywords if keyword.arg is not None
        }
        conflicts = sorted(existing.intersection(_MANAGED_SERVER_ARGS))
        if conflicts:
            raise PortIsolationError(
                "historical runner already sets managed port keyword(s): "
                + ", ".join(conflicts)
            )
        node.keywords.extend(
            ast.keyword(arg=name, value=ast.Constant(value=self.server_args[name]))
            for name in _MANAGED_SERVER_ARGS
        )
        self.injected_call_count += 1
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=_EFFECTIVE_VALIDATOR, ctx=ast.Load()),
                args=[node],
                keywords=[],
            ),
            node,
        )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--master-port", required=True, type=int)
    parser.add_argument("--scheduler-port", required=True, type=int)
    parser.add_argument("--nccl-port", required=True, type=int)
    parser.add_argument("--strict-ports", required=True, action="store_true")
    parser.add_argument("target_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def _instrument(
    source: bytes, target: Path, server_args: dict[str, int | bool]
) -> tuple[object, int]:
    try:
        tree = ast.parse(source, filename=str(target))
    except (SyntaxError, ValueError) as exc:
        raise PortIsolationError("target runner is not valid Python source") from exc
    injector = _PortInjector(server_args)
    tree = injector.visit(tree)
    if injector.injected_call_count != 1:
        raise PortIsolationError(
            "expected exactly one DiffGenerator.from_pretrained call, found "
            f"{injector.injected_call_count}"
        )
    ast.fix_missing_locations(tree)
    return (
        compile(tree, str(target), "exec", dont_inherit=True),
        injector.injected_call_count,
    )


def _assert_ports_available(ports: dict[str, int]) -> dict[str, object]:
    reservations: list[socket.socket] = []
    try:
        for name in _PORT_ARGUMENTS:
            port = ports[name]
            reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            reservation.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                reservation.bind(("", port))
                reservation.listen(1)
            except OSError as exc:
                reservation.close()
                raise PortIsolationError(
                    f"{name} managed port {port} is unavailable"
                ) from exc
            reservations.append(reservation)
    finally:
        for reservation in reservations:
            reservation.close()
    return {
        "status": "AVAILABLE",
        "checked_ports": [ports[name] for name in _PORT_ARGUMENTS],
        "check_scope": "simultaneous_ipv4_bind_before_strict_runtime_launch",
    }


def _effective_receipt(
    generator: object, expected: dict[str, int | bool]
) -> dict[str, object]:
    try:
        server_args = generator.server_args
        port_args = generator.port_args
        observed_server_args = {
            name: getattr(server_args, name) for name in _MANAGED_SERVER_ARGS
        }
        observed_port_args = {
            "master_port": port_args.master_port,
            "nccl_port": port_args.nccl_port,
        }
    except AttributeError as exc:
        raise PortIsolationError(
            "generator does not expose effective ServerArgs and PortArgs"
        ) from exc
    expected_port_args = {
        "master_port": expected["master_port"],
        "nccl_port": expected["nccl_port"],
    }
    if observed_server_args != expected or observed_port_args != expected_port_args:
        shutdown = getattr(generator, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
        raise PortIsolationError("effective SGLang ports differ from the strict contract")
    return {
        "schema_version": 1,
        "server_args": observed_server_args,
        "port_args": observed_port_args,
    }


def _rolloutbench_verify_effective_ports_v1(generator: object) -> object:
    if _EXPECTED_SERVER_ARGS is None:
        raise PortIsolationError("effective-port contract was not initialized")
    receipt = _effective_receipt(generator, _EXPECTED_SERVER_ARGS)
    print(
        "ROLLOUTBENCH_EFFECTIVE_PORTS "
        + json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return generator


def _execute(args: argparse.Namespace) -> None:
    global _EXPECTED_SERVER_ARGS
    raw_target = Path(args.target)
    if raw_target.is_symlink():
        raise PortIsolationError("target runner must not be a symlink")
    target = raw_target.resolve()
    if not target.is_file():
        raise PortIsolationError("target runner must be a regular file")
    ports = {
        "port": args.port,
        "master_port": args.master_port,
        "scheduler_port": args.scheduler_port,
        "nccl_port": args.nccl_port,
    }
    if any(
        type(value) is not int or not 1 <= value <= 65535
        for value in ports.values()
    ):
        raise PortIsolationError("managed ports must be decimal values in [1, 65535]")
    if len(set(ports.values())) != len(ports):
        raise PortIsolationError("managed ports must be distinct")
    if args.strict_ports is not True:
        raise PortIsolationError("strict port mode is required")
    server_args: dict[str, int | bool] = {**ports, "strict_ports": True}
    _EXPECTED_SERVER_ARGS = server_args

    source = target.read_bytes()
    code, injected_call_count = _instrument(source, target, server_args)
    port_preflight = _assert_ports_available(ports)

    forwarded = list(args.target_args)
    if forwarded[:1] == ["--"]:
        forwarded.pop(0)
    receipt = {
        "schema_version": 1,
        "adapter": str(Path(__file__).resolve()),
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "target": str(target),
        "target_sha256": hashlib.sha256(source).hexdigest(),
        "ports": ports,
        "strict_ports": True,
        "port_preflight": port_preflight,
        "injected_call_count": injected_call_count,
    }
    print(
        "ROLLOUTBENCH_PORT_ISOLATION "
        + json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        flush=True,
    )

    sys.argv = [str(target), *forwarded]
    sys.path[0] = str(target.parent)
    namespace = sys.modules["__main__"].__dict__
    namespace.update(
        {
            "__file__": str(target),
            "__name__": "__main__",
            "__package__": None,
            "__spec__": None,
            "__cached__": None,
        }
    )
    exec(code, namespace, namespace)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        _execute(_parse_args(argv))
    except PortIsolationError as exc:
        print(f"ROLLOUTBENCH_PORT_ISOLATION_ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
