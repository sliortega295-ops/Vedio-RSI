from __future__ import annotations

import argparse
import json
from pathlib import Path

from .freeze import freeze_suite
from .scheduler import SYSTEMS, simulate
from .schema import validate_suite_directory


DEFAULT_SUITE_DIR = Path("benchmarks/sana_video_2b_h100_v0")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rolloutbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="freeze the authoritative 35-episode suite")
    freeze.add_argument("--output", type=Path, default=DEFAULT_SUITE_DIR)
    freeze.add_argument("--repo-root", type=Path, default=Path.cwd())

    validate = subparsers.add_parser("validate-suite", help="validate a frozen suite")
    validate.add_argument("suite_dir", type=Path, nargs="?", default=DEFAULT_SUITE_DIR)
    validate.add_argument("--repo-root", type=Path, default=Path.cwd())

    simulation = subparsers.add_parser(
        "simulate", help="run a deterministic CPU-only contract simulation"
    )
    simulation.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    simulation.add_argument("--suite", type=Path, default=DEFAULT_SUITE_DIR)
    simulation.add_argument("--output", type=Path)
    simulation.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "simulate":
        validate_suite_directory(args.suite, repo_root=args.repo_root)
        payload = simulate(args.system, args.suite / "episodes.jsonl").as_dict()
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(json.dumps(payload["summary"], sort_keys=True))
        return 0
    if args.command == "freeze":
        suite_dir = freeze_suite(args.output, repo_root=args.repo_root)
        report = validate_suite_directory(suite_dir, repo_root=args.repo_root)
    else:
        report = validate_suite_directory(args.suite_dir, repo_root=args.repo_root)
    print(json.dumps(report.as_dict(), sort_keys=True))
    return 0
