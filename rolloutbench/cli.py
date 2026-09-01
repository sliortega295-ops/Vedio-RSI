from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .freeze import freeze_suite
from .scheduler import SYSTEMS, simulate
from .schema import validate_suite_directory
from .validators import build_quality_plan, compare_historical_oracle


DEFAULT_SUITE_DIR = Path("benchmarks/sana_video_2b_h100_v0")


def _write_json(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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

    quality_plan = subparsers.add_parser(
        "quality-plan", help="write the deterministic predeclared quality workload"
    )
    quality_plan.add_argument("--suite", type=Path, default=DEFAULT_SUITE_DIR)
    quality_plan.add_argument("--output", type=Path, required=True)
    quality_plan.add_argument("--repo-root", type=Path, default=Path.cwd())

    acceptance = subparsers.add_parser(
        "acceptance-check", help="compare a CPU acceptance replay with the hidden oracle"
    )
    acceptance.add_argument("--suite", type=Path, default=DEFAULT_SUITE_DIR)
    acceptance.add_argument("--simulation", type=Path, required=True)
    acceptance.add_argument("--output", type=Path)
    acceptance.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "quality-plan":
        validate_suite_directory(args.suite, repo_root=args.repo_root)
        protocol_bytes = (args.suite / "quality_protocol.json").read_bytes()
        protocol = json.loads(protocol_bytes)
        pairs = build_quality_plan(protocol, protocol["formal_cache_candidates"])
        pairs_bytes = json.dumps(
            pairs, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        payload = {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "quality_protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "pairs_sha256": hashlib.sha256(pairs_bytes).hexdigest(),
            "execution_status": "NOT_RUN",
            "performance_claim": False,
            "candidate_ids": protocol["formal_cache_candidates"],
            "candidate_count": len(protocol["formal_cache_candidates"]),
            "matched_pair_count": len(pairs),
            "pairs": pairs,
        }
        _write_json(args.output, payload)
        print(
            json.dumps(
                {"execution_status": "NOT_RUN", "matched_pair_count": len(pairs)},
                sort_keys=True,
            )
        )
        return 0
    if args.command == "acceptance-check":
        validate_suite_directory(args.suite, repo_root=args.repo_root)
        episodes = [
            json.loads(line)
            for line in (args.suite / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        suite = json.loads((args.suite / "suite.json").read_text(encoding="utf-8"))
        simulation = json.loads(args.simulation.read_text(encoding="utf-8"))
        payload = compare_historical_oracle(
            episodes,
            simulation.get("decisions", []),
            simulation.get("frontier"),
            suite["frontier_contracts"]["legacy_oracle"],
        )
        _write_json(args.output, payload)
        print(json.dumps({"status": payload["status"]}, sort_keys=True))
        return 0 if payload["status"] == "PASS" else 1
    if args.command == "simulate":
        validate_suite_directory(args.suite, repo_root=args.repo_root)
        payload = simulate(args.system, args.suite / "episodes.jsonl").as_dict()
        _write_json(args.output, payload)
        print(json.dumps(payload["summary"], sort_keys=True))
        return 0
    if args.command == "freeze":
        suite_dir = freeze_suite(args.output, repo_root=args.repo_root)
        report = validate_suite_directory(suite_dir, repo_root=args.repo_root)
    else:
        report = validate_suite_directory(args.suite_dir, repo_root=args.repo_root)
    print(json.dumps(report.as_dict(), sort_keys=True))
    return 0
