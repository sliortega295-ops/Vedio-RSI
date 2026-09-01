from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .acceptance import run_cpu_acceptance, verify_cpu_acceptance_pack
from .freeze import freeze_suite
from .h100_preflight import DEFAULT_PROFILE, build_preflight_spec, run_h100_preflight
from .preparation import prepare_experiment
from .runplan import build_experiment_plan, write_experiment_plan
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

    cpu_acceptance = subparsers.add_parser(
        "cpu-acceptance", help="write the four-system CPU contract evidence pack"
    )
    cpu_acceptance.add_argument("--suite", type=Path, default=DEFAULT_SUITE_DIR)
    cpu_acceptance.add_argument("--output-dir", type=Path, required=True)
    cpu_acceptance.add_argument("--repo-root", type=Path, default=Path.cwd())
    cpu_acceptance.add_argument(
        "--allow-dirty",
        action="store_true",
        help="development only: emit a PASS_DIRTY_NONREPRODUCIBLE pack",
    )

    verify_pack = subparsers.add_parser(
        "verify-evidence-pack", help="verify a CPU acceptance evidence pack"
    )
    verify_pack.add_argument("--pack-dir", type=Path, required=True)
    verify_pack.add_argument("--suite", type=Path, default=DEFAULT_SUITE_DIR)
    verify_pack.add_argument("--repo-root", type=Path, default=Path.cwd())
    verify_pack.add_argument("--allow-dirty", action="store_true")

    h100_preflight = subparsers.add_parser(
        "h100-preflight", help="run one read-only BAAI H100 readiness query"
    )
    h100_preflight.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    h100_preflight.add_argument("--output", type=Path, required=True)
    h100_preflight.add_argument("--repo-root", type=Path, default=Path.cwd())

    experiment_plan = subparsers.add_parser(
        "experiment-plan", help="write a frozen pilot or full H100 experiment plan"
    )
    experiment_plan.add_argument("--suite", type=Path, default=DEFAULT_SUITE_DIR)
    experiment_plan.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    experiment_plan.add_argument("--repo-root", type=Path, default=Path.cwd())
    experiment_plan.add_argument("--output", type=Path, required=True)
    experiment_plan.add_argument("--phase", choices=("pilot", "full"), required=True)
    experiment_plan.add_argument(
        "--repetitions",
        type=int,
        choices=(5,),
        required=True,
        help="formal plans predeclare five runs; summarize the first three before 4-5",
    )

    preparation = subparsers.add_parser(
        "prepare-experiment",
        help="prepare exact candidate worktrees and authority artifacts without GPU work",
    )
    preparation.add_argument("--plan", type=Path, required=True)
    preparation.add_argument("--suite", type=Path, default=DEFAULT_SUITE_DIR)
    preparation.add_argument("--experiment-root", type=Path, required=True)
    preparation.add_argument("--repo-root", type=Path, default=Path.cwd())
    preparation.add_argument(
        "--allow-dirty",
        action="store_true",
        help="development only: allow a dirty harness checkout",
    )

    formal_run = subparsers.add_parser(
        "run-formal",
        help="execute one externally authorized formal run with durable resume",
    )
    formal_run.add_argument("--plan", type=Path, required=True)
    formal_run.add_argument("--preparation", type=Path, required=True)
    formal_run.add_argument("--run-id", required=True)
    formal_run.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    formal_run.add_argument("--suite", type=Path, default=DEFAULT_SUITE_DIR)
    formal_run.add_argument("--repo-root", type=Path, default=Path.cwd())
    formal_run.add_argument("--state-root", type=Path, required=True)
    formal_run.add_argument("--authorization", type=Path, required=True)
    formal_run.add_argument(
        "--execute-authorized-gpu-plan",
        action="store_true",
        help="required acknowledgement; authorization validation still cannot be bypassed",
    )

    summarize = subparsers.add_parser(
        "summarize-system",
        help="aggregate completed repetitions for one system without GPU execution",
    )
    summarize.add_argument("--plan", type=Path, required=True)
    summarize.add_argument("--preparation", type=Path, required=True)
    summarize.add_argument("--state-root", type=Path, required=True)
    summarize.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    summarize.add_argument(
        "--completed-repetitions",
        type=int,
        choices=(3, 5),
        help=(
            "aggregate the first three or all five repetitions from one "
            "predeclared plan; defaults to the plan repetition count"
        ),
    )
    summarize.add_argument("--suite", type=Path, default=DEFAULT_SUITE_DIR)
    summarize.add_argument("--repo-root", type=Path, default=Path.cwd())
    summarize.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser(
        "compare-systems",
        help="validate and seal the exact four-system TTVF comparison",
    )
    compare.add_argument(
        "--result",
        type=Path,
        action="append",
        required=True,
        help="one sealed summarize-system result; provide exactly four",
    )
    compare.add_argument("--suite", type=Path, default=DEFAULT_SUITE_DIR)
    compare.add_argument("--repo-root", type=Path, default=Path.cwd())
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "compare-systems":
        from .aggregation import compare_system_results, write_system_comparison

        comparison = compare_system_results(
            args.result,
            args.suite,
            repo_root=args.repo_root,
        )
        receipt = write_system_comparison(args.output, comparison)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "summarize-system":
        from .aggregation import aggregate_system, write_system_result
        from .pilot_runner import load_run_context

        try:
            plan = json.loads(args.plan.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("cannot load experiment plan for aggregation") from exc
        if not isinstance(plan, dict) or not isinstance(plan.get("runs"), list):
            raise RuntimeError("experiment plan is malformed")
        plan_repetitions = plan.get("repetitions")
        planned_runs = sorted(
            (
                run
                for run in plan["runs"]
                if isinstance(run, dict) and run.get("system") == args.system
            ),
            key=lambda run: run.get("repeat_index", -1),
        )
        if (
            type(plan_repetitions) is not int
            or plan_repetitions not in {3, 5}
            or len(planned_runs) != plan_repetitions
            or [run.get("repeat_index") for run in planned_runs]
            != list(range(1, plan_repetitions + 1))
        ):
            raise RuntimeError("system aggregation run set is incomplete")
        completed_repetitions = (
            args.completed_repetitions
            if args.completed_repetitions is not None
            else plan_repetitions
        )
        if completed_repetitions > plan_repetitions:
            raise RuntimeError(
                "completed repetitions exceed the predeclared experiment plan"
            )
        run_ids = [
            str(run["run_id"])
            for run in planned_runs[:completed_repetitions]
        ]
        suite_path = args.suite
        if not suite_path.is_absolute():
            suite_path = args.repo_root / suite_path
        validate_suite_directory(suite_path, repo_root=args.repo_root)
        suite = json.loads(
            (suite_path / "suite.json").read_text(encoding="utf-8")
        )
        protocol = json.loads(
            (suite_path / "quality_protocol.json").read_text(encoding="utf-8")
        )
        contexts = [
            load_run_context(args.plan, args.preparation, run_id)
            for run_id in run_ids
        ]
        result = aggregate_system(
            contexts, args.state_root, suite, protocol
        )
        receipt = write_system_result(args.output, result)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "run-formal":
        if not args.execute_authorized_gpu_plan:
            raise RuntimeError(
                "run-formal requires --execute-authorized-gpu-plan"
            )
        from .formal_dispatch import dispatch_formal_run
        from .pilot_runner import SubprocessStageExecutor, load_run_context

        context = load_run_context(args.plan, args.preparation, args.run_id)
        profile_path = args.profile
        if not profile_path.is_absolute():
            profile_path = args.repo_root / profile_path
        suite_path = args.suite
        if not suite_path.is_absolute():
            suite_path = args.repo_root / suite_path
        try:
            profile = build_preflight_spec(
                profile_path, repo_root=args.repo_root
            )
            protocol = json.loads(
                (suite_path / "quality_protocol.json").read_text(encoding="utf-8")
            )
            authorization = json.loads(
                args.authorization.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("cannot load formal run profile/protocol/authorization") from exc
        if (suite_path / "suite.json").resolve() != Path(profile["suite_path"]).resolve():
            raise RuntimeError("formal run suite does not match the frozen H100 profile")
        lease_files = authorization.get("lease_files")
        if not isinstance(lease_files, dict):
            raise RuntimeError("launch authorization has no lease_files mapping")
        outcomes = dispatch_formal_run(
            context,
            SubprocessStageExecutor(),
            args.state_root,
            authorization_path=args.authorization.resolve(),
            lease_files=lease_files,
            profile=profile,
            quality_protocol=protocol,
        )
        print(
            json.dumps(
                {
                    "status": "COMPLETED",
                    "run_id": args.run_id,
                    "unit_outcome_count": len(outcomes),
                    "performance_claim": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "prepare-experiment":
        payload = prepare_experiment(
            args.plan,
            args.suite,
            args.experiment_root,
            repo_root=args.repo_root,
            require_clean=not args.allow_dirty,
        )
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "plan_id": payload["plan_id"],
                    "run_count": payload["run_count"],
                    "unique_episode_count": payload["unique_episode_count"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "experiment-plan":
        spec = build_preflight_spec(args.profile, repo_root=args.repo_root)
        suite_path = args.suite.resolve()
        if (suite_path / "suite.json").resolve() != Path(spec["suite_path"]).resolve():
            raise ValueError("experiment plan suite must match the frozen H100 profile")
        plan = build_experiment_plan(
            suite_path,
            scope=args.phase,
            repetitions=args.repetitions,
            gpu_uuids=tuple(item["uuid"] for item in spec["target_gpus"]),
            repo_root=args.repo_root,
        )
        write_receipt = write_experiment_plan(
            args.output, plan, repo_root=args.repo_root
        )
        print(
            json.dumps(
                {
                    **write_receipt,
                    "plan_id": plan["plan_id"],
                    "phase": plan["scope"],
                    "repetitions": plan["repetitions"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "h100-preflight":
        payload = run_h100_preflight(
            args.profile, repo_root=args.repo_root, output_path=args.output
        )
        print(
            json.dumps(
                {
                    "query_status": payload["query_status"],
                    "technical_ready": payload["technical_ready"],
                    "launch_authorized": payload["launch_authorized"],
                    "pilot_ready": payload["pilot_ready"],
                },
                sort_keys=True,
            )
        )
        return 0 if payload["query_status"] == "PASS" and payload["technical_ready"] else 1
    if args.command == "cpu-acceptance":
        payload = run_cpu_acceptance(
            args.suite,
            args.output_dir,
            repo_root=args.repo_root,
            require_clean=not args.allow_dirty,
        )
        print(json.dumps({"status": payload["status"]}, sort_keys=True))
        return 0 if payload["cpu_contract_status"] == "PASS" else 1
    if args.command == "verify-evidence-pack":
        payload = verify_cpu_acceptance_pack(
            args.pack_dir,
            args.suite,
            repo_root=args.repo_root,
            allow_dirty=args.allow_dirty,
        )
        print(json.dumps(payload, sort_keys=True))
        return 0
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
