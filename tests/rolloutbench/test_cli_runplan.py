from __future__ import annotations

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rolloutbench.cli import main


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_DIR = REPO_ROOT / "benchmarks" / "sana_video_2b_h100_v0"
PROFILE = SUITE_DIR / "h100_profile.json"


def _clean_source_receipt() -> dict[str, object]:
    return {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "tree_clean": True,
        "dirty_path_count": 0,
    }


class ExperimentPlanCliTests(unittest.TestCase):
    def test_writes_pilot_plan_idempotently_and_refuses_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "rolloutbench.runplan._source_receipt", return_value=_clean_source_receipt()
        ):
            output = Path(directory) / "plans" / "pilot.json"
            args = [
                "experiment-plan",
                "--suite",
                str(SUITE_DIR),
                "--profile",
                str(PROFILE),
                "--repo-root",
                str(REPO_ROOT),
                "--output",
                str(output),
                "--phase",
                "pilot",
                "--repetitions",
                "5",
            ]
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(0, main(args))
            receipt = json.loads(stdout.getvalue())
            self.assertEqual("WRITTEN", receipt["status"])
            self.assertEqual("pilot", receipt["phase"])
            self.assertEqual(5, receipt["repetitions"])

            plan = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("pilot", plan["scope"])
            self.assertEqual(20, len(plan["runs"]))
            self.assertEqual("NOT_RUN", plan["execution_status"])

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(0, main(args))
            output.write_text("conflict\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                main(args)

    def test_summarize_can_use_first_three_runs_from_predeclared_five(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "repetitions": 5,
                        "runs": [
                            {
                                "run_id": f"full-serial1-repeat-{index:02d}",
                                "system": "serial1",
                                "repeat_index": index,
                            }
                            for index in range(1, 6)
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            loaded_ids: list[str] = []

            def load_context(_plan: Path, _preparation: Path, run_id: str) -> str:
                loaded_ids.append(run_id)
                return run_id

            with (
                mock.patch(
                    "rolloutbench.pilot_runner.load_run_context",
                    side_effect=load_context,
                ),
                mock.patch(
                    "rolloutbench.aggregation.aggregate_system",
                    return_value={"status": "FULL_AGGREGATED"},
                ) as aggregate,
                mock.patch(
                    "rolloutbench.aggregation.write_system_result",
                    return_value={"status": "WRITTEN"},
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    0,
                    main(
                        [
                            "summarize-system",
                            "--plan",
                            str(plan_path),
                            "--preparation",
                            str(root / "preparation.json"),
                            "--state-root",
                            str(root / "state"),
                            "--system",
                            "serial1",
                            "--completed-repetitions",
                            "3",
                            "--suite",
                            str(SUITE_DIR),
                            "--repo-root",
                            str(REPO_ROOT),
                            "--output",
                            str(root / "result.json"),
                        ]
                    ),
                )
            self.assertEqual(
                [f"full-serial1-repeat-{index:02d}" for index in range(1, 4)],
                loaded_ids,
            )
            self.assertEqual(3, len(aggregate.call_args.args[0]))


if __name__ == "__main__":
    unittest.main()
