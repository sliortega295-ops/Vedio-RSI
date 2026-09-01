from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from rolloutbench.cli import main


class PreparationCliTests(unittest.TestCase):
    def test_prepare_experiment_routes_paths_and_prints_machine_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "plan.json"
            suite = root / "suite"
            experiment = root / "experiment"
            repository = root / "repo"
            expected = {
                "status": "READY",
                "plan_id": "abc",
                "run_count": 20,
                "unique_episode_count": 10,
                "gpu_execution": False,
                "vbench_execution": False,
            }
            with mock.patch(
                "rolloutbench.cli.prepare_experiment", return_value=expected
            ) as prepare:
                output = StringIO()
                with redirect_stdout(output):
                    result = main(
                        [
                            "prepare-experiment",
                            "--plan",
                            str(plan),
                            "--suite",
                            str(suite),
                            "--experiment-root",
                            str(experiment),
                            "--repo-root",
                            str(repository),
                            "--allow-dirty",
                        ]
                    )
            self.assertEqual(0, result)
            prepare.assert_called_once_with(
                plan,
                suite,
                experiment,
                repo_root=repository,
                require_clean=False,
            )
            self.assertEqual(
                {
                    "plan_id": "abc",
                    "run_count": 20,
                    "status": "READY",
                    "unique_episode_count": 10,
                },
                json.loads(output.getvalue()),
            )


if __name__ == "__main__":
    unittest.main()
