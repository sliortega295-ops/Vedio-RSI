from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from rolloutbench.leases import (
    LeaseContractError,
    acquire_cooperative_lease,
    release_cooperative_lease,
    validate_active_lease,
)


GPU_UUID = "GPU-83ed65f8-62e5-2a01-3471-8bfc752971d3"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _request(**overrides: str) -> dict[str, str]:
    value = {
        "authorization": "approved-ticket-123",
        "owner": "researcher",
        "host": "h100-node",
        "plan_id": "plan-abc",
        "run_id": "pilot-optroll2-repeat-01",
        "gpu_uuid": GPU_UUID,
        "lock_path": "/persistent/rolloutbench/locks/GPU-83ed65f8.lock",
    }
    value.update(overrides)
    return value


class CooperativeLeaseTests(unittest.TestCase):
    def test_acquire_and_read_only_validate_an_explicit_authorized_lease(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "leases" / "gpu-6.json"
            receipt = acquire_cooperative_lease(path, **_request())
            self.assertEqual("active", receipt["status"])
            self.assertFalse(receipt["ownership_claim"])
            self.assertNotIn("idle", json.dumps(receipt).lower())
            self.assertEqual(str(path.resolve()), receipt["lease_file"])
            self.assertTrue(Path(receipt["lock_path"]).is_absolute())
            self.assertEqual(receipt, validate_active_lease(path, **_request()))

            guard_path = (
                REPO_ROOT
                / "models/sana_video_2b_h100/baseline/gpu_guard.py"
            )
            module_spec = importlib.util.spec_from_file_location(
                "rolloutbench_test_gpu_guard", guard_path
            )
            assert module_spec is not None and module_spec.loader is not None
            guard = importlib.util.module_from_spec(module_spec)
            sys.modules[module_spec.name] = guard
            try:
                module_spec.loader.exec_module(guard)
            finally:
                sys.modules.pop(module_spec.name, None)
            loaded = guard.load_lease(path)
            self.assertEqual(GPU_UUID, loaded.gpu_uuid)
            self.assertEqual(Path(_request()["lock_path"]), loaded.lock_path)

    def test_same_acquisition_is_idempotent_but_conflicting_lease_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu-6.json"
            first = acquire_cooperative_lease(path, **_request())
            self.assertEqual(first, acquire_cooperative_lease(path, **_request()))
            with self.assertRaisesRegex(LeaseContractError, "refusing overwrite"):
                acquire_cooperative_lease(path, **_request(owner="another-owner"))

    def test_rejects_invalid_uuid_relative_lock_and_plan_run_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu-6.json"
            with self.assertRaisesRegex(LeaseContractError, "GPU UUID"):
                acquire_cooperative_lease(path, **_request(gpu_uuid="GPU-not-a-uuid"))
            with self.assertRaisesRegex(LeaseContractError, "lock_path"):
                acquire_cooperative_lease(path, **_request(lock_path="locks/lease.lock"))
            acquire_cooperative_lease(path, **_request())
            with self.assertRaisesRegex(LeaseContractError, "plan_id"):
                validate_active_lease(path, **_request(plan_id="other-plan"))
            with self.assertRaisesRegex(LeaseContractError, "run_id"):
                validate_active_lease(path, **_request(run_id="other-run"))

    def test_release_preserves_active_record_and_writes_idempotent_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu-6.json"
            active = acquire_cooperative_lease(path, **_request())
            release = release_cooperative_lease(path, **_request())
            self.assertEqual("released", release["status"])
            self.assertTrue(path.is_file())
            self.assertEqual(active, json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual(release, release_cooperative_lease(path, **_request()))
            with self.assertRaisesRegex(LeaseContractError, "released"):
                validate_active_lease(path, **_request())

    def test_validate_cannot_return_active_across_a_completed_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gpu-6.json"
            acquire_cooperative_lease(path, **_request())
            validation_loaded = threading.Event()
            allow_validation = threading.Event()
            release_finished = threading.Event()
            original_load = __import__("rolloutbench.leases", fromlist=["_load_record"])._load_record

            def release_after_validation_load(record_path: Path, label: str):
                result = original_load(record_path, label)
                if label == "lease record" and threading.current_thread().name == "validator":
                    validation_loaded.set()
                    allow_validation.wait(timeout=1)
                return result

            with patch("rolloutbench.leases._load_record", side_effect=release_after_validation_load):
                validator = threading.Thread(
                    name="validator", target=lambda: validate_active_lease(path, **_request())
                )
                validator.start()
                self.assertTrue(validation_loaded.wait(timeout=1))
                releaser = threading.Thread(
                    target=lambda: (
                        release_cooperative_lease(path, **_request()), release_finished.set()
                    )
                )
                releaser.start()
                self.assertFalse(release_finished.wait(timeout=0.1))
                allow_validation.set()
                validator.join(timeout=1)
                releaser.join(timeout=1)
            self.assertFalse(validator.is_alive())
            self.assertFalse(releaser.is_alive())
            with self.assertRaisesRegex(LeaseContractError, "released"):
                validate_active_lease(path, **_request())


if __name__ == "__main__":
    unittest.main()
