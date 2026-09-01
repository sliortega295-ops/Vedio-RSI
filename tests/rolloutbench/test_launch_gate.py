from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rolloutbench.launch_gate import (
    LaunchAuthorizationError,
    validate_launch_authorization,
)
from rolloutbench.leases import acquire_cooperative_lease
from rolloutbench.pilot_runner import RunContext


GPU_UUID = "GPU-83ed65f8-62e5-2a01-3471-8bfc752971d3"


class LaunchGateTests(unittest.TestCase):
    def _fixture(self, root: Path):
        now = datetime(2026, 9, 1, 16, 0, tzinfo=timezone.utc)
        context = RunContext(
            plan_id="plan-1",
            plan_sha256="a" * 64,
            run_sha256="b" * 64,
            preparation_sha256="c" * 64,
            run={
                "run_id": "pilot-serial1-repeat-01",
                "scope": "pilot",
                "workers": [{"worker_id": 0, "gpu_uuid": GPU_UUID}],
            },
            preparation={},
            plan_path=root / "plan.json",
            preparation_path=root / "preparation.json",
        )
        observed = now - timedelta(minutes=2)
        spec = {
            "profile_sha256": "1" * 64,
            "model_profile_sha256": "2" * 64,
            "suite_sha256": "3" * 64,
            "quality_protocol_sha256": "4" * 64,
            "artifacts_sha256": "5" * 64,
            "target_gpus": [{"uuid": GPU_UUID}],
        }
        preflight = {
            "schema_version": 1,
            "query_status": "PASS",
            "runtime_ready": True,
            "quality_ready": True,
            "two_gpu_idle_point_in_time": True,
            "technical_ready": True,
            "launch_authorized": False,
            "pilot_ready": False,
            **{
                key: value
                for key, value in spec.items()
                if key.endswith("_sha256")
            },
            "gpu_idle_scope": {
                "ownership_verified": False,
                "observed_at_utc": observed.isoformat(),
            },
            "observation": {"gpus": [{"uuid": GPU_UUID}]},
        }
        preflight_path = (root / "preflight.json").resolve()
        preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
        lock_path = (root / "locks" / "gpu.lock").resolve()
        lease_path = (root / "leases" / "gpu.json").resolve()
        acquire_cooperative_lease(
            lease_path,
            authorization="grant-1",
            owner="lyy",
            host="BAAI",
            plan_id=context.plan_id,
            run_id=context.run["run_id"],
            gpu_uuid=GPU_UUID,
            lock_path=str(lock_path),
        )
        authorization = {
            "schema_version": 1,
            "record_type": "gpu_launch_authorization",
            "status": "AUTHORIZED",
            "scope": "sol-rolloutbench-v0-formal-pilot",
            "ownership_verified": True,
            "authorization_id": "grant-1",
            "owner": "lyy",
            "issued_by": "cluster-scheduler",
            "authority_kind": "scheduler",
            "host": "BAAI",
            "plan_id": context.plan_id,
            "plan_sha256": context.plan_sha256,
            "run_id": context.run["run_id"],
            "run_sha256": context.run_sha256,
            "gpu_uuids": [GPU_UUID],
            "lock_paths": {GPU_UUID: str(lock_path)},
            "lease_files": {GPU_UUID: str(lease_path)},
            "preflight_observed_at_utc": observed.isoformat(),
            "issued_at_utc": (now - timedelta(minutes=1)).isoformat(),
            "expires_at_utc": (now + timedelta(hours=1)).isoformat(),
            "preflight_receipt_path": str(preflight_path),
            "preflight_receipt_sha256": hashlib.sha256(
                preflight_path.read_bytes()
            ).hexdigest(),
        }
        authorization_path = (root / "authorization.json").resolve()
        authorization_path.write_text(json.dumps(authorization), encoding="utf-8")
        return context, now, authorization_path, {GPU_UUID: lease_path}, spec

    def test_external_authorization_binds_fresh_preflight_and_active_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            context, now, path, leases, spec = self._fixture(Path(directory))
            receipt = validate_launch_authorization(
                context, path, leases, preflight_spec=spec, now=now
            )
        self.assertEqual("VALIDATED", receipt["status"])
        self.assertFalse(receipt["ownership_verified"])
        self.assertTrue(receipt["ownership_assertion_accepted"])
        self.assertEqual(
            "procedural_assertion_not_cryptographically_verified",
            receipt["authority_verification"],
        )
        self.assertEqual([GPU_UUID], receipt["gpu_uuids"])
        self.assertFalse(receipt["performance_claim"])

    def test_probe_cannot_self_authorize_and_expired_or_tampered_input_fails(self):
        mutations = {
            "ownership": lambda row, _now: row.__setitem__("ownership_verified", False),
            "expired": lambda row, now: row.__setitem__(
                "expires_at_utc", (now - timedelta(seconds=1)).isoformat()
            ),
            "preflight_sha": lambda row, _now: row.__setitem__(
                "preflight_receipt_sha256", "0" * 64
            ),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                context, now, path, leases, spec = self._fixture(Path(directory))
                authorization = json.loads(path.read_text(encoding="utf-8"))
                mutation(authorization, now)
                path.write_text(json.dumps(authorization), encoding="utf-8")
                with self.assertRaises(LaunchAuthorizationError):
                    validate_launch_authorization(
                        context, path, leases, preflight_spec=spec, now=now
                    )

    def test_missing_or_mismatched_active_lease_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            context, now, path, leases, spec = self._fixture(Path(directory))
            with self.assertRaisesRegex(
                LaunchAuthorizationError, "supplied lease files"
            ):
                validate_launch_authorization(
                    context, path, {}, preflight_spec=spec, now=now
                )
            lease_path = Path(leases[GPU_UUID])
            lease = json.loads(lease_path.read_text(encoding="utf-8"))
            lease["owner"] = "other"
            lease_path.write_text(json.dumps(lease), encoding="utf-8")
            with self.assertRaisesRegex(
                LaunchAuthorizationError, "active cooperative lease"
            ):
                validate_launch_authorization(
                    context, path, leases, preflight_spec=spec, now=now
                )

    def test_preflight_must_be_fresh_at_validation_time(self):
        for age in (timedelta(minutes=11), timedelta(hours=23)):
            with self.subTest(age=age), tempfile.TemporaryDirectory() as directory:
                context, now, path, leases, spec = self._fixture(Path(directory))
                authorization = json.loads(path.read_text(encoding="utf-8"))
                old = now - age
                authorization["preflight_observed_at_utc"] = old.isoformat()
                authorization["issued_at_utc"] = (old + timedelta(minutes=1)).isoformat()
                preflight_path = Path(authorization["preflight_receipt_path"])
                preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
                preflight["gpu_idle_scope"]["observed_at_utc"] = old.isoformat()
                preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
                authorization["preflight_receipt_sha256"] = hashlib.sha256(
                    preflight_path.read_bytes()
                ).hexdigest()
                path.write_text(json.dumps(authorization), encoding="utf-8")
                with self.assertRaisesRegex(LaunchAuthorizationError, "fresh preflight"):
                    validate_launch_authorization(
                        context, path, leases, preflight_spec=spec, now=now
                    )

    def test_expiry_must_follow_issue_time(self):
        with tempfile.TemporaryDirectory() as directory:
            context, now, path, leases, spec = self._fixture(Path(directory))
            authorization = json.loads(path.read_text(encoding="utf-8"))
            authorization["expires_at_utc"] = (
                now - timedelta(minutes=90)
            ).isoformat()
            path.write_text(json.dumps(authorization), encoding="utf-8")
            with self.assertRaisesRegex(LaunchAuthorizationError, "expired"):
                validate_launch_authorization(
                    context, path, leases, preflight_spec=spec, now=now
                )


if __name__ == "__main__":
    unittest.main()
