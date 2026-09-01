# Formal H100 launch authorization

Sol-RolloutBench never converts an idle `nvidia-smi` snapshot into permission to
use a GPU. A formal run needs three separate facts:

1. a preflight receipt less than ten minutes old;
2. a cooperative lease for every planned GPU UUID;
3. an authorization assertion issued by the user, cluster administrator or
   scheduler and bound to the exact plan and run hashes.

The current authorization file is **procedural**. Issuer strings are not backed
by a signature or trusted public key. Validation proves freshness, exact plan
binding and cooperative lease state; it does not cryptographically prove who
wrote the JSON. Do not describe this gate as technical self-authorization
prevention.

## Required shape

```json
{
  "schema_version": 1,
  "record_type": "gpu_launch_authorization",
  "status": "AUTHORIZED",
  "scope": "sol-rolloutbench-v0-formal-pilot",
  "ownership_verified": true,
  "authorization_id": "external-ticket-id",
  "owner": "operator-id",
  "issued_by": "cluster-scheduler",
  "authority_kind": "scheduler",
  "host": "BAAI",
  "plan_id": "<exact plan_id>",
  "plan_sha256": "<sha256 of plan.json>",
  "run_id": "pilot-serial1-repeat-01",
  "run_sha256": "<canonical run sha256>",
  "gpu_uuids": ["GPU-..."],
  "lock_paths": {"GPU-...": "/absolute/path/to/gpu.lock"},
  "lease_files": {"GPU-...": "/absolute/path/to/lease.json"},
  "preflight_observed_at_utc": "<ISO-8601 with timezone>",
  "issued_at_utc": "<ISO-8601 with timezone>",
  "expires_at_utc": "<ISO-8601 with timezone>",
  "preflight_receipt_path": "/absolute/path/to/preflight.json",
  "preflight_receipt_sha256": "<sha256>"
}
```

Use `sol-rolloutbench-v0-formal-full` for a full run. Authorization must satisfy
`observed <= issued <= now < expires`, the preflight must be at most ten minutes
old at each dispatch, and the authorization lifetime must be at most 24 hours.

The dispatcher validates authorization after taking its dispatcher lock and
again before every new GPU unit. The runtime then takes the physical GPU lock,
rejects a released lease, checks for live compute processes and confirms the
planned GPU UUID. Releasing a lease prevents subsequent units from starting.

## Non-authorizing preflight

This command is read-only and never launches a model:

```bash
python3 -m rolloutbench h100-preflight \
  --profile benchmarks/sana_video_2b_h100_v0/h100_profile.json \
  --repo-root . \
  --output /absolute/persistent/path/preflight.json
```

Even a passing receipt records `launch_authorized=false` and
`ownership_verified=false`. An external authority must create the authorization
and leases afterward.
