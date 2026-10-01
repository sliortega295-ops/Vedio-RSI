# Source adaptation provenance

`harness_site_changes.patch` is the exact `git diff --binary HEAD` from the
experiment Harness based on `d2c6407cc9b9133f3fff49fe4b561f14980d3f8b`.
It changes seven tracked files. `source_overlay/` contains the revised tracked
files and added A100 model/config, transport, observer, quality, image-input
and test code. All imported source bytes are recorded in the root publication
manifest. The archived `verify_delivery.py` rules were not replaced by this
patch.

## Upstream attribution and modification notice

The archived Sol-Video Inference Engine README declares its code Apache-2.0.
The [upstream license text](UPSTREAM-APACHE-2.0.txt) accompanies the source
snapshot. Upstream authorship is retained; this publication does not relicense
any model weights or excluded third-party runtime.

The following seven upstream-derived files in `source_overlay/` were modified
for this site's remote execution, Codex transport and collection route. They
are not pristine upstream files; the exact modifications are recorded in
`harness_site_changes.patch`, with the historical byte hashes in the manifest:

- `orchestration/bin/poll_executor.py`
- `orchestration/bin/resume_executor.py`
- `orchestration/bin/spawn_executor.py`
- `orchestration/run_orchestrated_experiment.py`
- `scripts/collect_run.py`
- `scripts/create_model_experiment.py`
- `scripts/launch_config.py`

## Audit scope

This is a historical audit snapshot. Runtime paths, UUIDs, narrow source/hash
approvals and authorization state depend on the original site. It needs a
reviewed configuration and cached model/runtime assets to run elsewhere.
The original site budget has expired. Do not treat this publication as a new
GPU authorization or invoke a broker/job launch from it.

The full SGLang runtime, isolated candidate runtime worktrees, environments,
model/evaluation weights and live broker state are excluded. Candidate source
patches available in the original small artifacts are included under
`../artifacts/candidate_source_diffs/`. Exact per-round source receipts and
limits appear in [the source-provenance report](../reports/source_provenance_candidates.md).
This snapshot alone cannot recreate every candidate from unpublished runtime
inputs.

The analysis scripts and four synthetic accounting tests are outside this
overlay, under `../scripts/` and `../tests/`. Running those tests uses no GPU
or network. Site-route tests in the overlay rely on historical image files
and approved artifacts; the reported 34/34 result belongs to the original
workspace test run and is not a clean-machine claim.
