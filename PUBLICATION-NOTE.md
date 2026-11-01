# Publication note — 2026-10-31

This is a curated results publication, not a new GPU run or a complete runtime-source release.

## Scope and chronology

- Publishes selected October 21–24 experiment/search reports, machine-readable outcomes and per-card summaries, plus a dated ResearchBoost design review.
- The synthesis and next-step document are new. Proposed work is `NOT_RUN`.
- Related [PR #1](https://github.com/sliortega295-ops/Vedio-RSI/pull/1) remains a historical October 1 snapshot. Related [PR #2](https://github.com/sliortega295-ops/Vedio-RSI/pull/2) is a separate FastRoll implementation with CPU-only validation. Neither PR is modified, merged, or retroactively validated by this publication.
- Source reports saying “not pushed” describe their original creation time. This note records their later publication; original outcomes, including failures, remain unchanged.
- A/B/C/D, DAG, SANA search and Wan use different trajectories and/or ports. They are not a single controlled scheduler benchmark or full Sol paper reproduction.

## Evidence custody and privacy

[PUBLICATION-MANIFEST.json](PUBLICATION-MANIFEST.json) records each imported source's project-relative identity, original SHA-256, published SHA-256, sizes and declared field projections. Original evidence stays unchanged. Private home/remote paths, node identifiers and GPU UUIDs are redacted. GPU UUID aliases are stable within this export; GPU indices remain intact.

Absolute Markdown links are rewritten only when their target is published here. Otherwise they become plain retained-artifact labels and are indexed in [UNPUBLISHED-REFERENCES.json](evidence/20261031/UNPUBLISHED-REFERENCES.json). Paths inside original JSON and code spans remain artifact references, not promises of public availability. `project://`, `remote-artifact://`, `scratch-artifact://` and `/workspace/` are provenance identifiers, not downloadable links.

Large analysis objects are projected onto declared aggregate fields; CSV record line endings are normalized to LF. SANA search omits nested full method/experience records; Wan audit omits full intervention payloads while its narrative preserves the interventions and `unattended_closed_loop=false`. The source index publishes IDs, URLs, hashes and code-path metadata, not full third-party source/README captures. It cannot by itself reproduce the original citation-text audit.

Not uploaded: model weights, raw videos or images, full per-second telemetry, raw Agent/API/session logs, credentials, large source archives, third-party code captures, and uncommitted runtime code from the experimental working tree. No model weights were downloaded or transferred in this publication task.

The public package is **sufficient to inspect the stated summaries and evidence boundaries**, not independently rerun all video/visual checks. Hash presence proves identity, not quality or portability. Historical CPU-test results are retained as historical receipts, not represented as tests rerun today.

## Verification in this PR

`tools/export_status.py` is an offline, explicit-allowlist evidence transform. It leaves source files unchanged and checks that JSON projection/redaction preserves numeric leaves. Re-export requires the private source evidence; it does not fetch it.

`tools/verify_status.py` runs offline with the Python standard library. It checks the imported file hashes and allowlist, JSON/CSV readability, published Markdown targets, obvious sensitive-path/credential patterns, and key cross-file result/status/arithmetic invariants. It also confirms that no model/video/image payload is part of this publication. This is publication validation, **not GPU reproduction or a complete security audit**.

```bash
python3 -B tools/verify_status.py
```

No experiment, global Memory update, runtime refactor, or third-party code vendoring is authorized by this results-only commit. Follow-up work needs its own implementation and verification record.
