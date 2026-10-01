# Publication note — 2026-10-01

This snapshot publishes evidence generated through 2026-09-14. The publication
does not execute a new GPU experiment or complete the outstanding A100
Kernel+Cache integration.

Historical reports contain statements such as “no push/PR,” “Cache blocked,”
or “first Kernel verifier failed.” These refer to the time each report was
written. They remain useful evidence. The Phase2 terminal report documents
later component reassessment; it does not rewrite the Phase1 result.

`PUBLICATION-MANIFEST.json` covers the 239 imported files. It records their
original source SHA256, published SHA256, size and category. Numerical JSON,
source snapshots and patches are copied byte-for-byte. Imported Markdown
links point to published relative files where available; 13 omitted links
are explicitly labeled “retained locally.” Those link transformations are
recorded in the manifest. Authored README, research-status and verification
files, and the accompanying upstream license text, are additional publication
material.

Original patch context lines and the historical acceptance checklist's final
blank line retain their whitespace for source-hash fidelity. Whitespace checks
on newly authored publication files pass; imported evidence is not reformatted.

Videos, sampled images, model/evaluation weights, environments, raw Agent
sessions, live broker queues and large raw telemetry are excluded. The raw
inputs remain in the experiment project; derived hashes do not make those
unpublished inputs independently available to an external reader. The
publication verifier checks the included bytes and arithmetic, not unseen
videos, original raw telemetry or paper-level quality.

The source overlay records the deployed site adaptation and test code.
Some files use historical local/remote paths and narrowly approved source
hashes. It is an audit snapshot, not an automatically runnable installation
for a different machine. The archived Harness base is
`d2c6407cc9b9133f3fff49fe4b561f14980d3f8b`; SGLang/model provenance and departures
from upstream are disclosed in the experiment reports. Full runtime trees
and current site authorization files are not included.

No proxy credential, model transfer, GPU submit or remote configuration change
is part of this publication. The public research claim remains preliminary:
the traces support rollout cost diagnosis and the need for evidence-bound
validation. They do not demonstrate a successful new rollout system, memory
transfer or retrieval benefit.
