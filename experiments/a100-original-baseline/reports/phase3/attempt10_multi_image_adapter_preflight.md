# Cache attempt 10 exact R03+R05 image adapter — prepared, not deployed

At 2026-09-14 08:17 UTC, the host-only site Codex transport was staged for
the already-rendered, hash-locked **R03+R05** 20-image set. This is a
transport adaptation, not an archived Harness quality decision. Master11
was still active when the coordinator source file was edited, but its running
Python process and the old site-goal broker had already loaded the old code;
there has been **no broker reload, manifest approval, Cache10 start, or real
image-bearing API call** from this change. It does not retroactively validate
Cache9's unsupported visual claims.

The prior coordinator adapter was byte-copied to
`artifacts/phase3/pre_image_adapter_r03r05/site_codex_goal_session.py`, SHA
`2bb4801acbb3d56d4569a15fe4f0f5060f8a93323e283639a274c92b2dda5dc8`,
before changing its source. The new
`repo/tools/symposium/site_codex_goal_session.py` SHA is
`ff0f89cd9faf4610165adcd9cded54f3485ca4f5bcb9197658c7a18f4e3345fc`.
Its entire functional delta is:

1. Pin `tools/symposium/site_phase3_multi_image.py` at the exact coordinator
   path and source SHA
   `d6f145a19b45c9f7a7295daa89d49d2e96b5c3bca7141c8a85136f19ecb036bd`.
2. For **only** `sana2ba100-cache_ca-0002`, attempt **10**, verify that
   module's path/SHA before import, then call its existing `inputs` validator.
   A missing/changed source, frame, image, receipt, or approval raises; it
   never falls through to the older R01/R02 single-round transport.
3. Add the fixed coordinator checkout to `sys.path` only for this branch so
   direct CLI invocation resolves that validated helper. No Executor worktree
   path is used. All other session/attempt routes are unchanged.

The already-tested module pins frozen-dense/R03/R05 source-frame digests,
both original LPIPS receipt SHA, renderer SHA, render manifest SHA
`a582270a33b97d2c8f5067ac3427ceee30de9dd9cec8dab7e42478323c0b390d`,
20 PNG SHA/size/path/order and all tile source SHA. The proposed exact
attachment manifest is
`artifacts/phase3/cache_image_transport/attachments/sana2ba100-cache_ca-0002-attempt-10-multi.json`,
SHA `d2e5982153d4cb1884d57985443a1cc8f06a31cde6fe1eee8f3787220866c416`.
It remains `status=prepared`, not approved. The older v1 R03-only attempt10
manifest SHA `b038f40d...` also remains prepared; the new exact route would
supersede it for attempt10 only after deployment, without overwriting it.

`PYTHONDONTWRITEBYTECODE=1 python3 -m unittest
tests.test_site_phase3_multi_image tests.test_site_phase3_multi_adapter -v`
passed **12/12** in 1.063 s. New adapter test source SHA is
`3c289e2406224a241a37c142538520d2072826a9584b58d3c177a68ae01b4905`.
Tests cover prepared-only rejection, exact session/attempt, 20-image R03 then
R05 ordering, bad module source SHA rejected before imported validation,
source-frame/render-image/LPIPS-receipt SHA rejection, and no v1 fallback.
`git diff --check` passed.

The real CLI dry-run with `--allow-prepared` returned one `--image` prefix
plus exactly **20 paths**, five prompts for each of R03 and R05,
**48,381,103 bytes** total, first
`r03-p01-overview.png` SHA `3799adfd...`, last `r05-p05-detail.png` SHA
`03137efb...`. The same CLI **without** `--allow-prepared` exited 1 with
`attachment identity/approval mismatch`, proving a real Agent start still
fails closed. The first integration CLI dry-run had exited nonzero with
`ModuleNotFoundError: tools`; the fixed coordinator-only import path produced
the PASS above. That failure is site-control preflight cost, not a candidate
or quality failure.

Before any activation: preserve Master11's current stream and Cache9's
unreceipted-death evidence; require all goal-broker request queues pending=0
and unresolved `.processing`=0; verify no other active Cache attempt; change
only this *multi* manifest's `status` to `approved` with pre/post SHA receipt;
re-run real CLI dry-run (not the prepared override); reload only the scoped
site-goal broker after checking its PID, children, and heartbeat; verify new
source SHA/heartbeat, then let the original Master retry native
`resume_executor --mode continue`. Cache10's actual `agent_api_begin`
must show all 20 approved image paths and immutable hashes. If Master11
must be stopped/restarted to adopt the route, that is a separately disclosed
site recovery, not native completion. No GPU grant or method gate is changed.
