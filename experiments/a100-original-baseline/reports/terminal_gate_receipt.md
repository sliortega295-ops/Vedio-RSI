# Terminal archived gate and hard-cap receipt

**Final status: PARTIAL.** The frozen dense run and real Master→independent Executor→poll→DELIVERY→first independent `verify_delivery.py` chain were executed. The first verifier **failed**, Cache remained quality-blocked, and no integrated delivery exists. No post-cap artifact is counted as a cap-contained verification PASS.

| UTC, 2026-09-13 | Direct receipt / boundary |
|---|---|
| 09:09:39.836269 | First GPU job; fixed 6-hour cap 15:09:39.836267 and host no-new-submit cutoff 14:48:39.836267. |
| 14:58:16.801720 | Last GPU worker (Kernel R10) terminal exit 0, not a hard-cap kill. R10's five requests sum 362.03 s and Executor rejected it as slower than retained R9's 358.39 s. |
| 15:05:04.640641 | Kernel Executor wrote `DELIVERY.json` for retained R9, SHA256 `8d6d91d3f9801174773a91ea99eccf960737416ea2d0dcca48a0b5da8ff89249`; no R10 frontier replacement. |
| 15:06:36.952 | Native Master poll recognized Kernel `delivered=true`, `delivery_status=complete`, and Executor no longer alive. |
| 15:06:50.427324 | Native Master `verify_delivery.py` exit **1**, `objective_ok=false`, issue `kernel_compile_bf16_max_autotune_ffn_temporal_linear:correctness_evidence_missing`. It independently recomputed `578.02 / 358.39 = 1.6128240185` and same timing scope, but no verifier-compatible structural evidence existed at that point. This is **not** a quality/method PASS. |
| 15:07:11.042 | Native Master `resume_executor.py` started independent Kernel attempt 8 to add compatible evidence; correction forbade another GPU job or benchmark rewrite. |
| 15:09:39.836267 | Exact hard cap. No GPU app or worker remained. The first verifier failure was still the only verifier result. |
| 15:09:54.643585 | Passive observer stopped after a safe-stop tail of about 15 s; post-cap samples are excluded from primary six-hour capacity fractions. |
| 15:10:31.100197 | Kernel attempt 8 wrote R9 run-local `equivalence.json`, SHA256 `277e08abd625fbbf8922d7348b2d8003612f92326da6c10fbd29bf19a2d19cfc`, **51.264 s after cap**. Its counts are source-derived structural expectations, not a runtime DiT counter. The attempt briefly cited the wrong R10 path before correcting to R9. This post-cap artifact was never passed through a second archived Master verifier; no retroactive promotion. |
| ~15:10:33 | Scoped host-broker stop of local Master/Kernel groups was acknowledged. Immediate `alive_after` can include zombie process entries; independent process inspection found only `Zs` worker entries with no active exec children. Remote compute-app query returned empty on all eight authorized UUIDs. |

The exact first verifier command, exit code, and unabridged JSON output are preserved in `repo/output/orchestrated/sana2ba100-20260913-093521/master/agent_attempt_05.timed.jsonl` (completed item 163):

```text
python3 orchestration/bin/verify_delivery.py --worktree '/home/lyy/Experiments/SolAgent/20260913-a100-original-baseline/repo/output/experiments/sana2ba100-kernel_aw-0002/worktree' --model sana_video_2b_a100 --tech kernel --baseline '/home/lyy/Experiments/SolAgent/20260913-a100-original-baseline/repo/output/orchestrated/sana2ba100-20260913-093521/BASELINE.json'
```

The authoritative `DELIVERY.json` remains unchanged from 15:05. There is no `INTEGRATED-DELIVERY.json` in `repo/output/`; Cache has no `DELIVERY.json` because aligned LPIPS/AlexNet and visual quality gating were not completed. The archived orchestration-lite Master/Executor path did **not** run the graph-only independent blind visual reviewer. Outer source/frame spot checks are explicitly separate and cannot repair this gate. No Agent/experiment action was resumed after scoped stop. The roughly 53-second post-cap local CPU-control tail is an observed cutoff-handling deviation, not an extension of the 6-hour GPU measurement or a valid new frontier result.

Terminal raw observer archive: `telemetry/terminal_1510/` checked transfer SHA256 `50d3d3c3cbfa0ded3d0fd63e718aecb01dc96aaa4e4bac736c374f5ebb9b3c78`. Primary exact-cap eight-card result: `reports/full_gpu_window_six_hour_cap.json` SHA256 `4d981007ae12dc2abab40a1e4f7a7de78e693f64a1b307096ac2278c258d9b4a`. Per-prompt stage/CPU lower bounds: `reports/timing_terminal.json` SHA256 `189bdd6fe6be65137731bae05401bc355db711d85ce3a303434a230d1923943f`. Local Codex timed-stream analysis: `reports/timing_agent_control_terminal.json` SHA256 `24bbc2b4757ef21f8db90b671fd6e85e4f4696ecfe7cf6ca1d3a0e38d18edb98`. Monotonic differences are within each host only; UTC joins have unknown clock-synchronization error.
