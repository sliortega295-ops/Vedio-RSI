# SANA-Video FP8 Executor Plan

1. Restore the lightweight Sol-Agent orchestration files into the accepted
   integrated baseline branch and preserve its Kernel R20 + Cache R12 artifacts.
2. Implement the registered `fp8_ffn` load-time transform and the SANA-specific
   H100 W8A8 E4M3 runtime wrapper. Keep the default OFF path unchanged and make
   capability/fallback behavior observable.
3. Register a quality-gated `fp8` executor with a six-round scope, delivery
   contract, and SANA default scheduling order `kernel, cache, fp8`.
4. Add an isolated FP8 config, overlay-copy contract entries, unit tests, and
   candidate/benchmark ledgers. Run all CPU/static tests and dry-run the
   experiment materializer.
5. Create a persistent remote 20260901 project without copying weights. Recheck
   GPU ownership, acquire an experiment-local lock, and run the component smoke.
6. If the smoke passes, run matching integrated OFF and FP8 ON generations,
   collect receipts, validate the video, and either promote the measured winner
   or record a structured negative.
7. Commit the self-contained branch and publish it only after code, provenance,
   and artifact checks pass.
