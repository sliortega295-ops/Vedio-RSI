## FP8 Executor Callable Evidence

- `scripts/sana/fp8_component_smoke.py`: mandatory H100 numerical/backend smoke
  before a full generation. Its timing is diagnostic only.
- `scripts/launch_config.py`: authoritative full-generation launcher using the
  experiment-local GPU lease.
- `scripts/collect_run.py` and `search/plan_eval.py --no-gemini`: durable
  benchmark/video collection and local quality evidence.

Do not spawn a second reviewer or call an external vision service. The executor
uses its own bounded visual inspection; the master independently rechecks the
delivered frames.
