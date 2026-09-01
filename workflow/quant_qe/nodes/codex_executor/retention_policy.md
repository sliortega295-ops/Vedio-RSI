## FP8 Retention Policy

Retain only a real native-FP8 full run that passes the component smoke, has no
silent fallback, preserves the frozen workload/video contract, avoids critical
visual corruption, and improves matching latency by at least 2%. One repeat is
required for a marginal 2-5% improvement; a larger result may be retained after
one full run but still needs a final confirmation before delivery.

Reject a concrete point for numerical/visual failure, no FP8 activation,
fallback-only execution, or measured regression. Mark missing hardware,
ownership, dependencies, or interrupted runs as blocked rather than algorithmic
failure. If all credible guarded FFN points fail, publish a structured negative.
