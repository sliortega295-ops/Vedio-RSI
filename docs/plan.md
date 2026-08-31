# Executable Kernel Search Plan

1. Verify the immutable baseline lock, fixed model/workload, runtime source
   hashes, active H100 lease, and clean frozen-start commit. Write the durable
   stage/source preflight. **Complete before implementation.**
2. Round 1: enable only default `torch.compile`; commit, launch through the local
   wrapper/flock, collect, assess, inspect authenticity, record, and gate.
3. Continue one hypothesis per round on the accumulated retained stack, starting
   with BF16 linear-attention aggregation and merged QKV, then exact invariant
   conditioning/RoPE preparation and bounded max-autotune if evidence supports
   them. Revert rejected implementation changes in a new traceable commit before
   the next candidate; never erase their commits or run artifacts.
4. Keep `TRAJECTORY.jsonl`, `KERNEL-PREFLIGHT.json`, and the canonical manifest
   current. After material stack changes, use the real full-generation run as
   the integrated gate. Preserve every failure/regression.
5. Stop only at target, hard cap 40, or a genuine plateau after roughly 3-4
   distinct no-new-best hypotheses. Write one honest `exact_fastest` delivery
   with real artifacts and the target-gap/remaining-ceiling explanation.
