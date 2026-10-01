"""Build a real local SANA generation plan without loading a GPU/model.

Usage: python examples/make_sana_plan.py --runtime /path/to/sol_runtime
       --model /path/to/local_snapshot --output /path/to/plan.json
Pins all source/assets. Model hashing reads local weights; no downloads occur.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastroll.cli import expand_candidate_matrix
from fastroll.contracts import atomic_json, pin_assets, pin_source

p = argparse.ArgumentParser()
p.add_argument("--runtime", type=Path, required=True)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--batch", default="sana-dense")
a = p.parse_args()
session = {
    "backend": "sana", "python": sys.executable,
    "source": pin_source(a.runtime), "assets": [pin_assets(a.model)],
    "config": {"model_path": str(a.model.resolve()), "compile": False, "warmup": True,
               "env": {"SGLANG_SANA_MINIMAL_IMPORT": "1"}},
}
samples = [{"prompt": prompt, "seed": seed, "height": 480, "width": 832,
            "num_frames": 81, "num_inference_steps": 50, "guidance_scale": 6.0, "fps": 16}
           for prompt in ("A dog runs across a grassy field.", "A camera slowly moves around a red toy car.")
           for seed in (42, 12345)]
plan = expand_candidate_matrix({"batch_id": a.batch, "samples": samples,
                               "candidates": [{"id": "dense", "session": session, "mode": "oneshot"}]})
atomic_json(a.output, plan)
print(f"Prepared {len(plan['tasks'])} tasks; no GPU execution. Real SANA adapter: GPU acceptance NOT_RUN.")
