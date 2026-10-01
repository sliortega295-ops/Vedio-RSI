#!/usr/bin/env python3
"""Verify published bytes, links and arithmetic; no network or GPU operations."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
A100 = ROOT / "experiments/a100-original-baseline"
RUNS = A100 / "repo/output/experiments"
SECRET = re.compile(r"gh[pousr]_[A-Za-z0-9]{15,}|sk-[A-Za-z0-9_-]{20,}|https?://[^\s@]+:[^\s@]+@|BEGIN [A-Z ]*PRIVATE KEY")
LINK = re.compile(r"\[([^\]\n]+)\]\(([^)\n]+)\)")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def close(actual: float, expected: float, *, abs_tol: float = 1e-6) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=abs_tol):
        raise ValueError(f"numeric mismatch: {actual} != {expected}")


def main() -> None:
    manifest = read(ROOT / "PUBLICATION-MANIFEST.json")
    if manifest["original_baseline_status"] != "PARTIAL" or manifest["integrated_run"] != "NOT_RUN":
        raise ValueError("unsupported integrated completion claim")
    for record in manifest["files"]:
        path = ROOT / record["path"]
        path.resolve().relative_to(ROOT.resolve())
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"] or len(data) != record["bytes"]:
            raise ValueError(f"publication bytes changed: {record['path']}")
        if not record["markdown_link_rewrites"] and record["sha256"] != record["source_sha256"]:
            raise ValueError(f"unreported source transformation: {record['path']}")
    files = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts]
    for path in files:
        if path.suffix in {".mp4", ".pth", ".safetensors", ".pt", ".bin"}:
            raise ValueError(f"excluded binary artifact: {path.relative_to(ROOT)}")
        text = path.read_text()
        if SECRET.search(text):
            raise ValueError(f"credential pattern: {path.relative_to(ROOT)}")
        if path.suffix == ".json":
            json.loads(text)
        if path.suffix == ".md":
            for label, target in LINK.findall(text):
                target = target.strip().strip("<>")
                if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                    continue
                linked = (path.parent / target.split("#", 1)[0]).resolve()
                linked.relative_to(ROOT.resolve())
                if not linked.exists():
                    raise ValueError(f"broken local link in {path.relative_to(ROOT)}: {label}")
    baseline = read(A100 / "repo/output/orchestrated/sana2ba100-20260913-093521/BASELINE.json")
    close(baseline["total_s"], 578.02)
    points = [
        ("kernel_aw", "20260913-140334-kernel_compile_bf16_max_autotune_ffn_temporal_linear", 358.39, None),
        ("cache_ca", "20260914-054324-cache_ca_r05_taylorseer_t0180_w10_o1", 371.36, "20260914T103201.266850+0000_3075761"),
        ("cache_ca", "20260914-083326-cache_ca_r08_taylorseer_t0180_w30_o1", 480.96, "20260914T103221.412243+0000_3076783"),
    ]
    point_result = []
    for technique, name, expected, attempt in points:
        run = RUNS / f"sana2ba100-{technique}-0002/worktree/runs" / name
        bench = read(run / "outputs/benchmark.json")
        close(bench["total_s"], expected)
        if bench["timing_scope"] != baseline["timing_scope"]:
            raise ValueError("request denominator mismatch")
        item = {"point": name, "request_s": expected,
                "request_speedup": baseline["total_s"] / expected}
        if attempt:
            quality = read(run / "outputs/quality.json")
            lpips = quality["judges"]["lpips"]
            if lpips["pairs_scored"] != 42 or lpips["status"] != "complete":
                raise ValueError("LPIPS sample scope changed")
            if "nvidia_gemini:api_key_missing" not in quality["promotion_blockers"]:
                raise ValueError("raw quality blocker was erased")
            receipt = read(run / "phase3_reverify_attempts" / attempt / "receipt.json")
            if receipt["collect"]["exit_code"] or receipt["plan_eval"]["exit_code"] or receipt["blocked_network_attempts"]:
                raise ValueError("fresh remote reassessment receipt failed")
            item.update(lpips_max=lpips["result"]["max"], lpips_pairs=42)
        point_result.append(item)
    phase1 = read(A100 / "reports/full_gpu_window_six_hour_cap.json")
    phase2 = read(A100 / "artifacts/phase3/phase2_full_window_terminal/full_window_cap.json")
    for window in (phase1, phase2):
        close(window["wall_s"], 21600)
        close(window["eight_gpu_capacity_s"], 172800)
        close(sum(window["seconds_by_concurrent_distinct_allocated_uuids"].values()), 21600)
        close(sum(item["allocated_gpu_s"] for item in window["per_gpu_uuid"].values()),
              window["allocated_distinct_uuid_gpu_s"])
        if window["open_job_count_at_snapshot"]:
            raise ValueError("unclosed candidate intervals")
    close(phase2["observer"]["covered_s"] + phase2["observer"]["unobserved_s"], 21600)
    pilot = read(ROOT / "experiments/h100-rollout-pilot/reports/PILOT-6340C3E.json")
    systems = pilot["systems"]
    close(systems["serial1"]["median_ttvf_s"] / systems["fifo2"]["median_ttvf_s"], systems["fifo2"]["speedup_vs_serial1"])
    close(systems["fifo2"]["median_ttvf_s"] / systems["optroll2"]["median_ttvf_s"], systems["optroll2"]["speedup_vs_fifo2"])
    if pilot["formal_boundary"]["formal_comparison_status"] != "NOT_RUN":
        raise ValueError("exploratory pilot relabeled formal")
    print(json.dumps({"status": "PASS", "imported_files": len(manifest["files"]),
                      "points": point_result,
                      "phase1_allocation_fraction": phase1["allocated_distinct_uuid_gpu_s"] / 172800,
                      "phase2_allocation_fraction": phase2["allocated_distinct_uuid_gpu_s"] / 172800,
                      "optroll2_speedup_vs_fifo2": systems["optroll2"]["speedup_vs_fifo2"],
                      "verification_scope": "included bytes, local links and arithmetic; raw videos/telemetry not independently re-evaluated"}, indent=2))


if __name__ == "__main__":
    main()
