"""Export a small, explicitly selected, privacy-filtered evidence snapshot.

This is a bulk publication transform, not an experiment runner. It never uses
the network, imports a model, changes source evidence, or copies binary assets.
Usage: python3 -B tools/export_status.py --source-root PROJECT --output-root .
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


ABCD = "20261021-abcd-native-execution"
DAG = "20261022-dag-native-execution"
SANA = "20261023-method-search"
WAN = "20261024-wan-search"
SEARCH = "20261024-search-debug"
AC = f"{ABCD}/campaigns/abcd-native-20261021-v6"
DC = f"{DAG}/campaigns/dag-native-20261022-v6"
SC = f"{SANA}/campaigns/method-search-20261023-six-family-002"
WC = f"{WAN}/campaigns/wan-search-20261024-002"
BC = f"{SANA}/crosschecks/cache-bridge-001"
ROOT = "evidence/20261031"

# (source relative to project, publication path, optional JSON top-level fields)
FILES = [
    (f"{ABCD}/reports/EXECUTION-STATUS.md", "abcd/REPORT.md", None),
    (f"{ABCD}/reports/FINAL-ACCEPTANCE.json", "abcd/FINAL-ACCEPTANCE.json", None),
    (f"{ABCD}/reports/BASELINE-CONSISTENCY.json", "abcd/BASELINE-CONSISTENCY.json", None),
    (f"{AC}/analysis/SUMMARY.json", "abcd/SUMMARY.json", None),
    (f"{DAG}/reports/FINAL-REPORT.md", "dag/REPORT.md", None),
    (f"{DC}/CAMPAIGN-VALIDITY.json", "dag/CAMPAIGN-VALIDITY.json", None),
    (f"{DC}/derived/P.json", "dag/METRICS.json", ["analysis_sha256", "metrics"]),
    (f"{DC}/derived/P-gpu.csv", "dag/gpu.csv", None),
    (f"{SANA}/reports/RESULTS-SIX-FAMILY-002.md", "sana/REPORT.md", None),
    (f"{SC}/derived/P.json", "sana/METRICS.json", ["analysis_sha256", "metrics"]),
    (f"{SC}/derived/P-gpu.csv", "sana/gpu.csv", None),
    (f"{SC}/derived/P-candidates.csv", "sana/candidates.csv", None),
    (f"{SC}/runs/P/TERMINAL.json", "sana/TERMINAL.json", None),
    (f"{SC}/runs/P/workspace/output/orchestrated/sana-20261023-164113/INTEGRATED-DELIVERY.json", "sana/INTEGRATED-DELIVERY.json", None),
    (f"{SC}/runs/P/method-search/REPORT.json", "sana/SEARCH-REPORT.json", [
        "schema", "run_id", "evidence_mode", "coverage", "funnel", "costs",
        "execution_status_counts", "boundary", "selection_slots_used", "selected_method_count",
    ]),
    (f"{BC}/reports/RESULTS.md", "quality-bridge/REPORT.md", None),
    (f"{BC}/artifacts/OUTCOMES.json", "quality-bridge/OUTCOMES.json", None),
    (f"{BC}/artifacts/QUALITY-CONFLICT-NOTICE.json", "quality-bridge/QUALITY-CONFLICT-NOTICE.json", None),
    (f"{BC}/artifacts/TIMING-SUMMARY.json", "quality-bridge/TIMING-SUMMARY.json", None),
    (f"{BC}/artifacts/RESOURCES.json", "quality-bridge/RESOURCES.json", None),
    (f"{BC}/artifacts/SOURCE-IMAGE-BINDING.json", "quality-bridge/SOURCE-IMAGE-BINDING.json", None),
    (f"{BC}/blind-workspace/verdicts.json", "quality-bridge/verdicts.json", None),
    (f"{WAN}/reports/RESULTS.md", "wan/REPORT.md", None),
    (f"{WAN}/reports/ASSET-AUDIT.md", "wan/ASSET-AUDIT.md", None),
    (f"{WAN}/reports/ENGINEERING-VALIDATION.md", "wan/ENGINEERING-VALIDATION.md", None),
    (f"{WC}/PROTOCOL.json", "wan/PROTOCOL.json", None),
    (f"{WC}/derived/P.json", "wan/METRICS.json", ["analysis_sha256", "metrics"]),
    (f"{WC}/derived/P-gpu.csv", "wan/gpu.csv", None),
    (f"{WC}/derived/P-gpu-stages.csv", "wan/gpu-stages.csv", None),
    (f"{WC}/derived/P-candidates.csv", "wan/candidates.csv", None),
    (f"{WC}/runs/P/VERIFIED-COMPONENTS.json", "wan/VERIFIED-COMPONENTS.json", None),
    (f"{WC}/runs/P/NO-FRONTIER-CHECK.json", "wan/NO-FRONTIER-CHECK.json", None),
    (f"{WC}/runs/P/workspace/output/orchestrated/wan21-20261024-164822/INTEGRATED-DELIVERY.json", "wan/INTEGRATED-DELIVERY.json", None),
    (f"{WC}/derived/SIX-FAMILY-AUDIT.json", "wan/AUDIT-SUMMARY.json", [
        "schema", "run_id", "timestamp", "status", "issues", "code_hash", "workload",
        "budget_usage", "funnel", "coverage", "selected_sources", "terminal",
        "integration_attempts", "experience_statuses",
        "unattended_closed_loop", "model_comparison_boundary", "cleanup_status",
        "receipt_validation_status", "boundary", "new_model_download_bytes",
        "weight_transfer_via_local_bytes",
    ]),
    (f"{SEARCH}/reports/SEARCH-DEBUG.md", "search/REPORT.md", None),
    (f"{SEARCH}/reports/AUDIT.json", "search/AUDIT.json", None),
    (f"{SEARCH}/reports/TEST-RESULTS.json", "search/TEST-RESULTS.json", None),
    (f"{SEARCH}/reports/methods.csv", "search/methods.csv", None),
    (f"{SEARCH}/reports/papers.csv", "search/papers.csv", None),
    (f"{SEARCH}/artifacts/catalog.json", "search/catalog.json", None),
    (f"{SEARCH}/artifacts/pending-leads.json", "search/pending-leads.json", None),
    (f"{SEARCH}/artifacts/historical-search-audit.json", "search/historical-search-audit.json", None),
]
for arm in "ABCD":
    FILES.append((f"{AC}/analysis/{arm}-gpu.csv", f"abcd/{arm}-gpu.csv", None))
    FILES.append((f"{AC}/analysis/{arm}-candidates.csv", f"abcd/{arm}-candidates.csv", None))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def redact(text: str, source_root: Path) -> str:
    # Preserve experiment-relative identities and all numbers. Actual private
    # paths, SSH identifiers and hardware UUIDs are not required by this report.
    text = text.replace(str(source_root) + "/", "project://SolAgent/")
    text = re.sub(r"/share/project/[^/\s\"'`]+/[^/\s\"'`]+/Experiments/SolAgent/",
                  "remote-artifact://SolAgent/", text)
    text = re.sub(r"/share/project/[^\s\"'`<>)]*", "<REMOTE_PROJECT_PATH>", text)
    text = re.sub(r"/home/[^/\s\"'`]+/", "<HOME>/", text)
    text = text.replace("/tmp/lyy-experiments/", "scratch-artifact://")
    text = re.sub(r"job-[a-z0-9-]+\.[^\s\"'`@]+@[^\s\"'`]+", "<SSH_TARGET>", text)
    text = re.sub(r"GPU-[0-9a-fA-F-]{16,}",
                  lambda m: "GPU-REDACTED-" + sha(m[0].encode())[:12], text)
    text = text.replace("c6cfcde5", "node-a").replace("1ffc9909", "node-b")
    return text


def numeric_leaves(obj, prefix=()):
    if isinstance(obj, dict):
        return {p: v for k, value in obj.items() for p, v in numeric_leaves(value, prefix + (k,)).items()}
    if isinstance(obj, list):
        return {p: v for i, value in enumerate(obj) for p, v in numeric_leaves(value, prefix + (i,)).items()}
    return {} if isinstance(obj, str) else {prefix: obj}


def export(source_root: Path, output_root: Path):
    source_root, output_root = source_root.resolve(), output_root.resolve()
    mapped = {str(source_root / src): f"{ROOT}/{dst}" for src, dst, _ in FILES}
    projected = {str(source_root / src) for src, _, fields in FILES if fields}
    manifest = {
        "schema": "sol-status-publication/v1", "publication_date": "2026-10-31",
        "scope": "Evidence publication; no new GPU experiment, no runtime-code release",
        "source_path_base": "project://SolAgent/", "files": [],
        "excluded": ["weights", "videos", "images", "raw agent/API logs", "credentials",
                     "full telemetry", "third-party source/README captures", "uncommitted runtime code"],
        "transformations": ["private paths and GPU UUIDs redacted", "selected Markdown links rewritten",
                            "unpublished links rendered as plain retained-artifact labels",
                            "CSV record line endings normalized to LF",
                            "large analysis JSON projected onto declared fields; numeric leaves preserved"],
    }
    omitted_links = []
    for src, dst, fields in FILES:
        source = source_root / src
        raw = source.read_bytes()
        content = raw.decode("utf-8")
        if source.suffix == ".csv":
            content = content.replace("\r\n", "\n")
        original = None
        if source.suffix == ".json":
            original = json.loads(content)
            if fields:
                original = {key: original[key] for key in fields}
            content = json.dumps(original, ensure_ascii=False, indent=2) + "\n"
        published = f"{ROOT}/{dst}"
        target = output_root / published
        if source.suffix == ".md":
            def link(match):
                label, address = match[1], match[2]
                if not address.startswith("/"):
                    return match[0]
                if address in mapped:
                    relative = os.path.relpath(output_root / mapped[address], target.parent)
                    if address in projected:
                        label += "（本次为字段节选）"
                    return f"[{label}]({relative})"
                omitted_links.append({"report": published, "label": label,
                                      "source_reference": redact(address, source_root)})
                return f"{label}（原始产物保留，未随本 PR 发布）"
            content = re.sub(r"!?\[([^\]]+)\]\(([^)]+)\)", link, content)
            content = (
                "> 2026-10-31 发布副本：保留原报告的实验日期、结论和历史状态；"
                "私有路径已替换，未发布产物不伪装为可访问链接。"
                "文末“未推送”等措辞指报告生成时；本次发布说明见根目录 PUBLICATION-NOTE.md。\n\n"
                + content
            )
        content = redact(content, source_root)
        if original is not None:
            assert numeric_leaves(original) == numeric_leaves(json.loads(content)), src
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        manifest["files"].append({"source": src, "published": published,
                                  "source_sha256": sha(raw), "published_sha256": sha(content.encode()),
                                  "source_bytes": len(raw), "published_bytes": len(content.encode()),
                                  "projection_fields": fields, "numeric_fields_preserved": original is not None})

    # Publish source identities, not redistributions of whole third-party files.
    source_path = source_root / SEARCH / "artifacts/evidence.json"
    raw = source_path.read_bytes()
    evidence = json.loads(raw)
    catalog = json.loads((source_root / SEARCH / "artifacts/catalog.json").read_text())
    referenced_paths = {}
    for method in catalog["methods"]:
        for item in method.get("code_evidence", []):
            referenced_paths.setdefault(item["source_id"], set()).add(item["path"])
    fields = ("id", "kind", "url", "repository", "commit_sha", "blob_sha",
              "captured_text_sha256", "commit_binding_checked")
    index = {"schema": "publication-source-index/v1", "sources": [],
             "boundary": "Metadata projection only. Full captured source text/trees are not redistributed. "
                         "This index does not independently reproduce the original citation-presence audit."}
    for source in evidence["sources"]:
        entry = {key: source[key] for key in fields if key in source}
        if source["id"] in referenced_paths:
            entry["catalog_referenced_paths"] = sorted(referenced_paths[source["id"]])
        index["sources"].append(entry)
    published = f"{ROOT}/search/SOURCE-INDEX.json"
    content = redact(json.dumps(index, ensure_ascii=False, indent=2) + "\n", source_root)
    (output_root / published).write_text(content)
    manifest["files"].append({"source": f"{SEARCH}/artifacts/evidence.json", "published": published,
                              "source_sha256": sha(raw), "published_sha256": sha(content.encode()),
                              "source_bytes": len(raw), "published_bytes": len(content.encode()),
                              "projection_fields": list(fields) + ["catalog_referenced_paths"],
                              "numeric_fields_preserved": False})
    (output_root / ROOT / "UNPUBLISHED-REFERENCES.json").write_text(
        json.dumps(omitted_links, ensure_ascii=False, indent=2) + "\n")
    (output_root / "PUBLICATION-MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"imported_files": len(manifest["files"]), "unpublished_links": len(omitted_links),
                      "published_bytes": sum(x["published_bytes"] for x in manifest["files"])}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    export(args.source_root, args.output_root)
