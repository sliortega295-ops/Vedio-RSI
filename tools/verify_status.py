"""Offline checks for this curated publication; no model or network operations."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import unittest
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/20261031"
SENSITIVE = (
    r"/home/[A-Za-z0-9_-]+/",
    r"/share/project/[A-Za-z0-9_-]+/",
    r"/tmp/lyy-experiments/",
    r"https?://[^\s/\"<>:]+:[^\s/\"<>@]+@",
    r"job-[a-f0-9-]+[^\s\"']*@ssh\.",
    r"GPU-[a-fA-F0-9]{8}-[a-fA-F0-9-]{20,}",
    r"(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]{20,}",
    r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
)


def load(relative):
    return json.loads((EVIDENCE / relative).read_text(encoding="utf-8"))


def public_files():
    # Explicit scope; ignored bytecode in unrelated historical directories is
    # not part of this publication and must never be staged with this snapshot.
    files = [ROOT / name for name in (
        "README.md", "PUBLICATION-NOTE.md", "PUBLICATION-MANIFEST.json", ".gitignore")]
    for directory in ("reports", "evidence", "tools"):
        files.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts)
    return sorted(files)


def sensitive_matches(text):
    return [pattern for pattern in SENSITIVE if re.search(pattern, text)]


class PublicationTests(unittest.TestCase):
    def test_manifest_hashes_sizes_and_allowlist(self):
        manifest = json.loads((ROOT / "PUBLICATION-MANIFEST.json").read_text())
        expected = set()
        self.assertEqual(len(manifest["files"]), 51)
        for record in manifest["files"]:
            path = ROOT / record["published"]
            self.assertTrue(path.resolve().is_relative_to(EVIDENCE.resolve()))
            self.assertFalse(path.is_symlink())
            self.assertNotIn(path, expected)
            expected.add(path)
            raw = path.read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record["published_sha256"], str(path))
            self.assertEqual(len(raw), record["published_bytes"], str(path))
            self.assertRegex(record["source_sha256"], r"^[a-f0-9]{64}$")
        actual = {p for p in EVIDENCE.rglob("*") if p.is_file()}
        self.assertEqual(actual, expected | {EVIDENCE / "UNPUBLISHED-REFERENCES.json"})

    def test_text_payload_and_parseability(self):
        allowed = {".md", ".json", ".csv", ".py"}
        for path in public_files():
            self.assertTrue(path.name == ".gitignore" or path.suffix in allowed, str(path))
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("\x00", text, str(path))
            if path.suffix == ".json":
                json.loads(text)
            if path.suffix == ".csv":
                rows = list(csv.reader(io.StringIO(text)))
                self.assertTrue(rows, str(path))
                self.assertTrue(all(len(row) == len(rows[0]) for row in rows), str(path))

    def test_obvious_sensitive_material(self):
        for path in public_files():
            # Tests/exporter necessarily contain the patterns they detect.
            if path.suffix != ".py":
                self.assertFalse(sensitive_matches(path.read_text()), str(path))

    def test_sensitive_detector_and_export_redaction(self):
        from export_status import redact
        examples = ["/home/example/private", "/share/project/example/private",
                    "https://example:secret@example.invalid/", "sk-" + "x" * 32,
                    "GPU-12345678-1234-1234-1234-123456789abc"]
        for text in examples:
            self.assertTrue(sensitive_matches(text))
        self.assertFalse(sensitive_matches("<HOME>/private GPU-REDACTED-abc123"))
        value = redact(examples[0], Path("/some/source"))
        self.assertEqual(value, "<HOME>/private")
        gpu = redact(examples[-1], Path("/some/source"))
        self.assertEqual(gpu, redact(examples[-1], Path("/some/source")))
        self.assertFalse(sensitive_matches(gpu))

    def test_markdown_relative_targets(self):
        for path in public_files():
            if path.suffix != ".md":
                continue
            text = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
            for address in re.findall(r"!?\[[^\]]+\]\(([^)]+)\)", text):
                address = address.strip("<>")
                parsed = urlsplit(address)
                if parsed.scheme or address.startswith("#"):
                    continue
                self.assertFalse(address.startswith("/"), f"{path}: {address}")
                target = (path.parent / unquote(parsed.path)).resolve()
                self.assertTrue(target.is_relative_to(ROOT), f"{path}: {address}")
                self.assertTrue(target.is_file(), f"{path}: {address}")

    def test_historical_and_invalid_trial_boundaries(self):
        arms = load("abcd/SUMMARY.json")["arms"]
        self.assertEqual([a["arm"] for a in arms], list("ABCD"))
        for arm in arms:
            self.assertEqual(arm["status"], "COMPLETE")
            self.assertEqual(arm["generated_work"]["retained_video_files"], 55)
            for point in arm["final_frontier"]:
                perf = point["performance"]
                self.assertTrue(math.isclose(perf["baseline_total_s"] / perf["config_total_s"],
                                             perf["speedup"], rel_tol=1e-8))
        validity = load("dag/CAMPAIGN-VALIDITY.json")
        self.assertEqual(validity["status"], "INVALID_INFRA_TRIAL")
        self.assertFalse(validity["comparison_eligible"])

    def test_sana_result_and_specific_quality_dispute(self):
        points = load("sana/INTEGRATED-DELIVERY.json")["frontier_points"]
        self.assertEqual(len(points), 1)
        point = points[0]
        self.assertEqual(point["quality"]["visual_overall"], "pass")
        perf = point["performance"]
        self.assertAlmostEqual(perf["speedup"], perf["baseline_total_s"] / perf["config_total_s"])
        self.assertAlmostEqual(perf["speedup"], 1.2933, places=4)
        conflict = load("quality-bridge/QUALITY-CONFLICT-NOTICE.json")
        self.assertEqual(conflict["status"], "DISPUTED_QUALITY")
        self.assertTrue(conflict["fresh_cache_five_video_sha256_identical"])
        self.assertEqual(len(conflict["affected_historical_candidates"]), 2)
        self.assertTrue(all(x.startswith("method-search-20261023-pilot-002/")
                            for x in conflict["affected_historical_candidates"]))

    def test_wan_components_do_not_imply_valid_composition(self):
        delivery = load("wan/INTEGRATED-DELIVERY.json")
        self.assertEqual(delivery["outcome"], "NO_VALID_FRONTIER")
        self.assertEqual(delivery["frontier_points"], [])
        self.assertEqual(len(load("wan/VERIFIED-COMPONENTS.json")), 5)
        self.assertEqual(len(delivery["evaluated_compositions"]), 2)
        for point in delivery["evaluated_compositions"]:
            self.assertEqual(point["decision"], "reject")
            self.assertEqual(point["quality"]["visual_overall"], "fail")
            perf = point["performance"]
            self.assertAlmostEqual(perf["speedup"], perf["baseline_total_s"] / perf["config_total_s"])
        metrics = load("wan/METRICS.json")["metrics"]
        self.assertEqual(metrics["generated_work"]["retained_video_files"], 65)
        self.assertFalse(metrics["formal_comparison_eligible"])
        self.assertAlmostEqual(metrics["assigned_gpu_hours"], metrics["formal_wall_seconds"] * 8 / 3600)
        self.assertLess(metrics["leased_gpu_hours"], metrics["assigned_gpu_hours"])
        self.assertLess(metrics["api"]["responses_wall_union_seconds"], metrics["formal_wall_seconds"])

    def test_search_is_static_and_counts_are_not_inflated(self):
        audit = load("search/AUDIT.json")
        self.assertEqual(audit["mode"], "search_only")
        self.assertEqual(audit["independent_methods"], 41)  # Includes 2 execution engines.
        self.assertEqual(sum(audit["code_counts"].values()), 41)
        self.assertEqual(audit["code_counts"]["implementation_present"], 28)
        for key in ("gpu_tasks", "local_validated_count", "memory_admissions"):
            self.assertEqual(audit[key], 0)
        self.assertEqual(len(load("search/catalog.json")["methods"]), 41)
        pending = Counter(x["status"] for x in load("search/pending-leads.json"))
        self.assertEqual(pending, {"paper_lead_pending_code_review": 15, "adjacent_or_out_of_scope": 10})
        source_index = load("search/SOURCE-INDEX.json")
        self.assertTrue(source_index["sources"])
        for source in source_index["sources"]:
            self.assertNotIn("text", source)
            self.assertNotIn("content", source)

    def test_each_card_has_a_published_row(self):
        names = [f"abcd/{arm}-gpu.csv" for arm in "ABCD"]
        names += [f"{group}/gpu.csv" for group in ("dag", "sana", "wan")]
        for name in names:
            rows = list(csv.DictReader(io.StringIO((EVIDENCE / name).read_text())))
            self.assertEqual(len(rows), 8, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
