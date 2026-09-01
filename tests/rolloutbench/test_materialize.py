from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rolloutbench.materialize import (
    AuthorityMaterializationError,
    materialize_candidate_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_DIR = REPO_ROOT / "benchmarks" / "sana_video_2b_h100_v0"


def _episode(episode_id: str) -> dict:
    for line in (SUITE_DIR / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["episode_id"] == episode_id:
            row.pop("golden")
            return row
    raise AssertionError(f"missing fixture episode {episode_id}")


class AuthorityMaterializationTests(unittest.TestCase):
    def test_materializes_kernel_and_cache_configs_from_their_authority_refs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for episode_id in ("K01", "C01"):
                with self.subTest(episode_id=episode_id):
                    receipt = materialize_candidate_artifacts(
                        _episode(episode_id), root, repo_root=REPO_ROOT
                    )
                    config = _episode(episode_id)["candidate"]["config"]
                    self.assertEqual(episode_id, receipt["episode_id"])
                    self.assertEqual(1, len(receipt["artifacts"]))
                    artifact = receipt["artifacts"][0]
                    self.assertEqual("config", artifact["kind"])
                    self.assertEqual(config["blob_sha256"], artifact["sha256"])
                    self.assertEqual(
                        config["blob_sha256"],
                        hashlib.sha256((root / artifact["relative_path"]).read_bytes()).hexdigest(),
                    )

    def test_materializes_preflight_probe_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = _episode("K15")
            receipt = materialize_candidate_artifacts(episode, root, repo_root=REPO_ROOT)
            source = episode["candidate"]["probe"]["source"]
            self.assertEqual(1, len(receipt["artifacts"]))
            artifact = receipt["artifacts"][0]
            self.assertEqual("probe_source", artifact["kind"])
            self.assertEqual(source["blob_sha256"], artifact["sha256"])
            self.assertTrue((root / artifact["relative_path"]).is_file())

    def test_same_content_is_idempotent_but_conflicting_existing_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = _episode("K01")
            first = materialize_candidate_artifacts(episode, root, repo_root=REPO_ROOT)
            self.assertEqual(first, materialize_candidate_artifacts(episode, root, repo_root=REPO_ROOT))
            target = root / first["artifacts"][0]["relative_path"]
            target.write_bytes(b"conflicting content")
            with self.assertRaisesRegex(AuthorityMaterializationError, "refusing overwrite"):
                materialize_candidate_artifacts(episode, root, repo_root=REPO_ROOT)

    def test_hash_mismatch_and_golden_input_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad_hash = copy.deepcopy(_episode("K01"))
            bad_hash["candidate"]["config"]["blob_sha256"] = "0" * 64
            bad_hash["candidate"]["config"]["authority_reported_sha256"] = "0" * 64
            with self.assertRaisesRegex(AuthorityMaterializationError, "SHA-256 mismatch"):
                materialize_candidate_artifacts(bad_hash, root, repo_root=REPO_ROOT)

            golden = _episode("K01")
            golden["golden"] = {"scheduler_visible": False}
            with self.assertRaisesRegex(AuthorityMaterializationError, "public descriptor"):
                materialize_candidate_artifacts(golden, root, repo_root=REPO_ROOT)

    def test_missing_authority_ref_or_path_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_ref = copy.deepcopy(_episode("K01"))
            missing_ref["candidate"]["authority_ref"] = "deadbeef"
            with self.assertRaisesRegex(AuthorityMaterializationError, "authority object is unavailable"):
                materialize_candidate_artifacts(missing_ref, root, repo_root=REPO_ROOT)

            missing_path = copy.deepcopy(_episode("K01"))
            missing_path["candidate"]["config"]["path"] = "config/does-not-exist.toml"
            with self.assertRaisesRegex(AuthorityMaterializationError, "authority object is unavailable"):
                materialize_candidate_artifacts(missing_path, root, repo_root=REPO_ROOT)

    def test_symlink_parent_cannot_escape_the_derived_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            (root / "K01").mkdir()
            (root / "K01" / "config").symlink_to(Path(outside), target_is_directory=True)
            with self.assertRaisesRegex(
                AuthorityMaterializationError, "symlink|escapes"
            ):
                materialize_candidate_artifacts(_episode("K01"), root, repo_root=REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
