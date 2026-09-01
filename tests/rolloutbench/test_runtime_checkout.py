from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from rolloutbench.runtime_checkout import (
    CRITICAL_RUNTIME_FILES,
    prepare_runtime_checkout,
    verify_runtime_receipt,
)
from rolloutbench.runtime_manifest import build_runtime_manifest


def _git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _episode(commit: str, *, golden: bool = False, preflight: bool = False) -> dict:
    result = {"candidate": {"candidate_commit": commit}}
    if preflight:
        result["candidate"].update(
            {"candidate_commit": "not_created_preflight_rejection", "parent_sha": commit, "probe": {"source": {}}}
        )
    if golden:
        result["golden"] = {"scheduler_visible": False}
    return result


class RuntimeCheckoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.repository = self.root / "source"
        self.repository.mkdir()
        _git("init", "--object-format=sha1", cwd=self.repository)
        _git("config", "user.email", "test@example.invalid", cwd=self.repository)
        _git("config", "user.name", "Runtime Checkout Test", cwd=self.repository)
        for relative in CRITICAL_RUNTIME_FILES:
            path = self.repository / "external" / "sol_runtime" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture {relative}\n", encoding="utf-8")
        _git("add", ".", cwd=self.repository)
        _git("commit", "-m", "fixture", cwd=self.repository)
        self.commit = _git("rev-parse", "HEAD", cwd=self.repository)
        self.worktrees = self.root / "worktrees"

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_creates_detached_checkout_and_is_idempotent(self) -> None:
        first = prepare_runtime_checkout(_episode(self.commit), self.repository, self.worktrees)
        target = self.worktrees / self.commit
        self.assertEqual("READY", first["status"])
        self.assertEqual(self.commit, first["runtime_ref"])
        self.assertEqual("candidate_commit", first["ref_role"])
        self.assertRegex(first["runtime_tree_oid"], r"^[0-9a-f]{40}$")
        self.assertEqual(str(target.resolve()), first["worktree_path"])
        self.assertTrue(target.is_dir())
        self.assertEqual(self.commit, _git("rev-parse", "HEAD", cwd=target))
        self.assertNotEqual(0, subprocess.run(["git", "symbolic-ref", "-q", "HEAD"], cwd=target).returncode)
        self.assertEqual(set(CRITICAL_RUNTIME_FILES), set(first["critical_runtime_file_sha256"]))
        self.assertEqual(first, prepare_runtime_checkout(_episode(self.commit), self.repository, self.worktrees))

    def test_rejects_bad_or_missing_candidate_commits(self) -> None:
        with self.assertRaisesRegex(ValueError, "lowercase"):
            prepare_runtime_checkout(_episode("A" * 40), self.repository, self.worktrees)
        with self.assertRaisesRegex(ValueError, "resolve"):
            prepare_runtime_checkout(_episode("0" * 40), self.repository, self.worktrees)

    def test_preflight_rejection_uses_the_exact_parent_runtime(self) -> None:
        receipt = prepare_runtime_checkout(
            _episode(self.commit, preflight=True), self.repository, self.worktrees
        )
        self.assertEqual(self.commit, receipt["runtime_ref"])
        self.assertEqual("parent_for_preflight_rejection", receipt["ref_role"])

    def test_rejects_existing_attached_or_dirty_worktrees(self) -> None:
        prepare_runtime_checkout(_episode(self.commit), self.repository, self.worktrees)
        target = self.worktrees / self.commit
        _git("checkout", "-b", "local-test-branch", cwd=target)
        with self.assertRaisesRegex(RuntimeError, "detached"):
            prepare_runtime_checkout(_episode(self.commit), self.repository, self.worktrees)
        _git("checkout", "--detach", self.commit, cwd=target)
        (target / "dirty.txt").write_text("dirty\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "clean"):
            prepare_runtime_checkout(_episode(self.commit), self.repository, self.worktrees)

    def test_rejects_a_receipt_forged_to_a_different_registered_worktree(self) -> None:
        receipt = prepare_runtime_checkout(_episode(self.commit), self.repository, self.worktrees)
        changed = self.repository / "external" / "sol_runtime" / CRITICAL_RUNTIME_FILES[0]
        changed.write_text("different runtime\n", encoding="utf-8")
        _git("add", ".", cwd=self.repository)
        _git("commit", "-m", "different runtime", cwd=self.repository)
        other_commit = _git("rev-parse", "HEAD", cwd=self.repository)
        other = self.worktrees / other_commit
        _git("worktree", "add", "--detach", str(other), other_commit, cwd=self.repository)
        forged = {**receipt, "worktree_path": str(other.resolve())}
        expected = build_runtime_manifest(self.repository, _episode(self.commit)["candidate"])
        with self.assertRaisesRegex(RuntimeError, "HEAD does not match"):
            verify_runtime_receipt(self.repository, forged, expected)

    def test_rejects_scheduler_hidden_golden_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "golden"):
            prepare_runtime_checkout(_episode(self.commit, golden=True), self.repository, self.worktrees)


if __name__ == "__main__":
    unittest.main()
