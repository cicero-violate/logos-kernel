"""Offline supply-chain negative controls. No remote Git operations."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import offline_pin_audit as a


def run(*args, cwd=None):
    return subprocess.check_output(["git", *args], cwd=cwd)


class OfflinePinAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "z3"
        self.repo.mkdir()
        run("init", "-q", cwd=self.repo)
        run("config", "user.name", "LocalFixture", cwd=self.repo)
        run("config", "user.email", "fixture@example.invalid", cwd=self.repo)
        run("remote", "add", "origin", "https://github.com/Z3Prover/z3", cwd=self.repo)
        (self.repo / "LICENSE.txt").write_text("MIT fixture, not a real Z3 license\n")
        (self.repo / "src").mkdir()
        (self.repo / "src" / "sample.py").write_text("def x(): return 3\n")
        run("add", "LICENSE.txt", "src/sample.py", cwd=self.repo)
        run("commit", "-qm", "local untrusted fixture", cwd=self.repo)
        self.rev = run("rev-parse", "HEAD", cwd=self.repo).decode().strip()
        self.row = dict(id="z3", repository_url="https://github.com/Z3Prover/z3",
                        immutable_git_commit_sha40=self.rev,
                        license_file_sha256=a.sha((self.repo / "LICENSE.txt").read_bytes()),
                        license_path="LICENSE.txt",
                        submodule_commit_shas={},
                        source_paths_to_verify=["src/sample.py"])

    def check(self):
        return a.verify_reference(self.row, self.root)

    def test_positive_local_integrity_not_origin_authentication(self):
        r = self.check()
        self.assertEqual(r["status"], "LOCALLY_PINNED")
        self.assertFalse(r["reviewer_identity_authenticated"])

    def test_missing_pin_is_hold(self):
        self.row["immutable_git_commit_sha40"] = None
        self.assertEqual(self.check()["status"], "HOLD")

    def test_corrupt_license_hash_is_hold(self):
        self.row["license_file_sha256"] = "a"*64
        self.assertIn("hash mismatch", self.check()["reason"])

    def test_mismatched_remote_metadata_is_hold(self):
        self.row["repository_url"] = "https://github.com/attacker/z3"
        self.assertIn("remote URL differs", self.check()["reason"])

    def test_missing_source_file_is_hold(self):
        self.row["source_paths_to_verify"] = ["missing.py"]
        self.assertEqual(self.check()["status"], "HOLD")

    def test_untracked_submodule_claim_is_hold(self):
        self.row["submodule_commit_shas"] = {"fake": "a"*40}
        self.assertIn("submodule Git-link pins mismatch", self.check()["reason"])

    def test_missing_explicit_submodule_manifest_is_hold(self):
        del self.row["submodule_commit_shas"]
        self.assertEqual(self.check()["status"], "HOLD")

    def test_fake_commit_is_hold(self):
        self.row["immutable_git_commit_sha40"] = "f"*40
        self.assertEqual(self.check()["status"], "HOLD")

    def test_path_traversal_refused(self):
        for value in ("../secret", "/etc/passwd", "src/../a", "src//a", ".git/config", "x\\y"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                a.safe_file(value)

    def test_symlink_cache_escape_is_hold(self):
        outside_temp = tempfile.TemporaryDirectory()
        self.addCleanup(outside_temp.cleanup)
        (self.root / "tscl").symlink_to(outside_temp.name)
        self.row["id"] = "tscl"
        self.assertIn("mirror escapes cache", self.check()["reason"])

    def test_oversized_local_blob_rejected(self):
        (self.repo / "LICENSE.txt").write_bytes(b"z" * (a.MAX_FILE_BYTES + 1))
        run("add", "LICENSE.txt", cwd=self.repo)
        run("commit", "-qm", "oversized fixture", cwd=self.repo)
        self.row["immutable_git_commit_sha40"] = run("rev-parse", "HEAD", cwd=self.repo).decode().strip()
        self.row["license_file_sha256"] = a.sha((self.repo / "LICENSE.txt").read_bytes())
        self.assertIn("oversized blob", self.check()["reason"])

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError):
            json.loads('{"id":1,"id":2}', object_pairs_hook=a.no_duplicates)

    def test_missing_license_blob_is_hold(self):
        self.row["license_path"] = "LICENSE-DOES-NOT-EXIST"
        self.assertEqual(self.check()["status"], "HOLD")


if __name__ == "__main__":
    unittest.main()
