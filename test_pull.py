import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dag


def task(task_id="t", deps=None):
    return {"kind": "task", "id": task_id, "title": "pull test",
            "priority": "P0", "depends_on": deps or [],
            "instruction": ["implement"], "acceptance": ["independent review"],
            "outputs": [f"artifacts/{task_id}/verification.json"],
            "avoid": ["self review"], "status": "planned"}


def repo(root):
    subprocess.run(["git", "-C", str(root), "init", "-q", "-b", "main"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Pull Test"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "pull-test@example.invalid"], check=True)
    body = {"seq": 1, "event": task()}
    row = {**body, "prev_sha256": dag.ZERO,
           "sha256": dag.hash_event(dag.ZERO, body)}
    (root / "instructions.jsonl").write_text(dag.canon(row) + "\n")
    subprocess.run(["git", "-C", str(root), "add", "--", "instructions.jsonl"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "fixture"], check=True)


def event_request(root, owner):
    location = root / "artifacts" / "t" / "candidate.json"
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text('{"candidate":"frozen"}\n')
    digest = hashlib.sha256(location.read_bytes()).hexdigest()
    dag.append({"kind": "review_request", "id": "t", "owner": owner,
                "candidate_path": "artifacts/t/candidate.json",
                "candidate_sha256": digest,
                "revision": subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()})
    return digest


class PullTests(unittest.TestCase):
    def test_pull_claims_implementation_once_then_independent_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo(root)
            with patch.object(dag, "ROOT", root), patch.object(dag, "LOG", root / "instructions.jsonl"):
                first = dag.pull("author")
                self.assertEqual((first["role"], first["task"], first["reused"]), ("IMPLEMENTER", "t", False))
                self.assertTrue(dag.pull("author")["reused"])
                digest = event_request(root, "author")
                self.assertEqual(dag.pull("author")["role"], "IMPLEMENTER")
                second = dag.pull("reviewer")
                self.assertEqual((second["role"], second["task"], second["candidate_sha256"]),
                                 ("VERIFIER", "t", digest))
                self.assertTrue(dag.pull("reviewer")["reused"])
                self.assertEqual(dag.pull("third")["role"], "HOLD")
                state, _, _ = dag.audit((root/"instructions.jsonl").read_text())
                self.assertEqual(state["t"]["owner"], "author")
                self.assertEqual(state["t"]["review"]["reviewer"], "reviewer")
                self.assertEqual(state["t"]["review"]["state"], "claimed")

    def test_blocked_last_owner_review_and_stale_request_fencing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            repo(root)
            with patch.object(dag, "ROOT", root), patch.object(dag, "LOG", root / "instructions.jsonl"):
                dag.pull("author")
                dag.append({"kind": "status", "id": "t", "status": "blocked", "agent": "author"})
                digest=event_request(root, "author")
                self.assertEqual(dag.pull("author")["role"], "HOLD")
                with self.assertRaisesRegex(ValueError, "own verification"):
                    dag.append({"kind": "review_claim", "id": "t", "reviewer": "author", "candidate_sha256": digest})
                self.assertEqual(dag.pull("independent")["role"], "VERIFIER")
                with self.assertRaisesRegex(ValueError, "only assigned reviewer"):
                    dag.append({"kind": "review_result", "id": "t", "reviewer": "forged",
                                "candidate_sha256": digest, "verdict": "HOLD"})
                with self.assertRaisesRegex(ValueError, "receipt path"):
                    dag.append({"kind": "review_result", "id": "t", "reviewer": "independent",
                                "candidate_sha256": digest, "verdict": "PASS"})
                with self.assertRaisesRegex(ValueError, "matching pending"):
                    dag.append({"kind": "review_claim", "id": "t", "reviewer": "another",
                                "candidate_sha256": digest})
                dag.append({"kind": "review_result", "id": "t", "reviewer": "independent",
                            "candidate_sha256": digest, "verdict": "HOLD", "note": "insufficient evidence"})
                self.assertEqual(dag.pull("independent")["role"], "HOLD")
                state, _, _=dag.audit((root/"instructions.jsonl").read_text())
                self.assertEqual(state["t"]["review"]["verdict"], "HOLD")
                self.assertIsNone(state["t"]["owner"])

    def test_event_lab_owner_alias_cannot_self_review(self):
        identity = "a" * 32
        events = [task(), {"kind": "claim", "id": "t", "owner": "chatgpt-session-" + identity},
                  {"kind": "review_request", "id": "t", "owner": "chatgpt-session-" + identity,
                   "candidate_path": "artifacts/t/candidate.json",
                   "candidate_sha256": "b" * 64, "revision": "c" * 40}]
        def as_log(items):
            previous = dag.ZERO
            lines = []
            for i, ev in enumerate(items, 1):
                body = {"seq": i, "event": ev}
                row = {**body, "prev_sha256": previous, "sha256": dag.hash_event(previous, body)}
                lines.append(dag.canon(row))
                previous = row["sha256"]
            return "\n".join(lines) + "\n"
        state, _, _ = dag.audit(as_log(events))
        with self.assertRaisesRegex(ValueError, "own verification"):
            dag.validate_event({"kind": "review_claim", "id": "t", "reviewer": identity,
                                "candidate_sha256": "b"*64}, state)
        selection = dag.choose_pull_event(state, identity)
        self.assertNotEqual(selection.get("kind"), "review_claim")

    def test_request_candidate_hash_is_checked_before_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            repo(root)
            with patch.object(dag, "ROOT", root), patch.object(dag, "LOG", root / "instructions.jsonl"):
                dag.pull("author")
                loc=root/"artifacts"/"t"/"candidate.json"
                loc.parent.mkdir(parents=True, exist_ok=True)
                loc.write_text("current")
                before=(root/"instructions.jsonl").read_bytes()
                with self.assertRaisesRegex(ValueError, "changed before review"):
                    dag.append({"kind": "review_request", "id": "t", "owner": "author",
                                "candidate_path": "artifacts/t/candidate.json",
                                "candidate_sha256": "0"*64, "revision": "a"*40})
                self.assertEqual((root/"instructions.jsonl").read_bytes(), before)


if __name__=="__main__":
    unittest.main()
