import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dag
from dag import ZERO, audit, append, canon, hash_event, validate_event


def row(i, prev, ev):
    body = {"seq": i, "event": ev}
    return {**body, "prev_sha256": prev, "sha256": hash_event(prev, body)}


def node(task_id, deps=()):
    return {
        "kind": "task", "id": task_id, "priority": "P0", "title": "test",
        "depends_on": list(deps), "instruction": ["execute"],
        "acceptance": ["verify"], "outputs": ["proof.json"],
        "avoid": ["drift"], "status": "planned",
    }


def log(events):
    prev = ZERO
    lines = []
    for i, event in enumerate(events, 1):
        current = row(i, prev, event)
        lines.append(canon(current))
        prev = current["sha256"]
    return "\n".join(lines) + "\n"


class DagTests(unittest.TestCase):
    def test_two_nodes(self):
        tasks, n, _ = audit(log([node("a"), node("b", ["a"])]))
        self.assertEqual(n, 2)
        self.assertEqual(tasks["b"]["record"]["depends_on"], ["a"])

    def test_tampering_detected(self):
        data = json.loads(log([node("a")]))
        data["event"]["title"] = "tampered"
        with self.assertRaisesRegex(ValueError, "hash changed"):
            audit(canon(data) + "\n")

    def test_forward_dependencies_rejected(self):
        with self.assertRaisesRegex(ValueError, "earlier tasks"):
            audit(log([node("b", ["a"])]))

    def test_cannot_claim_before_dependency_passes(self):
        with self.assertRaisesRegex(ValueError, "dependencies not passed"):
            audit(log([
                node("a"), node("b", ["a"]),
                {"kind": "claim", "id": "b", "owner": "real-agent-a"},
            ]))

    def test_claim_release_restores_ready(self):
        ts, _, _ = audit(log([
            node("a"),
            {"kind": "claim", "id": "a", "owner": "actual-agent"},
            {"kind": "release", "id": "a", "by": "actual-agent",
             "expected_owner": "actual-agent", "force": False, "reason": "handoff"},
        ]))
        self.assertEqual(ts["a"]["status"], "planned")
        self.assertIsNone(ts["a"]["owner"])
        self.assertTrue(dag.dependencies_passed(ts["a"], ts))

    def test_duplicate_claim_denied(self):
        with self.assertRaisesRegex(ValueError, "already claimed"):
            audit(log([
                node("a"), {"kind": "claim", "id": "a", "owner": "actual-agent"},
                {"kind": "claim", "id": "a", "owner": "other-real-agent"},
            ]))

    def test_unapproved_takeover_denied(self):
        with self.assertRaisesRegex(ValueError, "requires --force"):
            audit(log([
                node("a"), {"kind": "claim", "id": "a", "owner": "actual-agent"},
                {"kind": "release", "id": "a", "by": "different-agent",
                 "expected_owner": "actual-agent", "reason": "mistake"},
            ]))

    def test_expected_owner_fences_stale_release(self):
        with self.assertRaisesRegex(ValueError, "expected_owner mismatch"):
            audit(log([
                node("a"), {"kind": "claim", "id": "a", "owner": "actual-agent"},
                {"kind": "release", "id": "a", "by": "actual-agent", "expected_owner": "actual-agent"},
                {"kind": "claim", "id": "a", "owner": "new-agent"},
                {"kind": "release", "id": "a", "by": "actual-agent",
                 "expected_owner": "actual-agent", "force": True, "reason": "stale"},
            ]))

    def test_legacy_claim_released_without_rewriting(self):
        ts, n, _ = audit(log([
            node("a"),
            {"kind": "status", "id": "a", "agent": "worker-1", "status": "working", "note": ""},
            {"kind": "release", "id": "a", "by": "operator",
             "expected_owner": "worker-1", "force": True, "reason": "does not exist"},
        ]))
        self.assertEqual(n, 3)
        self.assertIsNone(ts["a"]["owner"])
        self.assertEqual(ts["a"]["status"], "planned")

    def test_placeholder_owner_rejected(self):
        with self.assertRaisesRegex(ValueError, "placeholder"):
            audit(log([node("a"), {"kind": "claim", "id": "a", "owner": "worker-1"}]))

    def test_unowned_task_cannot_fail(self):
        with self.assertRaisesRegex(ValueError, "only current task owner"):
            audit(log([node("a"), {"kind": "status", "id": "a",
                                   "status": "failed", "agent": "arbitrary-agent"}]))

    def test_claimed_task_can_fail_then_reopen(self):
        ts, _, _ = audit(log([
            node("a"),
            {"kind": "claim", "id": "a", "owner": "actual-agent"},
            {"kind": "status", "id": "a", "status": "failed", "agent": "actual-agent"},
            {"kind": "reopen", "id": "a", "by": "operator", "reason": "retry approved"},
        ]))
        self.assertEqual(ts["a"]["status"], "planned")
        self.assertIsNone(ts["a"]["owner"])

    def test_pass_requires_independent_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "evidence.json").write_text('{"result":"PASS"}')
            events = [
                node("a"),
                {"kind": "claim", "id": "a", "owner": "actual-agent"},
                {"kind": "status", "id": "a", "status": "passed", "agent": "actual-agent",
                 "verifier": "actual-agent", "evidence": "evidence.json"},
            ]
            with patch.object(dag, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "differ"):
                    audit(log(events))
                events[-1]["verifier"] = "independent-verifier"
                ts, _, _ = audit(log(events))
                self.assertEqual(ts["a"]["status"], "passed")
                self.assertIsNone(ts["a"]["owner"])

    @staticmethod
    def _git(root, *args):
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True, capture_output=True, text=True
        ).stdout.strip()

    def _init_git_ledger(self, root):
        self._git(root, "init", "-q", "-b", "main")
        self._git(root, "config", "user.name", "Ledger Test")
        self._git(root, "config", "user.email", "ledger-test@example.invalid")
        (root / "instructions.jsonl").write_text(log([node("initial")]))
        self._git(root, "add", "--", "instructions.jsonl")
        self._git(root, "commit", "-q", "-m", "initial ledger")

    def test_append_commits_each_event_without_other_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_git_ledger(root)
            # Other staged files may belong to a second agent.
            (root / "unrelated.txt").write_text("do not commit me\n")
            self._git(root, "add", "--", "unrelated.txt")
            with patch.object(dag, "ROOT", root), patch.object(dag, "LOG", root / "instructions.jsonl"):
                first = append(node("a"))
                second = append({"kind": "claim", "id": "a", "owner": "actual-agent"})
                third = append({"kind": "release", "id": "a", "by": "actual-agent",
                                "expected_owner": "actual-agent", "reason": "handoff"})
                tasks, n, head = audit(dag.LOG.read_text())
                self.assertEqual(n, 4)
                self.assertEqual(tasks["a"]["status"], "planned")
                self.assertIsNone(tasks["a"]["owner"])
                self.assertEqual(head, third["sha256"])
                self.assertNotEqual(first["git_commit"], second["git_commit"])
                self.assertNotEqual(second["git_commit"], third["git_commit"])
                self.assertEqual(self._git(root, "rev-list", "--count", "HEAD"), "4")
                committed = subprocess.check_output(["git", "-C", str(root),
                                                     "show", "HEAD:instructions.jsonl"])
                self.assertEqual(committed, dag.LOG.read_bytes())
                self.assertEqual(self._git(root, "diff", "--cached", "--name-only"), "unrelated.txt")
                listed = self._git(root, "ls-tree", "-r", "--name-only", "HEAD")
                self.assertNotIn("unrelated.txt", listed)

    def test_sync_uncommitted_append_recovers_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_git_ledger(root)
            with patch.object(dag, "ROOT", root), patch.object(dag, "LOG", root / "instructions.jsonl"):
                # Simulated crash after fsync and before Git commit.
                previous = audit(dag.LOG.read_text())[2]
                with dag.LOG.open("a") as stream:
                    stream.write(canon(row(2, previous, node("a"))) + "\n")
                with self.assertRaisesRegex(RuntimeError, "run 'python dag.py sync' first"):
                    append(node("b"))
                self.assertEqual(len(dag.LOG.read_text().splitlines()), 2)
                synced = dag.sync()
                self.assertEqual(synced["git_status"], "committed")
                append(node("b"))
                tasks, n, _ = audit(dag.LOG.read_text())
                self.assertEqual(n, 3)
                self.assertEqual(set(tasks), {"initial", "a", "b"})
                self.assertEqual(self._git(root, "rev-list", "--count", "HEAD"), "3")

    def test_concurrent_processes_serialize_appends_and_git_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_git_ledger(root)
            source_dir = str(Path(dag.__file__).resolve().parent)
            code = (
                "import sys; from pathlib import Path; "
                "sys.path.insert(0,sys.argv[1]); import dag; "
                "dag.ROOT=Path(sys.argv[2]); dag.LOG=dag.ROOT/'instructions.jsonl'; "
                "dag.append({'kind':'task','id':sys.argv[3],'priority':'P0',"
                "'title':'concurrency test','depends_on':[],'instruction':['run'],"
                "'acceptance':['verify'],'outputs':['receipt.json'],"
                "'avoid':['drift'],'status':'planned'})"
            )
            args = [sys.executable, "-c", code, source_dir, str(root)]
            a = subprocess.Popen(args + ["concurrent-a"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            b = subprocess.Popen(args + ["concurrent-b"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out_a, err_a = a.communicate(timeout=30)
            out_b, err_b = b.communicate(timeout=30)
            self.assertEqual(a.returncode, 0, err_a + out_a)
            self.assertEqual(b.returncode, 0, err_b + out_b)
            tasks, n, _ = audit((root / "instructions.jsonl").read_text())
            self.assertEqual(n, 3)
            self.assertEqual(set(tasks), {"initial", "concurrent-a", "concurrent-b"})
            self.assertEqual(self._git(root, "rev-list", "--count", "HEAD"), "3")
            committed = subprocess.check_output(["git", "-C", str(root),
                                                 "show", "HEAD:instructions.jsonl"])
            self.assertEqual(committed, (root / "instructions.jsonl").read_bytes())

    def test_failed_git_commit_is_recoverable_and_never_rewrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._init_git_ledger(root)
            hook = root / ".git/hooks/pre-commit"
            hook.write_text("#!/bin/sh\nexit 17\n")
            hook.chmod(0o755)
            with patch.object(dag, "ROOT", root), patch.object(dag, "LOG", root / "instructions.jsonl"):
                with self.assertRaisesRegex(RuntimeError, "APPENDED and fsynced"):
                    append(node("a"))
                self.assertEqual(len(dag.LOG.read_text().splitlines()), 2)
                self.assertEqual(self._git(root, "rev-list", "--count", "HEAD"), "1")
                hook.unlink()
                dag.sync()
                self.assertEqual(self._git(root, "rev-list", "--count", "HEAD"), "2")
                self.assertEqual(audit(dag.LOG.read_text())[1], 2)


if __name__ == "__main__":
    unittest.main()

