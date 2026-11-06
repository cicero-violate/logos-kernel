import json
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

    def test_append_only_claim_events_via_cli_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(dag, "ROOT", root), patch.object(dag, "LOG", root / "instructions.jsonl"):
                append(node("a"))
                append({"kind": "claim", "id": "a", "owner": "actual-agent"})
                append({"kind": "release", "id": "a", "by": "actual-agent",
                        "expected_owner": "actual-agent", "reason": "done"})
                ts, n, _ = audit(dag.LOG.read_text())
                self.assertEqual(n, 3)
                self.assertEqual(len(dag.LOG.read_text().splitlines()), 3)
                self.assertEqual(ts["a"]["status"], "planned")
                self.assertIsNone(ts["a"]["owner"])


if __name__ == "__main__":
    unittest.main()

