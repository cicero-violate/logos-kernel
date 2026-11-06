"""NC-028 bounded model-reranking negative controls.

Dependencies: PyTorch, pinned NC-007 controller weights, sandbox isolation.py.
Cannot prove unseen-goal invention or feedback learning.
"""
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import isolation

def propose(goal, rejected=None):
    payload={"seed":11,"goal":goal}
    if rejected is not None:
        payload["rejected_actions"]=rejected
    return isolation.run([str(HERE/"propose_worker.py")],json.dumps(payload),timeout=35)

class ModelRerankTests(unittest.TestCase):
    def test_distinct_frozen_model_alternative(self):
        first=propose("DOUBLE")
        self.assertEqual(first.returncode,0,first.stderr)
        original=json.loads(first.stdout.strip().splitlines()[-1])
        self.assertEqual(original["status"],"PROPOSED")
        second=propose("DOUBLE",[original["actions"]])
        self.assertEqual(second.returncode,0,second.stderr)
        revised=json.loads(second.stdout.strip().splitlines()[-1])
        self.assertEqual(revised["status"],"PROPOSED")
        self.assertNotEqual(revised["actions"],original["actions"])
        self.assertEqual(revised["actions"][-1],"HALT")
        self.assertLessEqual(len(revised["actions"]),5)
        self.assertTrue(revised["model_generated"])
        self.assertFalse(revised["learned_feedback"])
        self.assertFalse(revised["structurally_unseen_goal"])
    def test_invalid_feedback_fails_closed(self):
        for bad in ([["HALT"]], [["PYTHON","HALT"]], [["ADD_XX","HALT","ADD_X1"]], "ADD_XX"):
            cp=propose("DOUBLE",bad)
            self.assertNotEqual(cp.returncode,0)
            self.assertIn("invalid rejected",cp.stderr)
    def test_saved_probe_has_no_false_discovery_claim(self):
        path=HERE/"feedback_probe.json"
        if not path.exists():
            self.skipTest("feedback probe not yet executed")
        record=json.loads(path.read_text())
        self.assertEqual(record["status"],"NO_DISCOVERY")
        self.assertEqual(record["verification"],"IMPLEMENTER_ONLY_NOT_INDEPENDENT")
        self.assertEqual(len(record["trials"]),3)
        for trial in record["trials"]:
            self.assertEqual(trial["novel_discoveries"],0)
            self.assertFalse(trial["learned_feedback"])
            for attempt in trial["attempts"]:
                self.assertNotIn("expected",attempt["proposal_request_fields"])
                self.assertNotIn("counterexamples",attempt["proposal_request_fields"])
        injected=record["trials"][2]
        self.assertTrue(injected["fault_injection"])
        self.assertEqual(injected["revisions"],1)
        self.assertEqual(injected["attempts"][0]["result"],"COUNTEREXAMPLE")
        self.assertEqual(injected["attempts"][1]["result"],"COUNTEREXAMPLE")

    def test_unseen_goal_still_refused(self):
        cp=propose("UNSEEN_XOR_DOUBLE_PLUS_Y",[["ADD_XX","HALT"]])
        self.assertEqual(cp.returncode,0,cp.stderr)
        response=json.loads(cp.stdout.strip().splitlines()[-1])
        self.assertEqual(response["status"],"UNSUPPORTED_GOAL")
        self.assertFalse(response["model_generated"])

if __name__=="__main__":unittest.main()
