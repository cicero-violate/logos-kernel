"""Isolation, provenance, heldout and negative-feedback controls for NC-028.

Dependencies: Python 3 stdlib, PyTorch, the local lexical_model.py,
train_lexical.py, novel_trial.py, frozen NC-006/NC-007 execution dependencies.
"""
import hashlib
import json
from pathlib import Path
import resource
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arena
import isolation
from lexical_model import ACTIONS

class LexicalProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ckpt = HERE/"lexical_controller.pt"
        cls.digest = hashlib.sha256(cls.ckpt.read_bytes()).hexdigest()

    @classmethod
    def worker(cls, goal="UNSEEN_XOR_DOUBLE_PLUS_Y", feedback=None, **extra):
        cp = isolation.run([str(HERE/"lexical_model.py")],
            json.dumps({"goal":goal,"checkpoint_sha256":cls.digest,
                        "feedback":feedback if feedback is not None else [],**extra}),
            timeout=35)
        return cp

    def test_no_structural_target_in_training(self):
        import torch
        data = torch.load(self.ckpt, map_location="cpu", weights_only=True)
        self.assertNotIn("UNSEEN_XOR_DOUBLE_PLUS_Y", data["training_goals"])
        self.assertEqual(data["training_count"], 11)
        self.assertEqual(data["supervised_token_rows"], 30)
        self.assertEqual(data["training_token_correct"], 30)
        self.assertEqual(data["source_receipt_sha256"],
             arena.hash_file(HERE.parent/"NC-007"/"composition_receipt.json"))

    def test_real_model_unseen_proposal_and_gradient_response(self):
        one=self.worker()
        self.assertEqual(one.returncode,0,one.stderr)
        p=json.loads(one.stdout.strip().splitlines()[-1])
        self.assertEqual(p["status"],"PROPOSED")
        self.assertTrue(p["model_generated"])
        self.assertTrue(p["structurally_unseen_goal"])
        self.assertTrue(2<=len(p["actions"])<=5 and p["actions"][-1]=="HALT")
        self.assertTrue(all(a in ACTIONS for a in p["actions"]))
        # Protocol-only synthetic numeric mismatch; benchmark uses actual
        # execution-derived feedback, never fabricated sample labels.
        cx={"pair_index":0,"x":1,"y":2,"expected":4,"observed":3}
        fb={"actions":p["actions"],"correct":0,"tested":8,"counterexample":cx}
        qcp=self.worker(feedback=[fb])
        self.assertEqual(qcp.returncode,0,qcp.stderr)
        q=json.loads(qcp.stdout.strip().splitlines()[-1])
        self.assertEqual(q["status"],"PROPOSED")
        self.assertNotEqual(q["actions"],p["actions"])
        self.assertNotEqual(q["weights_before"],q["weights_after"])
        self.assertEqual(q["online_gradient_updates"],12)
        self.assertTrue(q["feedback_uses_labels"])
        self.assertFalse(q["feedback_uses_heldout_labels"])
        cx2=dict(cx,expected=65535)
        vcp=self.worker(feedback=[dict(fb,counterexample=cx2)])
        self.assertEqual(vcp.returncode,0,vcp.stderr)
        v=json.loads(vcp.stdout.strip().splitlines()[-1])
        self.assertNotEqual(q["weights_after"],v["weights_after"])

    def test_malformed_and_leaky_feedback_fails_closed(self):
        self.assertNotEqual(self.worker(expected=[1,2]).returncode,0)
        self.assertNotEqual(self.worker(goal="DOUBLE").returncode,0)
        self.assertNotEqual(self.worker(feedback=[{"actions":["HALT"],"correct":0,"tested":8}]).returncode,0)
        self.assertNotEqual(self.worker(feedback=[{"actions":["ADD_XX","HALT"],"correct":8,"tested":8}]).returncode,0)
        self.assertNotEqual(self.worker(feedback=[{"actions":["ADD_XX","HALT"],"correct":0,"tested":8,"target_actions":["ADD_XX","HALT"]}]).returncode,0)
        cx={"pair_index":0,"x":1,"y":2,"expected":4,"observed":3}
        for bad in (dict(cx,observed=4),dict(cx,x=-1),dict(cx,pair_index=8),
                    dict(cx,unexpected=1),dict(cx,expected="4")):
            self.assertNotEqual(self.worker(feedback=[{"actions":["ADD_XX","HALT"],
                "correct":0,"tested":8,"counterexample":bad}]).returncode,0)

    def test_cpu_limit_overrides_child_and_rejects_malformed(self):
        cp=isolation.run(["-c","import resource;print(resource.getrlimit(resource.RLIMIT_CPU))"],
                         cpu_seconds=3,timeout=10)
        self.assertEqual(cp.returncode,0,cp.stderr)
        self.assertIn("(3, 4)",cp.stdout)
        for invalid in (0,25,True,1.5):
            with self.assertRaises(ValueError):
                isolation.run(["-c","print(0)"],cpu_seconds=invalid)

    def test_saved_trial_matches_separate_splits(self):
        data=json.loads((HERE/"novel_feedback_results.json").read_text())
        self.assertEqual(data["status"],"NO_DISCOVERY")
        self.assertEqual(data["new_discoveries"],0)
        self.assertFalse(data["train_label_leakage"])
        self.assertEqual(data["cpu_budget_per_arm"],12)
        self.assertEqual(data["wall_budget_per_arm"],20)
        self.assertNotEqual(data["feedback_pairs_sha256"],data["heldout_pairs_sha256"])
        self.assertEqual(data["checkpoint_sha256"],self.digest)
        self.assertEqual(data["champion_sha256_unchanged"],
                         arena.hash_file(arena.ROOT.parent/
                            "rlt_split_algorithms_ablation_v1/snapshots/current_unified_v3.pt"))
        self.assertEqual(data["arm_totals"]["lexical_neural"]["heldout_tested"],16)
        self.assertTrue(data["cpu_matched"],data["compute_ratios"])
        self.assertTrue(data["wall_matched"],data["compute_ratios"])
        self.assertEqual(set(data["arm_totals"]),
                         {"lexical_neural","lexical_frozen","fixed_host","random_host"})
        for arm,totals in data["arm_totals"].items():
            expected_attempts=2 if arm.startswith("lexical_") else 4
            self.assertEqual(totals["attempts"],expected_attempts)
            self.assertEqual(totals["heldout_tested"],8*expected_attempts)
            self.assertLess(totals["cpu_seconds"],12)
            self.assertLess(totals["wall_seconds"],20)
            tested=[r for r in data["attempts"] if r["arm"]==arm and r["status"]=="TESTED"]
            self.assertEqual(len(tested),expected_attempts)
            self.assertEqual(totals["heldout_correct"],
                             sum(r["heldout_correct"] for r in tested))
            self.assertTrue(all(r["model_generated"]==arm.startswith("lexical_") for r in tested))
            if arm=="lexical_neural":
                first=tested[0]
                self.assertTrue(first["feedback_counterexamples"])
                cx=first["feedback_counterexamples"][0]
                self.assertEqual(cx["expected"],arena.truth(
                    "UNSEEN_XOR_DOUBLE_PLUS_Y",cx["x"],cx["y"]))
                self.assertNotEqual(cx["expected"],cx["observed"])
                self.assertEqual(tested[1]["feedback_method"],
                                 "numeric-counterexample-rejection-gradient")
            if arm=="lexical_frozen":
                self.assertEqual(tested[0]["actions"],tested[1]["actions"])
                self.assertEqual(tested[1]["feedback_method"],"none")

if __name__=="__main__":
    unittest.main()
