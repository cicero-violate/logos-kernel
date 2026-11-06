"""NC-004 checks: depends on causal_semantics.py + six matched checkpoints."""
from __future__ import annotations
import json
import sys
import unittest
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import causal_semantics as ex

class CausalSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.manifest=json.loads((HERE/"checkpoint.json").read_text())
        cls.report=json.loads((HERE/"semantic_receipt.json").read_text())
        cls.models={(arm,seed):ex.load(seed,arm) for arm in ex.ARMS for seed in ex.SEEDS}
        cls.x,cls.y,cls.ids=ex.truth_table()

    def test_source_champion_unchanged(self):
        inv=json.loads(ex.INV.read_text())
        self.assertEqual(ex.sha(ex.CHAMPION),inv["champion"]["selected_bundle_sha256"])
        self.assertEqual(self.manifest["original_champion_checkpoint_sha256"],ex.sha(ex.CHAMPION))
        self.assertFalse(self.manifest["promoted"])

    def test_pinned_code_files(self):
        self.assertEqual(self.manifest["source_files"][str(ex.BASELINE_SOURCE)],ex.sha(ex.BASELINE_SOURCE))
        self.assertEqual(self.manifest["source_files"][str(ex.ISA_SOURCE)],ex.sha(ex.ISA_SOURCE))

    def test_all_three_seeds_five_op_local_truth(self):
        self.assertEqual(len(self.x),320)
        for arm in ex.ARMS:
            for seed in ex.SEEDS:
                with self.subTest(arm=arm,seed=seed):
                    z=ex.metrics(self.models[(arm,seed)],self.x,self.y,self.ids)
                    self.assertEqual(z["numeric_correct"],320)
                    self.assertEqual(z["math_labels_correct"],320)

    def test_gradient_has_causal_path_only_for_coupled_arm(self):
        for seed in ex.SEEDS:
            g=ex.numeric_gradient(self.models[("coupled",seed)],self.x,self.y)
            n=ex.numeric_gradient(self.models[("detached",seed)],self.x,self.y)
            self.assertTrue(g["meaning_head_in_numeric_graph"])
            self.assertGreater(g["meaning_weight_gradient_l1"],1e-9)
            self.assertFalse(n["meaning_head_in_numeric_graph"])
            self.assertEqual(n["meaning_weight_gradient_l1"],0.)

    def test_intervention_changes_numeric_only_on_coupled_arm(self):
        for seed in ex.SEEDS:
            g=ex.semantic_intervention(self.models[("coupled",seed)],"coupled",self.x,self.y)
            n=ex.semantic_intervention(self.models[("detached",seed)],"detached",self.x,self.y)
            self.assertEqual(g["original_correct"],320)
            self.assertLess(g["intervened_correct"],320)
            self.assertGreater(g["outputs_changed"],0)
            self.assertEqual(n["intervened_correct"],320)
            self.assertEqual(n["outputs_changed"],0)

    def test_canonical_typed_math_AST_labels(self):
        for arm in ex.ARMS:
            for seed in ex.SEEDS:
                model=self.models[(arm,seed)]
                for op in ex.OPS:
                    predicted,ast=ex.predicted_ast(model,op)
                    self.assertEqual(predicted,op)
                    ex.isa.validate_ast(ast,16,"unsigned")
                    expected=ex.isa.lower({
                        "version":ex.isa.VERSION,
                        "width":16,"signedness":"unsigned",
                        "program":[ex.instruction(op),{"op":"HALT"}]
                    },0)["write"]["value"]
                    self.assertEqual(ast,expected)

    def test_no_learning_of_untrained_canonical_operations(self):
        self.assertEqual(set(ex.OPS),{"ADD","SHL","SHR","AND1","OR"})
        self.assertEqual(len(ex.isa.FIELDS),15)
        self.assertNotEqual(set(ex.OPS),set(ex.isa.FIELDS))
        self.assertEqual(self.report["scope"],"five_of_fifteen_canonical_ISA_primitives")

    def test_compositional_results_not_silently_assumed_correct(self):
        for arm in ex.ARMS:
            for seed in ex.SEEDS:
                row=self.report["models"][arm+"_seed"+str(seed)]
                self.assertEqual(len(row["compositions"]),6)
                for name,result in row["compositions"].items():
                    self.assertGreater(result["tested"],0)
                    self.assertGreaterEqual(result["correct"],0)
                    self.assertLessEqual(result["correct"],result["tested"])
                    self.assertLessEqual(len(result["failures"]),6)

    def test_checkpoint_includes_trainable_state_and_no_promotion(self):
        for row in self.manifest["models"]:
            obj=torch.load(HERE/row["checkpoint"],weights_only=True,map_location="cpu")
            self.assertEqual(ex.sha(HERE/row["checkpoint"]),row["checkpoint_sha256"])
            self.assertEqual(obj["trained_updates"],ex.STEPS)
            self.assertEqual(obj["source_sha256"],ex.sha(ex.BASELINE_SOURCE))
            self.assertIn("optimizer",obj)
            self.assertGreater(len(obj["optimizer"]["state"]),0)
            self.assertIn("seed_rng_python",obj)
            self.assertIn("seed_rng_numpy",obj)
            self.assertIn("seed_rng_torch",obj)
            self.assertIn("no_promotion",obj["checkpoint_kind"])

    def test_wrong_canonical_AST_rejected(self):
        try:
            ex.isa.validate_ast({"kind":"bv.mul","width":16,
                                 "signedness":"unsigned","args":[]},16,"unsigned")
        except ex.isa.ISAError:
            pass
        else:
            self.fail("Unknown AST node should fail validation")

if __name__=="__main__":
    unittest.main()
