"""NC-024 tests: Python3, torch, NC-023 symbolic optimizer and NC-020 benchmark."""
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import search_policy as p


def sample():
    src=ROOT/"artifacts/NC-020/equivalence_benchmark.jsonl"
    for line in src.read_text().splitlines():
        c=json.loads(line)
        if c["label"]=="equivalent" and c["domain"]=="integer" and c["family"]=="int_add_zero":
            return c
    raise RuntimeError("no bounded control")


class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.case=sample()

    def test_features_do_not_read_target_program(self):
        c=copy.deepcopy(self.case)
        options=list(p.ss.candidates(c["lhs"]))
        self.assertTrue(options)
        a=p.feature(options[0][0],c["lhs"],options[0][1],c)
        c["rhs"]={"unknown":"no target exposed"}
        b=p.feature(options[0][0],c["lhs"],options[0][1],c)
        self.assertEqual(a,b)
        self.assertEqual(len(a),p.FEAT_DIM)

    def test_shared_recurrent_attention_outputs_candidate_logits(self):
        net=p.RecurrentRewriteRanker()
        self.assertGreater(sum(v.numel() for v in net.parameters()),1000)
        logits=net(torch.zeros(4,p.FEAT_DIM))
        self.assertEqual(tuple(logits.shape),(4,))
        self.assertTrue(torch.isfinite(logits).all())

    def test_matched_proof_budget_and_all_modes(self):
        trained=p.RecurrentRewriteRanker()
        for arm in ("neural","untrained","heuristic","random"):
            result=p.ranked_search(self.case,arm,policy=trained if arm in ("neural","untrained") else None,
                                   seed=11,budget=3,pool_cap=16)
            self.assertLessEqual(result["explored_proofs"],3)
            self.assertTrue(result["verified"])
            same,proof=p.ss.equivalence(self.case["lhs"],result["optimized"],self.case)
            self.assertTrue(same)
            self.assertEqual(proof["method"],result["proof"]["method"])
            self.assertEqual(len(result["trace"]),result["explored_proofs"])

    def test_invalid_rewrite_cannot_be_promoted(self):
        falseproof={"method":"test_rejection"}
        with patch.object(p.ss,"equivalence",side_effect=lambda source,child,case:
                          (p.ss.eq(source,child),falseproof)):
            result=p.ranked_search(self.case,"heuristic",seed=11,budget=3,pool_cap=16)
        self.assertEqual(result["optimized_sha256"],p.ss.digest(self.case["lhs"]))
        self.assertEqual(result["verified_proposals"],0)
        self.assertEqual(result["optimized_dag_ops"],result["source_dag_ops"])

    def test_fixed_false_control_is_not_equivalent(self):
        case=copy.deepcopy(self.case)
        case["rhs"]={"kind":"int.const","value":881}
        self.assertFalse(p.ss.negative_control(case)["accepted"])


if __name__=="__main__":
    unittest.main()
