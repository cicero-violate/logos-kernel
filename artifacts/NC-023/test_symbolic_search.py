"""NC-023 adversarial tests. Dependencies: stdlib, symbolic_search, NC-020/NC-001."""
import json
import unittest
from pathlib import Path
from symbolic_search import ROOT,ref,root_rewrites,equivalence,negative_control,search,cost,intconst,op,digest

class TestSymbolicSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=[json.loads(x) for x in (ROOT/"artifacts/NC-020/equivalence_benchmark.jsonl").read_text().splitlines()]

    def find(self,fam,label="equivalent"):
        return next(c for c in self.cases if c["family"]==fam and c["label"]==label)

    def test_nested_factor_certified_and_cheaper(self):
        c=self.find("int_nested_factor")
        result=search(c,cap=32)
        self.assertEqual(result["proof"]["method"],"integer_polynomial_normal_form")
        self.assertLess(result["operations"]["dag_ops"],cost(c["lhs"])["dag_ops"])

    def test_absorption_finite_exhaustive(self):
        c=self.find("bv_absorption")
        r=search(c,cap=32)
        self.assertLess(r["operations"]["dag_ops"],cost(c["lhs"])["dag_ops"])
        self.assertEqual(r["proof"]["checked_assignments"],(1<<c["width"])**2)

    def test_double_shift_bv(self):
        c=self.find("bv_double_shift")
        r=search(c,cap=32)
        self.assertLess(r["operations"]["tree_nodes"],cost(c["lhs"])["tree_nodes"])
        self.assertEqual(r["proof"]["counterexample"],None)

    def test_wrong_integer_identity_rejected(self):
        c=self.find("int_nested_factor","non_equivalent")
        self.assertFalse(negative_control(c)["accepted"])

    def test_wrong_bv_identity_rejected(self):
        c=self.find("bv_absorption","non_equivalent")
        self.assertFalse(negative_control(c)["accepted"])

    def test_fp_not_admitted(self):
        c=self.find("fp64_reassociation","non_equivalent")
        self.assertEqual(negative_control(c)["reason"],"invalid/unsupported")

    def test_mixed_width_rejected(self):
        c=self.find("invalid_width","invalid_type")
        self.assertFalse(negative_control(c)["accepted"])

    def test_invalid_domain_and_wrong_equivalent_rejected(self):
        c=self.find("int_nested_factor")
        v=ref.alter(c["lhs"])
        self.assertFalse(equivalence(c["lhs"],v,c)[0])

    def test_rewrite_side_condition_precision(self):
        c=self.find("int_difference_squares")
        self.assertEqual(equivalence(c["lhs"],c["rhs"],c)[0],True)
        wrong=op("int.sub",c["lhs"],intconst(1))
        self.assertFalse(equivalence(c["lhs"],wrong,c)[0])

    def test_deterministic(self):
        c=self.find("int_nested_factor")
        self.assertEqual(search(c,cap=32),search(c,cap=32))

if __name__=="__main__":unittest.main()
