"""NC-020 tests; dependencies: Python 3 stdlib, benchmark.py, NC-001/canonical_isa.py,
and the three generated NC-020 JSON outputs. No model or GPU."""
import collections
import copy
import hashlib
import json
import unittest
import benchmark as b

class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=[json.loads(l) for l in (b.HERE/"equivalence_benchmark.jsonl").read_text().splitlines()]
        cls.meta=json.loads((b.HERE/"splits_and_semantics.json").read_text())
        cls.audit=json.loads((b.HERE/"proof_audit.json").read_text())
    def test_reproducible_generation(self):
        regenerated=b.produce()
        self.assertEqual(regenerated,self.cases)
        self.assertEqual(b.metadata(regenerated),self.meta)
        self.assertEqual(b.audit(regenerated),self.audit)
    def test_integrity_hashes(self):
        self.assertEqual(self.meta["dataset_sha256"],b.sha([
            {k:v for k,v in c.items() if k!="record_sha256"} for c in self.cases]))
        for c in self.cases:
            self.assertEqual(c["record_sha256"],b.sha({k:v for k,v in c.items() if k!="record_sha256"}))
        self.assertEqual(self.audit["case_set_sha256"],b.sha(sorted(c["id"] for c in self.cases)))
        for name,path in (("generator",b.HERE/"benchmark.py"),
                          ("isa",b.ROOT/"artifacts/NC-001/canonical_isa.py"),
                          ("semantic_objective",b.ROOT/"artifacts/NC-019/semantic_objective.json")):
            self.assertEqual(self.meta["source_sha256"][name],hashlib.sha256(path.read_bytes()).hexdigest())
    def test_both_labels_in_each_split(self):
        for split in ("train","validation","test"):
            cases=[c for c in self.cases if c["split"]==split and c["label"]!="invalid_type"]
            self.assertTrue(any(c["label"]=="equivalent" for c in cases))
            self.assertTrue(any(c["label"]=="non_equivalent" for c in cases))
    def test_family_and_structural_separation(self):
        for attr in ("family","structural_fingerprint"):
            d=collections.defaultdict(set)
            for c in self.cases:d[c[attr]].add(c["split"])
            self.assertTrue(all(len(s)==1 for s in d.values()))
        self.assertEqual(len({c["id"] for c in self.cases}),len(self.cases))
    def test_structural_difficulty(self):
        depths={c["difficulty"]["depth"] for c in self.cases}
        widths={c["width"] for c in self.cases if c["domain"]=="bitvector"}
        self.assertGreaterEqual(len(depths),4)
        self.assertEqual(widths,{3,4,5})
        self.assertEqual({c["difficulty"]["wrapping_layers"] for c in self.cases},{0,1,2})
    def test_integer_global_proofs(self):
        positives=negatives=0
        for c in self.cases:
            if c["domain"]!="integer":continue
            a,z=b.normalize_int(c["lhs"]),b.normalize_int(c["rhs"])
            if c["label"]=="equivalent":
                self.assertEqual(a,z);positives+=1
                self.assertEqual(c["proof"]["left_sha256"],c["proof"]["right_sha256"])
            else:
                self.assertNotEqual(a,z);negatives+=1
                env=c["proof"]["counterexample"]
                self.assertNotEqual(b.eval_int(c["lhs"],env),b.eval_int(c["rhs"],env))
        self.assertGreater(positives,0);self.assertGreater(negatives,0)
    def test_bitvector_finite_exhaustive(self):
        matched=mismatched=0
        for c in self.cases:
            if c["domain"]!="bitvector" or c["label"]=="invalid_type":continue
            witness,total=b.brute_bv(c["lhs"],c["rhs"],c["width"],c["signedness"])
            self.assertEqual(total,c["proof"]["checked_assignments"])
            self.assertEqual(witness,c["proof"]["counterexample"])
            if c["label"]=="equivalent":
                self.assertIsNone(witness);matched+=1
            else:
                self.assertIsNotNone(witness);mismatched+=1
        self.assertGreater(matched,0);self.assertGreater(mismatched,0)
    def test_wrong_integer_rewrite_rejected(self):
        x,y=b.var("x"),b.var("y")
        self.assertNotEqual(b.normalize_int(b.add(x,y)),b.normalize_int(b.add(x,x)))
        self.assertNotEqual(b.normalize_int(b.mul(x,b.add(y,b.const(1)))),
                            b.normalize_int(b.add(b.mul(x,y),b.const(1))))
    def test_overflow_and_signed_shift_negative_controls(self):
        x=b.bvr("X",4,"unsigned")
        inc=b.bvo("add",x,b.isa.lit(1,4,"unsigned"),width=4,signedness="unsigned")
        self.assertEqual(b.eval_bv(inc,15,0,4,"unsigned"),0)
        self.assertNotEqual(b.eval_bv(inc,15,0,4,"unsigned"),16)
        signed=b.bvr("X",4,"twos_complement")
        sar=b.bvo("sar",signed,width=4,signedness="twos_complement")
        shr=b.bvo("shr",signed,width=4,signedness="twos_complement")
        self.assertNotEqual(b.eval_bv(sar,8,0,4,"twos_complement"),
                            b.eval_bv(shr,8,0,4,"twos_complement"))
    def test_floating_counterexamples(self):
        for c in self.cases:
            if c["family"] not in ("fp64_reassociation","fp64_rounding"):continue
            def ev(n):
                if n["kind"]=="fp64.const":return n["value"]
                return ev(n["args"][0])+ev(n["args"][1])
            self.assertNotEqual(ev(c["lhs"]),ev(c["rhs"]))
            self.assertEqual(ev(c["lhs"]),c["proof"]["lhs_value"])
            self.assertEqual(ev(c["rhs"]),c["proof"]["rhs_value"])
    def test_type_mismatch_rejected(self):
        for c in self.cases:
            if c["label"]!="invalid_type":continue
            b.isa.validate_ast(c["lhs"],c["width"],c["signedness"])
            with self.assertRaises(b.isa.ISAError):
                b.isa.validate_ast(c["rhs"],c["width"],c["signedness"])
    def test_audit_fail_closed_on_tampering(self):
        # The report routine must not silently certify changed or unregistered
        # inputs, even if the caller supplies internally self-consistent fields.
        def check(mutator):
            rows=copy.deepcopy(self.cases)
            mutator(rows)
            with self.assertRaisesRegex(ValueError,"NC020_UNREGISTERED_OR_MODIFIED"):
                b.audit(rows)
        check(lambda rows:rows[0].__setitem__("record_sha256","0"*64))
        check(lambda rows:rows[0].__setitem__("label","non_equivalent"))
        check(lambda rows:rows[0].__setitem__("split","test"))
        check(lambda rows:rows[0].__setitem__("id","forged"))
        check(lambda rows:rows[0]["proof"].__setitem__("left_sha256","deadbeef"))
        width_index=next(i for i,row in enumerate(self.cases) if row["domain"]=="bitvector")
        check(lambda rows:rows[width_index]["lhs"].__setitem__("width",99))
        negative_index=next(i for i,row in enumerate(self.cases)
                            if row["label"]=="non_equivalent" and row["domain"]=="bitvector")
        check(lambda rows:rows[negative_index]["proof"].__setitem__("counterexample",None))
    def test_audit_refuses_unregistered_dataset(self):
        with self.assertRaisesRegex(ValueError,"NC020_UNREGISTERED_OR_MODIFIED"):
            b.audit([])
    def test_scope_disclosed(self):
        self.assertTrue(self.meta["splits"]["no_model_trained"])
        self.assertEqual(self.meta["splits"]["training_preexposure_absence"],"NOT_ESTABLISHED")
        self.assertEqual(self.audit["proof_coverage"]["independent_verifier_status"],"NOT_YET_VERIFIED")
        self.assertEqual(len(self.cases),self.audit["total_cases"])

if __name__=="__main__":unittest.main()
