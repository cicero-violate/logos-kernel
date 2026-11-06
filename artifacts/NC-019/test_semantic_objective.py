"""NC-019 reference/negative/control scorer tests.

Dependencies: Python3 stdlib, artifacts/NC-019/{reference_contract.py,
score_contract.py,benchmark_contract.json,semantic_objective.json,reference_receipt.json},
NC-001/canonical_isa.py and source/weight hashes.
"""
from __future__ import annotations
import copy
import hashlib
import itertools
import json
import random
import unittest
from pathlib import Path
import score_contract as s
import reference_contract as r
HERE=r.HERE

class SemanticContractTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.obj=json.loads((HERE/"semantic_objective.json").read_text())
  cls.contract=json.loads((HERE/"benchmark_contract.json").read_text())
  cls.receipt=json.loads((HERE/"reference_receipt.json").read_text())
 def test_reference_all_family_rows_have_positive_denominator(self):
  self.assertEqual(self.receipt["result"],"REFERENCE_CHECKS_PASS")
  self.assertEqual(len(self.contract["families"]),20)
  self.assertEqual((self.receipt["positive_families"],self.receipt["negative_families"]),(13,7))
  self.assertGreater(self.receipt["tested_cases_total"],80000)
  for row in self.receipt["rows"]:
   self.assertGreater(row["tested"],0)
   self.assertGreaterEqual(row["unique_valuations"],1)
   self.assertLessEqual(row["unique_valuations"],row["tested"])
 def test_finite_4bit_equivalences_and_counterexamples(self):
  for f in self.contract["families"]:
   four=[z for z in self.receipt["rows"] if z["family"]==f["id"] and z["width"]==4]
   self.assertEqual(len(four),len(f["supported_modes"]))
   for x in four:
    expected=16**len(f["variables"])
    self.assertEqual(x["tested"],expected)
    if f["expected_equivalence"]:
     self.assertEqual(x["equivalent_on_sample"],expected)
    else:
     self.assertLess(x["equivalent_on_sample"],expected)
     self.assertIsNotNone(x["counterexample"])
     z=x["counterexample"]
     self.assertNotEqual(z["lhs"],z["rhs"])
 def test_immutable_family_and_split_manifest(self):
  c=self.contract
  self.assertEqual(c["version"],r.VERSION)
  self.assertEqual(c["split_commitment_sha256"],
    r.fingerprint({"version":r.VERSION,"seeds":r.SEEDS,
     "widths":r.WIDTHS,"modes":r.MODES,
     "family_roster_sha256":c["family_roster_sha256"],
     "split_protocol":c["split_protocol"],"budgets":r.BUDGETS}))
  self.assertEqual(c["family_roster_sha256"],r.fingerprint(c["families"]))
  self.assertEqual(c["split_protocol"]["seeds"],[11,19,29])
  self.assertTrue(c["split_protocol"]["disjoint_structural_fingerprints"])
  self.assertFalse(c["split_protocol"]["raw_operand_disjointness"])
  self.assertEqual([len(c["split_protocol"]["splits"][z]) for z in r.ORDER],[5,2,13])
 def test_source_integrity_includes_champion_and_objective(self):
  dep=self.contract["dependencies"]
  self.assertGreaterEqual(len(dep),7)
  for key,record in dep.items():
   path=Path(record["path"])
   self.assertTrue(path.is_file())
   self.assertEqual(r.sha(path),record["sha256"],key)
   self.assertEqual(path.stat().st_size,record["bytes"])
  self.assertEqual(self.obj["id"],"NC-019")
  self.assertIn("NC-018",self.obj["precedence_rule"])
  self.assertIn("subproblem",self.obj["precedence_rule"])
 def test_equivalence_balanced_not_gamed_by_constant_predictor(self):
  cases=[{"case_id":x["id"],"truth":x["expected_equivalence"],"prediction":True}
         for x in self.contract["families"]]
  yes=s.equivalence_score(cases)
  self.assertEqual(yes["balanced_accuracy"],.5)
  labels={z["id"]:z["expected_equivalence"] for z in self.contract["families"]}
  registered=s.equivalence_score(cases,expected_labels=labels)
  self.assertTrue(registered["independent_label_registry_bound"])
  with self.assertRaises(s.InvalidScore):
   s.equivalence_score(cases[:-1],expected_labels=labels)
  self.assertEqual(yes["FP"],7)
  no=s.equivalence_score([{**x,"prediction":False} for x in cases])
  self.assertEqual(no["balanced_accuracy"],.5)
  self.assertEqual(no["FN"],13)
  none=s.equivalence_score([{**x,"prediction":None} for x in cases])
  self.assertEqual(none["balanced_accuracy"],0.0)
  self.assertEqual(none["correct"],0)
  reference=s.equivalence_score([{**x,"prediction":x["truth"]} for x in cases])
  self.assertEqual(reference["balanced_accuracy"],1.)
  self.assertEqual(reference["FP"],0)
 def test_equivalence_empty_label_and_bad_pred_rejected(self):
  with self.assertRaises(s.InvalidScore):
   s.equivalence_score([{"case_id":"p","truth":True,"prediction":True}])
  with self.assertRaises(s.InvalidScore):
   s.equivalence_score([{"case_id":"n","truth":False,"prediction":False}])
  cases=[{"case_id":"a","truth":True,"prediction":True},
         {"case_id":"b","truth":False,"prediction":False}]
  with self.assertRaises(s.InvalidScore):
   s.equivalence_score([cases[0],cases[0],cases[1]])
  with self.assertRaises(s.InvalidScore):
   s.equivalence_score([{**cases[0],"truth":1},cases[1]])
 def test_true_nonnequivalence_16bit_counterexample_without_training(self):
  bad=next(z for z in self.contract["families"] if z["id"]=="N_SAR_NOT_SHR")
  candidate={"a":32768}
  l=r.reference(bad["lhs"],candidate,16,"twos_complement")
  rr=r.reference(bad["rhs"],candidate,16,"twos_complement")
  self.assertNotEqual(l,rr)
  with self.assertRaises(r.isa.ISAError):
   r.reference(bad["lhs"],candidate,16,"unsigned")
 def test_test_heldout_structure_not_seen_in_training(self):
  shapes={split:{z["structural_fingerprint"] for z in self.contract["families"]
    if z["split"]==split} for split in r.ORDER}
  for a,b in itertools.combinations(r.ORDER,2):
   self.assertFalse(shapes[a]&shapes[b])
  self.assertGreaterEqual(len(shapes["test"]),8)
 def test_transfer_nonexecution_host_fallback_not_credit(self):
  heldout={"E_XOR_ASSOCIATIVITY","E_SUBTRACTION_REWRITE"}
  cases=[
   {"case_id":"a","family":"E_XOR_ASSOCIATIVITY","correct_value":9,"predicted_value":9,
      "halted":True,"execution_mode":"neural","postprediction_oracle_only":True},
   {"case_id":"b","family":"E_SUBTRACTION_REWRITE","correct_value":5,"predicted_value":5,
      "halted":True,"execution_mode":"symbolic","postprediction_oracle_only":True},
  ]
  result=s.transfer_score(cases,heldout)
  # Untrusted candidate claims yield ZERO credit by default.
  self.assertEqual((result["correct"],result["tested"]),(0,2))
  externally_observed=s.transfer_score(
   cases,heldout, trusted_execution_check=lambda item:item["case_id"]=="a",
   registered_case_ids={"a","b"})
  self.assertEqual((externally_observed["correct"],externally_observed["tested"]),(1,2))
  cases[0]["halted"]=False
  self.assertEqual(s.transfer_score(cases,heldout,
    trusted_execution_check=lambda item:item["case_id"]=="a",
    registered_case_ids={"a","b"})["correct"],0)
  with self.assertRaises(s.InvalidScore):
   s.transfer_score(cases,heldout,
     trusted_execution_check=lambda item:True,registered_case_ids={"a","b","c"})
  with self.assertRaises(s.InvalidScore):s.transfer_score(cases,{"UNSEEN"})
 def test_retention_requires_fixed_old_denominators(self):
  base={"ADD":{"correct":100,"tested":100,"case_set_sha256":"a"*64,"skill_contract_sha256":"c"*64},
        "XOR":{"correct":80,"tested":80,"case_set_sha256":"b"*64,"skill_contract_sha256":"d"*64}}
  current={"ADD":{**base["ADD"],"correct":99},"XOR":dict(base["XOR"])}
  score=s.retention_score(base,current)
  self.assertFalse(score["all_retained"])
  self.assertAlmostEqual(score["by_skill"]["ADD"]["forgetting"],.01)
  with self.assertRaises(s.InvalidScore):s.retention_score(base,{"ADD":current["ADD"]})
  with self.assertRaises(s.InvalidScore):s.retention_score(base,{
   "ADD":{**base["ADD"],"correct":99,"tested":101},"XOR":dict(base["XOR"])})
  changed_cases={"ADD":{**base["ADD"],"case_set_sha256":"f"*64},"XOR":dict(base["XOR"])}
  with self.assertRaises(s.InvalidScore):s.retention_score(base,changed_cases)
 def test_claims_demand_real_independent_proof_not_self_report(self):
  goals={"E_XOR_ASSOCIATIVITY","E_ADD_SUB_REASSOCIATION"}
  claim={"family":"E_XOR_ASSOCIATIVITY","proof_scope":"finite_exhaustive",
    "independently_verified":False,"verifier":"implementer",
    "receipt_sha256":"a"*64,"executable_result_correct":True,
    "source_mode":"neural_generated","reused_fixed_rom":False,"frozen_checkpoint_hash":"b"*64}
  score=s.verified_gain([claim],goals,set(),"implementer")
  self.assertEqual(score["verified_new_families"],0)
  claim["independently_verified"]=True
  claim["verifier"]="independent:real-runtime"
  # Even plausible verifier names and 64-char hashes are UNTRUSTED.
  score=s.verified_gain([claim],goals,set(),"implementer")
  self.assertEqual(score["verified_new_families"],0)
  # A trusted evaluator-only callback can authorize a simulated test fixture.
  # This is NOT an authentic model proof in production.
  score=s.verified_gain([claim],goals,set(),"implementer",
      trusted_proof_check=lambda item:item["family"]=="E_XOR_ASSOCIATIVITY")
  self.assertEqual(score["verified_new_families"],1)
  score=s.verified_gain([claim],goals,{"E_XOR_ASSOCIATIVITY"},"implementer")
  self.assertEqual(score["verified_new_families"],0)
 def test_fabricated_proof_metadata_never_credited(self):
  forged={"family":"E_XOR_ASSOCIATIVITY","proof_scope":"finite_exhaustive",
    "independently_verified":True,"verifier":"fictional:made-up-agent",
    "receipt_sha256":"a"*64,"executable_result_correct":True,
    "source_mode":"neural_generated","reused_fixed_rom":False,
    "frozen_checkpoint_hash":"b"*64}
  result=s.verified_gain([forged],{"E_XOR_ASSOCIATIVITY"},set(),"actual-owner")
  self.assertEqual(result["verified_new_families"],0)
  self.assertFalse(result["credited"])
  forged["frozen_checkpoint_hash"]="not-an-actual-checkpoint"
  result=s.verified_gain([forged],{"E_XOR_ASSOCIATIVITY"},set(),"actual-owner",
    trusted_proof_check=lambda item: True)
  self.assertEqual(result["verified_new_families"],0)

 def test_efficiency_nonexecution_and_missing_cost_rejected(self):
  cost={name:1 for name in s.COST_FIELDS}
  cost["total_elapsed_critical_path_seconds"]=4
  with self.assertRaises(s.InvalidScore):
   s.efficiency_score(2,cost)
  result=s.efficiency_score(2,cost,trusted_gain_check=lambda n:n==2)
  self.assertEqual(result["verified_families_per_second"],.5)
  self.assertEqual(result["energy_status"],"NOT_MEASURED")
  for change in ("total_elapsed_critical_path_seconds","proposals","verifier_calls"):
   missing=dict(cost);missing[change]=0
   with self.assertRaises(s.InvalidScore):
    s.efficiency_score(2,missing,trusted_gain_check=lambda n:n==2)
  missing=dict(cost);del missing["model_bytes"]
  with self.assertRaises(s.InvalidScore):s.efficiency_score(2,missing)
 def test_semantic_pareto_priority(self):
  a={"semantic_gates_passed":True,"gain":2,"retention":1,"total_cost":10}
  b={"semantic_gates_passed":True,"gain":2,"retention":1,"total_cost":20}
  self.assertTrue(s.dominates(a,b,["gain","retention"],["total_cost"]))
  self.assertFalse(s.dominates(b,a,["gain","retention"],["total_cost"]))
  a["semantic_gates_passed"]=False
  self.assertFalse(s.dominates(a,b,["gain","retention"],["total_cost"]))
  b["total_cost"]=None
  self.assertFalse(s.dominates(a,b,["gain","retention"],["total_cost"]))
 def test_seeds_make_reproducible_but_non_identical_samples(self):
  f=next(z for z in self.contract["families"] if z["id"]=="E_XOR_ASSOCIATIVITY")
  a=list(r.enumerate_valuations(f,8,"unsigned",11))
  b=list(r.enumerate_valuations(f,8,"unsigned",11))
  c=list(r.enumerate_valuations(f,8,"unsigned",19))
  self.assertEqual(a,b)
  self.assertNotEqual(a,c)
  self.assertEqual(len(a),256)
  self.assertEqual(len(c),256)

if __name__=="__main__":unittest.main()
