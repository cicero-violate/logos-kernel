"""NC006 bounded neural circuit registry tests.

Dependencies: Python 3 stdlib, PyTorch, artifacts/NC-006/verify_circuits.py,
NC-005/curriculum.py + stage06 checkpoints, NC000 inventory and
generated circuits.jsonl / proof_receipts.jsonl / library_manifest.json.
No mutation or training of neural checkpoints.
"""
import itertools
import json
import unittest
from pathlib import Path
import verify_circuits as v

def rows(name):
 p=v.HERE/name
 if not p.is_file():return None
 return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

class CircuitRegistryTests(unittest.TestCase):
 def test_exact_local_truth_reference_for_all_six_features(self):
  for op in v.OPS:
   patterns,features,reference=v.inputs(op)
   self.assertEqual(len(patterns),64)
   self.assertEqual(features.shape,(64,6))
   self.assertEqual(reference.shape,(64,2))
   self.assertEqual(len(set(patterns)),64)
   for i,(a,b,carry,prev,nxt,first) in enumerate(patterns):
    t=v.expected(op,a,b,carry,prev,nxt,first)
    self.assertEqual(list(t),reference[i].tolist())
    self.assertTrue(all(x in (0,1) for x in t))

 def test_subtraction_carry_is_no_borrow(self):
  self.assertEqual(v.expected("SUB",0,1,1,0,0,0),(1,0))
  self.assertEqual(v.expected("SUB",1,0,1,0,0,0),(1,1))
  self.assertEqual(v.expected("SUB",0,0,1,0,0,0),(0,1))
  self.assertEqual(v.expected("SUB",1,1,1,0,0,0),(0,1))

 def test_sar_preserves_sign_top_bit(self):
  self.assertEqual(v.expected("SAR",1,0,0,0,0,1),(1,0))
  self.assertEqual(v.expected("SAR",0,0,0,0,1,1),(0,0))
  self.assertEqual(v.expected("SAR",0,0,0,0,1,0),(1,0))
  self.assertEqual(v.expected("SHR",0,0,0,0,1,1),(1,0))

 def test_all_original_checkpoint_shas_pinned(self):
  manifest=json.loads((v.ROOT/"artifacts/NC-005/checkpoint_manifest.json").read_text())
  lookup={e["path"]:e for e in manifest["checkpoints"]}
  self.assertEqual(len(lookup),42)
  for seed in v.SEEDS:
   for arm in ("rehearsal","new_only"):
    path=v.c.CHECKPOINTS/f"{arm}_seed{seed}_stage06.pt"
    name=str(path.relative_to(v.ROOT))
    self.assertEqual(v.sha(path),lookup[name]["sha256"])

 def test_champion_snapshot_unchanged(self):
  manifest=json.loads((v.ROOT/"artifacts/NC-000/inventory.json").read_text())
  p=Path(manifest["roots"]["rlt_champion"])/"snapshots/current_unified_v3.pt"
  self.assertEqual(v.sha(p),manifest["champion"]["selected_bundle_sha256"])

 def test_registry_uses_exact_versioned_contract_with_excluded_controls(self):
  circuits=rows("circuits.jsonl")
  proofs=rows("proof_receipts.jsonl")
  if circuits is None or proofs is None:
   self.skipTest("no saved measurements yet; do not claim acceptance")
  self.assertEqual(len(circuits),61)
  self.assertEqual(len(proofs),60)
  ids={r["id"] for r in circuits}
  self.assertEqual(len(ids),61)
  pids={r["circuit_id"] for r in proofs}
  self.assertEqual(len(pids),60)
  self.assertEqual(pids,ids-{"symbolic:NC001:canonical_isa:v1"})
  for row in circuits:
   self.assertEqual(row["contract_version"],v.VERSION)
   if row["implementation_mode"]=="symbolic_reference":
    self.assertFalse(row["qualified_bounded"])
    self.assertEqual(row["checkpoint_sha256"],None)
   else:
    self.assertEqual(row["implementation_mode"],"neural_inference")
    self.assertTrue(row["checkpoint_sha256"])
    self.assertTrue(row["no_default_champion_promotion"])
 def test_registry_negative_arms_not_silently_promoted(self):
  circuits=rows("circuits.jsonl")
  if circuits is None:self.skipTest("measurements not complete")
  self.assertTrue(any(r["arm"]=="new_only" and
        r["library_status"]=="REJECTED_COUNTEREXAMPLE"
        for r in circuits if r["implementation_mode"]=="neural_inference"))
  qualified=[r for r in circuits if r.get("arm")=="rehearsal"]
  self.assertEqual(len(qualified),30)
  self.assertTrue(all(r["qualified_bounded"] for r in qualified))
  self.assertTrue(all(r["library_status"]=="CANDIDATE_AWAITING_DISTINCT_INDEPENDENT_VERIFICATION" for r in qualified))
 def test_receipt_full_denominators_and_failure_examples(self):
  proofs=rows("proof_receipts.jsonl")
  if proofs is None:self.skipTest("measurements not complete")
  for p in proofs:
   m=p["measurement"]
   if p["operation"]=="ZERO16":
    self.assertEqual(m["tested"],65536)
    self.assertEqual(m["positive_tested"],1)
    self.assertEqual(m["negative_tested"],65535)
    self.assertEqual(len(m["failed_cases"]),m["failure_count"])
   else:
    self.assertEqual(m["tested"],64)
    self.assertEqual(m["output_bit_tested"],128)
    self.assertEqual(m["failure_count"],m["tested"]-m["correct"])
    self.assertEqual(len(m["failure_examples"]),m["failure_count"],
      "every failing frozen neural input must be recorded, not top-k truncated")
    self.assertTrue(m["float_logit_finiteness"])
   self.assertTrue(p["no_fallback_or_synthetic_result"])
 def test_manifest_contract_and_original_preservation(self):
  p=v.HERE/"library_manifest.json"
  if not p.is_file():self.skipTest("measurements not complete")
  m=json.loads(p.read_text())
  self.assertEqual(m["task"],"NC-006")
  self.assertEqual(m["status"],"IMPLEMENTER_PROPOSED_NOT_INDEPENDENTLY_VERIFIED")
  self.assertEqual(m["neural_circuits_count"],60)
  self.assertEqual(m["neural_local_qualified_rehearsal"],27)
  self.assertEqual(m["zero_qualified_rehearsal"],3)
  self.assertFalse(m["checkpoint_weights_modified"])
  self.assertEqual(m["independent_verifier_status"],"PENDING")
  for k,digest in m["source_sha256"].items():
   self.assertEqual(v.sha(m["source_files"][k]),digest)

class BranchHostBoundaryTests(unittest.TestCase):
 def test_exhaustive_jz_jnz_neural_predicate_host_branch(self):
  p=v.HERE/"branch_conformance.json"
  self.assertTrue(p.is_file())
  result=json.loads(p.read_text())
  self.assertEqual(result["task"],"NC-006")
  self.assertEqual(result["execution_mode"],"neural_is_zero_predicate_plus_host_predicate_to_PC_update")
  self.assertFalse(result["learned_PC"])
  self.assertFalse(result["model_invention"])
  self.assertEqual(set(result["models"]),{"seed11","seed19","seed29"})
  for seed,record in result["models"].items():
   self.assertEqual(record["JZ"]["correct"],65536)
   self.assertEqual(record["JZ"]["tested"],65536)
   self.assertEqual(record["JNZ"]["correct"],65536)
   self.assertEqual(record["JNZ"]["tested"],65536)
   self.assertEqual(record["inverted_neural_zero_control"]["JZ_correct"],0)
   self.assertEqual(record["inverted_neural_zero_control"]["JNZ_correct"],0)
   self.assertFalse(record["JZ"]["failures"])
   self.assertEqual(record["JZ"]["branch_target"],"host")
   self.assertEqual(record["JNZ"]["branch_target"],"host")

if __name__=="__main__":unittest.main()
