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
  self.assertEqual(m["status"],"INDEPENDENT_BOUNDED_VERIFIED_NOT_PROMOTED")
  self.assertEqual(m["neural_circuits_count"],60)
  self.assertEqual(m["neural_local_qualified_rehearsal"],27)
  self.assertEqual(m["zero_qualified_rehearsal"],3)
  self.assertFalse(m["checkpoint_weights_modified"])
  self.assertEqual(m["independent_verifier_status"],"PASS")
  for k,digest in m["source_sha256"].items():
   self.assertEqual(v.sha(m["source_files"][k]),digest)

class IndependentReleaseTests(unittest.TestCase):
 def test_authenticated_verifier_and_two_stage_failure_history(self):
  import hashlib
  base=v.HERE
  release=json.loads((base/"independent_receipt.json").read_text())
  audit1=json.loads((base/"codex_verifier_final.json").read_text())
  audit2=json.loads((base/"codex_verifier_v2_final.json").read_text())
  trace1=[json.loads(z) for z in (base/"codex_verifier_trace.jsonl").read_text().splitlines() if z.strip()]
  trace2=[json.loads(z) for z in (base/"codex_verifier_v2_trace.jsonl").read_text().splitlines() if z.strip()]
  id1=next(x["thread_id"] for x in trace1 if x.get("type")=="thread.started")
  id2=next(x["thread_id"] for x in trace2 if x.get("type")=="thread.started")
  self.assertNotEqual(id1,id2)
  self.assertEqual(release["prior_codex_failed_verifier"],"codex-cli:"+id1)
  self.assertEqual(release["distinct_codex_verifier"],"codex-cli:"+id2)
  self.assertNotEqual(release["implementer_runtime_session"],"codex-cli:"+id2)
  self.assertEqual(audit1["verdict"],"FAIL")
  self.assertEqual(audit2["verdict"],"PASS")
  self.assertEqual(release["prior_fail_retained"]["omitted"],185)
  self.assertEqual(release["corrective_proof"]["preserved_failures"],487)
  self.assertEqual(release["corrective_proof"]["omitted"],0)
  self.assertEqual(release["implementation_audited_commit"],audit2["audited_commit"])
  for item in release["evidence"].values():
   path=v.ROOT/item["path"]
   self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),item["sha256"])

 def test_only_bounded_neural_rehearsal_circuits_accepted(self):
  import hashlib
  base=v.HERE
  accepted=json.loads((base/"accepted_library.json").read_text())
  manifest=json.loads((base/"library_manifest.json").read_text())
  source=[json.loads(z) for z in (base/"circuits.jsonl").read_text().splitlines() if z.strip()]
  receipts={z["circuit_id"]:z for z in rows("proof_receipts.jsonl")}
  eligible={z["id"]:z for z in source if z.get("arm")=="rehearsal" and z.get("qualified_bounded")}
  self.assertEqual(len(eligible),30)
  self.assertEqual(len(accepted["accepted_neural"]),30)
  self.assertEqual(accepted["excluded_neural_count"],30)
  self.assertFalse(accepted["champion_promoted"])
  self.assertEqual(accepted["state"],"BOUNDED_VERIFIED_NO_CHAMPION_PROMOTION")
  self.assertEqual({z["id"] for z in accepted["accepted_neural"]},set(eligible))
  self.assertTrue(all(z["arm"]=="rehearsal" and z["implementation_mode"]=="neural_inference"
                      for z in accepted["accepted_neural"]))
  self.assertEqual(manifest["accepted_neural_count"],30)
  self.assertEqual(manifest["excluded_neural_count"],30)
  self.assertEqual(manifest["independent_verification"]["accepted_library_sha256"],
      hashlib.sha256((base/"accepted_library.json").read_bytes()).hexdigest())
  for z in accepted["accepted_neural"]:
   self.assertEqual(z["checkpoint_sha256"],eligible[z["id"]]["checkpoint_sha256"])
   proof=receipts[z["id"]]
   self.assertEqual(proof["measurement"]["correct"],proof["measurement"]["tested"])
   self.assertEqual(proof["proof_status"],"FINITE_EXHAUSTIVE_OBSERVED")

 def test_full_negative_failure_sets_preserved(self):
  proof=rows("proof_receipts.jsonl")
  fails=[r for r in proof if r["operation"]!="ZERO16" and r["measurement"]["failure_count"]]
  self.assertEqual(len(fails),23)
  self.assertEqual(sum(z["measurement"]["failure_count"] for z in fails),487)
  self.assertEqual(sum(len(z["measurement"]["failure_examples"]) for z in fails),487)
  self.assertEqual(sorted({z["circuit_id"].split(":")[0] for z in fails}),["new_only"])

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
