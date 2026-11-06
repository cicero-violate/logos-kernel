#!/usr/bin/env python3
"""NC-019 preregistration of semantic-first benchmarks and scoring.

Dependencies: Python >=3.10 standard library, artifacts/NC-001/canonical_isa.py,
immutable NC-000/NC-004/NC-005 artifacts, and frozen RLT champion checkpoint.
No training, no pretrained evaluation, no neural/ROM substitutions.

Writes exactly:
 artifacts/NC-019/semantic_objective.json
 artifacts/NC-019/benchmark_contract.json
 artifacts/NC-019/reference_receipt.json
"""
from __future__ import annotations
import hashlib
import itertools
import json
import random
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"artifacts/NC-001"))
import canonical_isa as isa
VERSION="logos.semantic-objective/1"
SEEDS=(11,19,29)
WIDTHS=(4,8,16)
MODES=("unsigned","twos_complement")
VARS=("a","b","c")
REGISTER={"a":"X","b":"Y","c":"T"}
OP={"add","sub","and","or","xor","and1","shl","shr","sar"}
ORDER=["train","validation","test"]
BUDGETS={
 "per_model_seed":{"optimizer_updates":2250,"minibatch_size":256,"training_seeds":list(SEEDS),
                    "number_of_seeds":3,"effective_training_examples_are_logged":True,
                    "equal_sample_exposure_claim_allowed":False},
 "per_formula_family":{"proposal_evaluations_max":1024,"search_expansions_max":1024,
                       "verifier_calls_max":1024,"wall_clock_seconds_max":900,
                       "max_model_inference_calls":8192},
 "reference_test":{"width4":"all 16^arity assignments, all modes",
                   "width8":"256 deterministically sampled assignments per family/seed/mode",
                   "width16":"256 deterministically sampled assignments per family/seed/mode",
                   "seeds":list(SEEDS),"batch_repeat_handling":"retain duplicates and report unique valuations",
                   "no_model_test_access_before_freeze":True},
 "hardware_policy":"log real CPU/GPU model, peak allocated and RSS bytes, package versions; time limits are ceilings not claims of achieved speed",
}
def var(name):return {"var":name}
def const(value):return {"const":value}
def call(op,*args):return {"op":op,"args":list(args)}
a,b,c=(var(z) for z in VARS)
zero=const(0)
# Every family is a structural equivalence claim, NOT automatically a formal proof.
FAMILIES=(
 ("T_ADD_COMM","train",True,call("add",a,b),call("add",b,a),("a","b")),
 ("T_XOR_CANCEL","train",True,call("xor",call("xor",a,b),b),a,("a","b")),
 ("T_AND_IDEMPOTENCE","train",True,call("and",a,a),a,("a",)),
 ("T_SHIFT_DOUBLE","train",True,call("shl",a),call("add",a,a),("a",)),
 ("T_SUBTRACT_ZERO","train",True,call("sub",a,zero),a,("a",)),
 ("V_OR_ABSORPTION","validation",True,call("or",a,call("and",a,b)),a,("a","b")),
 ("V_SUB_CANCEL","validation",True,call("sub",call("add",a,b),b),a,("a","b")),
 ("E_DISTRIB_AND_OVER_OR","test",True,
  call("or",call("and",a,b),call("and",a,c)),
  call("and",a,call("or",b,c)),("a","b","c")),
 ("E_ADD_SUB_REASSOCIATION","test",True,
  call("add",a,call("sub",b,c)),
  call("sub",call("add",a,b),c),("a","b","c")),
 ("E_XOR_ASSOCIATIVITY","test",True,
  call("xor",a,call("xor",b,c)),
  call("xor",call("xor",a,b),c),("a","b","c")),
 ("E_SUBTRACTION_REWRITE","test",True,
  call("sub",call("sub",a,b),c),
  call("sub",a,call("add",b,c)),("a","b","c")),
 ("E_XOR_CHAIN_CANCEL","test",True,
  call("xor",call("xor",a,b),call("xor",b,c)),
  call("xor",a,c),("a","b","c")),
 ("E_OR_DISTRIB_AND","test",True,
  call("and",call("or",a,b),call("or",a,c)),
  call("or",a,call("and",b,c)),("a","b","c")),
 ("N_ADD_OR","test",False,call("add",a,b),call("or",a,b),("a","b")),
 ("N_SUB_REVERSE","test",False,call("sub",a,b),call("sub",b,a),("a","b")),
 ("N_SHIFT_ROUNDING","test",False,
  call("shr",call("add",a,b)),call("add",call("shr",a),call("shr",b)),("a","b")),
 ("N_XOR_OR","test",False,call("xor",a,b),call("or",a,b),("a","b")),
 ("N_INVALID_DISTRIBUTION","test",False,
  call("and",call("or",a,b),c),call("or",a,call("and",b,c)),("a","b","c")),
 ("N_SAR_NOT_SHR","test",False,call("sar",a),call("shr",a),("a",)),
 ("N_SUB_WRONG_GROUP","test",False,
  call("sub",a,call("sub",b,c)),
  call("sub",call("sub",a,b),c),("a","b","c")),
)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for chunk in iter(lambda:f.read(1<<20),b""):h.update(chunk)
 return h.hexdigest()
def canonical(value):
 return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
def fingerprint(value):return hashlib.sha256(canonical(value).encode()).hexdigest()
def domain_seed(identifier,split,width,mode,seed):
 material=canonical([VERSION,identifier,split,width,mode,seed]).encode()
 return int.from_bytes(hashlib.sha256(material).digest()[:8],"big")
def variables(expr):
 if type(expr) is not dict:raise ValueError("not a typed expression node")
 if "var" in expr:
  assert set(expr)=={"var"} and expr["var"] in VARS
  return {expr["var"]}
 if "const" in expr:
  assert set(expr)=={"const"} and type(expr["const"]) is int and expr["const"]>=0
  return set()
 assert set(expr)=={"op","args"} and expr["op"] in OP
 arity=2 if expr["op"] in ("add","sub","and","or","xor") else 1
 assert type(expr["args"]) is list and len(expr["args"])==arity
 result=set()
 for child in expr["args"]:result.update(variables(child))
 return result

def typed(expr,width,mode):
 if "var" in expr:
  return {"kind":"bv.read","name":REGISTER[expr["var"]],
    "width":width,"signedness":mode}
 if "const" in expr:
  return {"kind":"bv.const","value":expr["const"],
    "width":width,"signedness":mode}
 return {"kind":"bv."+expr["op"],"args":[typed(z,width,mode) for z in expr["args"]],
   "width":width,"signedness":mode}

def native_eval(expr,registers,width,mode):
 mask=(1<<width)-1
 if "var" in expr:return registers[expr["var"]]
 if "const" in expr:return expr["const"]&mask
 name=expr["op"]
 x=native_eval(expr["args"][0],registers,width,mode)
 y=native_eval(expr["args"][1],registers,width,mode) if len(expr["args"])==2 else 0
 if name=="add":return (x+y)&mask
 if name=="sub":return (x-y)&mask
 if name=="and":return x&y
 if name=="or":return x|y
 if name=="xor":return x^y
 if name=="and1":return x&1
 if name=="shl":return (x<<1)&mask
 if name=="shr":return x>>1
 if name=="sar":
  if mode!="twos_complement":raise ValueError("SAR requires signedness")
  return ((x>>1)|(x&(1<<(width-1))))&mask
 raise ValueError("unknown operator")

def reference(expr,vals,width,mode):
 ast=typed(expr,width,mode)
 isa.validate_ast(ast,width,mode)
 regs={"X":0,"Y":0,"R":0,"T":0}
 for key,val in vals.items():regs[REGISTER[key]]=val
 result=isa.eval_ast(ast,regs,width,mode)
 assert type(result) is int
 assert result==native_eval(expr,vals,width,mode), ("independent oracle mismatch",expr,vals)
 return result

def enumerate_valuations(fam,width,mode,seed):
 """Width4 exhaustive; widths8/16 fixed PRNG sample, record repeats."""
 names=fam["variables"]
 if width==4:
  yield from (dict(zip(names,assignment)) for assignment in itertools.product(range(16),repeat=len(names)))
 else:
  rng=random.Random(domain_seed(fam["id"],fam["split"],width,mode,seed))
  for _ in range(256):
   yield {key:rng.randrange(1<<width) for key in names}

def family_structural_signature(expr):
 if "var" in expr:return ["variable"]
 if "const" in expr:return ["constant",expr["const"]]
 return [expr["op"],*[family_structural_signature(a) for a in expr["args"]]]

def source_files():
 baserlt=Path("/workspace/experiments/rlt_split_algorithms_ablation_v1")
 files={
  "NC001_canonical_ISA":ROOT/"artifacts/NC-001/canonical_isa.py",
  "NC004_semantic_experiment":ROOT/"artifacts/NC-004/causal_semantics.py",
  "NC004_checkpoint_inventory":ROOT/"artifacts/NC-004/checkpoint.json",
  "NC005_training_report":ROOT/"artifacts/NC-005/training_report.json",
  "NC005_independent_receipt":ROOT/"artifacts/NC-005/independent_receipt.json",
  "NC000_source_inventory":ROOT/"artifacts/NC-000/inventory.json",
  "champion_weights":baserlt/"snapshots/current_unified_v3.pt",
 }
 inv=json.loads(files["NC000_source_inventory"].read_text())
 assert sha(files["champion_weights"])==inv["champion"]["selected_bundle_sha256"]
 return {key:{"path":str(value),"sha256":sha(value),"bytes":value.stat().st_size}
         for key,value in files.items()}

def metric_definitions():
 return {
  "semantic_equivalence":{
   "metric":"balanced_accuracy_with_family_macro_report",
   "inputs":"typed candidate classification of positive and negative AST pairs, same width/signedness",
   "positive":"ground-truth equivalent under declared bitvector semantics",
   "negative":"one exact typed counterexample under same semantics",
   "TP":"equivalent pairs correctly accepted","TN":"non-equivalent pairs correctly rejected",
   "FP":"non-equivalent pairs incorrectly accepted","FN":"equivalent pairs incorrectly rejected",
   "formula":"0.5 * [TP/(TP+FN) + TN/(TN+FP)]",
   "denominators":"TP+FN > 0 AND TN+FP > 0, otherwise invalid score",
   "require":"record TP,TN,FP,FN, abstentions, per-family numerators, proof scopes, counterexamples and externally supplied complete case-label registry",
   "nonexecution":"invalid; count as incorrect on that case, no abstention bonus",
  },
  "unseen_composition":{
   "metric":"exact_verified_output_accuracy_on_family_heldout",
   "formula":"number_of_exact_heldout_outputs_with_valid_type_and_termination / total_registered_heldout_cases",
   "denominators":"registered_heldout_cases > 0, exact expected case ID set must match; report by family, width, sign and seed",
   "split":"structurally held-out entire formula families, no access during training/model selection",
   "nonexecution":"0 correct; timeout, missing output or type mismatch counts as failed case",
   "external_execution_provenance":"candidate-supplied neural execution flags are untrusted; independent evaluator supplies trusted_execution_check callback, otherwise zero transfer credit",
   "not_allowed":"training-seen primitive truth rows cannot count as unseen composition",
  },
  "verified_capability_gain":{
   "metric":"newly certified distinct operators or rewrite families, not examples or cached programs",
   "formula":"count(new_versioned_heldout_families_with_independent_equivalence_proof_and_correct_executable_result AND absent_from_frozen_baseline) / count(preregistered_new_goal_families)",
   "denominators":"pre_registered_new_goal_families > 0",
   "baseline":"freeze code, checkpoint, family list and proof scope before training",
   "proof":"each gain needs typed contract, exact checkpoint/source hash, independent proof or explicit finite-exhaustive scope, counterexample search, non-regression",
   "rejected":"memorizing exact programs, source ROM reuse, self-reported discovery, sampled agreement labeled proof",
   "external_proof_authentication":"No candidate proof/verifier name/hash can earn gain by itself. A trusted independent verifier callback must inspect real proof, runtime identity, checkpoint and source; otherwise credit zero.",
  },
  "retention":{
   "metric":"qualified_previous_capability_replay_accuracy",
   "formula":"correct_and_verified_old_skill_cases_at_stage_t / frozen_registered_old_skill_cases",
   "denominators":"old_skill_cases > 0; frozen case_set_sha256 and skill_contract_sha256 must match before and after; every prior verified skill separately and every stage",
   "forgetting_formula":"max(0, accuracy_before - accuracy_after)",
   "qualifier":"old skill must have frozen pre-stage independent receipt with scope and checkpoint",
   "gate":"zero regression on exactly proven prior scope; otherwise report lost cases and no promotion",
  },
  "search_learning_efficiency":{
   "metric":"verified_discovery_per_fully_accounted_compute_and_pareto_front",
   "formula":"distinct_verified_new_families / total_wall_seconds, with numerator and denominator separately reported",
   "denominators":"total_wall_seconds > 0, proposals > 0, verifier_calls > 0; any positive gain numerator requires trusted_gain_check from independent evaluator",
   "cost_vector":["training_seconds","model_inference_seconds","proposal_seconds","search_seconds",
                  "verification_seconds","total_elapsed_critical_path_seconds",
                  "peak_cpu_ram_bytes","peak_gpu_ram_bytes","model_bytes",
                  "proposals","search_expansions","verifier_calls","total_energy_joules_if_measured"],
   "total_cost_identity":"training + inference + proposal + search + verification wall-component times may overlap; total_elapsed is independently timed, do not sum overlapping times",
   "dominance":"A dominates B only if A passes all semantic/retention gates, is no worse on all reported capability scores/cost metrics, and strictly better on at least one",
   "priority":"Never optimize cost by skipping execution, semantic checks, verification or training while calling results equal",
   "missing_energy":"null and labelled NOT_MEASURED; never treat missing as zero",
  }
 }
def objective():
 return {
  "schema":1,"id":"NC-019","version":VERSION,"preregistered":True,
  "research_priority":["semantic_correctness","unseen_composition","verified_capability_gain",
                       "retention","search_learning_efficiency"],
  "optimization":"hard-gated lexicographic correctness/verified transfer first; Pareto efficiency among passing models only",
  "scalar_meaning_score":"NONE; unrelated measured dimensions remain separate",
  "metric_definitions":metric_definitions(),
  "promotion_hard_gates":[
   "semantic equivalence evaluation with nonempty TP+FN and TN+FP and all negative controls disclosed",
   "no false equivalence accepted in independently certified exhaustive/proof scope",
   "at least one structurally unseen test family independently checked before calling transfer",
   "all previously independently qualified operations retained at their original proof scopes",
   "new capability credited only when nontrivial new family is absent from frozen baseline and certified independently",
   "all training/inference/proposal/search/verification compute and memory costs recorded, no hidden oracle/host substitution",
   "three matched seeds reported individually with seed-specific failure cases; do not hide regressions in averages",
   "independent actual verifier identity, exact source/checkpoint hashes and rollback plan required",
   "reversible champion promotion only after downstream dedicated release task; NC019 itself never promotes",
  ],
  "anti_metrics":[
    "training or test recall of primitive truth-table rows is NOT unseen understanding",
    "classifying equivalent ASTs without executing valid predictions is NOT composed behavior",
    "a constant yes/no predictor is rejected using both semantic labels",
    "a syntactic match or raw string equality is NOT general semantic equivalence",
    "searching more candidates or caching prior solutions is NOT verified capability gain",
    "a small parameter count or 0 seconds due to no execution is NOT an efficiency improvement",
    "meaning-channel gradients/rotation are causal pathway evidence, not a universal scalar measure of understanding",
    "source RLT checkpoint contains multiplication training; reuse is not zero-shot discovery",
  ],
  "predecessor":"NC-018",
  "precedence_rule":"NC-019 semantic-first research correctness, unseen verified transfer and retention dominate NC-018 compute-minimization; NC-018 is retained as an execution-cost subproblem and its original task/ledger remains untouched.",
  "downstream":["NC-020","NC-021","NC-022","NC-023","NC-024","NC-025","NC-026","NC-027"],
  "model_and_rom_limit":"NC019 defines evaluation, does not establish learned program generation or fuse host ROM with Transformer",
 }
def build_contract():
 records=[]
 for name,split,equiv,left,right,names in FAMILIES:
  supported=list(MODES) if name!="N_SAR_NOT_SHR" else ["twos_complement"]
  assert set(variables(left)|variables(right))==set(names),(name,"vars")
  for mode in supported:
   for width in WIDTHS:
    isa.validate_ast(typed(left,width,mode),width,mode)
    isa.validate_ast(typed(right,width,mode),width,mode)
  sig=fingerprint([family_structural_signature(left),family_structural_signature(right)])
  records.append({
   "id":name,"split":split,"expected_equivalence":equiv,"lhs":left,"rhs":right,
   "variables":list(names),"supported_modes":supported,
   "structural_fingerprint":sig,"label_rule":"reference finite exhaustive at width4; wider sampled observations are not proofs",
   "negative_control":not equiv,
   "negative_witness_required":not equiv,
  })
 ids=[z["id"] for z in records]
 assert len(ids)==len(set(ids))
 splits={kind:sorted([z["id"] for z in records if z["split"]==kind]) for kind in ORDER}
 assert all(splits.values())
 sigs=[x["structural_fingerprint"] for x in records]
 assert len(sigs)==len(set(sigs)),"distinct heldout formulas have identical skeletons"
 # The benchmark is a preregistered contract. NC020 will construct the larger challenge suite.
 doc={
   "schema":1,"task":"NC-019","version":VERSION,"pre_registered":True,
   "mathematical_semantics":{
     "domain":"fixed-width bitvectors over raw unsigned encodings",
     "widths":list(WIDTHS),"signedness_modes":list(MODES),
     "overflow":"all ADD/SUB/SHL wrap modulo 2^width; operations AND/OR/XOR bitwise",
     "SHR":"logical right shift",
     "SAR":"sign-preserving arithmetic right shift, two's-complement only",
     "cast_policy":"no implicit type changes, width mixing, boolean-as-int or rounding extrapolation",
     "typed_AST_contract":"NC001 canonical_isa.validate_ast; use only supported bv.const/read/add/sub/and/or/xor/and1/shl/shr/sar",
     "variable_mapping":REGISTER,"values":"raw bitpatterns in 0 .. 2^width-1",
     "integers_not_real_numbers":"integer unbounded or float identities are never assumed to apply to wraparound bitvectors",
     "proof_strength":"4bit exhaustive finite enumeration only; 8/16bit sampled tests are evidence but not full mathematical proof",
   },
   "budget_ceiling_not_observed_metrics":BUDGETS,
   "split_protocol":{
      "train":"only named train families and their generated inputs available to model optimizer",
      "validation":"named validation families available for selection; never update model using validation targets",
      "test":"named test families and negative controls hidden from optimizer/hyperparameter search until frozen checkpoint commit",
      "disjoint_families":True,"disjoint_structural_fingerprints":True,
      "raw_operand_disjointness":False,
      "reason_raw_values_overlap":"4bit exhaustive finite spaces necessarily include repeated raw values; no assertion of old NC004/NC005 pretraining-disjointness",
      "canonical_case_id":"sha256(version,family_id,split,width,mode,seed,assignment_index)",
      "rng":"Python random.Random seed from first 64 bits of sha256(canonical_json([version,id,split,width,mode,seed]))",
      "determinism":"width4 enumerate cartesian 0..15; width8/16 sample with replacement 256 per family/seed/mode",
      "splits":splits,"seeds":list(SEEDS),"source_hash_pin":"See immutable source_hashes in this document",
      "labels":"equivalent pairs=positive, inequivalent pairs=negative with a concrete witness",
      "test_immutability":"This contract must be committed before NC020 models/datasets or candidate optimization; changes require new version and new append-only task",
   },
   "families":records,
   "negative_controls":{
      "always_yes":{"expected":"FAIL on all negative examples; cannot produce balanced accuracy=1"},
      "always_no":{"expected":"FAIL on all positive examples; cannot produce balanced accuracy=1"},
      "string_equality":{"expected":"FAIL on syntactically distinct equivalent pairs"},
      "no_execution":{"expected":"0 correct on executed-output tests; no wall-time=0 efficiency gain"},
      "semantic_domain_mismatch":{"expected":"reject SAR with unsigned, reject width-mismatch and nonfinite/wrong AST"},
      "training_family_lookup":{"expected":"fails transfer credit on strictly heldout structural families"},
      "raw_optimizer_time_only":{"expected":"fails total-system cost accounting if search or verification omitted"},
      "fabricated_proof":{"expected":"no verified capability gain without independently checked proof"},
      "numerical_counterexamples":["N_ADD_OR","N_SUB_REVERSE","N_SHIFT_ROUNDING","N_XOR_OR",
          "N_INVALID_DISTRIBUTION","N_SAR_NOT_SHR","N_SUB_WRONG_GROUP"],
   },
   "dependencies":source_files(),
   "contract_type":"pre-registered candidate model evaluation, not learned evidence",
   "future_NC020":"Will supply separate typed AST equivalence benchmark with proof-status receipts and independent verifier",
 }
 doc["family_roster_sha256"]=fingerprint(records)
 doc["split_commitment_sha256"]=fingerprint({"version":VERSION,"seeds":SEEDS,
     "widths":WIDTHS,"modes":MODES,"family_roster_sha256":doc["family_roster_sha256"],
     "split_protocol":doc["split_protocol"],"budgets":BUDGETS})
 return doc

def audit():
 obj=objective()
 contract=build_contract()
 negatives=[z for z in contract["families"] if not z["expected_equivalence"]]
 positives=[z for z in contract["families"] if z["expected_equivalence"]]
 assert len(negatives)>=3 and len(positives)>=3
 auditrows=[]
 for family in contract["families"]:
  for width in WIDTHS:
   for mode in family["supported_modes"]:
    allvals=list(enumerate_valuations(family,width,mode,11))
    assert allvals
    # First audit seed; no trained model can use these results as a transfer claim.
    successes=0;counter=None
    for val in allvals:
     result=reference(family["lhs"],val,width,mode)==reference(family["rhs"],val,width,mode)
     successes+=int(result)
     if not result and counter is None:counter={"variables":val,"lhs":reference(family["lhs"],val,width,mode),
          "rhs":reference(family["rhs"],val,width,mode)}
    if family["expected_equivalence"]:
     assert successes==len(allvals),(family["id"],width,mode,"expected true")
    elif width==4:
     assert counter is not None,(family["id"],"negative has no 4bit witness")
    auditrows.append({"family":family["id"],"split":family["split"],"width":width,
     "signedness":mode,"seed":11 if width>4 else None,
     "tested":len(allvals),"equivalent_on_sample":successes,
     "non_equivalent_on_sample":len(allvals)-successes,
     "unique_valuations":len({tuple(sorted(v.items())) for v in allvals}),
     "counterexample":counter,
     "finite_exhaustive_domain":width==4,
     "observed_scope":"FINITE_EXHAUSTIVE" if width==4 else "SAMPLED_ONLY"})
 seen={r["family"] for r in auditrows}
 assert seen=={r["id"] for r in contract["families"]}
 # A fixed small labeled fixture, not fabricated model accuracy.
 fixtures=[]
 for label,group in ((True,positives),(False,negatives)):
  for f in group:
   row=next(z for z in auditrows if z["family"]==f["id"] and z["width"]==4)
   fixtures.append({"family":f["id"],"label":label,
     "positive_finite_4bit_domain_verified":bool(label and row["non_equivalent_on_sample"]==0),
     "negative_witness":row["counterexample"]})
 P=sum(int(x["label"]) for x in fixtures)
 N=len(fixtures)-P
 assert P>0 and N>0
 # Baselines do not execute any neural model.
 baselines={
  "always_yes":{"TP":P,"FN":0,"TN":0,"FP":N},
  "always_no":{"TP":0,"FN":P,"TN":N,"FP":0},
  "correct_reference_label_fixture":{"TP":P,"FN":0,"TN":N,"FP":0}
 }
 for z in baselines.values():
  assert z["TP"]+z["FN"]==P and z["TN"]+z["FP"]==N
  z["balanced_accuracy"]=(z["TP"]/P+z["TN"]/N)/2
 receipt={
  "task":"NC-019","result":"REFERENCE_CHECKS_PASS",
  "reference_interpreters":"NC001 typed AST evaluator and independent pure Python bitvector evaluator; checked equal for each tested value",
  "seed_scope":"first seed 11 for wider sample; all 4bit valuations enumerated independent of RNG; later NC020 uses all 3 seeds",
  "rows":auditrows,"positive_families":P,"negative_families":N,
  "tested_cases_total":sum(z["tested"] for z in auditrows),
  "finite_exhaustive_rows":sum(z["finite_exhaustive_domain"] for z in auditrows),
  "width8_and_16_rows_sampled":sum(not z["finite_exhaustive_domain"] for z in auditrows),
  "fixture_baselines":baselines,
  "source_hashes":contract["dependencies"],
  "split_commitment_sha256":contract["split_commitment_sha256"],
  "no_neural_model_evaluated":True,
  "no_automatic_global_proof_claim":True,
  "no_champion_promotion":True,
 }
 return obj,contract,receipt

def main():
 obj,contract,receipt=audit()
 HERE.mkdir(parents=True,exist_ok=True)
 for name,data in (("semantic_objective.json",obj),("benchmark_contract.json",contract),
                   ("reference_receipt.json",receipt)):
  (HERE/name).write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")
 print(json.dumps({"task":"NC-019","reference":receipt["result"],
   "positive_families":receipt["positive_families"],
   "negative_families":receipt["negative_families"],
   "reference_cases":receipt["tested_cases_total"],
   "exhaustive_finite_rows":receipt["finite_exhaustive_rows"],
   "wider_sampled_rows":receipt["width8_and_16_rows_sampled"],
   "baseline_accuracy":{k:v["balanced_accuracy"] for k,v in receipt["fixture_baselines"].items()},
   "source_hashes":len(contract["dependencies"]),
   "split_commitment_sha256":contract["split_commitment_sha256"]},indent=2))
if __name__=="__main__":main()
