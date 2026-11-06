"""NC-019 strict benchmark scorer, not model inference or an understanding measure.

Dependencies: Python >= 3.10 standard library, benchmark_contract.json,
semantic_objective.json. All results require declared nonempty denominators;
abstention, nonexecution and missing cost metadata are never free successes.
"""
from __future__ import annotations
import math

class InvalidScore(ValueError):
 pass

def valid_count(v,what):
 if type(v) is not int or v < 0:raise InvalidScore(f"{what}: nonnegative integer needed")
 return v
def nonempty(n,what):
 valid_count(n,what)
 if n == 0:raise InvalidScore(f"{what}: empty denominator")
 return n
def safe_float(x,what):
 if type(x) not in (int,float) or not math.isfinite(x) or x < 0:
  raise InvalidScore(f"{what}: must be finite nonnegative number")
 return float(x)

def equivalence_score(cases, *, expected_labels=None):
 """cases: unique preregistered cases labelled True/False, prediction bool/None.

 None, timeout, missing or invalid output is abstention and scored wrong.
 Strict bool prevents Python bool-is-int aliasing. Keep TP/TN/FP/FN plus abstentions.
 """
 if expected_labels is not None:
  if type(expected_labels) is not dict or not expected_labels:
   raise InvalidScore("nonempty externally registered case labels required")
  if any(type(k) is not str or type(v) is not bool for k,v in expected_labels.items()):
   raise InvalidScore("external expected labels must map unique IDs to booleans")
 ids=set();P=N=TP=TN=FP=FN=AP=AN=0
 for item in cases:
  if type(item) is not dict or not {"case_id","truth","prediction"} <= set(item):
   raise InvalidScore("case_id/truth/prediction required")
  case_id=item["case_id"]
  if type(case_id) is not str or not case_id or case_id in ids:
   raise InvalidScore("unique nonempty case IDs required")
  ids.add(case_id)
  truth=item["truth"]
  if expected_labels is not None and (case_id not in expected_labels or
      expected_labels[case_id] is not truth):
   raise InvalidScore("unknown or candidate-modified oracle truth")
  if type(truth) is not bool:raise InvalidScore("truth must be explicit boolean")
  pred=item["prediction"]
  if truth:
   P+=1
   if type(pred) is bool and pred:TP+=1
   elif type(pred) is bool:FN+=1
   else:AP+=1
  else:
   N+=1
   if type(pred) is bool and not pred:TN+=1
   elif type(pred) is bool:FP+=1
   else:AN+=1
 if expected_labels is not None and ids!=set(expected_labels):
  raise InvalidScore("missing required registered equivalence cases")
 nonempty(P,"equivalence positive cases")
 nonempty(N,"equivalence negative cases")
 assert TP+FN+AP==P and TN+FP+AN==N
 return {"TP":TP,"TN":TN,"FP":FP,"FN":FN,"abstain_positive":AP,
   "abstain_negative":AN,"positive_denominator":P,"negative_denominator":N,
   "balanced_accuracy":(TP/P+TN/N)/2,
   "positive_recall":TP/P,"negative_recall":TN/N,
   "correct":TP+TN,"tested":P+N,
   "all_negative_false_equivalences_rejected":FP==0,
   "scope":"registered typed pairs only, reported by family, width, signedness and seed",
   "independent_label_registry_bound":expected_labels is not None,
   "scientific_promotion_requires_external_labels":True}

def transfer_score(cases,heldout_family_ids, *, trusted_execution_check=None, registered_case_ids=None):
 """Complete executable outputs on structurally unseen families; no empty tasks."""
 if trusted_execution_check is not None and not callable(trusted_execution_check):
  raise InvalidScore("external execution checker must be callable")
 if type(heldout_family_ids) not in (set,frozenset) or not heldout_family_ids:
  raise InvalidScore("heldout family set must be nonempty")
 seen=set();correct=0;total=0;per={}
 for x in cases:
  if type(x) is not dict or not {"case_id","family","correct_value","predicted_value",
                                  "halted","execution_mode"} <= set(x):
   raise InvalidScore("typed executable transfer case required")
  fam=x["family"];cid=x["case_id"]
  if fam not in heldout_family_ids:raise InvalidScore("transfer case not in heldout family")
  if type(cid) is not str or not cid or cid in seen:raise InvalidScore("duplicate transfer case")
  seen.add(cid)
  expected=x["correct_value"];got=x["predicted_value"]
  if type(expected) is not int:raise InvalidScore("oracle expected bitvector must be integer")
  valid=type(got) is int and type(x["halted"]) is bool and x["halted"] is True
  # Symbolic host/oracle may be independently correct, but MUST NOT earn neural transfer.
  neural=x["execution_mode"]=="neural" and x.get("postprediction_oracle_only") is True
  # Candidate-provided neural/postprediction flags are UNTRUSTED self-reports.
  # Only a caller-controlled independent evaluation check can credit them.
  independently_observed=bool(trusted_execution_check is not None and
                              registered_case_ids is not None and
                              trusted_execution_check(x) is True)
  match=valid and neural and independently_observed and got==expected
  correct+=int(match);total+=1
  slot=per.setdefault(fam,{"correct":0,"tested":0})
  slot["correct"]+=int(match);slot["tested"]+=1
 if registered_case_ids is not None:
  if not isinstance(registered_case_ids,(set,frozenset)) or not registered_case_ids:
   raise InvalidScore("external heldout case registry must be nonempty")
  if seen!=registered_case_ids:raise InvalidScore("missing expected heldout test cases")
 nonempty(total,"unseen transfer")
 missing=set(heldout_family_ids)-set(per)
 if missing:raise InvalidScore(f"missing preregistered heldout families: {sorted(missing)}")
 return {"correct":correct,"tested":total,"accuracy":correct/total,
   "by_family":per,"nonexecution_count_as_failed":True,
   "host_oracle_answer_not_credited_as_neural":True,
   "independent_execution_trace_verification_required":True,
   "full_registered_dataset_bound":registered_case_ids is not None}

def verified_gain(claims,preregistered_goals,frozen_baseline_families,implementer_identity,
                  *, trusted_proof_check=None):
 if trusted_proof_check is not None and not callable(trusted_proof_check):
  raise InvalidScore("trusted proof verifier must be an evaluator-only callable")
 if not isinstance(preregistered_goals,(set,frozenset)) or not preregistered_goals:
  raise InvalidScore("pre-registered goal families required")
 if not isinstance(frozen_baseline_families,(set,frozenset)):
  raise InvalidScore("baseline must be recorded")
 if type(implementer_identity) is not str or not implementer_identity:
  raise InvalidScore("real implementer identity required")
 scored=set();rejected={}
 for item in claims:
  if not isinstance(item,dict) or type(item.get("family")) is not str:
   raise InvalidScore("family claim required")
  family=item["family"]
  if family not in preregistered_goals:
   raise InvalidScore("claim outside declared goals")
  if family in scored or family in rejected:raise InvalidScore("duplicate gain claim")
  # Invariant: a candidate's proof flag, verifier name or SHA string is NEVER
  # sufficient. No trusted evaluator callback means zero credited gain.
  independently_authenticated=bool(trusted_proof_check is not None and
                                    trusted_proof_check(item) is True)
  frozen_hash=item.get("frozen_checkpoint_hash")
  disqualify=(
   not independently_authenticated or
   family in frozen_baseline_families or
   item.get("proof_scope") not in ("finite_exhaustive","smt_unsat_certificate","proof_checked") or
   item.get("independently_verified") is not True or
   type(item.get("verifier")) is not str or not item["verifier"] or
   item["verifier"]==implementer_identity or
   type(item.get("receipt_sha256")) is not str or len(item["receipt_sha256"])!=64 or
   any(k not in "0123456789abcdef" for k in item["receipt_sha256"]) or
   item.get("executable_result_correct") is not True or
   item.get("source_mode")!="neural_generated" or
   item.get("reused_fixed_rom") is not False or
   type(frozen_hash) is not str or len(frozen_hash)!=64 or
   any(k not in "0123456789abcdef" for k in frozen_hash)
  )
  if disqualify:rejected[family]="absent real independent proof/new neural capability/or source baseline"
  else:scored.add(family)
 return {"verified_new_families":len(scored),"preregistered_goal_families":len(preregistered_goals),
  "verified_gain_fraction":len(scored)/len(preregistered_goals),
  "credited":sorted(scored),"rejected":rejected,
  "requires_external_independent_receipt_authentication":True,
  "candidate_self_reports_never_credited_without_trusted_evaluator":True,
  "assumption":"trusted_proof_check must be supplied by an actual external verifier that checks proof trace, runtime provenance, full checkpoint hashes and result; this scorer cannot authenticate identities from string fields"}

def retention_score(before,after):
 """Strict all-previous-qualified-case replay: no silent missing cases."""
 if type(before) is not dict or type(after) is not dict:
  raise InvalidScore("skill checkpoint baselines required")
 if not before or before.keys()!=after.keys():
  raise InvalidScore("all prior verified skills must be retested")
 out={}
 for skill,b in before.items():
  a=after[skill]
  for label,row in (("before",b),("after",a)):
   if type(row) is not dict:raise InvalidScore("invalid skill record")
   total=nonempty(row.get("tested"),f"{skill} {label} denominator")
   hits=valid_count(row.get("correct"),f"{skill} {label} correct")
   if hits>total:raise InvalidScore("more correct than tested")
  for field in ("case_set_sha256","skill_contract_sha256"):
   for item in (a,b):
    val=item.get(field)
    if type(val) is not str or len(val)!=64 or any(z not in "0123456789abcdef" for z in val):
     raise InvalidScore("retention requires frozen exact case-set/contract hashes")
   if a[field]!=b[field]:raise InvalidScore("retention cases or contract changed")
  if b["tested"]!=a["tested"]:
   raise InvalidScore("retention denominator changed")
  old=b["correct"]/b["tested"];new=a["correct"]/a["tested"]
  out[skill]={"before_accuracy":old,"after_accuracy":new,
     "forgetting":max(0.0,old-new),"regression":a["correct"]<b["correct"],
     "lost_net_correct_cases":max(0,b["correct"]-a["correct"]),
     "denominator":b["tested"]}
 return {"by_skill":out,"all_retained":all(not z["regression"] for z in out.values()),
  "qualified_prior_skills":len(out)}

COST_FIELDS=("training_seconds","model_inference_seconds","proposal_seconds","search_seconds",
 "verification_seconds","total_elapsed_critical_path_seconds","peak_cpu_ram_bytes",
 "peak_gpu_ram_bytes","model_bytes","proposals","search_expansions","verifier_calls")

def efficiency_score(new_verified_families,cost, *, trusted_gain_check=None):
 valid_count(new_verified_families,"verified gain numerator")
 if trusted_gain_check is not None and not callable(trusted_gain_check):
  raise InvalidScore("gain audit callback must be callable")
 if new_verified_families > 0 and (trusted_gain_check is None or
      trusted_gain_check(new_verified_families) is not True):
  raise InvalidScore("verified gain numerator has no trusted external attestation")
 if type(cost) is not dict:raise InvalidScore("cost vector required")
 for field in COST_FIELDS:
  if field not in cost:raise InvalidScore(f"missing cost: {field}")
  safe_float(cost[field],field)
 elapsed=cost["total_elapsed_critical_path_seconds"]
 if elapsed<=0 or cost["verifier_calls"]<=0 or cost["proposals"]<=0:
  raise InvalidScore("cannot earn cheap discovery by nonexecution/nonverification")
 energy=cost.get("total_energy_joules_if_measured")
 if energy is not None:safe_float(energy,"energy")
 return {"verified_families":new_verified_families,"total_elapsed_seconds":elapsed,
   "verified_families_per_second":new_verified_families/elapsed,
   "cost":{k:cost[k] for k in COST_FIELDS},
   "total_energy_joules_if_measured":energy,
   "energy_status":"NOT_MEASURED" if energy is None else "MEASURED",
   "nonoverlapping_wall_time_not_assumed":True,
   "not_a_substitute_for_semantic_passing_gates":True,
   "efficiency_is_arithmetic_only":"gain callback binds externally checked numerator; every cost field still requires independent measurement audit",
   "gain_numerator_externally_authenticated":new_verified_families==0 or trusted_gain_check is not None}

def dominates(a,b,required_scores,required_costs):
 """Semantic-gated Pareto: must not claim dominance with absent dimensions."""
 if not a.get("semantic_gates_passed") or not b.get("semantic_gates_passed"):
  return False
 keys=set(required_scores)|set(required_costs)
 if not keys:return False
 better=False
 for k in required_scores:
  if k not in a or k not in b or a[k] is None or b[k] is None:
   return False
  av=safe_float(a[k],k);bv=safe_float(b[k],k)
  if av<bv:return False
  better |= av>bv
 for k in required_costs:
  if k not in a or k not in b or a[k] is None or b[k] is None:
   return False
  av=safe_float(a[k],k);bv=safe_float(b[k],k)
  if av>bv:return False
  better |= av<bv
 return bool(better)
