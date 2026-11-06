#!/usr/bin/env python3
"""NC-003 source-level and dataset leakage audit; NOT formal absence proof.

Dependencies: Python 3 stdlib; frozen source RLT/ISA Python modules and training
receipts; NC-000 inventory; no model execution or checkpoint modification.
"""
import ast
import hashlib
import json
import random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RLT=Path('/workspace/experiments/rlt_split_algorithms_ablation_v1')
TR=RLT/'neural_isa_transfer_v1'
SH=TR/'unified_shared_v1'
HERE=Path(__file__).resolve().parent
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as fd:
  for block in iter(lambda:fd.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def imports(path):
 tree=ast.parse(Path(path).read_text())
 names=set()
 for node in ast.walk(tree):
  if isinstance(node,ast.Import):names.update(n.name for n in node.names)
  if isinstance(node,ast.ImportFrom) and node.module:names.add(node.module)
 return sorted(names)
def source_rec(path):
 return {"source":str(path),"sha256":sha(path),
         "imports":imports(path),"line_count":len(path.read_text().splitlines())}

def main():
 paths={
 "shared_model":SH/"model.py",
 "shared_training":SH/"train.py",
 "source_math_encoder":TR/"rlt_micro.py",
 "source_math_encoder_v2":TR/"rlt_micro_v2.py",
 "qualified_runtime":SH/"runtime.py",
 "teacher_controller":TR/"vendor/transformer_controller/train_controller.py",
 "independent_host_used_historically":TR/"verify_hybrid_independent.py",
 "NC003_new_host":HERE/"replay_independent.py",
 }
 rec={name:source_rec(p) for name,p in paths.items()}
 checker=rec["NC003_new_host"]["imports"]
 forbidden=("verify_shared","verify_hybrid_independent","runtime","train_controller","train","train_causal")
 assert all(not (name==bad or name.endswith("."+bad)) for name in checker for bad in forbidden)
 trainer=paths["shared_training"].read_text()
 source=paths["shared_model"].read_text()
 teacher=paths["teacher_controller"].read_text()
 runtime=paths["qualified_runtime"].read_text()
 assert 'F.cross_entropy(lo,ops[ix])' in trainer
 assert 'F.cross_entropy(lp,nextp[ix])' in trainer
 assert 'F.cross_entropy(lh,halts[ix])' in trainer
 assert 'dataset(' in trainer
 assert 'def next_pc' in teacher
 assert 'for pc' in teacher or 'for pc,' in teacher
 assert 'self.micro.encoder' in source
 assert 'self.op_head(z)' in source and 'self.pc_head(z)' in source
 assert 'reference=a*b' in runtime
 assert 'record.update' in runtime and 'neural_prediction_correct' in runtime
 train_records={}
 for arm in ('jointly_tuned','frozen_shared'):
  train_records[arm]={}
  for seed in (11,19,29):
   item=json.loads((SH/'outputs'/f'seed{seed}'/f'{arm}_train.json').read_text())
   train_records[arm][str(seed)]={"direct_MUL_answer_targets_reported":item.get("math_MUL_answer_labels"),
      "train_state_examples":item.get("train_state_examples"),
      "heldout_state_examples":item.get("heldout_state_examples"),
      "updates":item.get("updates"),
      "source_record_sha256":sha(SH/'outputs'/f'seed{seed}'/f'{arm}_train.json')}
 r=random.Random(11052026+991122)
 random_pairs=[(r.randrange(256),r.randrange(256)) for _ in range(4096)]
 edge=(0,1,2,3,7,15,31,63,127,128,255)
 eight=random_pairs+[(a,b) for a in edge for b in edge]
 assert len(eight)==4217
 unique=len(set(eight))
 controls={
  "report_version":1,
  "classification":"SOURCE_AUDIT_AND_EVIDENCE_BOUNDARY_NOT_FORMAL_NONLEAKAGE_PROOF",
  "implementation":"NC003 host only calls frozen neural logits and updates registers using predicted instruction; no teacher opcode/PC is substituted",
  "source_records":rec,
  "dataset_counts":{"eightbit_evaluations":4217,"eightbit_unique_operand_pairs":unique,
       "eightbit_duplicate_evaluations":len(eight)-unique,
       "eightbit_full_domain":65536,"eightbit_coverage_percent":100*unique/65536,
       "same_operand_list_reused_across_all_seeds":True,
       "fourbit_all_256_pairs":True,
       "original_pretraining_operand_overlap_excluded":False,
       "new_2048_disjoint_set_not_proven_disjoint_from_original_pretraining":True},
  "training_provenance":train_records,
  "risk_table":[
   {"boundary":"neural arithmetic","observation":"Source arithmetic uses trained RLT attention logits with local bit features and predicted math-code embedding; 320/320 patterns supervised","leakage":"Complete local 6-bit truth table is in training; cannot call truth-table transfer novel reasoning","verdict":"KNOWN_TRAINED"},
   {"boundary":"neural controller","observation":"Predicts opcode, PC and HALT from shared attention over fixed ROM, register tokens and current PC","leakage":"Teacher-forced controller states and ROM exposed in training; no independent program invention","verdict":"KNOWN_TEACHER_SUPERVISED"},
   {"boundary":"host routing","observation":"Host dispatches predicted decoded opcode to fixed AND1/ADD/SHL/SHR local steps, wires X/Y/R/T registers, loops at most 75","leakage":"Python supplies program logic, bit indexing and register updates; host work is essential to multiplication","verdict":"HOST_PROVIDED_PROGRAM"},
   {"boundary":"program ROM","observation":"8 named fixed operations JZ_B,AND1,JZ_T,ADD,SHL,SHR,JMP,HALT provided in controller input","leakage":"No generated program and no learned arbitrary CFG","verdict":"FIXED_PROGRAM"},
   {"boundary":"reference oracle","observation":"Only computes product a*b AFTER neural rollout in verifier; compares halted flag and value","leakage":"Code inspection cannot establish lack of hidden training contamination; tests must be causally intervened","verdict":"POSTPREDICTION_ONLY_IN_NEW_VERIFIER"},
   {"boundary":"source original RLT","observation":"Pretrained seven-task Transformer contains multiplication supervision before transfer","leakage":"No pristine zero-MUL-training baseline, no source pretraining pair exclusion","verdict":"PRETRAINED_MUL_PRIOR"}
  ],
  "intervention_contract":{"control":"8 fixed operand probes",
     "flip_decoded_PC_prediction":"must change correctness; no oracle substituted",
     "zero_neural_arithmetic":"must degrade numeric correctness; no symbolic arithmetic fallback",
     "unmodified":"control accuracy should exceed both ablations"},
  "exhaustive_plan":{
      "status":"PROPOSED_NOT_EXECUTED",
      "domain":"unsigned 8-bit ordered operand pairs (a,b), total=256*256=65536",
      "enumeration":"index 0..65535 -> a=floor(index/256), b=index mod 256",
      "shards":64,"pairs_per_shard":1024,
      "seeds":[11,19,29],
      "arms":["jointly_tuned"],
      "verify_fields":["checkpoint/source SHA","predicted register result","neural halted","cycle_count <=75","predicted opcode/PC trace","postprediction oracle"],
      "counterexample_handling":"Persist every failure with operands, full trace and hash; any failed/unhalted case blocks whole-arm unconditional correctness and champion promotion",
      "branch_coverage_plan":"Instrument PC=0 JZ_B, PC=2 JZ_T with zero/nonzero values, explicit PC=6 loop, PC=7 HALT; include off-distribution register snapshots for all eight PCs and both branch directions, with PC/flag correctness and 75-step bounds",
      "teacher_forcing_leakage_plan":"Cross-check train/validation controller state hashes, exclude known train pairs, create unseen combinations of PC, registers, and branch directions; report pretraining overlap as UNKNOWN until recovered",
      "acceptance_scope":"Full 65,536 operand-pair enumeration would still not prove all arbitrary machine/register/CFG states, general math reasoning, or autonomous program synthesis"
   },
  "limitations":["Source AST/syntax checks and interventions only support stated boundaries, not a formal information-flow proof",
       "All 320 local cases were supervised during training",
       "The controller saw fixed teacher ROM and simulated program states",
       "Eight-bit 4217-case dataset has duplicates; report evaluations versus unique pairs",
       "Historical source RLT pretrained with MUL",
       "No off-distribution eight-bit exhaustive run done in NC003 unless documented separately",
       "No champion promotion permitted before separate independent verifier"]
 }
 path=HERE/'host_leakage_audit.json'
 path.write_text(json.dumps(controls,sort_keys=True,indent=2)+"\n")
 print(json.dumps({"task":"NC-003","source_files":len(rec),
    "eightbit_evaluations":len(eight),"unique_pairs":unique,
    "duplicates":len(eight)-unique,"known_training_issues":len(controls["risk_table"]),
    "exhaustive_status":controls["exhaustive_plan"]["status"],
    "sha256":sha(path)},indent=2))
if __name__=="__main__":main()
