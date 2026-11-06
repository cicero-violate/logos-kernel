"""Generate NC-001 implementation test receipt; NOT independent-agent verification.
Dependencies: canonical_isa.py, test_isa.py, Python 3 stdlib, original NC-000
inventory.json and frozen RLT checkpoint for integrity.
"""
from pathlib import Path
import datetime
import hashlib
import io
import json
import random
import subprocess
import sys
import unittest
import canonical_isa as isa
import test_isa

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()

def main():
 suite=unittest.defaultTestLoader.loadTestsFromModule(test_isa)
 buffer=io.StringIO()
 result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(suite)
 tests={"tests":result.testsRun,"failures":len(result.failures),
        "errors":len(result.errors),"skips":len(result.skipped),
        "status":"PASS" if result.wasSuccessful() else "FAIL",
        "details":buffer.getvalue().splitlines()}
 contract=isa.spec()
 specpath=HERE/"isa_spec.json"
 assert json.loads(specpath.read_text())==contract,"spec/source disagreement"
 inv=json.loads((ROOT/"artifacts/NC-000/inventory.json").read_text())
 champion=Path(inv["roots"]["rlt_champion"])/"snapshots/current_unified_v3.pt"
 expected=inv["champion"]["selected_bundle_sha256"]
 actual=sha(champion)
 baseline={"checkpoint":str(champion),"expected_sha256":expected,"sha256":actual,
           "unchanged":actual==expected}
 rng=random.Random(20261105)
 p={"version":isa.VERSION,"width":16,"signedness":"unsigned",
    "program":[
     {"op":"JZ","src":"Y","target":7},
     {"op":"AND1","dst":"T","src":"Y"},
     {"op":"JZ","src":"T","target":4},
     {"op":"ADD","dst":"R","a":"R","b":"X"},
     {"op":"SHL","dst":"X","src":"X"},
     {"op":"SHR","dst":"Y","src":"Y"},
     {"op":"JMP","target":0},
     {"op":"HALT"}]}
 cases4=[(a,b) for a in range(16) for b in range(16)]
 cases8=[(rng.randrange(256),rng.randrange(256)) for _ in range(256)]
 benches={}
 for name,rows in (("fourbit_exhaustive",cases4),("eightbit_sampled",cases8)):
  failures=[]
  for a,b in rows:
   x=isa.reference(p,{"X":a,"Y":b,"R":0,"T":0},max_steps=75)
   if not x["halted"] or x["registers"]["R"]!=a*b:
    failures.append({"a":a,"b":b,"result":x["registers"]["R"],
                     "expected":a*b,"status":x["status"]})
  benches[name]={"tested":len(rows),"correct":len(rows)-len(failures),
                 "failed_cases":failures[:20],"result":"PASS" if not failures else "FAIL"}
 record={
  "task":"NC-001","schema":1,"result":"IMPLEMENTER_PASS" if result.wasSuccessful() and
       actual==expected and all(x["result"]=="PASS" for x in benches.values())
       else "IMPLEMENTER_FAIL",
  "independent_verification":"PENDING_DISTINCT_REAL_AGENT",
  "interpretation":"symbolic reference only; zero neural training or model prediction",
  "utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
  "source_dependencies":{
    "canonical_isa.py":sha(HERE/"canonical_isa.py"),
    "test_isa.py":sha(HERE/"test_isa.py"),
    "isa_spec.json":sha(specpath),
    "baseline_inventory.json":sha(ROOT/"artifacts/NC-000/inventory.json")},
  "run_command":"cd /workspace/experiments/logos-kernel && PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-001/make_conformance.py",
  "runtime":{"python":sys.version.split()[0],"requires":["Python 3 standard library"]},
  "validation":{"version":isa.VERSION,"opcodes":sorted(isa.FIELDS),
                 "opcode_count":len(isa.FIELDS),"registers":list(isa.REGS),
                 "width_range":[1,64],"signedness":["unsigned","twos_complement"],
                 "explicit_halt_required":True,"branch_target_checks":True},
  "tests":tests,"fixed_rom_reference_benchmarks":benches,
  "champion":baseline,
  "limits":["ISA contract is engineered, not learned from data.",
            "ROM-based multiplication does not establish neural program generation.",
            "4-bit evaluation is exhaustive only for 16x16 operands; 8-bit is sampled.",
            "AST perturbation test is symbolic wiring, not a neural causality ablation.",
            "Separate real agent must independently audit before NC-001 PASS.",
            "No model weights were trained, saved, altered, or promoted."]}
 output=HERE/"conformance.json"
 output.write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
 print(json.dumps({"result":record["result"],"test_count":tests["tests"],
                   "test_errors":tests["errors"],"test_failures":tests["failures"],
                   "symbolic_4bit":benches["fourbit_exhaustive"],
                   "symbolic_8bit":{k:v for k,v in benches["eightbit_sampled"].items() if k!="failed_cases"},
                   "champion_unchanged":baseline["unchanged"],
                   "conformance_sha256":sha(output)},indent=2))
 if record["result"]!="IMPLEMENTER_PASS": sys.exit(2)
if __name__=="__main__":
 main()
