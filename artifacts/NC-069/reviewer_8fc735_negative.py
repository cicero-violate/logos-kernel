"""NC-069 independent heldout-label leakage negative control.
Dependencies: Python 3, torch, sibling train_eval.py and causal_benchmark.py, NC-001 ISA.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import train_eval as te

def run():
    fixed={"inputs":{"X":0,"Y":0,"R":0,"T":0},"factual":[0,0,0,0]}
    observed=[]
    for answer in ([1,1,1,1],[2,2,2,2]):
        samples=[{**fixed,"target":answer} for _ in range(4)]
        result=te.baselines({"heldout_combo":samples})["heldout_combo"]["majority"]
        observed.append({"answer":answer,"correct":result["correct"],"total":result["total"]})
        assert result["correct"]==4
    assert observed[0]["answer"]!=observed[1]["answer"]
    return {"task":"NC-069","control":"test-target-derived majority baseline",
      "verdict":"FAIL","same_inference_inputs":True,"mutated_heldout_labels":True,
      "observed":observed,"expected":"baseline majority estimated on training data only",
      "cause":"train_eval.baselines majority(samples) counts s[target] from the same held-out samples it scores",
      "scope":"synthetic adversarial control; other arms and causal model not tested"}

if __name__=="__main__": print(json.dumps(run(),indent=2,sort_keys=True))
