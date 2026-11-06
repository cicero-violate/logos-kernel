"""Only execute a bounded typed proposal on NC-006 frozen accepted primitives.

Dependencies: NC-007/compose.py, NC-005/curriculum.py and checkpoints,
NC-006 frozen accepted library, NC-001/canonical_isa.py, PyTorch, NumPy.
No target program or task answer enters this child.
"""
import importlib.util, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
src=HERE.parent/"NC-007"/"compose.py"
spec=importlib.util.spec_from_file_location("nc007_compose",src)
compose=importlib.util.module_from_spec(spec)
spec.loader.exec_module(compose)
def main():
    d=json.load(sys.stdin)
    if set(d)!={"seed","actions","pairs"}: raise ValueError("invalid execution input schema")
    seed=d["seed"]
    tokens=d["actions"]
    pairs=d["pairs"]
    if type(seed) is not int or seed not in (11,19,29): raise ValueError("invalid seed")
    if not isinstance(tokens,list) or not 2<=len(tokens)<=5 or tokens[-1]!="HALT" or tokens.count("HALT")!=1:
        raise ValueError("HALT bound rejected")
    if any(type(t) is not str or t not in compose.ACTIONS for t in tokens):
        raise ValueError("unregistered action")
    if not isinstance(pairs,list) or not 1<=len(pairs)<=16 or any(
        not isinstance(p,list) or len(p)!=2 or any(type(v) is not int or v<0 or v>65535 for v in p)
        for p in pairs):
        raise ValueError("operand bound rejected")
    compose.torch.set_num_threads(1)
    model,reg,info=compose.load_frozen(seed)
    outputs,proofs,elapsed=compose.execute_neural(model,tokens,pairs,reg)
    print(json.dumps({"outputs":outputs,"proof_circuit_ids":proofs,
                      "exec_ns":elapsed,"frozen_sha256":info["checkpoint_sha256"]}))
if __name__=="__main__":main()
