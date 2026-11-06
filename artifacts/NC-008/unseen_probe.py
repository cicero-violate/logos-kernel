"""Zero-shot, no-update transfer test; not scientific task invention.

Dependencies: invent.py and its frozen PyTorch/NumPy/NC-007 assets,
policy_*.pt, proposals.jsonl.
"""
import json
import torch
import invent
from pathlib import Path

HERE=Path(__file__).resolve().parent
UNSEEN = {
    "OR_SHIFT":lambda x,y: (x | y) >> 1,
    "XOR_ADD_Y":lambda x,y: ((x ^ y) + y) & invent.MASK,
    "XOR_MASK_ONE":lambda x,y: ((x & 1) ^ y) & invent.MASK,
}
def main():
    torch.set_num_threads(1)
    records=[]
    for source in invent.TASKS:
        payload=torch.load(HERE/("policy_"+source.lower()+".pt"),map_location="cpu",weights_only=True)
        model=invent.SpecTransformer()
        model.load_state_dict(payload["state_dict"],strict=True)
        model.eval()
        for goal,truth in UNSEEN.items():
            pairs=invent.probes(43837+sum(map(ord,goal)),invent.PROBES)
            spec=invent.bit_features(invent.examples(truth,pairs))
            predicted=invent.proposal(model,spec,1,greedy=True)[0]
            validation=invent.probes(98372+sum(map(ord,goal)),256)
            validation=[p for p in validation if p not in pairs]
            actual=[invent.symbolic(predicted,x,y) for x,y in validation]
            correct=sum(p==truth(x,y) for p,(x,y) in zip(actual,validation))
            records.append({"source_policy":source,"unseen_goal":goal,"program":predicted,
                            "updates_on_unseen":0,"correct":correct,"tested":len(validation)})
    obj={"schema":invent.SCHEMA,"purpose":"zero-shot unseen-task negative control",
         "note":"Evaluated by independent host arithmetic only; successful results would require fresh frozen-neural verification; no oracle-supplied program",
         "results":records}
    (HERE/"unseen_probe.json").write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"zero_shot_successes":sum(x["correct"]==x["tested"] for x in records),
                      "trials":len(records),"total_correct":sum(x["correct"] for x in records),
                      "total_tested":sum(x["tested"] for x in records)}))
if __name__=="__main__":main()
