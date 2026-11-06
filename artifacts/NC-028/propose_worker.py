"""Frozen NC-007 controller inference. No target recipe or oracle is loaded.

Dependencies: PyTorch, checkpoint controller_seed{11,19,29}.pt, Python stdlib.
Input: JSON {seed, goal, rejected_actions?}; output: bounded frozen-model proposals.

Rejected-action feedback excludes prior complete candidates; it does NOT
update neural weights, add unknown goal representations, or establish discovery.
"""
import json, sys
import torch
from torch import nn
from pathlib import Path
HERE=Path(__file__).resolve().parent
CACHED=HERE.parent/"NC-007"/"checkpoints"
GOALS=("DOUBLE","INC","PARITY","XOR_DOUBLE","AND_PLUS_X","SUB_XOR_X",
       "OR_SHIFT_RIGHT","ADD_XOR_Y","XOR_ADD_X","DOUBLE_PLUS_Y","TRIPLE")
ACTIONS=("ADD_XX","ADD_X1","AND1_X","XOR_XY","SHL_R","AND_XY","ADD_RX",
         "SUB_XY","XOR_RX","OR_XY","SHR_R","ADD_XY","XOR_RY","ADD_RY","HALT")
DIM=32
STEPS=5
class FrozenController(nn.Module):
    def __init__(self):
        super().__init__()
        self.goal=nn.Embedding(len(GOALS),DIM)
        self.previous=nn.Embedding(len(ACTIONS)+1,DIM)
        self.position=nn.Embedding(STEPS,DIM)
        self.decoder=nn.Sequential(nn.Linear(3*DIM,96),nn.Tanh(),
                                   nn.Linear(96,96),nn.Tanh(),nn.Linear(96,len(ACTIONS)))
    def forward(self,g,p,prev):
        return self.decoder(torch.cat((self.goal(g),self.position(p),self.previous(prev)),dim=-1))

def next_distinct(model, goal, rejected):
    """Bounded CPU beam of frozen-model tokens; no oracle/target access."""
    rejected = {tuple(a) for a in rejected}
    beam = [(0.0, (), len(ACTIONS))]
    completed = []
    with torch.no_grad():
        for pos in range(STEPS):
            following = []
            for score, seq, prev in beam:
                logits = model(torch.tensor([GOALS.index(goal)]),
                               torch.tensor([pos]), torch.tensor([prev]))
                logp = torch.log_softmax(logits[0], dim=-1)
                choices = torch.topk(logp, k=4).indices.tolist()
                for idx in choices:
                    candidate = seq + (ACTIONS[idx],)
                    val = score + float(logp[idx].item())
                    if ACTIONS[idx] == "HALT":
                        if len(candidate) >= 2 and candidate not in rejected:
                            completed.append((val, candidate))
                    elif pos + 1 < STEPS:
                        following.append((val, candidate, idx))
            following.sort(key=lambda item: (-item[0], item[1]))
            beam = following[:32]
    completed.sort(key=lambda item: (-item[0], item[1]))
    return list(completed[0][1]) if completed else None


def validate_rejections(data):
    if set(data) not in ({"seed", "goal"}, {"seed", "goal", "rejected_actions"}):
        raise ValueError("invalid proposal input schema")
    prior = data.get("rejected_actions", [])
    if type(prior) is not list or len(prior) > 4:
        raise ValueError("invalid rejected_actions")
    for candidate in prior:
        if (type(candidate) is not list or not 2 <= len(candidate) <= STEPS
                or any(type(tok) is not str or tok not in ACTIONS for tok in candidate)
                or candidate[-1] != "HALT"
                or "HALT" in candidate[:-1]):
            raise ValueError("invalid rejected program")
    return prior


def main():
    data=json.load(sys.stdin)
    prior = validate_rejections(data)
    if type(data["seed"]) is not int or not isinstance(data["goal"],str):
        raise ValueError("invalid proposal input schema")
    seed,goal=data["seed"],data["goal"]
    if seed not in (11,19,29):
        raise ValueError("unregistered seed")
    if goal not in GOALS:
        print(json.dumps({"status":"UNSUPPORTED_GOAL","actions":[],"model_generated":False}))
        return
    torch.set_num_threads(1)
    payload=torch.load(CACHED/f"controller_seed{seed}.pt",map_location="cpu",weights_only=False)
    if payload["architecture"]!="MacroController" or payload["seed"]!=seed:
        raise ValueError("checkpoint metadata mismatch")
    if tuple(payload["action_vocabulary"])!=ACTIONS or tuple(payload["goals"])!=GOALS:
        raise ValueError("vocabulary mismatch")
    model=FrozenController().cpu()
    model.load_state_dict(payload["model"],strict=True)
    model.eval()
    prev=len(ACTIONS)
    actions=[]
    with torch.no_grad():
        for pos in range(STEPS):
            token=ACTIONS[int(model(torch.tensor([GOALS.index(goal)]),
                                   torch.tensor([pos]),torch.tensor([prev])).argmax())]
            actions.append(token)
            if token=="HALT":break
            prev=ACTIONS.index(token)
    if prior:
        # A negative verdict can exclude complete prior programs. The new
        # tokens still come exclusively from frozen-controller logits.
        # This is bounded model-guided reranking, not learned feedback.
        actions = next_distinct(model, goal, prior)
    status = ("NO_ALTERNATIVE" if actions is None
              else "PROPOSED" if actions[-1] == "HALT" and len(actions)>1
              else "INVALID_HALT")
    print(json.dumps({"status":status,"actions":actions or [],
                      "model_generated":bool(actions),
                      "feedback_mode":"excluded_prior_programs_only",
                      "learned_feedback":False,
                      "structurally_unseen_goal":False,
                      "source":"NC-007 supervised frozen controller","seed":seed,"goal":goal}))
if __name__=="__main__":main()
