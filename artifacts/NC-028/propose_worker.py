"""Frozen NC-007 controller inference. No target recipe or oracle is loaded.

Dependencies: PyTorch, checkpoint controller_seed{11,19,29}.pt, Python stdlib.
Input: JSON {seed, goal}; output: neural autoregressive action tokens only.
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

def main():
    data=json.load(sys.stdin)
    if set(data)!={"seed","goal"} or type(data["seed"]) is not int or not isinstance(data["goal"],str):
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
    status="PROPOSED" if actions[-1]=="HALT" and len(actions)>1 else "INVALID_HALT"
    print(json.dumps({"status":status,"actions":actions,"model_generated":True,
                      "source":"NC-007 supervised frozen controller","seed":seed,"goal":goal}))
if __name__=="__main__":main()
