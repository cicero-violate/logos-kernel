"""NC-062 sandbox-only 4-bit example-conditioned CPU neural policy.

Executed as Python -I -B -c SOURCE under NC-028 Landlock/seccomp, never
as a direct host process. Input JSON contains numeric examples and known
verified-library training labels ONLY; it never contains a goal answer,
sealed table, withheld counterexample, thresholds, oracle, or reveal.
Dependencies: Python>=3.10 stdlib, PyTorch CPU (already in NC-028 allowlist).
"""
import hashlib
import json
import random
import sys
import os
os.environ["USER"]="sandbox"
os.environ["LOGNAME"]="sandbox"
os.environ["TORCHINDUCTOR_CACHE_DIR"]="/workspace/experiments/logos-kernel/artifacts/NC-028/scratch/torch_cache"

import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.set_grad_enabled(True)

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(24, 24), nn.Tanh(), nn.Linear(24, 4))
    def forward(self, x):
        return self.net(x)

def features(examples):
    if type(examples) is not list or len(examples)!=8:
        raise ValueError("exactly eight public examples required")
    rows=[]
    for e in examples:
        if type(e) is not dict or set(e)!=set(("X","Y","R")) or any(
            type(e[k]) is not int or not 0<=e[k]<16 for k in ("X","Y","R")):
            raise ValueError("4-bit example schema")
        rows.extend([e["X"]/15.0, e["Y"]/15.0, e["R"]/15.0])
    return rows

def digest(model):
    h=hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().contiguous().numpy().tobytes())
    return h.hexdigest()

def unpack(model, value):
    if value is None:return
    if type(value) is not dict or set(value)!=set(model.state_dict()):
        raise ValueError("model state shape")
    tensors={}
    for name,origin in model.state_dict().items():
        t=torch.tensor(value[name], dtype=torch.float32)
        if t.shape!=origin.shape or not torch.isfinite(t).all():
            raise ValueError("invalid model state")
        tensors[name]=t
    model.load_state_dict(tensors,strict=True)

def pack(model):
    return {k:v.detach().tolist() for k,v in sorted(model.state_dict().items())}

def main():
    data=json.load(sys.stdin)
    if type(data) is not dict or set(data)!=set(("mode","seed","state","training","queries")):
        raise ValueError("invalid worker request")
    mode=data["mode"];seed=data["seed"]
    if mode not in ("init","train","infer") or type(seed) is not int or not 0<=seed<=100000:
        raise ValueError("invalid mode/seed")
    torch.manual_seed(seed)
    random.seed(seed)
    model=Policy().cpu()
    unpack(model,data["state"])
    before=digest(model)
    updates=0
    if mode=="train":
        train=data["training"]
        if type(train) is not list or not 1<=len(train)<=128:
            raise ValueError("bounded training")
        values=[];labels=[]
        for item in train:
            if type(item) is not dict or set(item)!=set(("examples","label")):
                raise ValueError("invalid training record")
            features_row=features(item["examples"])
            y=item["label"]
            if type(y) is not int or not 0<=y<4:
                raise ValueError("invalid class")
            values.append(features_row);labels.append(y)
        x=torch.tensor(values,dtype=torch.float32)
        y=torch.tensor(labels,dtype=torch.long)
        opt=torch.optim.SGD(model.parameters(),lr=0.08)
        for _ in range(8):
            opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x),y)
            loss.backward()
            opt.step()
            updates+=1
    if mode=="infer":
        if data["training"]:
            raise ValueError("inference cannot receive training labels")
        if type(data["queries"]) is not list or len(data["queries"])>40:
            raise ValueError("bounded query count")
        model.eval()
        with torch.no_grad():
            probs=F.softmax(model(torch.tensor([features(e) for e in data["queries"]],dtype=torch.float32)),dim=1)
            order=probs.argsort(dim=1,descending=True).tolist()
        choices=order
    else:
        if data["queries"]: raise ValueError("no infer on training or initialization")
        choices=[]
    print(json.dumps({"mode":mode,"seed":seed,"model_sha256_before":before,
       "model_sha256_after":digest(model),"model_state":pack(model) if mode!="infer" else None,
       "choices":choices,"gradient_updates":updates,
       "claimed_new_operator":False},sort_keys=True))
if __name__=="__main__":main()
