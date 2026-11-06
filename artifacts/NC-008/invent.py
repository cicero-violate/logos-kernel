#!/usr/bin/env python3
"""NC-008: bounded verifier-reward-trained Transformer program proposals.

Dependencies: Python 3, torch, numpy, NC-007/compose.py, frozen NC-005
checkpoints, NC-006 accepted library/manifest, NC-001 canonical_isa.
No target instruction sequences, pretrained LLM, external solver, or network.
This is *bounded guided stochastic synthesis*, not independent invention of a
primitive or zero-shot generalization. The frozen neural ISA executes finalists.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "artifacts/NC-007"))
import compose

MASK = 65535
SEED = 2108
ROUNDS = 36
BATCH = 64
PROBES = 20
HELDOUT = 256
OPS = ("ADD_X1", "ADD_XX", "XOR_XY", "AND_XY", "OR_XY",
       "AND1_X", "ADD_RX", "ADD_RY", "XOR_RX", "XOR_RY",
       "SHL_R", "SHR_R", "HALT")
BODY = OPS[:-1]
START = len(OPS)
# Behavioral specification only; no target instructions or accepted solution.
TASKS = {
    "INC": lambda x,y: (x + 1) & MASK,
    "DOUBLE": lambda x,y: (x + x) & MASK,
    "BOOLEAN_COMPOSITE": lambda x,y: ((x & y) ^ x) & MASK,
    "SHIFT_COMPOSITE": lambda x,y: ((x ^ y) << 1) & MASK,
}
CORNER = [(0,0), (MASK,0), (0,MASK), (MASK,MASK), (0,1), (1,0),
          (1,1), (0xAAAA,0x5555), (0x5555,0xAAAA), (32768,32767)]
SCHEMA = "logos.nc008.invention/1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def probes(seed, count):
    rng = random.Random(seed)
    out = list(CORNER)
    while len(out) < count:
        p = (rng.randrange(65536), rng.randrange(65536))
        if p not in out:
            out.append(p)
    return out[:count]


def examples(target, pairs):
    return [(x,y,target(x,y)) for x,y in pairs]


def bit_features(rows):
    return torch.tensor([[[float((v >> b) & 1) for v in row for b in range(16)]
                          for row in rows]], dtype=torch.float32)


class SpecTransformer(nn.Module):
    """Attention from a next-token query to input/output behavior examples."""
    def __init__(self):
        super().__init__()
        d = 48
        self.io = nn.Linear(48, d)
        self.step = nn.Embedding(2, d)
        self.prev = nn.Embedding(len(OPS)+1, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4,
                                           dim_feedforward=96, dropout=0,
                                           batch_first=True, norm_first=False)
        self.attention = nn.TransformerEncoder(layer, num_layers=2,
                                               enable_nested_tensor=False)
        self.readout = nn.Linear(d, len(OPS))
    def forward(self, specs, steps, prev):
        b = len(steps)
        memories = self.io(specs)
        query = self.step(steps) + self.prev(prev)
        encoded = self.attention(torch.cat((memories, query[:,None,:]), dim=1))
        return self.readout(encoded[:, -1])


def symbolic(program, x, y):
    """Independent bounded host score; never substitutes for neural final test."""
    r = 0
    for token in program:
        if token == "HALT":
            break
        op,a,b = compose.ACTIONS[token]
        state = {"X": x, "Y": y, "K": 1, "R": r}
        v = state[a]
        w = state[b] if b is not None else 0
        if op == "ADD": r = (v + w) & MASK
        elif op == "XOR": r = v ^ w
        elif op == "AND": r = v & w
        elif op == "OR": r = v | w
        elif op == "AND1": r = v & 1
        elif op == "SHL": r = (v << 1) & MASK
        elif op == "SHR": r = v >> 1
        else: raise ValueError(op)
    return r


def score(program, rows):
    outcomes = [symbolic(program, x, y) for x,y,_ in rows]
    exact = sum(pred == expected for pred,(_,_,expected) in zip(outcomes,rows))
    bitwise = sum(16 - (pred ^ expected).bit_count()
                  for pred,(_,_,expected) in zip(outcomes,rows))
    return exact, bitwise / (16 * len(rows))


def proposal(policy, spec, n, greedy=False):
    x = spec.expand(n,-1,-1)
    chosen = []
    previous = torch.full((n,), START, dtype=torch.long)
    with torch.no_grad():
        for t in range(2):
            logits = policy(x, torch.full((n,), t), previous)
            if t == 0:
                logits[:, OPS.index("HALT")] = -1e9
            index = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
            chosen.append(index)
            previous = index
    result = []
    for i in range(n):
        a = OPS[int(chosen[0][i])]
        b = OPS[int(chosen[1][i])]
        result.append([a,"HALT"] if b == "HALT" else [a,b,"HALT"])
    return result


def fit_policy(policy, opt, spec, selected):
    # Self-imitation of proposals which the verifier has ranked, not supplied
    # teacher recipes or symbolic program synthesis.
    opt.zero_grad(set_to_none=True)
    count = len(selected)
    x = spec.expand(count,-1,-1)
    first = torch.tensor([OPS.index(p[0]) for p in selected])
    second = torch.tensor([OPS.index(p[1]) for p in selected])
    start = torch.full((count,), START, dtype=torch.long)
    step0 = torch.zeros(count,dtype=torch.long)
    step1 = torch.ones(count,dtype=torch.long)
    a = policy(x,step0,start)
    a = a.clone()
    a[:,OPS.index("HALT")] = -1e9
    b = policy(x,step1,first)
    loss = F.cross_entropy(a,first) + F.cross_entropy(b,second)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(),1.0)
    opt.step()
    return float(loss.detach())


def run_one(name, target, frozen, registry):
    spec_pairs = probes(SEED + sum(map(ord,name)), PROBES)
    rows = examples(target,spec_pairs)
    spec = bit_features(rows)
    torch.manual_seed(SEED + sum(map(ord,name)))
    policy = SpecTransformer()
    opt = torch.optim.AdamW(policy.parameters(),lr=0.003,weight_decay=0)
    rng = random.Random(SEED + 1000 + sum(map(ord,name)))
    # Uniform baseline has same proposal count and probes; no policy or oracle
    # program sequence is given. It is reported, not misnamed neural invention.
    control_first = None
    for k in range(ROUNDS*BATCH):
        a = rng.choice(BODY)
        b = rng.choice(OPS)
        program = [a,"HALT"] if b=="HALT" else [a,b,"HALT"]
        if score(program,rows)[0] == len(rows):
            control_first = k+1
            break
    history = []
    best = None
    best_metric = (-1,-1.)
    first_exact = None
    for round_index in range(ROUNDS):
        programs = proposal(policy,spec,BATCH)
        rankings = []
        for i,p in enumerate(programs):
            exact, bitwise = score(p,rows)
            rankings.append(((exact,bitwise), i, p))
        rankings.sort(key=lambda r:(r[0][0],r[0][1]),reverse=True)
        if rankings[0][0] > best_metric:
            best_metric,best = rankings[0][0],rankings[0][2]
        if first_exact is None:
            for metric,i,p in rankings:
                if metric[0] == len(rows):
                    first_exact = round_index*BATCH+i+1
                    break
        selected = [p for _,_,p in rankings[:12]]
        loss = fit_policy(policy,opt,spec,selected)
        history.append({"round": round_index+1, "best_exact":rankings[0][0][0],
                        "loss":round(loss,6)})
    greedy = proposal(policy,spec,1,greedy=True)[0]
    # A verified candidate must be the actual Transformer greedy decoding,
    # NOT the external reranker-selected best solution.
    score_train = score(greedy,rows)[0]
    pairs = probes(SEED + 50000 + sum(map(ord,name)),HELDOUT)
    overlap = set(spec_pairs)&set(pairs)
    pairs = [p for p in pairs if p not in overlap]
    # Guarantee exactly 256 disjoint unseen operand pairs.
    extra = probes(SEED + 70000 + sum(map(ord,name)),HELDOUT+PROBES+10)
    for p in extra:
        if len(pairs)>=HELDOUT: break
        if p not in spec_pairs and p not in pairs:pairs.append(p)
    assert len(pairs)==HELDOUT and not (set(spec_pairs)&set(pairs))
    expected = [target(x,y) for x,y in pairs]
    references = []
    if greedy[-1] == "HALT" and score_train == len(rows):
        actual,refs,elapsed_ns = compose.execute_neural(frozen,greedy,pairs,registry)
        correct = sum(a==b for a,b in zip(actual,expected))
        references = refs
    else:
        correct,elapsed_ns = 0,0
    ck = HERE / ("policy_"+name.lower()+".pt")
    torch.save({"schema": SCHEMA,"seed":SEED,"task":name,
                "state_dict":policy.state_dict(),"optimizer":opt.state_dict(),
                "spec_examples":rows,"rounds":ROUNDS,
                "torch_rng":torch.get_rng_state()},ck)
    return {"task":name,"target_specification":"behavioral input/output examples; no target program provided",
            "probe_count":len(rows),"probe_sha256":hashlib.sha256(canonical(rows).encode()).hexdigest(),
            "heldout_count":len(pairs),"heldout_pairs_sha256":hashlib.sha256(canonical(pairs).encode()).hexdigest(),
            "neural_isa_correct":correct,"neural_isa_tested":len(pairs),
            "greedy_program":greedy,"greedy_probe_exact":score_train,
            "best_sampled_program":best,"best_sampled_probe_exact":best_metric[0],
            "first_exact_model_proposal_attempt":first_exact,
            "first_exact_uniform_control_attempt":control_first,
            "sample_budget":ROUNDS*BATCH,
            "gradient_updates":ROUNDS,"last_loss":history[-1]["loss"],
            "proposal_checkpoint":str(ck.relative_to(ROOT)),
            "proposal_checkpoint_sha256":sha(ck),
            "frozen_primitive_ids":references,"elapsed_neural_execution_ns":elapsed_ns,
            "status":"SUPPORTED_BOUNDED" if correct==HELDOUT else "FAILED",
            "history":history}


def main():
    torch.set_num_threads(1)
    random.seed(SEED)
    torch.manual_seed(SEED)
    frozen,registry,provenance = compose.load_frozen(11)
    traces = [run_one(name,target,frozen,registry) for name,target in TASKS.items()]
    out = HERE/"proposals.jsonl"
    out.write_text("".join(canonical(t)+"\n" for t in traces))
    receipt = {
        "schema":SCHEMA,"task":"NC-008",
        "status":"IMPLEMENTER_TESTED_PENDING_INDEPENDENT",
        "method":"Transformer stochastic neural candidate proposals; reward-ranked policy distillation; greedy decoder; independent heldout frozen-neural execution",
        "not_a_claim_of":"zero-shot invention, new primitive discovery, policy superiority over uniform search, AGI or proof over all 2^32 input pairs",
        "source_sha256":{"invent.py":sha(HERE/"invent.py"),
                         "nc007_compose.py":sha(ROOT/"artifacts/NC-007/compose.py"),
                         "nc006_library":sha(ROOT/"artifacts/NC-006/accepted_library.json")},
        "frozen_provenance":provenance,
        "model":"2-layer 4-head 48-width TransformerEncoder over task IO samples and previous-action query",
        "policy_parameters":sum(p.numel() for p in SpecTransformer().parameters()),
        "training":"36 rounds per task of 64 model samples, train on top 12 self-proposed programs under probe reward; no target program labels",
        "sample_budget_per_task":ROUNDS*BATCH,"task_count":len(traces),
        "supported_count":sum(t["status"]=="SUPPORTED_BOUNDED" for t in traces),
        "heldout_neural_correct":sum(t["neural_isa_correct"] for t in traces),
        "heldout_neural_total":sum(t["neural_isa_tested"] for t in traces),
        "proposals_sha256":sha(out),
        "limits":["Task-specific policy adaptation has access to reference I/O examples and verifier feedback.",
                  "Policy training and uniform baseline both search bounded program vocabulary; sampled programs are not newly invented primitives.",
                  "Frozen local neural ISA, not a jointly trained Transformer, performs numeric execution.",
                  "Proof consists of finite disjoint heldout operand tests, not symbolic equivalence.",
                  "Host handles registers, HALT and primitive wiring; neural model only chooses program tokens.",
                  "No claim that self-trained policy beats uniform baseline or generalizes to unseen task families.",
                  "Local Python/PyTorch execution is not hostile-code isolation.",
                  "No champion promotion."] }
    (HERE/"invention_receipt.json").write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print(canonical({"supported":receipt["supported_count"],"heldout_correct":receipt["heldout_neural_correct"],
                     "heldout_total":receipt["heldout_neural_total"],
                     "tasks":[[t["task"],t["greedy_program"],t["status"],t["first_exact_uniform_control_attempt"]] for t in traces]}))


if __name__=="__main__":main()
