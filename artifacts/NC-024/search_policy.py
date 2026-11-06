"""NC-024: RLT-style recurrent attention ranks *symbolically generated* rewrites.

Dependencies: Python>=3.10, PyTorch, ../NC-023/symbolic_search.py,
../NC-020/benchmark.py, ../NC-001/canonical_isa.py.
Distinct from the original RLT champion; never loads/modifies its weights.
The model cannot propose rules, claim proofs, or see the optimized target RHS.
"""
from __future__ import annotations
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "artifacts/NC-023"))
import symbolic_search as ss

RULE_NAMES = tuple(r[0] for r in ss.RULES)
KINDS = ("int.add", "int.sub", "int.mul", "bv.add", "bv.sub",
         "bv.xor", "bv.and", "bv.or", "bv.shl", "int.const", "int.var", "bv.read")
FEAT_DIM = len(RULE_NAMES) + 2 + 2 * len(KINDS) + 2


def feature(rule, parent, candidate, case):
    """Candidate AST shapes only; no target RHS, proof or oracle outputs."""
    out = [float(rule == r) for r in RULE_NAMES]
    out.extend([float(case["domain"] == "integer"), float(case["domain"] == "bitvector")])
    for ast in (parent, candidate):
        counts = dict.fromkeys(KINDS, 0)
        def visit(node):
            if node["kind"] in counts:
                counts[node["kind"]] += 1
            for child in node.get("args", []):
                visit(child)
        visit(ast)
        out.extend(min(counts[k], 32) / 32 for k in KINDS)
    out.extend([min(ss.num_nodes(parent), 96) / 96, min(ss.num_nodes(candidate), 96) / 96])
    assert len(out) == FEAT_DIM
    return out


class RecurrentRewriteRanker(nn.Module):
    """Shared recurrent candidate-attention cell, z_L/z_H, two outer cycles."""
    def __init__(self, hidden=32):
        super().__init__()
        self.embed = nn.Linear(FEAT_DIM, hidden)
        self.attn = nn.MultiheadAttention(hidden, 4, dropout=0, batch_first=True)
        self.low = nn.LayerNorm(hidden)
        self.high = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, hidden))
        self.readout = nn.Linear(hidden, 1)

    def forward(self, matrix):
        h = self.embed(matrix.unsqueeze(0))
        z_h, z_l = h, torch.zeros_like(h)
        for _ in range(2):
            for _ in range(2):
                update, _ = self.attn(z_h + z_l, z_h + z_l, z_h + z_l, need_weights=False)
                z_l = self.low(z_l + update + self.ff(z_l + update))
            z_h = self.high(z_h + z_l)
        return self.readout(z_h).squeeze(0).squeeze(-1)


def root_training_set(case):
    options = {}
    for rule, child in ss.candidates(case["lhs"]):
        h = ss.digest(child)
        if h != ss.digest(case["lhs"]):
            options.setdefault(h, (rule, child))
    options = [options[k] for k in sorted(options)][:48]
    if len(options) < 2:
        return None
    features = torch.tensor([feature(rule, case["lhs"], child, case)
                             for rule, child in options], dtype=torch.float32)
    # Label is a deterministic training-only *static cost*, not an optimized target.
    label = min(range(len(options)), key=lambda i: (ss.score(options[i][1]), i))
    return features, label


def fit(train_cases, seed, epochs=2):
    torch.manual_seed(seed)
    policy = RecurrentRewriteRanker()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=.004, weight_decay=0)
    prepared = [root_training_set(case) for case in train_cases]
    prepared = [x for x in prepared if x is not None]
    if not prepared:
        raise ValueError("no trainable root choices")
    start = time.process_time()
    losses = []
    for epoch in range(epochs):
        random.Random(seed + epoch).shuffle(prepared)
        for matrix, label in prepared:
            optimizer.zero_grad(set_to_none=True)
            logits = policy(matrix)
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([label]))
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    return policy, {"training_cases": len(prepared), "epochs": epochs,
                    "gradient_updates": len(losses),
                    "training_cpu_seconds": round(time.process_time() - start, 6),
                    "last_loss": round(losses[-1], 6)}


def ranked_search(case, mode, policy=None, seed=0, budget=8, pool_cap=64):
    """Same queue, allowed rules, verifier, proposal cap and proof budget for every arm."""
    assert mode in ("neural", "untrained", "heuristic", "random")
    src = case["lhs"]
    ss.validate_domain(src, case)
    source_hash = ss.digest(src)
    pool, visited = {}, {source_hash: {"ast": src, "parent": None}}
    generated, proposal_events, valid = 0, [], 0
    rng = random.Random(seed ^ int(case["id"][:8], 16))
    best = src
    def add(parent):
        nonlocal generated
        ph = ss.digest(parent)
        for rule, child in ss.candidates(parent):
            generated += 1
            h = ss.digest(child)
            if h == source_hash or h in visited or h in pool:
                continue
            pool[h] = (rule, parent, ph, child)
        # Global deterministically truncated pool: same generated opportunities per arm.
        if len(pool) > pool_cap:
            keep = set(sorted(pool)[:pool_cap])
            for k in list(pool):
                if k not in keep:
                    del pool[k]
    add(src)
    start_wall = time.perf_counter_ns()
    start_cpu = time.process_time()
    choice_cpu = 0.0
    while pool and len(proposal_events) < budget:
        keys = sorted(pool)
        before = time.process_time()
        if mode == "heuristic":
            selected = min(keys, key=lambda h: (ss.score(pool[h][3]), h))
        elif mode == "random":
            selected = rng.choice(keys)
        else:
            if policy is None:
                raise ValueError("policy required")
            with torch.no_grad():
                matrix = torch.tensor([feature(pool[h][0], pool[h][1], pool[h][3], case)
                                       for h in keys], dtype=torch.float32)
                logits = policy(matrix)
                selected = keys[int(logits.argmax())]
        choice_cpu += time.process_time() - before
        rule, parent, parent_hash, child = pool.pop(selected)
        try:
            ok, proof = ss.equivalence(src, child, case)
        except (TypeError, ValueError, KeyError, ss.ref.isa.ISAError):
            ok, proof = False, {"method": "rejected_invalid_candidate"}
        if ok:
            valid += 1
            visited[selected] = {"ast": child, "parent": parent_hash, "rule": rule}
            if ss.score(child) < ss.score(best):
                best = child
            add(child)
        proposal_events.append({"rule": rule, "candidate_sha256": selected,
                                "parent_sha256": parent_hash, "accepted": bool(ok),
                                "proof_method": proof.get("method"),
                                "candidate_dag_ops": ss.cost(child)["dag_ops"]})
    final_ok, final_proof = ss.equivalence(src, best, case)
    if not final_ok:
        raise AssertionError("unproved output")
    return {"optimized_sha256": ss.digest(best), "optimized": best,
            "verified": final_ok, "proof": final_proof,
            "source_dag_ops": ss.cost(src)["dag_ops"],
            "optimized_dag_ops": ss.cost(best)["dag_ops"],
            "explored_proofs": len(proposal_events), "verified_proposals": valid,
            "generated_proposals": generated, "ranker_cpu_s": round(choice_cpu, 8),
            "full_cpu_s": round(time.process_time() - start_cpu, 8),
            "full_wall_s": round((time.perf_counter_ns() - start_wall) / 1e9, 8),
            "trace": proposal_events}
