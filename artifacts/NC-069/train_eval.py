"""NC-069 experiment: does interventional training let a small CPU model answer unseen interventions?

Arms share one architecture, sample budget and step budget, and differ only in training data or input:
  interventional   trained on intervened and plain runs
  passive_tokens   trained on plain runs only, then shown intervention tokens it never trained on
  passive_rewrite  trained on plain runs only, then shown the program already rewritten by the oracle's rule
  shuffled_control interventional data with intervention tokens shuffled across each batch
  leaked_control   the answer is an input feature, which should score near 100%
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import causal_benchmark as cb  # noqa: E402

REGS = cb.isa.REGS
SPECIAL = ["<pad>", "<cls>"]
VOCAB = {t: i for i, t in enumerate(SPECIAL + list(cb.isa.FIELDS) + list(REGS)
                                     + [f"v{i}" for i in range(cb.VALUES)] + list(cb.KINDS)
                                     + [f"step{i}" for i in range(cb.MAX_PROGRAM + 1)])}
PAD = VOCAB["<pad>"]
SLOTS = 4
PROGRAM_SLOTS = (cb.MAX_PROGRAM + 1) * SLOTS  # One spare row for an inserted instruction.
LENGTH = 1 + PROGRAM_SLOTS + len(REGS) + 2 + SLOTS + len(REGS)
INTERVENTION_START = 1 + PROGRAM_SLOTS + len(REGS)
INTERVENTION_END = INTERVENTION_START + 2 + SLOTS
ARMS = ("interventional", "passive_tokens", "passive_rewrite", "shuffled_control", "leaked_control")
BUCKETS = ("heldout_combo", "heldout_program", "factual")


def _instruction(item):
    tokens = [VOCAB[item["op"]]]
    for field in cb.isa.FIELDS[item["op"]]:
        value = item[field]
        tokens.append(VOCAB[value] if field in ("dst", "src", "a", "b") else VOCAB[f"v{value}"])
    return tokens + [PAD] * (SLOTS - len(tokens))


def encode(sample, rewrite=False, leak=False):
    program, inputs, intervention = sample["program"], sample["inputs"], sample["intervention"]
    if rewrite:
        program, inputs = cb.apply_intervention(program, inputs, intervention)
        intervention = {"kind": "none"}
    tokens = [VOCAB["<cls>"]]
    rows = [_instruction(item) for item in program["program"]]
    rows += [[PAD] * SLOTS] * (cb.MAX_PROGRAM + 1 - len(rows))
    tokens += [token for row in rows for token in row]
    tokens += [VOCAB[f"v{inputs[name]}"] for name in REGS]
    tokens += [VOCAB[intervention["kind"]], VOCAB[f"step{intervention['step']}"] if "step" in intervention else PAD]
    if "instruction" in intervention:
        tokens += _instruction(intervention["instruction"])
    elif "register" in intervention:
        tokens += [VOCAB[intervention["register"]], VOCAB[f"v{intervention['value']}"], PAD, PAD]
    else:
        tokens += [PAD] * SLOTS
    tokens += [VOCAB[f"v{value}"] for value in sample["target"]] if leak else [PAD] * len(REGS)
    return tokens


def tensors(samples, **kwargs):
    x = torch.tensor([encode(s, **kwargs) for s in samples], dtype=torch.long)
    return x, torch.tensor([s["target"] for s in samples], dtype=torch.long)


class Model(nn.Module):
    def __init__(self, width=64, layers=2, heads=4):
        super().__init__()
        self.tokens = nn.Embedding(len(VOCAB), width, padding_idx=PAD)
        self.positions = nn.Embedding(LENGTH, width)
        layer = nn.TransformerEncoderLayer(width, heads, width * 2, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.heads = nn.ModuleList(nn.Linear(width, cb.VALUES) for _ in REGS)

    def forward(self, x):
        h = self.tokens(x) + self.positions(torch.arange(x.shape[1]))
        h = self.encoder(h, src_key_padding_mask=x == PAD)[:, 0]
        return torch.stack([head(h) for head in self.heads], dim=1)


def wilson(correct, total, z=1.96):
    if not total:
        return [None, None]
    p = correct / total
    d = 1 + z * z / total
    c = (p + z * z / (2 * total)) / d
    m = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [round(c - m, 4), round(c + m, 4)]


def score(predicted, samples):
    """Exact four-register accuracy overall and where the intervention changed the outcome."""
    target = [s["target"] for s in samples]
    correct = [p == t for p, t in zip(predicted, target)]
    effect = [c for c, s in zip(correct, samples) if s["target"] != s["factual"]]
    return {"correct": sum(correct), "total": len(correct), "accuracy": round(sum(correct) / len(correct), 4),
            "ci95": wilson(sum(correct), len(correct)), "effect_correct": sum(effect), "effect_total": len(effect),
            "effect_accuracy": round(sum(effect) / len(effect), 4) if effect else None,
            "effect_ci95": wilson(sum(effect), len(effect))}


def predict(model, x, batch=512):
    model.eval()
    with torch.no_grad():
        return [row.tolist() for start in range(0, len(x), batch)
                for row in model(x[start:start + batch]).argmax(-1)]


def train_arm(arm, seed, train, tests, args, out):
    torch.manual_seed(seed)
    x, y = tensors(train, leak=arm == "leaked_control")
    model = Model()
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    gen = torch.Generator().manual_seed(seed)
    cpu = time.process_time()
    wall = time.time()
    model.train()
    for _ in range(args.steps):
        idx = torch.randint(0, len(x), (args.batch,), generator=gen)
        xb = x[idx]
        if arm == "shuffled_control":
            perm = torch.randperm(args.batch, generator=gen)
            xb = xb.clone()
            xb[:, INTERVENTION_START:INTERVENTION_END] = xb[perm, INTERVENTION_START:INTERVENTION_END]
        loss = nn.functional.cross_entropy(model(xb).flatten(0, 1), y[idx].flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    cpu = time.process_time() - cpu
    wall = time.time() - wall
    results = {}
    for bucket, samples in tests.items():
        xt, _ = tensors(samples, rewrite=arm == "passive_rewrite", leak=arm == "leaked_control")
        results[bucket] = score(predict(model, xt), samples)
    path = out / f"{arm}_seed{seed}.pt"
    torch.save({"state": model.state_dict(), "arm": arm, "seed": seed}, path)
    held = results["heldout_combo"]
    return {"arm": arm, "seed": seed, "parameters": params, "train_samples": len(train), "steps": args.steps,
            "cpu_seconds": round(cpu, 1), "wall_seconds": round(wall, 1), "final_loss": round(loss.item(), 4),
            "checkpoint": path.name, "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "cpu_seconds_per_heldout_correct": round(cpu / held["correct"], 4) if held["correct"] else None,
            "buckets": results}


def baselines(tests):
    def majority(samples):
        counts = [torch.bincount(torch.tensor([s["target"][i] for s in samples]), minlength=cb.VALUES).argmax().item()
                  for i in range(len(REGS))]
        return counts
    out = {}
    for bucket, samples in tests.items():
        out[bucket] = {
            "copy_input": score([[s["inputs"][n] for n in REGS] for s in samples], samples),
            "factual_oracle": score([s["factual"] for s in samples], samples),
            "majority": score([majority(samples)] * len(samples), samples)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--train-samples", type=int, default=30000)
    p.add_argument("--test-samples", type=int, default=4000)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 19, 29])
    p.add_argument("--out", type=Path, default=HERE)
    a = p.parse_args()
    torch.set_num_threads(2)
    a.out.mkdir(parents=True, exist_ok=True)
    tests = {b: cb.generate(1000, a.test_samples, b) for b in BUCKETS}
    report = {"schema": "logos.nc069.results/1", "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
              "held_out_combinations": sorted({s["combo"] for s in tests["heldout_combo"]}),
              "baselines": baselines(tests), "runs": []}
    target = a.out / "results.json"
    for seed in a.seeds:
        train = {"mixed": cb.generate(seed, a.train_samples, "train"),
                 "passive": cb.generate(seed, a.train_samples, "passive_train")}
        for arm in ARMS:
            data = train["passive"] if arm.startswith("passive") else train["mixed"]
            run = train_arm(arm, seed, data, tests, a, a.out)
            report["runs"].append(run)
            target.write_text(json.dumps(report, indent=1))
            b = run["buckets"]
            print(arm, seed, "combo", b["heldout_combo"]["accuracy"], "effect", b["heldout_combo"]["effect_accuracy"],
                  "program", b["heldout_program"]["accuracy"], "factual", b["factual"]["accuracy"],
                  "cpu", run["cpu_seconds"], flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
