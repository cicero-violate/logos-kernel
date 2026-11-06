"""NC-082: a Transformer reads a program it has never seen and steps through it, on learned arithmetic.

Unlike the earlier controller, whose ROM tokens were constant, the program is an input that varies per example. Training
uses random programs with exact one-step targets only. Test programs are held out by hash and run without teacher forcing.
The host keeps registers, feeds predictions forward and applies operators; it never resolves PC, opcode or operands.
"""
import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("neural_quotient", HERE.parent / "NC-080" / "neural_quotient.py")
nq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nq)

OPS = ("ADD", "SUB", "AND", "OR", "XOR", "SHL", "SHR", "LOADI", "JZ", "JMP", "HALT")
ALU = ("ADD", "SUB", "AND", "OR", "XOR", "SHL", "SHR")
OP = {name: i for i, name in enumerate(OPS)}
REGS = 8
MAX_LEN = 24
MUL_INPUTS = 256
STEP_LIMIT = 160


def random_program(rng):
    length = rng.randint(2, MAX_LEN - 1)
    body = []
    for pc in range(length):
        roll = rng.random()
        if roll < 0.12:
            body.append(("JZ", rng.randrange(REGS), 0, 0, rng.randint(0, length)))
        elif roll < 0.2:
            body.append(("JMP", 0, 0, 0, rng.randint(0, length)))
        elif roll < 0.35:
            body.append(("LOADI", 0, 0, rng.randrange(REGS), rng.randrange(256)))
        else:
            name = rng.choice(ALU)
            b = 0 if name in ("SHL", "SHR") else rng.randrange(REGS)
            body.append((name, rng.randrange(REGS), b, rng.randrange(REGS), 0))
    return [*body, ("HALT", 0, 0, 0, 0)]


def program_key(program):
    return hashlib.sha256(json.dumps(program).encode()).hexdigest()


def multiply_program():
    """r0=X, r1=Y, result in r2. Shift-and-add over 8-bit registers, written in this ISA and never used in training."""
    return [("LOADI", 0, 0, 3, 1), ("LOADI", 0, 0, 2, 0), ("JZ", 1, 0, 0, 9), ("AND", 1, 3, 4, 0), ("JZ", 4, 0, 0, 6), ("ADD", 2, 0, 2, 0),
            ("SHL", 0, 0, 0, 0), ("SHR", 1, 0, 1, 0), ("JMP", 0, 0, 0, 2), ("HALT", 0, 0, 0, 0)]


def compile_concept(expr):
    """Compile an expression to straight-line code with r0=X, r1=Y and the result in r2, or None if it does not fit."""
    order, seen = [], set()
    def walk(e):
        if e in seen:
            return
        seen.add(e)
        for part in e[1:]:
            if isinstance(part, tuple):
                walk(part)
        if e[0] not in ("X", "Y"):
            order.append(e)
    walk(expr)
    last_use = {}
    for i, e in enumerate(order):
        for part in e[1:]:
            if isinstance(part, tuple):
                last_use[part] = i
    last_use[expr] = len(order)
    where = {("X",): 0, ("Y",): 1}
    free = list(range(3, REGS)) + [2]
    body = []
    for i, e in enumerate(order):
        args = [p for p in e[1:] if isinstance(p, tuple)]
        if not free:
            return None
        dst = free.pop(0)
        if e[0] == "K":
            body.append(("LOADI", 0, 0, dst, e[1] & 255))
        elif e[0] in ("SHL", "SHR"):
            body.append((e[0], where[args[0]], 0, dst, 0))
        else:
            body.append((e[0], where[args[0]], where[args[1]], dst, 0))
        where[e] = dst
        for part in set(args):
            if part not in (("X",), ("Y",)) and last_use.get(part) == i:
                free.append(where[part])
    if expr in (("X",), ("Y",)):
        return None
    if where[expr] != 2:
        body.append(("OR", where[expr], where[expr], 2, 0))
    body.append(("HALT", 0, 0, 0, 0))
    return body if len(body) <= MAX_LEN else None


def exact_step(item, regs, pc):
    name, a, b, dst, imm = item
    if name == "JZ":
        return imm if regs[a] == 0 else pc + 1
    if name == "JMP":
        return imm
    return pc if name == "HALT" else pc + 1


def exact_alu(name, x, y):
    return {"ADD": (x + y) & 255, "SUB": (x - y) & 255, "AND": x & y, "OR": x | y, "XOR": x ^ y, "SHL": (x << 1) & 255, "SHR": x >> 1}[name]


def exact_run(program, x, y):
    regs = [0] * REGS
    regs[0], regs[1] = x, y
    pc = 0
    for _ in range(STEP_LIMIT):
        item = program[pc]
        if item[0] == "HALT":
            return regs[2]
        if item[0] in ALU:
            regs[item[3]] = exact_alu(item[0], regs[item[1]], regs[item[2]])
        elif item[0] == "LOADI":
            regs[item[3]] = item[4]
        pc = exact_step(item, regs, pc)
    return None


def encode_program(program):
    out = np.zeros((MAX_LEN, 5), dtype=np.int64)
    out[:, 0] = len(OPS)  # Padding opcode.
    for i, (name, a, b, dst, imm) in enumerate(program):
        out[i] = (OP[name], a, b, dst, imm)
    return out


class Controller(nn.Module):
    def __init__(self, dim=96, layers=3, heads=4):
        super().__init__()
        self.op = nn.Embedding(len(OPS) + 1, dim)
        self.a, self.b, self.dst = (nn.Embedding(REGS, dim) for _ in range(3))
        self.imm = nn.Embedding(256, dim)
        self.position = nn.Embedding(MAX_LEN + 1, dim)
        self.pc = nn.Embedding(MAX_LEN + 1, dim)
        self.reg_bits = nn.Linear(8, dim)
        self.reg_id = nn.Embedding(REGS, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0.0, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.heads = nn.ModuleDict({"op": nn.Linear(dim, len(OPS)), "a": nn.Linear(dim, REGS), "b": nn.Linear(dim, REGS), "dst": nn.Linear(dim, REGS),
                                    "imm": nn.Linear(dim, 256), "next": nn.Linear(dim, MAX_LEN + 1)})
        self.blank = False

    def forward(self, program, pc, regs):
        n = program.shape[0]
        toks = self.op(program[:, :, 0]) + self.a(program[:, :, 1]) + self.b(program[:, :, 2]) + self.dst(program[:, :, 3]) + self.imm(program[:, :, 4])
        if self.blank:
            toks = toks * 0
        toks = toks + self.position(torch.arange(MAX_LEN))
        bits = ((regs[:, :, None] >> torch.arange(8)) & 1).float()
        reg_toks = self.reg_bits(bits) + self.reg_id(torch.arange(REGS))
        h = torch.cat((self.cls.expand(n, -1, -1), self.pc(pc)[:, None], reg_toks, toks), 1)
        z = self.encoder(h)[:, 0]
        return {k: head(z) for k, head in self.heads.items()}


def batch_rows(rng, size):
    programs, pcs, regs, targets = [], [], [], {k: [] for k in ("op", "a", "b", "dst", "imm", "next")}
    for _ in range(size):
        program = random_program(rng)
        pc = rng.randrange(len(program))
        values = [0 if rng.random() < 0.25 else rng.randrange(256) for _ in range(REGS)]
        item = program[pc]
        programs.append(encode_program(program))
        pcs.append(pc)
        regs.append(values)
        for key, value in zip(("op", "a", "b", "dst", "imm"), (OP[item[0]], *item[1:])):
            targets[key].append(value)
        targets["next"].append(exact_step(item, values, pc))
    return (torch.tensor(np.stack(programs)), torch.tensor(pcs), torch.tensor(regs),
            {k: torch.tensor(v) for k, v in targets.items()})


def train(seed, steps, batch, held_out):
    torch.manual_seed(seed)
    rng = random.Random(f"NC-082/train/{seed}")
    model = Controller()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    started = time.process_time()
    collisions = 0
    for _ in range(steps):
        program, pc, regs, targets = batch_rows(rng, batch)
        out = model(program, pc, regs)
        loss = sum(nn.functional.cross_entropy(out[k], targets[k]) for k in out)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    model.eval()
    # Random programs are generated fresh, so check by construction on a large sample that none match held-out programs.
    check = random.Random(f"NC-082/train/{seed}")
    collisions = sum(program_key(random_program(check)) in held_out for _ in range(20000))
    return model, {"steps": steps, "batch": batch, "final_loss": round(loss.item(), 4), "train_cpu_seconds": round(time.process_time() - started, 1),
                   "parameters": sum(p.numel() for p in model.parameters()), "held_out_collisions_in_20000_sampled_programs": collisions}


class NeuralALU:
    """NC-080's learned operators, applied bit by bit through the network's own decisions."""
    def __init__(self, seed):
        self.net = nq.NeuralOperators()
        state = torch.load(HERE.parent / "NC-080" / f"operators_seed{seed}.pt", weights_only=True)["state"]
        self.net.load_state_dict(state)
        self.net.eval()
        nq.network_table(self.net)
        self.table = self.net._table.numpy().astype(int)

    def __call__(self, name, x, y):
        table = self.table[nq.OPS.index(name)]
        carry, out = 0, 0
        for i in range(8):
            a, b = (x >> i) & 1, (y >> i) & 1
            previous = (x >> (i - 1)) & 1 if i else 0
            following = (x >> (i + 1)) & 1 if i < 7 else 0
            bit, carry = table[a * 16 + b * 8 + carry * 4 + previous * 2 + following]
            out |= int(bit) << i
        return out


@torch.no_grad()
def rollout(model, alu, program, inputs):
    """No teacher forcing: every opcode, operand, destination and next PC comes from the Transformer."""
    encoded = torch.tensor(encode_program(program))
    n = len(inputs)
    regs = torch.zeros((n, REGS), dtype=torch.long)
    regs[:, 0] = torch.tensor([x for x, _ in inputs])
    regs[:, 1] = torch.tensor([y for _, y in inputs])
    pc = torch.zeros(n, dtype=torch.long)
    done = torch.zeros(n, dtype=torch.bool)
    steps = torch.zeros(n, dtype=torch.long)
    for _ in range(STEP_LIMIT):
        live = (~done).nonzero().flatten()
        if not len(live):
            break
        out = model(encoded[None].expand(len(live), -1, -1), pc[live], regs[live])
        picks = {k: v.argmax(1) for k, v in out.items()}
        for j, index in enumerate(live.tolist()):
            name = OPS[picks["op"][j].item()]
            steps[index] += 1
            if name == "HALT":
                done[index] = True
                continue
            if name in ALU:
                a, b, dst = (picks[k][j].item() for k in ("a", "b", "dst"))
                regs[index, dst] = alu(name, regs[index, a].item(), regs[index, b].item())
            elif name == "LOADI":
                regs[index, picks["dst"][j].item()] = picks["imm"][j].item()
            pc[index] = picks["next"][j].item()
    return regs[:, 2].tolist(), done.tolist(), steps.tolist()


def evaluate_program(model, alu, program, inputs):
    got, done, steps = rollout(model, alu, program, inputs)
    want = [exact_run(program, x, y) for x, y in inputs]
    correct = sum(g == w and d for g, w, d in zip(got, want, done))
    return {"correct": correct, "total": len(inputs), "halted": sum(done), "max_steps": max(steps)}


def test_programs(path, limit):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    concepts, seen = [], set()
    for row in rows:
        expr = nq.to_tuple(row["expression"])
        parts = set()
        nq.parts(expr, parts)
        for part in sorted(parts, key=repr):
            if part[0] in ("X", "Y", "K"):
                continue
            program = compile_concept(part)
            if program is not None and program_key(program) not in seen and len(program) >= 3:
                seen.add(program_key(program))
                concepts.append(program)
    concepts.sort(key=lambda p: (len(p), program_key(p)))
    return concepts[:: max(1, len(concepts) // limit)][:limit], len(concepts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 19, 29])
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--concepts", type=int, default=120)
    p.add_argument("--out", type=Path, default=HERE)
    a = p.parse_args()
    torch.set_num_threads(2)
    a.out.mkdir(parents=True, exist_ok=True)
    snapshot = a.out / "library_snapshot.jsonl"
    if a.library.resolve() != snapshot.resolve():
        snapshot.write_bytes(a.library.read_bytes())
    concepts, available = test_programs(snapshot, a.concepts)
    mul = multiply_program()
    held_out = {program_key(q) for q in concepts} | {program_key(mul)}
    rng = random.Random("NC-082/inputs")
    inputs = [(rng.randrange(256), rng.randrange(256)) for _ in range(16)]
    mul_inputs = [(rng.randrange(256), rng.randrange(256)) for _ in range(MUL_INPUTS)]
    report = {"schema": "logos.nc082.results/1", "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, "library_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
              "concept_programs_available": available, "concept_programs_tested": len(concepts), "program_lengths": [min(map(len, concepts)), max(map(len, concepts))] if concepts else None,
              "boundary": "The Transformer predicts opcode, operands, destination, immediate and next PC. NC-080 operators compute ALU bits. The host keeps registers, feeds predictions forward and iterates bits.", "seeds": {}}
    for seed in a.seeds:
        model, info = train(seed, a.steps, a.batch, held_out)
        alu = NeuralALU(seed)
        rows = [evaluate_program(model, alu, q, inputs) for q in concepts]
        perfect = sum(r["correct"] == r["total"] for r in rows)
        multiply = evaluate_program(model, alu, mul, mul_inputs)
        model.blank = True
        blank_rows = [evaluate_program(model, alu, q, inputs) for q in concepts[:: max(1, len(concepts) // 20)]]
        blank_mul = evaluate_program(model, alu, mul, mul_inputs[:64])
        model.blank = False
        checkpoint = a.out / f"controller_seed{seed}.pt"
        torch.save({"state": model.state_dict(), "seed": seed}, checkpoint)
        report["seeds"][str(seed)] = {**info, "concept_programs_fully_correct": perfect, "concept_rollouts_correct": sum(r["correct"] for r in rows), "concept_rollouts_total": sum(r["total"] for r in rows),
                                      "per_length": {str(n): [sum(r["correct"] == r["total"] for r, q in zip(rows, concepts) if len(q) == n), sum(len(q) == n for q in concepts)] for n in sorted({len(q) for q in concepts})},
                                      "multiply": multiply, "blank_program_control": {"programs_fully_correct": sum(r["correct"] == r["total"] for r in blank_rows), "programs": len(blank_rows), "multiply": blank_mul},
                                      "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        (a.out / "controller_results.json").write_text(json.dumps(report, indent=1))
        print(seed, "loss", info["final_loss"], "concepts", perfect, "of", len(concepts), "multiply", multiply["correct"], "of", multiply["total"], "halted", multiply["halted"], "blank", report["seeds"][str(seed)]["blank_program_control"]["programs_fully_correct"], blank_mul["correct"], "cpu", info["train_cpu_seconds"], flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
