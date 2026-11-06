"""NC-010: a Transformer writes programs with branches and loops from input-output examples alone.

Training shows bit-loop skeletons over most operator combinations. Every program with a held-out combination is left out, so
shift-and-add multiplication is never shown. Asked for multiplication from examples, the model must recombine what it learned.
This is recombination of taught parts, not invention from nothing. An arm trained without skeletons shows how much they matter.
Candidates are checked on the examples, then on all 65,536 pairs, then run on NC-082's machine. Nothing else is accepted.
"""
import argparse
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("nc082_controller", HERE.parent / "NC-082" / "controller.py")
c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c)

STEPS = ("SHL", "SHR", None)
COMBINE = ("ADD", "SUB", "XOR", "OR", "AND")
INITS = (0, 1, 255)
HELD_OUT = (("ADD", "SHL"), ("XOR", "SHL"), ("ADD", "SHR"), ("OR", "SHL"))  # With every initial value and loop register.
SANITY = (("SUB", "SHL", 0), ("AND", "SHR", 255), ("XOR", "SHR", 0), ("OR", "SHR", 1), ("ADD", None, 0), ("AND", "SHL", 255))
MAX_INSTRUCTIONS = 12
EXAMPLES = 16
BUDGET = 4096
ROUND = 128
OP_TOKENS = {name: i + 3 for i, name in enumerate(c.OPS)}
REG_BASE = 3 + len(c.OPS)
IMM_BASE = REG_BASE + c.REGS
VOCAB = IMM_BASE + 256
PAD, BOS, EOS = 0, 1, 2
FIELDS = 5
TARGET = MAX_INSTRUCTIONS * FIELDS + 2


def skeleton(combine, step, init, rng=None, loop=1):
    """A bit loop: for each bit of one input, combine the other into r2 if the bit is set, then transform it and shift."""
    rng = rng or random.Random(0)
    other = 1 - loop
    mask, temp = rng.sample(range(3, c.REGS), 2)
    loads = [("LOADI", 0, 0, mask, 1), ("LOADI", 0, 0, 2, init)]
    if rng.random() < .5:
        loads.reverse()
    shift = ("SHR", loop, 0, loop, 0)
    transform = [(step, other, 0, other, 0)] if step else []
    tail = [*transform, shift] if rng.random() < .5 else [shift, *transform]
    after = 7 + len(tail)  # The HALT: two loads, JZ, AND, JZ, combine, the tail, then JMP.
    body = [*loads, ("JZ", loop, 0, 0, after), ("AND", loop, mask, temp, 0), ("JZ", temp, 0, 0, 6), (combine, 2, other, 2, 0), *tail, ("JMP", 0, 0, 0, 2), ("HALT", 0, 0, 0, 0)]
    return body


def is_held_out(combine, step):
    return (combine, step) in HELD_OUT


def combos():
    return [(a, b) for a in COMBINE for b in STEPS]


def inputs_for_examples():
    rng = random.Random("NC-010/examples")
    fixed = [(0, 0), (1, 1), (2, 3), (3, 5), (7, 9), (15, 16), (255, 1), (1, 255), (128, 2), (2, 128)]
    return fixed + [(rng.randrange(256), rng.randrange(256)) for _ in range(EXAMPLES - len(fixed))]


EXAMPLE_INPUTS = inputs_for_examples()


def behaviour(program):
    out = [c.exact_run(program, x, y) for x, y in EXAMPLE_INPUTS]
    return None if any(v is None for v in out) else tuple(out)


def held_out_goals():
    """Each held-out combination's function, from canonical programs used only as the exam's reference, never in training."""
    return {f"{a}_{b}": skeleton(a, b, 0) for a, b in HELD_OUT}


def sanity_goals():
    return {f"{a}_{b}_{i}": skeleton(a, b, i) for a, b, i in SANITY}


def source_tokens(outputs):
    tokens = []
    for (x, y), value in zip(EXAMPLE_INPUTS, outputs):
        tokens += [IMM_BASE + x, IMM_BASE + y, IMM_BASE + value]
    return tokens


def canonical(program):
    """Unused fields become zero, so each instruction has one spelling."""
    out = []
    for name, a, b, dst, imm in program:
        if name in ("SHL", "SHR"):
            out.append((name, a, 0, dst, 0))
        elif name in c.ALU:
            out.append((name, a, b, dst, 0))
        elif name == "LOADI":
            out.append((name, 0, 0, dst, imm))
        elif name == "JZ":
            out.append((name, a, 0, 0, imm))
        elif name == "JMP":
            out.append((name, 0, 0, 0, imm))
        else:
            out.append((name, 0, 0, 0, 0))
    return out


def encode(program):
    tokens = [BOS]
    for name, a, b, dst, imm in canonical(program):
        tokens += [OP_TOKENS[name], REG_BASE + a, REG_BASE + b, REG_BASE + dst, IMM_BASE + imm]
    tokens.append(EOS)
    return tokens + [PAD] * (TARGET - len(tokens))


NAMES = {v: k for k, v in OP_TOKENS.items()}


def decode(tokens):
    """Strict: a malformed, unterminated or out-of-range program returns None."""
    program = []
    for start in range(0, len(tokens), FIELDS):
        if tokens[start] == EOS:
            break
        chunk = tokens[start:start + FIELDS]
        if len(chunk) < FIELDS or chunk[0] not in NAMES:
            return None
        name = NAMES[chunk[0]]
        if not all(REG_BASE <= t < IMM_BASE for t in chunk[1:4]) or not IMM_BASE <= chunk[4] < VOCAB:
            return None
        program.append((name, chunk[1] - REG_BASE, chunk[2] - REG_BASE, chunk[3] - REG_BASE, chunk[4] - IMM_BASE))
    else:
        return None
    if not program or program[-1][0] != "HALT" or len(program) > MAX_INSTRUCTIONS:
        return None
    if any(n in ("JZ", "JMP") and i >= len(program) for n, _, _, _, i in program):
        return None
    return program


def type_mask():
    """Which tokens are legal at each target position. Structure is given; every choice within it is the model's."""
    ops = torch.zeros(VOCAB, dtype=torch.bool)
    ops[list(OP_TOKENS.values()) + [EOS]] = True
    regs = torch.zeros(VOCAB, dtype=torch.bool)
    regs[REG_BASE:IMM_BASE] = True
    imms = torch.zeros(VOCAB, dtype=torch.bool)
    imms[IMM_BASE:] = True
    return [ops, regs, regs, regs, imms]


MASKS = type_mask()


def make_data(seed, count, use_skeletons, held_tables):
    """Skeletons from non-held combinations plus random halting programs, with no held-out behaviour on any example set."""
    rng = random.Random(f"NC-010/data/{seed}/{use_skeletons}")
    allowed = [(a, b) for a, b in combos() if not is_held_out(a, b)]
    sources, targets, seen, dropped = [], [], set(), 0
    tries = 0
    while len(sources) < count and tries < count * 40:
        tries += 1
        from_skeleton = use_skeletons and rng.random() < .7
        if from_skeleton:
            a, b = rng.choice(allowed)
            program = skeleton(a, b, rng.choice(INITS), rng, rng.choice((0, 1)))
        else:
            program = c.random_program(rng)
        if len(program) > MAX_INSTRUCTIONS:
            continue
        out = behaviour(program)
        if out is None or len(set(out)) < 3:
            continue
        if out in held_tables:
            dropped += 1
            continue
        key = c.program_key(canonical(program))
        if key in seen and not from_skeleton:  # The skeleton space is small, so repeats keep it 70% of the mix.
            continue
        seen.add(key)
        sources.append(source_tokens(out))
        targets.append(encode(program))
    return torch.tensor(sources), torch.tensor(targets), {"programs": len(sources), "dropped_for_held_out_behaviour": dropped, "tries": tries}


class Writer(nn.Module):
    def __init__(self, dim=96, layers=2, heads=4):
        super().__init__()
        self.token = nn.Embedding(VOCAB, dim, padding_idx=PAD)
        self.source_position = nn.Embedding(EXAMPLES * 3, dim)
        self.target_position = nn.Embedding(TARGET, dim)
        self.net = nn.Transformer(dim, heads, layers, layers, dim * 2, dropout=0.0, batch_first=True, activation="gelu")
        self.head = nn.Linear(dim, VOCAB)

    def memory(self, source):
        return self.net.encoder(self.token(source) + self.source_position(torch.arange(source.shape[1])))

    def step(self, target, memory):
        h = self.token(target) + self.target_position(torch.arange(target.shape[1]))
        mask = nn.Transformer.generate_square_subsequent_mask(target.shape[1])
        return self.head(self.net.decoder(h, memory, tgt_mask=mask, tgt_is_causal=True))


def train(seed, data, steps, batch):
    started = time.process_time()
    torch.manual_seed(seed)
    source, target = data
    model = Writer()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        pick = torch.randint(len(source), (batch,), generator=gen)
        s, t = source[pick], target[pick]
        logits = model.step(t[:, :-1], model.memory(s))
        loss = nn.functional.cross_entropy(logits.reshape(-1, VOCAB), t[:, 1:].reshape(-1), ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    model.eval()
    return model, {"steps": steps, "final_loss": round(loss.item(), 4), "parameters": sum(p.numel() for p in model.parameters()), "train_cpu_seconds": round(time.process_time() - started, 1)}


@torch.no_grad()
def sample(model, outputs, count, gen, greedy_first=False):
    source = torch.tensor([source_tokens(outputs)]).repeat(count, 1)
    memory = model.memory(source)
    target = torch.full((count, 1), BOS, dtype=torch.long)
    for position in range(TARGET - 1):
        logits = model.step(target, memory)[:, -1]
        logits[:, ~MASKS[position % FIELDS]] = -1e9
        pick = torch.multinomial(torch.softmax(logits, 1), 1, generator=gen).flatten()
        if greedy_first:
            pick[0] = logits[0].argmax()
        target = torch.cat((target, pick[:, None]), 1)
    return [decode(row) for row in target[:, 1:].tolist()]


def exact_table(program):
    return [c.exact_run(program, x, y) for x in range(256) for y in range(256)]


def computes(program, table):
    """All 65,536 pairs. This is the only acceptance test, and it stops at the first difference."""
    index = 0
    for x in range(256):
        for y in range(256):
            if c.exact_run(program, x, y) != table[index]:
                return False
            index += 1
    return True


def search(propose, examples, table, budget):
    """Spend up to budget candidates. Fitting the examples is only a filter before the exhaustive check."""
    used, malformed, fit_but_wrong = 0, 0, 0
    while used < budget:
        for program in propose():
            if used >= budget:
                break
            used += 1
            if program is None:
                malformed += 1
            elif behaviour(program) == examples:
                if computes(program, table):
                    return {"solved": True, "candidates": used, "malformed": malformed, "fit_but_wrong": fit_but_wrong, "program": program}
                fit_but_wrong += 1
    return {"solved": False, "candidates": used, "malformed": malformed, "fit_but_wrong": fit_but_wrong, "program": None}


def run_on_neural_machine(program, seed):
    model = c.Controller()
    model.load_state_dict(torch.load(HERE.parent / "NC-082" / f"controller_seed{seed}.pt", weights_only=True)["state"])
    model.eval()
    alu = c.NeuralALU(seed)
    rng = random.Random("NC-010/neural")
    edges = (0, 1, 2, 3, 15, 16, 127, 128, 170, 254, 255)
    pairs = [(rng.randrange(256), rng.randrange(256)) for _ in range(4096)] + [(x, y) for x in edges for y in edges]
    correct = 0
    for start in range(0, len(pairs), 512):
        part = pairs[start:start + 512]
        got, done, _ = c.rollout(model, alu, program, part)
        correct += sum(d and g == c.exact_run(program, x, y) for (x, y), g, d in zip(part, got, done))
    return {"correct": correct, "total": len(pairs)}


def clean(result):
    return {**result, "program": [list(i) for i in result["program"]] if result["program"] else None}


def random_batch(rng):
    """Whole random programs within the length limit. Cutting one short could strand a jump past its end."""
    out = []
    while len(out) < ROUND:
        program = c.random_program(rng)
        if len(program) <= MAX_INSTRUCTIONS:
            out.append(program)
    return out


def run_seed(seed, a, goals, tables, held_tables):
    report = {}
    for arm, use_skeletons in (("skeletons", True), ("random_programs_only", False)):
        source, target, info = make_data(seed, a.programs, use_skeletons, held_tables)
        checkpoint = a.out / f"writer_{arm}_seed{seed}.pt"
        if a.reuse_checkpoints:
            model = Writer()
            model.load_state_dict(torch.load(checkpoint, weights_only=True)["state"])
            model.eval()
            trained = {"reused_checkpoint": True, "steps": a.steps, "parameters": sum(p.numel() for p in model.parameters())}
        else:
            model, trained = train(seed, (source, target), a.steps, a.batch)
            torch.save({"state": model.state_dict(), "seed": seed}, checkpoint)
        gen = torch.Generator().manual_seed(seed)
        rows = {}
        for name, program in goals.items():
            examples = behaviour(program)
            first = {"on": True}
            def propose():
                out = sample(model, examples, ROUND, gen, greedy_first=first["on"])
                first["on"] = False
                return out
            rows[name] = clean(search(propose, examples, tables[name], a.budget))
            print(seed, arm, name, rows[name]["solved"], rows[name]["candidates"], "fit-wrong", rows[name]["fit_but_wrong"], flush=True)
        row = {"data": info, "training": trained, "goals": rows, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        if arm == "skeletons":
            controls = {}
            others = list(goals)
            for name in held_out_goals():
                other = others[(others.index(name) + 1) % len(others)]
                shown = behaviour(goals[other])
                controls[name] = clean(search(lambda: sample(model, shown, ROUND, gen), behaviour(goals[name]), tables[name], a.budget))
                print(seed, "other_goal_examples", name, controls[name]["solved"], flush=True)
            row["other_goal_examples_control"] = controls
            row["neural_machine"] = {name: run_on_neural_machine([tuple(i) for i in rows[name]["program"]], seed) for name in held_out_goals() if rows[name]["solved"]}
            print(seed, "neural machine", row["neural_machine"], flush=True)
        report[arm] = row
        (a.out / f"invention_receipt_seed{seed}.json").write_text(json.dumps(report, indent=1))  # A later failure must not lose this arm.
    rng = random.Random(f"NC-010/random/{seed}")
    report["random_programs_control"] = {name: clean(search(lambda: random_batch(rng), behaviour(goals[name]), tables[name], a.budget)) for name in held_out_goals()}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 19, 29])
    p.add_argument("--programs", type=int, default=60000)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--budget", type=int, default=BUDGET)
    p.add_argument("--reuse-checkpoints", action="store_true", help="Evaluate saved writers instead of training.")
    p.add_argument("--out", type=Path, default=HERE)
    a = p.parse_args()
    torch.set_num_threads(2)
    a.out.mkdir(parents=True, exist_ok=True)
    goals = {**held_out_goals(), **sanity_goals()}
    tables = {name: exact_table(program) for name, program in goals.items()}
    held_tables = {behaviour(program) for program in held_out_goals().values()}
    assert all(tables["ADD_SHL"][x * 256 + y] == (x * y) & 255 for x in range(256) for y in range(256))
    report = {"schema": "logos.nc010.results/1", "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, "held_out_combinations": HELD_OUT, "sanity_goals": list(sanity_goals()), "seeds": {}}
    for seed in a.seeds:
        report["seeds"][str(seed)] = run_seed(seed, a, goals, tables, held_tables)
        (a.out / f"invention_receipt_seed{seed}.json").write_text(json.dumps(report["seeds"][str(seed)], indent=1))
    (a.out / "program_invention_receipt.json").write_text(json.dumps(report, indent=1))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
