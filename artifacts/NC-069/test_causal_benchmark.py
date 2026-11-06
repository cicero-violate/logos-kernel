"""Run: PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-069/test_causal_benchmark.py"""
import random
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import causal_benchmark as cb  # noqa: E402

MASK = cb.VALUES - 1


def independent_run(program, inputs, intervention):
    """A separately written evaluator that applies interventions during execution, not by rewriting."""
    regs = dict(inputs)
    if intervention["kind"] == "change_input":
        regs[intervention["register"]] = intervention["value"]
    body = program["program"]
    pc = 0
    while True:
        item = body[pc]
        if intervention["kind"] == "set_register" and intervention["step"] == pc:
            regs[intervention["register"]] = intervention["value"]
        skip = intervention["kind"] == "skip_step" and intervention["step"] == pc
        if intervention["kind"] == "replace_instruction" and intervention["step"] == pc:
            item = intervention["instruction"]
        op = item["op"]
        if op == "HALT":
            return [regs[n] for n in ("X", "Y", "R", "T")]
        next_pc = pc + 1
        if not skip:
            if op == "LOADI":
                regs[item["dst"]] = item["imm"]
            elif op == "MOV":
                regs[item["dst"]] = regs[item["src"]]
            elif op in ("ADD", "SUB", "AND", "OR", "XOR"):
                a, b = regs[item["a"]], regs[item["b"]]
                regs[item["dst"]] = {"ADD": a + b, "SUB": a - b, "AND": a & b, "OR": a | b, "XOR": a ^ b}[op] & MASK
            elif op == "AND1":
                regs[item["dst"]] = regs[item["src"]] & 1
            elif op == "SHL":
                regs[item["dst"]] = (regs[item["src"]] << 1) & MASK
            elif op == "SHR":
                regs[item["dst"]] = regs[item["src"]] >> 1
            elif op == "SAR":
                value = regs[item["src"]]
                regs[item["dst"]] = (value >> 1) | (value & (1 << (cb.WIDTH - 1)))
            elif op == "JMP":
                next_pc = item["target"]
            elif op == "JZ" and regs[item["src"]] == 0:
                next_pc = item["target"]
            elif op == "JNZ" and regs[item["src"]] != 0:
                next_pc = item["target"]
        pc = next_pc


class BenchmarkTests(unittest.TestCase):
    def test_oracle_matches_a_separately_written_evaluator(self):
        rng = random.Random(1)
        seen = set()
        for _ in range(5000):
            program = cb.random_program(rng)
            inputs = {n: rng.randrange(cb.VALUES) for n in ("X", "Y", "R", "T")}
            intervention = cb.random_intervention(rng, program)
            seen.add(intervention["kind"])
            self.assertEqual(cb.outcome(program, inputs, intervention),
                             independent_run(program, inputs, intervention), intervention)
        self.assertEqual(seen, set(cb.KINDS[1:]))

    def _program(self, *body):
        return cb.isa.validate({"version": cb.isa.VERSION, "width": cb.WIDTH, "signedness": cb.SIGN,
                                "program": [*body, {"op": "HALT"}]})

    def test_set_register_applies_on_a_path_that_jumps_to_the_step(self):
        program = self._program({"op": "JMP", "target": 2}, {"op": "LOADI", "dst": "X", "imm": 9},
                                {"op": "MOV", "dst": "R", "src": "X"})
        inputs = {"X": 1, "Y": 0, "R": 0, "T": 0}
        at_target = {"kind": "set_register", "step": 2, "register": "X", "value": 5}
        self.assertEqual(cb.outcome(program, inputs, at_target)[2], 5)
        skipped = {"kind": "set_register", "step": 1, "register": "X", "value": 5}
        self.assertEqual(cb.outcome(program, inputs, skipped), cb.outcome(program, inputs))

    def test_skip_keeps_flags_and_targets_valid(self):
        program = self._program({"op": "ADD", "dst": "R", "a": "X", "b": "Y"}, {"op": "JNZ", "src": "R", "target": 3},
                                {"op": "LOADI", "dst": "T", "imm": 7})
        inputs = {"X": 1, "Y": 1, "R": 0, "T": 0}
        self.assertEqual(cb.outcome(program, inputs, {"kind": "skip_step", "step": 0}), [1, 1, 0, 7])

    def test_splits_are_disjoint_on_programs_and_held_out_combinations(self):
        train = cb.generate(0, 800, "train") + cb.generate(0, 400, "passive_train")
        combos = cb.generate(0, 400, "heldout_combo")
        seen = cb.generate(0, 400, "heldout_program")
        train_programs = {cb.isa.encode(s["program"]) for s in train}
        for s in combos + seen:
            self.assertNotIn(cb.isa.encode(s["program"]), train_programs)
        train_combos = {s["combo"] for s in train}
        for s in combos:
            self.assertTrue(cb.is_heldout_combo(s["combo"]))
            self.assertNotIn(s["combo"], train_combos)
        for s in seen:
            self.assertFalse(cb.is_heldout_combo(s["combo"]))

    def test_generation_is_deterministic_and_passive_splits_never_intervene(self):
        self.assertEqual(cb.generate(3, 50, "train"), cb.generate(3, 50, "train"))
        self.assertNotEqual(cb.generate(3, 50, "train"), cb.generate(4, 50, "train"))
        for split in ("passive_train", "factual"):
            self.assertTrue(all(s["intervention"]["kind"] == "none" for s in cb.generate(1, 50, split)))

    def test_interventions_change_some_outcomes_but_not_all(self):
        samples = cb.generate(0, 800, "heldout_program")
        changed = sum(s["target"] != s["factual"] for s in samples)
        self.assertGreater(changed, 80)
        self.assertLess(changed, 790)


if __name__ == "__main__":
    unittest.main()
