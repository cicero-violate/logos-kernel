"""Run: OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-082/test_controller.py"""
import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import controller as c  # noqa: E402


class ControllerTests(unittest.TestCase):
    def test_multiply_program_really_multiplies(self):
        program = c.multiply_program()
        rng = random.Random(1)
        for _ in range(300):
            x, y = rng.randrange(256), rng.randrange(256)
            self.assertEqual(c.exact_run(program, x, y), (x * y) & 255)

    def test_compiled_concepts_match_exact_evaluation(self):
        rng = random.Random(2)
        expressions = [("SUB", ("SHL", ("ADD", ("X",), ("Y",))), ("X",)), ("ADD", ("SHL", ("SHL", ("X",))), ("X",)), ("XOR", ("K", 7), ("SHR", ("Y",))),
                       ("ADD", ("ADD", ("X",), ("Y",)), ("ADD", ("X",), ("Y",)))]
        for expr in expressions:
            program = c.compile_concept(expr)
            self.assertIsNotNone(program)
            for _ in range(100):
                x, y = rng.randrange(256), rng.randrange(256)
                self.assertEqual(c.exact_run(program, x, y), c.nq.reference_value(expr, x, y) if hasattr(c.nq, "reference_value") else self._value(expr, x, y))

    def test_branching_programs_compute_what_they_claim(self):
        rng = random.Random(6)
        for name, (program, want) in c.branching_programs().items():
            self.assertLessEqual(len(program), c.MAX_LEN)
            for _ in range(300):
                x, y = rng.randrange(256), rng.randrange(256)
                self.assertEqual(c.exact_run(program, x, y), want(x, y), name)

    def test_branch_curriculum_balances_the_zero_test(self):
        rng = random.Random(7)
        _, pc, regs, targets = c.batch_rows(rng, 2000, branch_fraction=1.0)
        self.assertTrue(all(o == c.OP["JZ"] for o in targets["op"].tolist()))
        tested = regs[torch.arange(2000), targets["a"]]
        self.assertAlmostEqual((tested == 0).float().mean().item(), 0.5, delta=0.05)

    def _value(self, expr, x, y):
        import numpy as np
        return int(c.nq.exact_eval(expr, {}) [x * 256 + y])

    def test_random_programs_stay_valid_and_never_equal_held_out_programs(self):
        rng = random.Random(3)
        held = {c.program_key(c.multiply_program())}
        for _ in range(3000):
            program = c.random_program(rng)
            self.assertEqual(program[-1][0], "HALT")
            self.assertLessEqual(len(program), c.MAX_LEN)
            self.assertNotIn(c.program_key(program), held)
            for item in program:
                if item[0] in ("JZ", "JMP"):
                    self.assertLess(item[4], len(program))

    def test_blanking_removes_program_information(self):
        model = c.Controller(dim=16, layers=1, heads=2).eval()
        rng = random.Random(4)
        a = torch.tensor(c.encode_program(c.random_program(rng)))[None]
        b = torch.tensor(c.encode_program(c.random_program(rng)))[None]
        pc = torch.zeros(1, dtype=torch.long)
        regs = torch.zeros((1, c.REGS), dtype=torch.long)
        with torch.no_grad():
            self.assertFalse(torch.equal(model(a, pc, regs)["op"], model(b, pc, regs)["op"]))
            model.blank = True
            self.assertTrue(torch.equal(model(a, pc, regs)["op"], model(b, pc, regs)["op"]))

    def test_neural_alu_matches_exact_operations(self):
        alu = c.NeuralALU(11)
        rng = random.Random(5)
        for _ in range(500):
            x, y = rng.randrange(256), rng.randrange(256)
            for name in c.ALU:
                self.assertEqual(alu(name, x, y), c.exact_alu(name, x, y), name)

    def test_rollout_uses_only_model_predictions(self):
        """A model that always predicts HALT must halt at once, whatever the program says."""
        class AlwaysHalt(torch.nn.Module):
            def forward(self, program, pc, regs):
                n = len(pc)
                out = {k: torch.zeros(n, size) for k, size in (("op", len(c.OPS)), ("a", c.REGS), ("b", c.REGS), ("dst", c.REGS), ("imm", 256), ("next", c.MAX_LEN + 1))}
                out["op"][:, c.OP["HALT"]] = 1
                return out
        got, done, steps = c.rollout(AlwaysHalt(), c.NeuralALU(11), c.multiply_program(), [(3, 5)])
        self.assertEqual((got, done, steps), ([0], [True], [1]))


if __name__ == "__main__":
    unittest.main()
