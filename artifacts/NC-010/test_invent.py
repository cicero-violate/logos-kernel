"""Run: OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-010/test_invent.py"""
import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import invent as n  # noqa: E402

c = n.c


class InventTests(unittest.TestCase):
    def test_skeleton_computes_multiplication_in_every_spelling(self):
        rng = random.Random(1)
        for _ in range(30):
            program = n.skeleton("ADD", "SHL", 0, rng, rng.choice((0, 1)))
            for _ in range(40):
                x, y = rng.randrange(256), rng.randrange(256)
                self.assertEqual(c.exact_run(program, x, y), (x * y) & 255)

    def test_skeleton_without_a_transform_jumps_correctly(self):
        rng = random.Random(2)
        for _ in range(20):
            program = n.skeleton("XOR", None, 0, rng, 1)
            for _ in range(40):
                x, y = rng.randrange(256), rng.randrange(256)
                self.assertEqual(c.exact_run(program, x, y), x if bin(y).count("1") % 2 else 0)

    def test_tokens_round_trip(self):
        rng = random.Random(3)
        for _ in range(100):
            program = n.canonical(n.skeleton(rng.choice(n.COMBINE), rng.choice(n.STEPS), rng.choice(n.INITS), rng))
            tokens = n.encode(program)[1:]
            self.assertEqual(n.decode(tokens), program)

    def test_malformed_programs_are_refused(self):
        good = n.encode(n.skeleton("SUB", "SHL", 0))[1:]
        self.assertIsNone(n.decode(good[:7]))
        no_halt = list(good)
        no_halt[n.FIELDS * 9] = n.OP_TOKENS["JMP"]
        self.assertIsNone(n.decode(no_halt))
        far = list(good)
        far[n.FIELDS * 8 + 4] = n.IMM_BASE + 200
        self.assertIsNone(n.decode(far))

    def test_training_data_never_contains_a_held_out_behaviour_or_program(self):
        held = {n.behaviour(p) for p in n.held_out_goals().values()}
        source, target, info = n.make_data(1, 1500, True, held)
        self.assertEqual(info["programs"], 1500)
        for s, t in zip(source.tolist(), target.tolist()):
            outputs = tuple(v - n.IMM_BASE for v in s[2::3])
            self.assertNotIn(outputs, held)
            program = n.decode(t[1:])
            self.assertEqual(n.behaviour(program), outputs)
            self.assertNotEqual(c.exact_run(program, 7, 9), 63 if c.exact_run(program, 3, 5) == 15 and c.exact_run(program, 255, 3) == 253 else -1)

    def test_only_a_program_right_on_every_pair_is_accepted(self):
        goal = n.skeleton("ADD", "SHL", 0)
        table = n.exact_table(goal)
        near = [("LOADI", 0, 0, 2, 0), ("HALT", 0, 0, 0, 0)]
        examples = n.behaviour(goal)
        bad = n.search(lambda: [near], examples, table, 3)
        self.assertFalse(bad["solved"])
        table2 = list(table)
        table2[200 * 256 + 201] ^= 1
        fooled = n.search(lambda: [goal], examples, table2, 3)
        self.assertFalse(fooled["solved"])
        self.assertEqual(fooled["fit_but_wrong"], 3)
        self.assertTrue(n.search(lambda: [goal], examples, table, 3)["solved"])

    def test_random_control_programs_always_run(self):
        """Regression: cutting programs to length once stranded a jump past the end and crashed the control."""
        rng = random.Random(9)
        for _ in range(8):
            for program in n.random_batch(rng):
                self.assertLessEqual(len(program), n.MAX_INSTRUCTIONS)
                n.behaviour(program)

    def test_sampling_obeys_the_token_types_and_budget(self):
        torch.manual_seed(1)
        model = n.Writer(dim=16, layers=1, heads=2).eval()
        out = n.sample(model, n.behaviour(n.skeleton("SUB", "SHL", 0)), 4, torch.Generator().manual_seed(1))
        self.assertEqual(len(out), 4)
        result = n.search(lambda: [None, None], (0,), [0] * 65536, 5)
        self.assertEqual((result["candidates"], result["malformed"]), (5, 5))


if __name__ == "__main__":
    unittest.main()
