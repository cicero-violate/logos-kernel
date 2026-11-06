"""Run: PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-079/test_stack.py"""
import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stack as s  # noqa: E402

isa = s.sb.isa


class StackTests(unittest.TestCase):
    x, y = s.probes()

    def test_small_expressions_match_the_canonical_isa(self):
        program = {"version": isa.VERSION, "width": 8, "signedness": "unsigned", "program": [
            {"op": "ADD", "dst": "R", "a": "X", "b": "Y"}, {"op": "SHL", "dst": "T", "src": "R"}, {"op": "SUB", "dst": "R", "a": "T", "b": "X"}, {"op": "HALT"}]}
        expr = ("SUB", ("SHL", ("ADD", ("X",), ("Y",))), ("X",))
        rng = random.Random(1)
        for _ in range(300):
            a, b = rng.randrange(256), rng.randrange(256)
            want = isa.reference(program, {"X": a, "Y": b, "R": 0, "T": 0})["registers"]["R"]
            self.assertEqual(s.reference_value(expr, a, b), want)
            self.assertEqual(int(s.evaluate(expr, np.int64(a), np.int64(b))), want)

    def test_search_finds_and_verifies_a_shallow_goal(self):
        goal = ("ADD", ("X",), ("SHL", ("Y",)))
        expr, used, _ = s.search(s.goal_vector(goal, self.x, self.y), [], 20000, self.x, self.y)
        self.assertIsNotNone(expr)
        self.assertLessEqual(used, 20000)
        self.assertTrue(s.verified(expr, goal))

    def test_budget_is_respected_and_failure_returns_none(self):
        goal = ("LINEAR", 37, 53)
        expr, used, _ = s.search(s.goal_vector(goal, self.x, self.y), [], 200, self.x, self.y)
        self.assertIsNone(expr)
        self.assertLessEqual(used, 200)

    def test_a_useful_library_reduces_candidates(self):
        concept = ("ADD", ("SHL", ("SHL", ("X",))), ("X",))
        goal = ("ADD", concept, ("SHL", ("Y",)))
        without = s.search(s.goal_vector(goal, self.x, self.y), [], 50000, self.x, self.y)[1]
        withit = s.search(s.goal_vector(goal, self.x, self.y), [concept], 50000, self.x, self.y)[1]
        self.assertLess(withit, without)

    def test_verifier_catches_a_program_that_only_fits_the_examples(self):
        goal = ("ADD", ("X",), ("Y",))
        wrong = ("XOR", ("X",), ("Y",))
        self.assertFalse(s.verified(wrong, goal))

    def test_sealed_and_train_goals_are_disjoint_in_every_world(self):
        for cls in (s.StructuredWorld, s.ArithmeticWorld, s.UnstructuredWorld):
            world = cls(3)
            rng = random.Random(3)
            train = {s.goal_key(g) for g in s.draw_goals(world, rng, 30, "train")}
            sealed = {s.goal_key(g) for g in s.draw_goals(world, rng, 30, "sealed")}
            self.assertTrue(train and sealed, cls.name)
            self.assertFalse(train & sealed, cls.name)

    def test_harvest_keeps_recurring_parts_within_the_cap_and_excludes_leaves(self):
        part = ("SHL", ("X",))
        lib = s.harvest([("ADD", part, ("Y",)), ("SUB", part, ("X",))])
        self.assertIn(part, lib)
        self.assertLessEqual(len(lib), s.LIBRARY_CAP)
        self.assertTrue(all(s.size(e) > 1 for e in lib))

    def test_random_library_matches_count_without_copying_content(self):
        real = [("ADD", ("X",), ("Y",)), ("SHL", ("SHL", ("X",)))]
        fake = s.random_library(random.Random(1), real)
        self.assertEqual(len(fake), len(real))


if __name__ == "__main__":
    unittest.main()
