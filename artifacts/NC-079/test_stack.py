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

    def test_random_library_matches_exact_sizes(self):
        # Reviewer counterexample: requested sizes 3 and 5 produced 4 and 7.
        targets = (2, 3, 3, 4, 5, 7, 9, 13, 25, 40)
        like = [self._of_size(n) for n in targets]
        for trial in range(30):
            fake = s.random_library(random.Random(trial), like)
            self.assertEqual([s.size(e) for e in fake], list(targets))

    @staticmethod
    def _of_size(n):
        expr = ("X",)
        for _ in range(n - 1):
            expr = ("SHL", expr)
        return expr

    def test_random_arm_matches_the_real_arm_at_every_generation(self):
        """Regression: the random arm once copied sizes from its own solutions, which fall behind the real arm's."""
        import argparse
        import run_compounding as rc
        import run_curriculum as cu
        x, y = s.probes()
        a = argparse.Namespace(generations=4, budget=3000, train_goals=10, sealed_goals=3, retention_goals=3)
        for world in (s.StructuredWorld, s.UnstructuredWorld):
            real, grown = rc.run_arm(world, "library", 11, a, x, y)
            fake, _ = rc.run_arm(world, "random_library", 11, a, x, y, grown)
            s.assert_matched_sizes(real, fake)
        real, grown = cu.run_arm("library", 11, a, x, y)
        fake, _ = cu.run_arm("random_library", 11, a, x, y, grown)
        s.assert_matched_sizes(real, fake)
        self.assertGreater(real[-1]["library_size"], 0, "the check must not pass on empty libraries")
        real[-1] = {**real[-1], "library_size": real[-1]["library_size"] + 1}
        with self.assertRaises(ValueError):
            s.assert_matched_sizes(real, fake)

    def test_separate_evaluator_covers_every_pair(self):
        """A separate evaluator wrong on one pair must fail verification, wherever that pair is."""
        from unittest import mock
        expr, goal = ("ADD", ("X",), ("Y",)), ("LINEAR", 1, 1)
        self.assertTrue(s.verified(expr, goal))
        real, calls = s.reference_value, []
        for bad in ((0, 0), (255, 254), (17, 200)):
            def wrong(e, a, b, bad=bad):
                calls.append(1)
                return (real(e, a, b) + 1) & 255 if (a, b) == bad else real(e, a, b)
            with mock.patch.object(s, "reference_value", wrong):
                self.assertFalse(s.verified(expr, goal), bad)
        calls.clear()
        with mock.patch.object(s, "reference_value", lambda e, a, b: (calls.append((a, b)), real(e, a, b))[1]):
            self.assertTrue(s.verified(expr, goal))
        self.assertEqual(len(set(calls)), 65536)  # It recurses, so count distinct input pairs, not calls.

    def test_every_arm_gets_the_same_exam(self):
        # Reviewer counterexample: arms shared only 0 to 3 of 30 goals.
        for cls in (s.StructuredWorld, s.ArithmeticWorld, s.UnstructuredWorld):
            world = cls(11)
            first = s.draw_goals(world, s.exam_rng(world.name, 11, 0, "sealed"), 30, "sealed")
            again = s.draw_goals(cls(11), s.exam_rng(world.name, 11, 0, "sealed"), 30, "sealed")
            self.assertEqual(first, again)
            self.assertNotEqual(first, s.draw_goals(world, s.exam_rng(world.name, 11, 1, "sealed"), 30, "sealed"))


if __name__ == "__main__":
    unittest.main()
