"""Run: OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-081/test_selector.py"""
import random
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import selector as sel  # noqa: E402

lib = sel.lib
CONCEPTS = [("SHL", ("X",)), ("ADD", ("X",), ("Y",)), ("SUB", ("SHL", ("X",)), ("Y",)), ("XOR", ("X",), ("SHR", ("Y",))), ("ADD", ("SHL", ("SHL", ("X",))), ("X",))]


class SelectorTests(unittest.TestCase):
    def test_no_sealed_side_concept_is_ever_a_label(self):
        path = Path(sel.__file__).parent / "library_seed11.jsonl"
        concepts = lib.Library(path).concepts()
        sealed = [i for i, c in enumerate(concepts) if sel.is_sealed_key(sel.function_key(c))]
        self.assertTrue(sealed, "the snapshot must contain sealed-side concepts for this test to mean anything")
        _, labels, _ = sel.make_dreams(11, concepts, 400)
        self.assertEqual(labels[:, sealed].sum().item(), 0)
        self.assertGreater(labels.sum().item(), 400)

    def test_a_changed_library_is_refused_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp) / "library.py"
            changed.write_bytes(sel.LIB.read_bytes() + b"\n")
            with self.assertRaisesRegex(SystemExit, "pinned library mismatch"):
                sel.load_pinned(changed, sel.LIB_SHA256)
        self.assertEqual(sel.load_pinned(sel.LIB, sel.LIB_SHA256).MASK, lib.MASK)

    def test_second_evaluator_covers_the_whole_domain(self):
        calls, real = [], lib.run_stack
        def count(code, x, y):
            calls.append(1)
            return real(code, x, y)
        with mock.patch.object(lib, "run_stack", count):
            self.assertTrue(lib.verify(("ADD", ("X",), ("Y",)), ("linear", 1, 1, 1)))
        self.assertEqual(len(calls), 65536)

    def test_calibration_reports_measured_cpu_and_refuses_a_false_match(self):
        seen = []
        def linear(budget):
            seen.append(budget)
            return {"solved": 1, "cpu_seconds": budget / 100}
        good = sel.calibrate(linear, 100.0, 5000)
        self.assertTrue(good["matched"])
        self.assertLessEqual(abs(good["cpu_seconds"] - 100), 10)
        self.assertEqual(len(seen), 2)
        stuck = sel.calibrate(lambda budget: {"solved": 0, "cpu_seconds": 300.0}, 100.0, 5000)
        self.assertFalse(stuck["matched"])
        self.assertEqual(len(stuck["attempts"]), sel.CALIBRATION_TRIES)

    def test_function_key_matches_the_library_goal_key(self):
        rng = random.Random(1)
        for _ in range(40):
            a, b = rng.randrange(1, 4), rng.randrange(1, 4)
            goal = ("linear", 1, 1, 1)
            self.assertEqual(sel.function_key(("ADD", ("X",), ("Y",))), lib.goal_table_key(goal))
            expr = ("ADD", ("SHL", ("X",)), ("Y",))
            self.assertEqual(sel.function_key(expr), lib.goal_table_key(("linear", 2, 1, 1)))

    def test_exam_is_deterministic_sealed_and_distinct(self):
        goals = sel.exam(3, 30)
        self.assertEqual(goals, sel.exam(3, 30))
        self.assertNotEqual(goals, sel.exam(4, 30))
        self.assertTrue(all(lib.is_sealed_side(g) for g in goals))
        self.assertEqual(len({lib.goal_table_key(g) for g in goals}), 30)

    def test_training_contains_no_sealed_side_functions_and_labels_are_right(self):
        rng = random.Random(2)
        x, y, skipped = sel.make_dreams(0, CONCEPTS, 200)
        self.assertEqual(x.shape[0], 200)
        self.assertGreater(skipped, 0)
        self.assertTrue(((y.sum(1) >= 1) & (y.sum(1) <= 3)).all())
        for _ in range(100):
            expr, picks = sel.dream(rng, CONCEPTS, list(range(len(CONCEPTS))))
            for index in picks:
                self.assertIn(CONCEPTS[index], list(lib.subexpressions(expr)))

    def test_features_carry_the_input_output_pairing(self):
        out = np.arange(sel.EXAMPLES) % 256
        a = sel.features(out)
        self.assertEqual(a.shape, (sel.EXAMPLES, 3 * sel.BITS))
        self.assertFalse(np.array_equal(a, sel.features(out[::-1])))

    def test_model_ignores_example_order(self):
        import torch
        model = sel.Selector(5)
        x = torch.rand(2, sel.EXAMPLES, 3 * sel.BITS)
        order = torch.randperm(sel.EXAMPLES)
        self.assertTrue(torch.allclose(model(x), model(x[:, order]), atol=1e-5))

    def test_a_right_answer_needs_exhaustive_verification(self):
        xs, ys = lib.probes()
        self.assertEqual(sel.solve(("linear", 3, 0, 1), [], 5000, xs, ys)[0], True)
        self.assertEqual(sel.solve(("linear", 37, 53, 1), [], 50, xs, ys)[0], False)


if __name__ == "__main__":
    unittest.main()
