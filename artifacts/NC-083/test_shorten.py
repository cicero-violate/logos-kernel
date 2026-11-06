"""Run: OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-083/test_shorten.py"""
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shorten as sh  # noqa: E402

X, Y = ("X",), ("Y",)
BLOATED = ("SUB", ("ADD", ("ADD", X, Y), X), X)  # Equals X + Y.


class ShortenTests(unittest.TestCase):
    def test_a_changed_library_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp) / "library.py"
            changed.write_bytes((sh.HERE / "la015_library_pinned.py").read_bytes() + b"\n")
            with self.assertRaisesRegex(SystemExit, "pinned library mismatch"):
                sh.load_module(changed, "x", sh.LIB_SHA256)

    def test_values_match_the_stack_machine(self):
        rng = random.Random(1)
        def build(depth):
            if depth == 0 or rng.random() < .2:
                return rng.choice((X, Y, ("K", rng.randrange(256))))
            if rng.random() < .25:
                return (rng.choice(sh.UNARY), build(depth - 1))
            return (rng.choice(sh.BINARY), build(depth - 1), build(depth - 1))
        for _ in range(40):
            expr = build(4)
            code = sh.lib.compile_stack(expr)
            got = sh.values(expr)
            for i in rng.sample(range(65536), 200):
                self.assertEqual(int(got[i]), sh.lib.run_stack(code, int(sh.ALL_X[i]), int(sh.ALL_Y[i])))

    def test_dag_ops_counts_shared_parts_once(self):
        shared = ("ADD", X, Y)
        self.assertEqual(sh.dag_ops(("XOR", shared, shared)), 2)
        self.assertEqual(sh.tree_size(("XOR", shared, shared)), 7)

    def test_frame_accepts_a_correct_shorter_program(self):
        result, _ = sh.shrink(BLOATED, lambda subtrees: [[("ADD", X, Y)] for _ in subtrees])
        self.assertEqual(result, ("ADD", X, Y))

    def test_frame_rejects_a_shorter_program_that_is_wrong_on_one_pair(self):
        """A proposer can offer anything; a wrong proposal must never change the program."""
        wrong = ("ADD", Y, X)  # Not a part of the program, so only the offer is made wrong.
        real = sh.values
        def one_pair_off(expr, x=sh.ALL_X, y=sh.ALL_Y, memo=None):
            out = real(expr, x, y, memo)
            if expr == wrong and len(out) == 65536:
                out = out.copy()
                out[12345] ^= 1
            return out
        with mock.patch.object(sh, "values", one_pair_off):
            result, _ = sh.shrink(BLOATED, lambda subtrees: [[wrong] for _ in subtrees])
        self.assertEqual(result, BLOATED)

    def test_final_check_uses_the_second_evaluator_on_every_pair(self):
        for bad in ((0, 0), (255, 254), (17, 200)):
            real = sh.lib.run_stack
            flip = ("ADD", X, Y)
            flipped = sh.lib.compile_stack(flip)
            def wrong(code, x, y, bad=bad):
                value = real(code, x, y)
                return (value + 1) & 255 if code == flipped and (x, y) == bad else value
            with mock.patch.object(sh.lib, "run_stack", wrong):
                self.assertFalse(sh.confirmed_equal(BLOATED, flip), bad)

    def test_tokens_round_trip_and_malformed_tokens_are_refused(self):
        for expr in (BLOATED, ("XOR", ("K", 200), ("SHL", Y)), X):
            self.assertEqual(sh.decode(sh.encode(expr)), expr)
        self.assertIsNone(sh.decode([sh.TOKEN["ADD"], sh.TOKEN["X"]]))
        self.assertIsNone(sh.decode([sh.TOKEN["X"], sh.TOKEN["Y"]]))
        self.assertIsNone(sh.decode([]))

    def test_exam_programs_are_real_held_out_and_verified(self):
        exam, available = sh.load_exam(sh.HERE / "library_seed11.jsonl")
        self.assertGreater(available, 0)
        self.assertGreaterEqual(len(exam), 10)
        for program in exam:
            self.assertTrue(sh.is_held_out(sh.function_key(program)))
            self.assertGreaterEqual(sh.dag_ops(program), 3)

    def test_table_finds_true_shorter_forms(self):
        table = sh.Table(max_tried=30000, max_entries=4000)
        found = table.lookup(BLOATED)
        self.assertLess(sh.cost(found), sh.cost(BLOATED))
        self.assertTrue(np.array_equal(sh.values(found), sh.values(BLOATED)))

    def test_training_pairs_never_contain_a_held_out_function(self):
        table = sh.Table(max_tried=60000, max_entries=8000)
        source, target, skipped = sh.make_pairs(table, 300, 1)
        self.assertGreater(skipped, 0)
        for row in target.tolist():
            tokens = [t for t in row[1:] if t not in (0, sh.TOKEN["EOS"])]
            self.assertFalse(sh.is_held_out(sh.function_key(sh.decode(tokens))))

    def test_nc023_adapter_preserves_meaning_and_can_shorten(self):
        program = ("XOR", ("XOR", ("ADD", X, Y), Y), Y)
        result = sh.run_nc023([program])
        self.assertEqual(result["errors"], 0)
        self.assertEqual(result["total_ops"], 1)

    def test_calibration_reports_measured_cpu_and_refuses_a_false_match(self):
        good = sh.calibrate(lambda b: {"total_ops": 1, "cpu_seconds": b / 100}, 100.0, 5000)
        self.assertTrue(good["matched"])
        stuck = sh.calibrate(lambda b: {"total_ops": 0, "cpu_seconds": 300.0}, 100.0, 5000)
        self.assertFalse(stuck["matched"])


if __name__ == "__main__":
    unittest.main()
