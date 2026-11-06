"""Run: PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-078/test_synth.py"""
import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth  # noqa: E402


class SynthTests(unittest.TestCase):
    def test_fast_evaluator_matches_the_canonical_interpreter(self):
        rng = random.Random(3)
        for _ in range(120):
            body = synth.random_body(rng)
            self.assertTrue(synth.verify_with_reference(body, synth.evaluate(body)), body)

    def test_reference_check_rejects_a_wrong_table(self):
        body = [{"op": "ADD", "dst": "R", "a": "X", "b": "Y"}]
        table = synth.evaluate(body).copy()
        table[7] ^= 1
        self.assertFalse(synth.verify_with_reference(body, table))

    def test_named_goals_have_the_stated_meaning_and_known_solution(self):
        tables = synth.named_tables()
        self.assertEqual(int(tables["INC"][synth.XS.tolist().index(15)]), 0)
        inc = [{"op": "LOADI", "dst": "T", "imm": 1}, {"op": "ADD", "dst": "R", "a": "X", "b": "T"}]
        self.assertTrue(np.array_equal(synth.evaluate(inc), tables["INC"]))
        self.assertTrue(synth.verify_with_reference(inc, tables["INC"]))

    def test_training_set_contains_no_heldout_or_named_behaviour(self):
        train = synth.training_programs(0, 3000)
        named = {synth.table_key(t) for t in synth.named_tables().values()}
        unseen = {synth.table_key(t) for t in synth.unseen_goals(100).values()}
        for body, table in train:
            key = synth.table_key(table)
            self.assertNotIn(key, named)
            self.assertNotIn(key, unseen)
            self.assertFalse(synth.is_heldout_function(table))
            self.assertTrue(np.array_equal(synth.evaluate(body), table))

    def test_splits_are_deterministic_and_unseen_goals_are_distinct(self):
        a = synth.unseen_goals(50)
        self.assertEqual({k: v.tobytes() for k, v in a.items()}, {k: v.tobytes() for k, v in synth.unseen_goals(50).items()})
        self.assertEqual(len({v.tobytes() for v in a.values()}), 50)
        self.assertTrue(all(synth.is_heldout_function(v) for v in a.values()))

    def test_enumeration_is_shortest_first(self):
        first = next(synth.enumerate_shortest_first())
        self.assertEqual(len(first), 1)
        self.assertEqual(len(synth.all_instructions()), 4 * (16 + 20 + 80))


if __name__ == "__main__":
    unittest.main()
