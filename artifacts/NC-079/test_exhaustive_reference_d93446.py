"""Negative control: sample-only independent verification can miss a single bad input."""
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import stack as s
import exhaustive_reference_d93446 as full

class ExhaustiveReferenceTests(unittest.TestCase):
    def test_full_scalar_domain_accepts_true_and_rejects_false_program(self):
        good=("ADD",("X",),("Y",))
        wrong=("XOR",("X",),("Y",))
        self.assertTrue(full.verify(good,good))
        self.assertFalse(full.verify(wrong,good))

    def test_scalar_corner_not_sampled_by_existing_gate(self):
        rng=random.Random(1)
        checked={(rng.randrange(256),rng.randrange(256)) for _ in range(1024)}
        missing=next((x,y) for x in range(256) for y in range(256) if (x,y) not in checked)
        program=("ADD",("X",),("Y",))
        original=s.reference_value
        def faulty_reference(expr,x,y):
            v=original(expr,x,y)
            return (v+1)&255 if (x,y)==missing else v
        with patch.object(s,"reference_value",faulty_reference):
            self.assertTrue(s.verified(program,program,sample_seed=1))
            self.assertFalse(full.verify(program,program))
if __name__=="__main__":unittest.main()
