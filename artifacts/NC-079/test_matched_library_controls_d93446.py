"""Independent wiring/negative controls for NC-079 across three seeds and all worlds.

Mocks search to keep this a fast functional contract test, NOT a scientific
campaign rerun. The actual campaign must separately be rerun and verified.
Dependencies: stdlib, NumPy, NC-079 stack/run_compounding/run_curriculum, NC-061.
"""
import argparse
import random
import sys
import numpy as np
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stack as s
import run_compounding as compounding
import run_curriculum as curriculum

SEEDS=(11,19,29)
GENS=5

def shape(expr):
    return s.size(expr)

def exact_match(real, fake):
    a=sorted(map(shape,real))
    b=sorted(map(shape,fake))
    if a!=b:
        raise AssertionError(f"random control shape mismatch: real={a}, random={b}")
    return a

def harvest_progress(solutions):
    # Synthesized, deterministic reference libraries of increasing counts/sizes.
    return [sized(i+2) for i in range(min(len(solutions),s.LIBRARY_CAP))]

def sized(n):
    e=("X",)
    for _ in range(n-1):
        e=("SHL",e)
    return e

def mock_search(*args, **kwargs):
    return (("SHL",("X",)), 1, 0.0)

def mock_verified(*args,**kwargs):
    return True

def mock_generation(world,library,budget,sealed_rng,train_rng,x,y,train_count,sealed_count,seen):
    # Covers the real -> random-library wiring, not model performance.
    row={"library_size":len(library),"largest_solution":2,
         "generation":0,"solve_rate":0.0,"solved":0,"total":sealed_count,
         "candidates_used":0,"cpu_seconds":0,"training_overlap":0,
         "over_budget_goals":0}
    return row, [("SHL",("X",))]

def mock_goals(world,rng,count,split,excluded=()):
    return [("LINEAR",1,1)]*count

class MatchedRandomControlTests(unittest.TestCase):
    def test_exact_multisets_and_reject_equal_count_wrong_shape(self):
        real=[sized(2),sized(5),sized(8)]
        fake=s.random_library(random.Random(1),real)
        self.assertEqual(exact_match(real,fake),[2,5,8])
        wrong=[sized(2),sized(5),sized(7)]
        self.assertEqual(len(real),len(wrong))
        with self.assertRaises(AssertionError):
            exact_match(real,wrong)

    def test_all_main_worlds_and_seeds_reference_shapes(self):
        a=argparse.Namespace(generations=GENS,budget=1,train_goals=1,
                             sealed_goals=1,retention_goals=0)
        for world in compounding.WORLDS:
            for seed in SEEDS:
                trace=[]
                orig_random=s.random_library
                def audited(rng,like):
                    fake=orig_random(rng,like)
                    trace.append({"expected":list(map(shape,like)),
                                  "actual":list(map(shape,fake))})
                    exact_match(like,fake)
                    return fake
                with patch.object(s,"run_generation",mock_generation), \
                     patch.object(s,"harvest",harvest_progress), \
                     patch.object(s,"search",mock_search), \
                     patch.object(s,"random_library",audited):
                    real,grown=compounding.run_arm(world,"library",seed,a,None,None)
                    fake,_=compounding.run_arm(world,"random_library",seed,a,None,None,grown)
                self.assertEqual(len(trace),GENS)
                self.assertEqual([r["library_size"] for r in real],
                                 [r["library_size"] for r in fake])
                self.assertTrue(all(t["expected"]==t["actual"] for t in trace))
                self.assertGreater(max(len(t["actual"]) for t in trace),0)

    def test_curriculum_three_seeds_reference_shapes(self):
        a=argparse.Namespace(generations=GENS,budget=1,train_goals=2,
                             sealed_goals=1,retention_goals=0)
        for seed in SEEDS:
            trace=[]
            orig_random=s.random_library
            def audited(rng,like):
                fake=orig_random(rng,like)
                trace.append((list(map(shape,like)),list(map(shape,fake))))
                exact_match(like,fake)
                return fake
            with patch.object(s,"draw_goals",mock_goals), \
                 patch.object(s,"search",mock_search), \
                 patch.object(s,"verified",mock_verified), \
                 patch.object(s,"harvest",harvest_progress), \
                 patch.object(s,"random_library",audited):
                real,grown=curriculum.run_arm("library",seed,a,np.asarray([0],dtype=np.int64),np.asarray([0],dtype=np.int64))
                fake,_=curriculum.run_arm("random_library",seed,a,np.asarray([0],dtype=np.int64),np.asarray([0],dtype=np.int64),grown)
            self.assertEqual(len(trace),GENS)
            self.assertTrue(all(a==b for a,b in trace))
            self.assertEqual([r["library_size"] for r in real],
                             [r["library_size"] for r in fake])
            self.assertGreater(max(len(a) for a,_ in trace),0)

if __name__=="__main__":
    unittest.main()
