"""Meaningful law-search counterexamples, typed contracts, and lower orders."""
import copy
from fractions import Fraction as F
import random
import unittest
import discover as d


def case(values, times=None, queries=None):
    n=len(values)
    return {"case_id":"test", "shape":"scalar", "observed_values":values,
            "observed_times":times or list(range(n)),
            "query_times":queries or list(range(n,n+8))}


class DiscoveryTests(unittest.TestCase):
    def test_minimum_polynomial_orders(self):
        for degree in range(1,5):
            c=case([3+t+2*t**degree for t in range(20)])
            out=d.forecast(c,"mdl_search")
            self.assertEqual(out["models"][0]["kind"],"polynomial")
            self.assertEqual(out["models"][0]["order"],degree)
            self.assertEqual(out["forecast"],[float(3+t+2*t**degree) for t in c["query_times"]])
            for trace in out["traces"][0]:
                if trace["kind"]=="polynomial" and trace["order"]<degree:
                    self.assertFalse(trace["accepted"])

    def test_nonuniform_time_not_sample_index(self):
        times=[i*(i+1)//2 for i in range(24)]
        out=d.forecast(case([7-2*t+3*t*t for t in times[:16]],times[:16],times[16:]),"mdl_search")
        self.assertEqual(out["models"][0]["order"],2)
        self.assertEqual(out["forecast"],[float(7-2*t+3*t*t) for t in times[16:]])

    def test_geometric_not_polynomial(self):
        out=d.forecast(case([3**t for t in range(18)]),"mdl_search")
        self.assertEqual(out["models"][0]["kind"],"ar")
        self.assertEqual(out["models"][0]["coeff"],["3"])
        self.assertEqual(out["forecast"],[float(3**t) for t in range(18,26)])

    def test_nonlinear_modulus_is_inferred(self):
        values=[7]
        for _ in range(37): values.append((values[-1]**2+5)%101)
        out=d.forecast(case(values[:30]),"mdl_search")
        self.assertEqual(out["models"][0]["kind"],"quadratic_mod")
        self.assertEqual(out["models"][0]["modulus"],101)
        self.assertEqual(out["models"][0]["offset"],5)
        self.assertEqual(out["forecast"],values[30:])

    def test_noise_does_not_become_high_order_polynomial(self):
        rng=random.Random(329)
        out=d.forecast(case([rng.randrange(-50,51) for _ in range(24)]),"mdl_search")
        self.assertIsNone(out["models"][0])
        self.assertTrue(out["failures"][0])
        self.assertLessEqual(out["candidate_checks"],d.MAX_CHECKS)

    def test_metadata_and_future_truth_rejected(self):
        c=case(list(range(16)))
        c["reference_values"]=[16,17]
        with self.assertRaisesRegex(ValueError,"public-only"): d.forecast(c,"mdl_search")

    def test_prefix_check_rejects_interpolation_overfit(self):
        c=case([i*i if i<8 else i*i+7 for i in range(18)])
        out=d.forecast(c,"mdl_search")
        self.assertFalse(any(t["accepted"] and t["kind"]=="polynomial" for t in out["traces"][0]))

    def test_query_values_do_not_affect_model_selection(self):
        c=case([i*i for i in range(18)])
        other=copy.deepcopy(c);other["query_times"]=[100+i for i in range(8)]
        self.assertEqual(d.forecast(c,"mdl_search")["models"],d.forecast(other,"mdl_search")["models"])

    def test_typed_input_rejections(self):
        for bad in (float("nan"),float("inf"),True):
            c=case(list(range(16)));c["observed_values"][3]=bad
            with self.assertRaises(ValueError): d.forecast(c,"mdl_search")
        c=case(list(range(16)));c["query_times"]=[0]
        with self.assertRaises(ValueError):d.forecast(c,"mdl_search")

    def test_finite_search_budget_all_arms(self):
        for arm in d.ARMS:
            out=d.forecast(case([i*i for i in range(20)]),arm)
            self.assertLessEqual(out["candidate_checks"],128)
            self.assertEqual(len(out["forecast"]),8)
            for trace in out["traces"]:
                costs=[row["law_size"] for row in trace]
                self.assertEqual(costs,sorted(costs))


if __name__=="__main__":unittest.main()
