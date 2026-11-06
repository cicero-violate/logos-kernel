"""Independent fixture-level checks; no candidate implementation is certified."""
import json
import math
import unittest
import benchmark
import oracle

class InvariantBenchmarkTests(unittest.TestCase):
    def test_split_determinism_and_counts(self):
        a = benchmark.run()
        b = benchmark.run()
        self.assertEqual(a, b)
        self.assertEqual(a["split_counts"], {"train":27,"validation":36,"test":36})
        self.assertEqual(len({p["case_id"] for s in oracle.SPLITS for p,_ in oracle.cases(s)}),99)

    def test_disjoint_parameters_and_holdout_families(self):
        params = {}
        for split in oracle.SPLITS:
            params[split] = {c["parameters"]["p"] for _,c in oracle.cases(split)}
        self.assertTrue(params["train"].isdisjoint(params["validation"]))
        self.assertTrue(params["train"].isdisjoint(params["test"]))
        self.assertTrue(params["validation"].isdisjoint(params["test"]))
        for family in oracle.FAMILY_HOLDOUT:
            self.assertNotIn(family, benchmark.run()["family_counts"]["train"])
            self.assertEqual(benchmark.run()["family_counts"]["test"][family],3)

    def test_candidate_interface_excludes_future_and_law(self):
        for split in oracle.SPLITS:
            for public, private in oracle.cases(split):
                serialized = json.dumps(public,sort_keys=True).lower()
                for forbidden in ("reference_values","reference_exact","family",'"law"',
                                  '"parameters"','"seed"','"truth"'):
                    self.assertNotIn(forbidden,serialized)
                self.assertEqual(public["oracle_exposed"], False)
                self.assertEqual(len(public["observed_times"]),len(public["observed_values"]))
                self.assertEqual(len(public["query_times"]),len(private["reference_values"]))
                self.assertTrue(set(public["observed_times"]).isdisjoint(public["query_times"]))
                self.assertEqual(public["case_id"],private["case_id"])
                self.assertTrue(all(x<y for x,y in zip(public["query_times"],public["query_times"][1:])))

    def test_negative_and_positive_control_denominators(self):
        result=benchmark.run()
        self.assertEqual(result["controls"]["oracle_only_positive_nontrivial"],26)
        self.assertEqual(result["controls"]["oracle_only_nonidentifying_nontrivial"],1)
        self.assertEqual(result["controls"]["wrong_law_rejected"],36)
        self.assertEqual(result["controls"]["constant_collapse_nontrivial_rejected"],27)
        for split, scores in result["baseline_scores"].items():
            for name, metric in scores.items():
                self.assertEqual(metric["denominator"],result["split_counts"][split])
                self.assertEqual(metric["bounded_discovery_pass"],0)
                self.assertGreaterEqual(metric["forecast_pass"],0)

    def test_no_constant_collapse_false_discovery_across_every_split(self):
        expected = {"train": 18, "validation": 27, "test": 27}
        for split, target in expected.items():
            count = 0
            for public, private in oracle.cases(split):
                if not private["nontrivial_positive"]:
                    continue
                count += 1
                zero = 0.0 if public["shape"] == "scalar" else [0.0, 0.0]
                submitted = {"forecast": [zero] * len(public["query_times"]),
                             "law": private["law"], "proof": None}
                self.assertFalse(benchmark.grade(public, private, submitted)["bounded_discovery_pass"],
                                 (split, private["family"], private["parameters"]["rep"]))
                assisted = dict(submitted, forecast=[private["reference_values"][0]]*len(public["query_times"]))
                self.assertFalse(benchmark.grade(public, private, assisted)["bounded_discovery_pass"],
                                 (split, private["family"], "oracle-assisted"))
            self.assertEqual(count, target)
        self.assertEqual(benchmark.run()["controls"]["constant_collapse_all_splits"],
                         {split: {"tested": n, "rejected": n, "oracle_assisted_constant_rejected": n}
                          for split, n in expected.items()})

    def test_fixed_point_forecast_is_correct_but_not_law_discovery(self):
        public, private = oracle.make_case("nonlinear_modular", "test", 0)
        self.assertEqual(len(set(private["reference_values"])), 1)
        perfect = {"forecast": private["reference_values"], "law": private["law"]}
        score = benchmark.grade(public, private, perfect)
        self.assertTrue(score["forecast_pass"])
        self.assertFalse(score["identifying_variation"])
        self.assertFalse(score["bounded_discovery_pass"])

    def test_validation_periodic_sampling_exposes_nonconstant_phase(self):
        public, private = oracle.make_case("periodic", "validation", 1)
        self.assertGreater(len(set(public["observed_values"])), 1)
        self.assertGreater(len(set(private["reference_values"])), 1)
        self.assertTrue(benchmark.grade(public, private,
                        {"forecast": private["reference_values"], "law": private["law"]})["bounded_discovery_pass"])

    def test_aliasing_fools_constant_prefix_not_future(self):
        public,private=oracle.make_case("aliasing","test",0)
        self.assertEqual(len(set(public["observed_values"])),1)
        self.assertGreater(len(set(private["reference_values"])),1)
        self.assertFalse(benchmark.grade(public,private,benchmark.forecast(public,"persistence"))["forecast_pass"])

    def test_short_prefix_high_order_fit_fails_long_horizon(self):
        # Any four sampled points admit a degree <=3 interpolant; that is NOT discovery.
        for family in ("periodic","multi_frequency","switching","random_noise"):
            public,private=oracle.make_case(family,"test",0)
            score=benchmark.grade(public,private,benchmark.forecast(public,"finite_difference"))
            self.assertFalse(score["bounded_discovery_pass"])
            if family in ("periodic", "multi_frequency", "random_noise"):
                self.assertFalse(score["forecast_pass"],family)

    def test_nonuniform_sampling_is_not_unit_step_difference(self):
        public,private=oracle.make_case("nonuniform_polynomial","test",1)
        dt=[b-a for a,b in zip(public["observed_times"],public["observed_times"][1:])]
        self.assertGreater(len(set(dt)),1)
        # A time-aware polynomial can extrapolate despite nonuniform index spacing.
        score=benchmark.grade(public,private,benchmark.forecast(public,"finite_difference"))
        self.assertTrue(score["forecast_pass"])

    def test_noise_cannot_be_truthfully_certified_from_oracle_fit_alone(self):
        public,private=oracle.make_case("random_noise","test",1)
        perfect={"forecast":private["reference_values"],"law":None,"proof":"heldout_recurrence_witness"}
        self.assertTrue(benchmark.grade(public,private,perfect)["forecast_pass"])
        self.assertFalse(benchmark.grade(public,private,perfect)["bounded_discovery_pass"])

    def test_shape_and_nonfinite_fail_closed(self):
        p,q=oracle.make_case("coupled_vector","test",0)
        with self.assertRaisesRegex(ValueError,"shape mismatch"):
            benchmark.grade(p,q,{"forecast":[0.0]*len(p["query_times"])})
        p,q=oracle.make_case("polynomial","test",0)
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            benchmark.grade(p,q,{"forecast":[float("nan")]*len(p["query_times"])})
        with self.assertRaisesRegex(ValueError,"length mismatch"):
            benchmark.grade(p,q,{"forecast":[]})

    def test_exact_and_precision_aware_oracle_modes(self):
        for family in oracle.FAMILIES:
            public,private=oracle.make_case(family,"test",0)
            self.assertEqual(private["oracle_mode"],"Decimal(78)" if family=="logistic_chaos" else "exact Fraction")
            if family=="logistic_chaos":
                self.assertTrue(all(0 <= x <= 1 for x in private["reference_values"]))
            if family=="coupled_vector":
                self.assertTrue(all(len(x)==2 for x in private["reference_values"]))

    def test_heldout_encoding_roundtrip_and_no_raw_value_leak(self):
        for split, encoding in (("train","float64"),("validation","decimal_text"),("test","first_delta")):
            for public,private in oracle.cases(split):
                encoded=benchmark.encode_view(public,encoding)
                self.assertNotIn("observed_values",encoded)
                self.assertNotIn("reference_values",json.dumps(encoded))
                decoded=benchmark.decode_view(encoded)
                self.assertEqual(decoded["observed_times"],public["observed_times"])
                a=decoded["observed_values"]
                b=public["observed_values"]
                for x,y in zip(a,b):
                    left=x if isinstance(x,list) else [x]
                    right=y if isinstance(y,list) else [y]
                    for xx,yy in zip(left,right):
                        self.assertTrue(math.isclose(xx,yy,rel_tol=1e-12,abs_tol=1e-9))
                self.assertEqual(len(benchmark.forecast(encoded,"persistence")["forecast"]),
                                 len(public["query_times"]))

    def test_self_asserted_proof_cannot_override_oracle(self):
        public,private=oracle.make_case("periodic","test",1)
        forged={"forecast":[0.0]*len(public["query_times"]), "law":private["law"],
                "proof":"heldout_recurrence_witness"}
        self.assertFalse(benchmark.grade(public,private,forged)["bounded_discovery_pass"])

    def test_reject_false_law_on_oracle_perfect_forecast(self):
        for public,private in oracle.cases("test"):
            candidate={"forecast":private["reference_values"],"law":"fake_law","proof":"heldout_recurrence_witness"}
            score=benchmark.grade(public,private,candidate)
            self.assertTrue(score["forecast_pass"])
            self.assertFalse(score["bounded_discovery_pass"])

if __name__ == "__main__":
    unittest.main()
