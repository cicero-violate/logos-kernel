"""NC-056 isolation and evaluation negative controls; stdlib + bubblewrap."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import discover
import evaluate


class EvaluationTests(unittest.TestCase):
    def test_fixed_seed_numeric_reproduction_without_identifier_leak(self):
        a=evaluate.generate(123);b=evaluate.generate(123)
        self.assertEqual(len(a),26)
        for (p,q),(r,s) in zip(a,b):
            self.assertEqual(q,s)
            self.assertNotEqual(p["case_id"],r["case_id"])
            self.assertEqual({k:v for k,v in p.items() if k!="case_id"},{k:v for k,v in r.items() if k!="case_id"})
            self.assertEqual(set(p),{"case_id","shape","observed_times","observed_values","query_times"})

    def test_bad_forecast_cannot_claim_polynomial_identity(self):
        public,private=next((p,q) for p,q in evaluate.generate(123) if q["family"]=="polynomial")
        result=discover.forecast(public,"mdl_search")
        self.assertTrue(evaluate.grade(public,private,result)["bounded_law_recovery"])
        result["forecast"]=[0]*len(result["forecast"])
        self.assertFalse(evaluate.grade(public,private,result)["bounded_law_recovery"])

    def test_noisy_or_aliased_constant_is_not_law_discovery(self):
        for public,private in evaluate.generate(123):
            if private["family"] in ("noisy_polynomial","aliasing","random_noise","constant"):
                result=discover.forecast(public,"mdl_search")
                self.assertFalse(evaluate.grade(public,private,result)["bounded_law_recovery"])

    def test_sandbox_hides_workspace_and_oracle(self):
        source=b'''import importlib.util,json,os,sys
assert not os.path.exists('/workspace/experiments/logos-kernel/instructions.jsonl')
assert importlib.util.find_spec('oracle') is None
assert importlib.util.find_spec('evaluate') is None
assert not any(k.endswith('TOKEN') or k.endswith('API_KEY') for k in os.environ)
for line in sys.stdin:
 p=json.loads(line);print(json.dumps({'case_id':p['case_id'],'isolated':True}))
'''
        self.assertTrue(evaluate.isolated(source,[{"case_id":"probe"}])[0]["isolated"])

    def test_isolation_fails_closed(self):
        with patch.object(evaluate.shutil,"which",return_value=None):
            with self.assertRaisesRegex(RuntimeError,"no unsandboxed fallback"):
                evaluate.isolated(b"print(1)",[])

    def test_real_participant_runs_isolated(self):
        p,q=next((p,q) for p,q in evaluate.generate(456) if q["family"]=="polynomial")
        output=evaluate.isolated((Path(__file__).parent/"discover.py").read_bytes(),[p])[0]
        self.assertEqual(len(output["arms"]),6)
        self.assertTrue(evaluate.grade(p,q,output["arms"][0])["bounded_law_recovery"])


if __name__=="__main__":unittest.main()
