"""NC-062 isolated parent policy tests.

Dependencies: Python 3 stdlib, run_loop.py, NC-001 canonical ISA,
NC-009 promote.py and registry, NC-061 scoreboard.py, NC-028 sandbox library.
These tests DO NOT claim independent reviewer identity.
"""
from __future__ import annotations
import copy
import hashlib
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_loop as r

class TestNC062(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library=r.verified_initial_library()

    def test_four_templates_exhaustively_compiled(self):
        self.assertEqual(len(self.library),4)
        for item in self.library:
            self.assertEqual(len(r.sb.truth_table(item["program"])),256)
            self.assertEqual(item["verified_cases"],256)

    def test_unknown_token_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError,"UNMAPPED_ACTION"):
            r.compile_program(["FABRICATED"])
        with self.assertRaisesRegex(RuntimeError,"ACTION_LENGTH"):
            r.compile_program([])
        with self.assertRaisesRegex(RuntimeError,"ACTION_LENGTH"):
            r.compile_program(["ADD_XX"]*5)

    def test_verification_rejects_sample_only_fit(self):
        # Zero output exactly fits all eight deliberately chosen x=y=0
        # examples of an ADD(X,Y) function, but fails the full 256 table.
        true=[{"op":"ADD","dst":"R","a":"X","b":"Y"}]
        fake=[{"op":"LOADI","dst":"R","imm":0}]
        table=r.sb.truth_table(true)
        examples=[{"X":0,"Y":0,"R":0}]*8
        self.assertEqual(len(examples),8)
        test={"table":table,"examples":examples}
        self.assertEqual(r.proof(test,fake)["verdict"],"COUNTEREXAMPLE")
        self.assertEqual(r.proof(test,true)["verdict"],"VERIFIED")

    def test_bad_candidate_rankings_fail_closed(self):
        got=r.sb.sealed_goals({"generator":r.sb.GENERATOR,
                    "seed":"NC062/test-only",
                    "nonce":"not-secret","thresholds":{**r.sb.THRESHOLDS,"goals":1}})
        with self.assertRaisesRegex(RuntimeError,"MODEL_RANKING_INVALID"):
            r.decide_discovery(got,[[0,0,1,2]],self.library,1,11)
        with self.assertRaisesRegex(RuntimeError,"MODEL_RANKING_INVALID"):
            r.decide_discovery(got,[[0,1,2,99]],self.library,1,11)

    def test_vetted_templates_cannot_be_falsely_new(self):
        for index,entry in enumerate(self.library):
            goal={"table":r.sb.truth_table(entry["program"]),"examples":[]}
            rankings=[index]+[i for i in range(4) if i!=index]
            records,accepted,new=r.decide_discovery([goal],[rankings],self.library,1,11)
            self.assertEqual((accepted,new),(1,0))
            self.assertTrue(records[0]["accepted"])
            self.assertFalse(records[0]["new_discovery"])

    def test_registry_chain_tamper_rejected(self):
        rows=[__import__("json").loads(z) for z in
             (r.NC9/"operator_promotions.jsonl").read_text().splitlines()]
        r.promote.verify_chain(rows)
        malicious=copy.deepcopy(rows)
        malicious[0]["operator_id"]="FORGED"
        with self.assertRaises(ValueError):
            r.promote.verify_chain(malicious)
        with self.assertRaises(ValueError):
            r.promote.verify_chain(rows[1:])

    def test_sealed_commitment_tamper_rejected(self):
        commit,reveal=r.sb.new_commitment()
        self.assertEqual(r.sb.verify_reveal(commit,reveal),reveal)
        for key in ("seed","nonce","generator"):
            mod=copy.deepcopy(reveal);mod[key]+="tampered"
            with self.assertRaises(ValueError):
                r.sb.verify_reveal(commit,mod)
        mod=copy.deepcopy(reveal)
        mod["thresholds"]["cpu_tolerance"]=999
        with self.assertRaises(ValueError):
            r.sb.verify_reveal(commit,mod)

    def test_no_seal_in_candidate_payload(self):
        # Explicit 5-key child request schema; no goal table/answer/reveal.
        src=r.WORKER
        self.assertIn('set(("mode","seed","state","training","queries"))',src)
        self.assertIn('type(data)',src)
        self.assertNotIn('secrets.token_hex',src)
        self.assertNotIn('sealed_goals(',src)

    def test_score_denominator_and_cpu_provenance(self):
        g=r.sb.sealed_goals({"generator":r.sb.GENERATOR,"seed":"NC062/test-score",
                    "nonce":"test","thresholds":{**r.sb.THRESHOLDS,"goals":2}})
        scores=r.score_cached(g,[[0,1,2,3]]*2,self.library,[],2.5)
        self.assertEqual(scores["total"],2)
        self.assertEqual(scores["cpu_seconds"],2.5)
        self.assertEqual(scores["actual_inference_worker_calls"],1)
        self.assertEqual(scores["candidates_used"],2)
        self.assertEqual(scores["over_budget_goals"],0)

    def test_scientific_nonclaims(self):
        self.assertEqual(len(r.SEEDS),3)
        self.assertEqual(r.GENS,(1,2,3))
        self.assertEqual(r.BUDGET["max_proposals_per_goal"],4)
        self.assertTrue(r.CHAMPION.is_file())
        self.assertNotIn("target program",r.WORKER.lower())
        self.assertEqual(r.BUDGET["promotion"].split(";")[1].strip(),"no NC-009 mutation")

if __name__=="__main__":
    unittest.main()
