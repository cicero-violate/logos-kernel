"""NC-028 regression tests; no network access or champion modification.

Dependencies: isolation.py, propose_worker.py, execute_worker.py, arena.py,
NC-007 controller checkpoints and NC-005/006 frozen primitive library.
"""
import hashlib, json, sys
from pathlib import Path
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import arena, isolation

class ReleaseTests(unittest.TestCase):
    def test_no_new_goal_invention(self):
        cp=isolation.run([str(HERE/"propose_worker.py")],
            json.dumps({"seed":11,"goal":"UNSEEN_XOR_DOUBLE_PLUS_Y"}),timeout=35)
        self.assertEqual(cp.returncode,0,cp.stderr)
        x=json.loads(cp.stdout.strip().splitlines()[-1])
        self.assertEqual(x["status"],"UNSUPPORTED_GOAL")
        self.assertFalse(x["model_generated"])
        self.assertEqual(x["actions"],[])
    def test_no_program_fallback_or_halt_bypass(self):
        before=arena.hash_file(arena.ROOT.parent/"rlt_split_algorithms_ablation_v1/snapshots/current_unified_v3.pt")
        for bad in (["ADD_XX"],["HALT"],["ADD_XX","HALT","ADD_XX"],["PYTHON","HALT"]):
            cp=isolation.run([str(HERE/"execute_worker.py")],
                json.dumps({"seed":11,"actions":bad,"pairs":[[3,7]]}),timeout=35)
            self.assertNotEqual(cp.returncode,0)
            self.assertIn(("HALT bound" if bad!=["PYTHON","HALT"] else "unregistered action"),cp.stderr)
        after=arena.hash_file(arena.ROOT.parent/"rlt_split_algorithms_ablation_v1/snapshots/current_unified_v3.pt")
        self.assertEqual(before,after)
    def test_provenance_gate_rejects_tampered_checkpoint(self):
        genuine=arena.hash_file
        def bad_hash(path):
            if str(path).endswith("/controller_seed11.pt"):
                return "0"*64
            return genuine(path)
        with patch.object(arena,"hash_file",side_effect=bad_hash):
            with self.assertRaisesRegex(RuntimeError,"controller checkpoint hash mismatch"):
                arena.main()
    def test_ground_truth_from_expression_not_reference_program(self):
        for goal in arena.GOALS:
            sample=arena.pairs(11,goal)
            self.assertEqual(len(sample),len(set(tuple(p) for p in sample)))
            self.assertEqual(len(sample),8)
            for x,y in sample:
                self.assertEqual(arena.truth(goal,x,y) & arena.MASK,arena.truth(goal,x,y))
    def test_manifest_and_results_if_present(self):
        result=HERE/"pilot_results.json"
        trace=HERE/"proposal_trace.jsonl"
        if not result.exists() or not trace.exists():
            self.skipTest("pilot not yet completed")
        rows=[json.loads(line) for line in trace.read_text().splitlines()]
        self.assertEqual(len(rows),3*4*3)
        self.assertEqual(len({(r["seed"],r["goal"],r["arm"]) for r in rows}),len(rows))
        self.assertTrue(all(r["tested"]==8 and r["correct"]+len(r["failures"])==8 for r in rows))
        for r in rows:
            self.assertGreaterEqual(r["execution_cost"]["cpu_seconds"],0)
            self.assertGreaterEqual(r["proposal_cost"]["cpu_seconds"],0)
            self.assertEqual(r["heldout_pairs_sha256"],
              hashlib.sha256(json.dumps(arena.pairs(r["seed"],r["goal"])).encode()).hexdigest())
            self.assertEqual(r["model_generated"],r["arm"]=="supervised_neural_controller" and bool(r["actions"]))
            if r["model_generated"]:
                self.assertEqual(r["actions"][-1],"HALT")
        summary=json.loads(result.read_text())
        self.assertEqual(summary["status"],"NO_DISCOVERY")
        for arm,v in summary["counts"].items():
            selected=[r for r in rows if r["arm"]==arm]
            self.assertEqual(v["correct"],sum(r["correct"] for r in selected))
            self.assertEqual(v["tested"],sum(r["tested"] for r in selected))
        self.assertEqual(summary["new_discoveries"],0)

if __name__=="__main__": unittest.main()
