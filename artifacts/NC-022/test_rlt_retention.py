"""NC-022 tests; depends on Python3, PyTorch, NumPy, rlt_rehearsal.py,
and NC-022 task-local data; original RLT source + frozen trained bundle."""
import json
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import rlt_rehearsal as r

class RetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=json.loads((r.HERE/"checkpoint_manifest.json").read_text())
        cls.ret=json.loads((r.HERE/"rlt_retention_matrix.json").read_text())
        cls.ab=json.loads((r.HERE/"exposure_matched_ablation.json").read_text())
        cls.ledger=json.loads((r.HERE/"train_and_holdout_ledger.json").read_text())
    def test_original_champion_frozen(self):
        r.validate_contract()
        self.assertEqual(r.sha_file(r.SOURCE),r.P["source_snapshot_sha256"])
        self.assertEqual(self.m["source_sha256"],r.source_hashes())
        self.assertFalse(self.m["promoted"])
        self.assertTrue(self.m["no_champion_modification"])
        self.assertEqual(r.P["model_parameters"],25089)
    def test_three_seed_three_arm_five_stage(self):
        self.assertEqual(self.m["checkpoint_count"],45)
        self.assertEqual(self.m["evaluations"],len(self.ret["records"]))
        self.assertEqual(self.m["seed_count"],3)
        self.assertEqual(self.m["arm_count"],3)
        self.assertEqual(self.m["stage_count"],5)
        keys={(i["seed"],i["arm"],i["stage"]) for i in self.m["checkpoints"]}
        expected={(seed,arm,stage) for seed in r.SEEDS for arm in r.ARMS for stage in range(5)}
        self.assertEqual(keys,expected)
    def test_exposure_match_exact_not_approximate(self):
        for seed in r.SEEDS:
            ledger=self.ledger[str(seed)]["blocks"]
            check=r.matched_exposure_check(ledger)
            self.assertEqual(check,self.ab["matched_exposure_proof"][str(seed)])
            self.assertEqual(check,self.ret["exposure_matching"][str(seed)])
            self.assertTrue(check["exact_operand_task_width_matched"])
            self.assertEqual(check["rehearsal_vs_block_replay_l1_difference"],0)
            self.assertEqual(check["per_seed_arm_total_examples"],6912)
            self.assertFalse(check["new_only_per_task_exposure_matched"])
    def test_train_holdout_pair_and_reverse_disjointness(self):
        for seed in r.SEEDS:
            led=self.ledger[str(seed)]
            train={r.pairkey(*pair) for blocks in led["blocks"].values()
                   for block in blocks for pair in block["pairs"]}
            val=[r.pairkey(*pair) for group in led["groups"].values() for pair in group]
            self.assertEqual(len(val),len(set(val)))
            self.assertFalse(train.intersection(val))
            self.assertEqual(len(val),sum(r.P["evaluation_groups"].values()))
    def test_frozen_checkpoint_package_complete(self):
        for ck in self.m["checkpoints"]:
            file=r.ROOT/ck["path"]
            self.assertEqual(r.sha_file(file),ck["sha256"])
            pkg=torch.load(file,map_location="cpu",weights_only=False)
            self.assertEqual(pkg["schema"],r.SCHEMA)
            self.assertEqual(pkg["task"],"NC-022")
            self.assertEqual(pkg["seed"],ck["seed"])
            self.assertEqual(pkg["arm"],ck["arm"])
            self.assertEqual(pkg["stage"],ck["stage"])
            self.assertEqual(pkg["source_sha256"],r.source_hashes())
            for field in ("model","optimizer","random_python","random_numpy","random_torch"):
                self.assertIn(field,pkg)
            self.assertFalse(pkg["promoted"])
            model=r.TaskConditionedRLT()
            model.load_state_dict(pkg["model"],strict=True)
            self.assertEqual(r.model_state_sha(model),ck["model_state_sha256"])
            if ck["stage"]>0:
                self.assertGreater(len(pkg["optimizer"]["state"]),0)
    def test_manifest_trace_and_report_hashes(self):
        self.assertEqual(self.m["train_ledger_sha256"],r.sha_file(r.HERE/"train_and_holdout_ledger.json"))
        self.assertEqual(self.m["retention_matrix_sha256"],r.sha_file(r.HERE/"rlt_retention_matrix.json"))
        self.assertEqual(self.m["exposure_matched_ablation_sha256"],r.sha_file(r.HERE/"exposure_matched_ablation.json"))
        for ck in self.m["checkpoints"]:
            trace=r.ROOT/ck["evaluation_trace"]["path"]
            self.assertEqual(r.sha_file(trace),ck["evaluation_trace"]["sha256"])
            rows=[json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual(len(rows),41)
            self.assertEqual(sum(len(z["pairs"]) for z in rows),ck["evaluation_trace"]["predictions"])
    def test_raw_ground_truth_and_exact_score(self):
        score={(x["seed"],x["arm"],x["stage"],x["kind"],x["group"],x["task"]):x for x in self.ret["records"]}
        tested=0
        for ck in self.m["checkpoints"]:
            rows=[json.loads(line) for line in
                  (r.ROOT/ck["evaluation_trace"]["path"]).read_text().splitlines()]
            for item in rows:
                if item["kind"]=="direct":
                    truth=[r.truth(x,y,item["width"],item["task"]) for x,y in item["pairs"]]
                else:
                    truth=[r.compose_truth(p,r.COMPOSITIONS[item["task"]],item["width"])
                           for p in item["pairs"]]
                self.assertEqual(item["truth"],truth)
                got=item["prediction"]
                self.assertEqual(len(got),len(truth))
                self.assertTrue(all(0<=x<=r.MASK(item["width"]) for x in got))
                n=len(got);cnt=sum(x==y for x,y in zip(got,truth))
                rec=score[(item["seed"],item["arm"],item["stage"],item["kind"],item["group"],item["task"])]
                self.assertEqual((rec["correct"],rec["tested"]),(cnt,n))
                self.assertAlmostEqual(rec["accuracy"],cnt/n)
                self.assertEqual(rec["pair_set_sha256"],r.digest(item["pairs"]))
                tested+=n
        self.assertGreater(tested,70000)
    def test_three_arms_identical_before_curriculum_branch(self):
        states={(z["seed"],z["arm"],z["stage"]):z["model_state_sha256"] for z in self.m["checkpoints"]}
        for seed in r.SEEDS:
            for stage in (0,1):
                self.assertEqual(len({states[seed,a,stage] for a in r.ARMS}),1)
    def test_regression_and_rehearsal_calculation(self):
        recomputed=r.score_summary(self.ret["records"])
        self.assertEqual(recomputed,self.ret["summary"])
        self.assertEqual(recomputed["claim"],self.ab["claim"])
        self.assertEqual(len(recomputed["by_seed_arm_stage"]),45)
        self.assertEqual(len(recomputed["rehearsal_comparisons"]),9)
        self.assertEqual(len(recomputed["restoration"]),9)
        if recomputed["claim"]=="NO_GAIN":
            self.assertFalse(recomputed["strict_reproduced_gain"])
    def test_update_budget_and_exposure(self):
        c=self.ab["compute_accounting"]
        self.assertEqual(c["total_optimizer_updates"],1296)
        self.assertEqual(c["total_training_examples"],62208)
        for seed in r.SEEDS:
            for arm in r.ARMS:
                group=sorted((c for c in self.m["checkpoints"] if c["seed"]==seed and c["arm"]==arm),
                             key=lambda x:x["stage"])
                self.assertEqual([x["updates_total"] for x in group],[0,24,72,120,144])
    def test_out_of_scope_controls(self):
        self.assertEqual(set(r.COMPOSITIONS),set(r.P["independent_composition_probes"]))
        self.assertTrue(self.ret["neural_inference_without_symbolic_fallback"])
        self.assertFalse(self.ret["promoted"])
        self.assertEqual(r.truth(255,1,8,"add"),0)
        self.assertNotEqual(r.truth(255,1,8,"add"),r.truth(255,1,8,"xor"))
        self.assertEqual(r.compose_truth((37,8),r.COMPOSITIONS["add_then_sub"],8),37)
        with self.assertRaises(ValueError):
            r.truth(1,2,8,"INVENTED_OPERATOR")
if __name__=="__main__": unittest.main()
