"""Dependencies: Python3, NumPy, PyTorch, compose.py, NC-001, frozen NC-005
checkpoints, NC-006 library, NC-007 controller checkpoints and trace JSONL."""
import collections
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import compose as c

class CompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads((c.HERE/"composition_receipt.json").read_text())
        cls.rows=[json.loads(x) for x in (c.HERE/"composition_traces.jsonl").read_text().splitlines()]
    def test_three_seed_train_and_reload(self):
        self.assertEqual([x["seed"] for x in self.report["training"]],[11,19,29])
        for entry in self.report["training"]:
            self.assertEqual(entry["training_updates"],c.UPDATES)
            self.assertEqual(entry["supervision_rows"],len(c.teaching_rows()))
            model=c.reload_controller(entry["seed"],entry)
            for goal,target in c.GOALS.items():
                self.assertEqual(tuple(c.predict(model,goal)["actions"]),target)
    def test_supervised_only(self):
        self.assertEqual(len(c.GOALS),11)
        self.assertTrue(all(v[-1]=="HALT" for v in c.GOALS.values()))
        self.assertIn("no unseen goal families",self.report["controller_input_scope"])
        with self.assertRaises(ValueError):
            c.predict(c.MacroController(),"UNSEEN_GOAL")
    def test_counts_and_failures(self):
        self.assertEqual(len(self.rows),36)
        self.assertEqual(collections.Counter(r["goal"] for r in self.rows),
                         {name:3 for name in (*c.GOALS,"HOST_LOOP_TRIPLE")})
        self.assertEqual(sum(r["tested"] for r in self.rows if r["goal"] in c.GOALS),8448)
        self.assertEqual(sum(r["tested"] for r in self.rows if r["goal"]=="HOST_LOOP_TRIPLE"),768)
        for row in self.rows:
            self.assertEqual(row["tested"],256)
            self.assertEqual(row["correct"],256)
            self.assertFalse(row["failure_examples"])
            self.assertGreater(row["duration_ns"],0)
    def test_accepted_registry_only(self):
        lib=json.loads(c.LIB_PATH.read_text())
        accepted={x["id"]:x for x in lib["accepted_neural"]}
        self.assertEqual(len(accepted),30)
        for row in self.rows:
            if row["goal"]=="HOST_LOOP_TRIPLE":continue
            ids=row["source_proof_circuit_ids"]
            self.assertEqual(len(ids),row["program_length"])
            for ident in ids:
                entry=accepted[ident]
                self.assertEqual(entry["seed"],row["seed"])
                self.assertEqual(entry["arm"],"rehearsal")
                self.assertEqual(entry["implementation_mode"],"neural_inference")
                self.assertEqual(entry["checkpoint_sha256"],row["frozen_checkpoint_sha256"])
    def test_original_weights_unchanged(self):
        manifest=json.loads(c.LIB_MANIFEST.read_text())
        self.assertEqual(c.file_sha(c.LIB_PATH),
                         manifest["independent_verification"]["accepted_library_sha256"])
        self.assertEqual(c.file_sha(c.NEURAL_SRC),manifest["source_sha256"]["NC005_curriculum"])
        self.assertEqual(c.file_sha(c.CHECKPOINT_MANIFEST),manifest["checkpoint_manifest_sha256"])
        self.assertEqual(c.file_sha(manifest["source_files"]["original_RLT_champion"]),
                         manifest["original_champion_sha256"])
        self.assertTrue(self.report["no_model_promotion"])
    def test_reference_oracle_and_overflow(self):
        pairs=[(0,17),(1,19),(7,2),(65535,13)]
        self.assertEqual(c.reference_values(c.GOALS["DOUBLE"],pairs),
                         [(x*2)&c.MASK for x,y in pairs])
        self.assertEqual(c.reference_values(c.GOALS["INC"],pairs),
                         [(x+1)&c.MASK for x,y in pairs])
        self.assertEqual(c.reference_values(c.GOALS["PARITY"],pairs),
                         [x&1 for x,y in pairs])
    def test_real_neural_inference_not_symbolic_fallback(self):
        model,registry,_=c.load_frozen(11)
        pairs=[(0,3),(1,4),(2,5),(15,8),(65535,0)]
        pred,refs,elapsed=c.execute_neural(model,c.GOALS["DOUBLE"],pairs,registry)
        self.assertEqual(pred,[(x*2)&c.MASK for x,y in pairs])
        self.assertGreater(elapsed,0)
        wrong,_,_=c.execute_neural(model,("AND1_X","HALT"),pairs,registry)
        self.assertNotEqual(wrong,pred)
        self.assertEqual(len(refs),1)
    def test_invalid_program_fails_closed(self):
        model,registry,_=c.load_frozen(11)
        for program in ((),("ADD_XX",),("HALT",),("BAD","HALT"),
                        ("HALT","ADD_XX"),("ADD_XX","HALT","HALT")):
            with self.subTest(program=program):
                with self.assertRaises(ValueError):
                    c.execute_neural(model,program,[(1,2)],registry)
    def test_disjoint_dag_and_cfg(self):
        cfg=c.runtime_loop_cfg()
        self.assertTrue(cfg["cycle_present"])
        self.assertTrue(cfg["host_controlled"])
        self.assertFalse(cfg["model_generated"])
        self.assertIn(["branch","add"],cfg["edges"])
        for row in self.rows:
            if row["goal"]=="HOST_LOOP_TRIPLE":
                self.assertTrue(row["not_learned_control"])
                continue
            graph=row["data_dependency_DAG"]
            self.assertTrue(graph["acyclic"])
            for source,target in graph["edges"]:
                self.assertLess(int(source[1:]),int(target[1:]))
    def test_metadata_and_hashes(self):
        self.assertEqual(c.file_sha(c.HERE/"composition_traces.jsonl"),
                         self.report["trace_file_sha256"])
        self.assertEqual(c.file_sha(c.HERE/"compose.py"),
                         self.report["source_sha256"]["compose.py"])
        for entry in self.report["training"]:
            self.assertEqual(c.file_sha(c.ROOT/entry["checkpoint_path"]),
                             entry["checkpoint_sha256"])
    def test_frozen_cases_deterministic(self):
        for seed in c.SEEDS:
            pairs=c.heldout_pairs(seed)
            self.assertEqual(len(pairs),256)
            self.assertEqual(len(set(pairs)),256)
            self.assertEqual(pairs,c.heldout_pairs(seed))
    def test_nonclaims(self):
        self.assertTrue(self.report["frozen_original_neural_weights_unchanged"])
        self.assertTrue(self.report["data_dependency_dags_separate_from_runtime_cfg"])
        self.assertGreater(self.report["elapsed_training_and_eval_seconds"],0)
        self.assertTrue(any("Not a Transformer" in x for x in self.report["limit"]))
if __name__=="__main__":unittest.main()
