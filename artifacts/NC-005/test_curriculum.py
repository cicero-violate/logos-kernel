"""NC-005 correctness and preservation tests, independent of any result claims.

Depends on Python 3 stdlib, PyTorch, NumPy, artifacts/NC-005/curriculum.py,
NC-004/causal_semantics.py and frozen coupled_seed{11,19,29}.pt checkpoints,
NC-000/inventory.json. Does NOT retrain or rewrite checkpoints.
"""
import json
import pathlib
import random
import unittest
import torch
import curriculum as n

class CurriculumContract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1)

 def test_stage_order_and_matched_budget(self):
  self.assertEqual([s[0] for s in n.STAGES],
     ["BOOLEAN","CARRY_ADD","SHIFTS","COMPARE_SUB",
      "PC_BRANCH","SEQUENCES","LOOPS"])
  self.assertEqual(n.ARMS,("rehearsal","new_only"))
  self.assertEqual(n.SEEDS,(11,19,29))
  self.assertEqual(n.BATCH,256)
  self.assertEqual(n.STAGE_UPDATES,sum(s[2] for s in n.STAGES))
  self.assertEqual(len(n.OPS),9)
  self.assertEqual(len(set(n.OPS)),9)
  self.assertIn("SUB",n.OPS)
  self.assertIn("SAR",n.OPS)

 def test_local_truth_tables_9_by_64(self):
  x,y,ids=n.local_table()
  self.assertEqual(x.shape,(9*64,15))
  self.assertEqual(y.shape,(9*64,2))
  self.assertEqual(ids.shape,(9*64,))
  for op in n.OPS:
   self.assertEqual((ids==n.INDEX[op]).sum().item(),64)
  for op in n.OPS:
   for a,b,c,p,q,f in ((0,0,0,0,0,0),(1,1,1,1,1,1),
       (0,1,0,1,0,1),(1,0,1,0,1,0)):
    actual=n.primitive_target(op,a,b,c,p,q,f)
    self.assertEqual(len(actual),2)
    self.assertTrue(all(v in (0,1) for v in actual))
  self.assertEqual(n.primitive_target("SUB",0,1,1,0,0,0),(1,0))
  self.assertEqual(n.primitive_target("SUB",1,0,1,0,0,0),(1,1))
  self.assertEqual(n.primitive_target("SAR",1,0,0,0,0,1),(1,0))
  self.assertEqual(n.primitive_target("SAR",0,0,0,0,1,0),(1,0))

 def test_existing_model_warmstart_tensor_topology_and_hashes(self):
  x,y,ids=n.local_table()
  for seed in n.SEEDS:
   m=n.warm_start(seed)
   self.assertEqual(m.bit[0].weight.shape,(64,15))
   self.assertEqual(m.meaning.weight.shape,(9,9))
   self.assertEqual(m.is_zero[0].weight.shape,(32,16))
   old=n.bit_accuracy(m,x,y,ids,n.OLD)
   self.assertEqual(old["tested"],5*64)
   # Preserve the imported original trained old-op skills; no silent zeroing.
   self.assertGreaterEqual(old["correct"],300, f"warmstart_seed_{seed}")

 def test_oracle_sampled_sequences_mathematical_wiring(self):
  for name,ops in n.sequence_programs().items():
   self.assertEqual(len(ops),2)
   self.assertTrue(all(op in n.OPS for op,_,__ in ops))
   for a,b in ((0,0),(1,2),(255,255),(65535,1)):
    result=n.reference_sequence(ops,[(a,b)])
    self.assertEqual(len(result),1)
    self.assertTrue(0<=result[0]<=65535)

 def test_branch_words_and_target_partition(self):
  words=[0,1,2,255,256,32768,65535]
  self.assertEqual(n.words(n.bits(words)),words)
  self.assertEqual(n.take_branch(n.warm_start(11),"JMP",n.bits([0,1])),[True,True])
  self.assertEqual(n.take_branch(n.warm_start(11),"HALT",n.bits([0,1])),[False,False])

 def test_training_uses_real_intermediate_program_traces(self):
  import trace_training
  for seed in n.SEEDS:
   traces=trace_training.create(seed,n)
   info=traces["sources"]
   self.assertEqual(info["sequence_training_operand_pairs"],48)
   self.assertEqual(info["sequence_local_transition_examples"],4608)
   self.assertEqual(info["loop_training_operand_pairs"],96)
   self.assertGreater(info["loop_local_transition_examples"],10000)
   self.assertEqual(info["unseen_sequence_structure_test"],
                    ["subtract_then_xor","xor_then_double"])
   for phase in ("SEQUENCES","LOOPS"):
    x,y,ids=traces[phase]
    self.assertEqual(x.shape[1],len(n.OPS)+6)
    self.assertEqual(y.shape[1],2)
    self.assertEqual(len(x),len(y))
    self.assertEqual(len(x),len(ids))
    for row in (0,3,17,len(x)-1):
     f=x[row].tolist()
     op=n.OPS[int(ids[row])]
     values=[int(z) for z in f[len(n.OPS):]]
     self.assertEqual(tuple(int(z) for z in y[row].tolist()),
                      n.primitive_target(op,*values))

 def test_labels_exactly_bitvector_mod_width(self):
  r=random.Random(153)
  for k in range(300):
   a=r.randrange(1<<n.WIDTH);b=r.randrange(1<<n.WIDTH)
   for op in ("ADD","SUB","XOR","OR","AND"):
    expected=n.reference_sequence([(op,"X","Y")],[(a,b)])[0]
    truth=({"ADD":lambda:(a+b)&n.MASK,"SUB":lambda:(a-b)&n.MASK,
            "XOR":lambda:a^b,"OR":lambda:a|b,"AND":lambda:a&b}[op])()
    self.assertEqual(expected,truth)

 def test_training_does_not_promote_old_weights(self):
  inv=json.loads(n.CHAMPION.read_text())
  self.assertEqual(n.sha(n.base.CHAMPION),inv["champion"]["selected_bundle_sha256"])
  for seed in n.SEEDS:
   source=n.base.HERE/f"coupled_seed{seed}.pt"
   self.assertTrue(source.is_file())
   record=json.loads((n.base.HERE/"checkpoint.json").read_text())
   entry=next(z for z in record["models"] if z["arm"]=="coupled" and z["seed"]==seed)
   self.assertEqual(n.sha(source),entry["checkpoint_sha256"])

class SavedExperiment(unittest.TestCase):
 def test_saved_receipt_invariants_when_present(self):
  path=n.HERE/"training_report.json"
  if not path.is_file():
   self.skipTest("training not completed: do not claim experimental PASS")
  doc=json.loads(path.read_text())
  self.assertEqual(set(doc["results"]),
    {f"{arm}_seed{seed}" for arm in n.ARMS for seed in n.SEEDS})
  self.assertEqual(doc["status"],"IMPLEMENTER_TRAINING_COMPLETE")
  for name,res in doc["results"].items():
   self.assertEqual(len(res["stages"]),len(n.STAGES))
   self.assertEqual(res["stages"][-1]["updates_cumulative"],n.STAGE_UPDATES)
   for stage in res["stages"]:
    ck=n.ROOT/stage["checkpoint"]["path"]
    self.assertTrue(ck.is_file(),f"missing checkpoint {ck}")
    self.assertEqual(n.sha(ck),stage["checkpoint"]["sha256"])
    self.assertEqual(stage["old_five_retention"]["tested"],320)
    self.assertEqual(len(stage["local_by_op"]),9)
    for op in n.OPS:
     self.assertEqual(stage["local_by_op"][op]["tested"],64)
   first=n.ROOT/res["stages"][0]["checkpoint"]["path"]
   payload=torch.load(first,map_location="cpu",weights_only=True)
   self.assertEqual(payload["task"],"NC-005")
   self.assertIn("optimizer",payload)
   self.assertIn("rng_torch",payload)
   self.assertEqual(payload["nc004_source_checkpoint_sha256"],
                    n.sha(n.base.HERE/f"coupled_seed{res['seed']}.pt"))

if __name__=="__main__":
 unittest.main()
