"""NC-008 replay and negative controls.

Depends on invent.py, NC-007 compose.py and its frozen NC-005/006 assets,
PyTorch, NumPy, training checkpoints and proposals.jsonl.
"""
import json
import unittest
from pathlib import Path
import torch
import invent

HERE = Path(__file__).resolve().parent


class IndependentReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.rows = [json.loads(x) for x in (HERE/"proposals.jsonl").read_text().splitlines()]
        cls.frozen,cls.registry,_ = invent.compose.load_frozen(11)

    def test_greedy_replay_and_new_samples(self):
        for row in self.rows:
            with self.subTest(task=row["task"]):
                payload = torch.load(invent.ROOT/row["proposal_checkpoint"],
                                     map_location="cpu",weights_only=True)
                self.assertEqual(payload["task"], row["task"])
                self.assertEqual(invent.sha(invent.ROOT/row["proposal_checkpoint"]),
                                 row["proposal_checkpoint_sha256"])
                model = invent.SpecTransformer()
                model.load_state_dict(payload["state_dict"],strict=True)
                model.eval()
                spec = invent.bit_features(payload["spec_examples"])
                program = invent.proposal(model,spec,1,greedy=True)[0]
                self.assertEqual(program,row["greedy_program"])
                self.assertEqual(program[-1],"HALT")
                self.assertLessEqual(len(program),3)
                fresh = invent.probes(99342+sum(map(ord,row["task"])),37)
                # Ensure a fresh independent evaluation split with no training examples.
                used = {(x,y) for x,y,_ in payload["spec_examples"]}
                fresh = [p for p in fresh if p not in used]
                actual,refs,_ = invent.compose.execute_neural(self.frozen,program,fresh,self.registry)
                expected = [invent.TASKS[row["task"]](x,y) for x,y in fresh]
                self.assertEqual(actual,expected)
                self.assertEqual(len(refs),len(program)-1)

    def test_wrong_program_is_rejected_by_behavior(self):
        for row in self.rows:
            with self.subTest(task=row["task"]):
                pairs = invent.probes(124231+sum(map(ord,row["task"])),40)
                altered = list(row["greedy_program"])
                altered[-2] = "AND1_X"
                got,_,_ = invent.compose.execute_neural(self.frozen,altered,pairs,self.registry)
                expected = [invent.TASKS[row["task"]](x,y) for x,y in pairs]
                self.assertNotEqual(got,expected)

    def test_primitive_provenance_tamper_rejected(self):
        p = self.rows[0]["greedy_program"]
        tampered = {k:dict(v) for k,v in self.registry.items()}
        tampered[p[0]]["checkpoint_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"UNVERIFIED_PRIMITIVE_PROVENANCE"):
            invent.compose.execute_neural(self.frozen,p,[(1,2)],tampered)

    def test_policy_proposals_depend_on_weights_and_behavior(self):
        altered_count=0
        for row in self.rows:
            payload = torch.load(invent.ROOT/row["proposal_checkpoint"],
                                 map_location="cpu",weights_only=True)
            model = invent.SpecTransformer()
            model.load_state_dict(payload["state_dict"],strict=True)
            model.eval()
            spec = invent.bit_features(payload["spec_examples"])
            prior = invent.proposal(model,spec,1,greedy=True)[0]
            with torch.no_grad():
                base = model(spec,torch.zeros(1,dtype=torch.long),
                             torch.tensor([invent.START]))
                altered = spec.clone()
                altered[:,:,32:48] = 1-altered[:,:,32:48]
                changed = model(altered,torch.zeros(1,dtype=torch.long),
                                torch.tensor([invent.START]))
                self.assertGreater(float((base-changed).abs().max()),1e-7)
                model.readout.weight.zero_()
                model.readout.bias.zero_()
            ablated = invent.proposal(model,spec,1,greedy=True)[0]
            altered_count += int(ablated != prior)
        self.assertGreaterEqual(altered_count,3)

    def test_truthful_limits_and_uniform_baseline(self):
        receipt=json.loads((HERE/"invention_receipt.json").read_text())
        self.assertEqual(receipt["supported_count"],4)
        self.assertEqual(receipt["proposals_sha256"],invent.sha(HERE/"proposals.jsonl"))
        self.assertTrue(all(row["first_exact_uniform_control_attempt"] is not None
                            for row in self.rows))
        self.assertEqual(len(self.rows),4)
        self.assertTrue(all(row["neural_isa_correct"]==256 for row in self.rows))


if __name__ == "__main__": unittest.main()
