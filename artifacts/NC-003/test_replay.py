"""NC-003 replay harness correctness tests, not neural generalization evidence.

Dependencies: Python 3 stdlib, torch, replay_independent.py, host_leakage_audit.py,
saved NC-000 inventory, frozen source architecture modules (imported, not executed
for synthetic host tests).
"""
import json
import random
import unittest
from pathlib import Path
import torch
import replay_independent as runner
import host_leakage_audit as leakage

class SyntheticFixedRomStub:
    """Synthetic reference logits, only to test host wiring; NEVER a trained model."""
    @staticmethod
    def __call__(states,pc):
        n=len(pc)
        op=torch.zeros(n,8)
        nextpc=torch.zeros(n,8)
        halt=torch.zeros(n,2)
        b=runner.words(states[:,1,:])
        t=runner.words(states[:,3,:])
        for i,k in enumerate(pc.tolist()):
            next_id=runner.reference_next(k,b[i],t[i])
            op[i,k]=10.0
            nextpc[i,next_id]=10.0
            halt[i,1 if k==7 else 0]=10.0
        return op,nextpc,halt
    @staticmethod
    def local_logits(ids,features):
        result=torch.empty(len(ids),2)
        math=torch.zeros(len(ids),5)
        for i,op in enumerate(ids.tolist()):
            a,b,c,p,q,f=(int(v) for v in features[i].tolist())
            pair=(((a+b+c)&1,int(a+b+c>=2)) if op==0
                  else (p,0) if op==1 else (q,0) if op==2
                  else (a&f,0) if op==3 else (a|b,0))
            for col,v in enumerate(pair):
                result[i,col]=10.0 if v else -10.0
            math[i,op]=10.0
        return result,math

class HostHarnessTests(unittest.TestCase):
    def test_bitword_encode_decode_width16(self):
        r=random.Random(5)
        inputs=[0,1,255,256,32768,65535,*[r.randrange(65536) for _ in range(50)]]
        self.assertEqual(runner.words(runner.bits(inputs)),inputs)
    def test_reference_branch_targets(self):
        self.assertEqual(runner.reference_next(0,0,1),7)
        self.assertEqual(runner.reference_next(0,3,0),1)
        self.assertEqual(runner.reference_next(2,3,0),4)
        self.assertEqual(runner.reference_next(2,3,1),3)
        self.assertEqual(runner.reference_next(6,0,0),0)
        self.assertEqual(runner.reference_next(7,0,0),7)
        self.assertEqual([runner.reference_next(n,10,10) for n in (1,3,4,5)], [2,4,5,6])
    def test_stub_program_carries_and_halts_exhaustive_4bit(self):
        reference=SyntheticFixedRomStub()
        cases=[(a,b) for a in range(16) for b in range(16)]
        result=runner.reproduce(reference,cases)
        self.assertEqual(result['correct'],256)
        self.assertEqual(result['halted'],256)
        self.assertEqual(result['unhalted'],0)
        self.assertEqual(result['op_rom_agreement']['correct'],
                         result['op_rom_agreement']['tested'])
        self.assertEqual(result['pc_teacher_agreement']['correct'],
                         result['pc_teacher_agreement']['tested'])
        self.assertTrue(result['observed_branch_state_counts']['PC0_zero'])
        self.assertTrue(result['observed_branch_state_counts']['PC0_nonzero'])
        self.assertTrue(result['observed_branch_state_counts']['PC2_zero'])
        self.assertTrue(result['observed_branch_state_counts']['PC2_nonzero'])
    def test_stub_controls_prevent_false_reference_substitution(self):
        reference=SyntheticFixedRomStub()
        probes=[(13,11),(255,255),(37,63),(128,128),(16,16),(7,9),(127,128),(3,2)]
        control=runner.reproduce(reference,probes)
        poison=runner.reproduce(reference,probes,poison_pc=True)
        noarithmetic=runner.reproduce(reference,probes,zero_local=True)
        self.assertEqual(control['correct'],8)
        self.assertLess(poison['correct'],control['correct'])
        self.assertLess(noarithmetic['correct'],control['correct'])
    def test_historical_sample_duplicates_and_coverage(self):
        data=json.loads((Path(__file__).parent/'host_leakage_audit.json').read_text())
        counts=data['dataset_counts']
        self.assertEqual(counts['eightbit_evaluations'],4217)
        self.assertEqual(counts['eightbit_unique_operand_pairs'],4084)
        self.assertEqual(counts['eightbit_duplicate_evaluations'],133)
        self.assertEqual(counts['eightbit_full_domain'],65536)
        self.assertFalse(counts['original_pretraining_operand_overlap_excluded'])
        self.assertEqual(data['exhaustive_plan']['status'],'PROPOSED_NOT_EXECUTED')
    def test_source_borders(self):
        rec=json.loads((Path(__file__).parent/'host_leakage_audit.json').read_text())
        for name in ('shared_model','shared_training','source_math_encoder',
                     'source_math_encoder_v2','qualified_runtime','teacher_controller',
                     'independent_host_used_historically','NC003_new_host'):
            entry=rec['source_records'][name]
            self.assertEqual(entry['sha256'],leakage.sha(entry['source']))
        self.assertNotIn('verify_hybrid_independent',rec['source_records']['NC003_new_host']['imports'])
        self.assertNotIn('verify_shared',rec['source_records']['NC003_new_host']['imports'])
    def test_original_champion_checksum(self):
        manifest=json.loads((runner.ROOT/'artifacts/NC-000/inventory.json').read_text())
        expected=manifest['champion']['selected_bundle_sha256']
        self.assertEqual(runner.sha(runner.SOURCE/'snapshots/current_unified_v3.pt'),expected)

if __name__=='__main__':unittest.main()

class SavedFrozenReplayTests(unittest.TestCase):
 def test_full_saved_checkpoint_denominators_and_failures(self):
  p=Path(__file__).parent/'verification.json'
  self.assertTrue(p.is_file(),'full frozen replay must run before this test')
  d=json.loads(p.read_text())
  self.assertEqual(d['status'],'IMPLEMENTER_PASS_QUALIFIED_ARM')
  self.assertEqual(set(d['arm_results']),{'jointly_tuned','frozen_shared'})
  for arm,rows in d['arm_results'].items():
   self.assertEqual([r['seed'] for r in rows],[11,19,29])
   for row in rows:
    self.assertEqual((row['local_isa']['correct'],row['local_isa']['tested']),(320,320))
    self.assertEqual((row['opcode_semantics']['correct'],row['opcode_semantics']['tested']),(320,320))
    self.assertEqual((row['fourbit']['correct'],row['fourbit']['tested']),(256,256))
    self.assertEqual(row['eightbit']['tested'],4217)
    self.assertEqual(row['eightbit']['halted'],4217)
    self.assertEqual(row['eightbit']['unhalted'],0)
    self.assertEqual(row['eightcase_control']['correct'],8)
    self.assertEqual(row['pc_corruption_control']['correct'],0)
    self.assertEqual(row['zero_arithmetic_control']['correct'],0)
    self.assertGreater(row['eightbit']['branch_counts']['PC0_zero'],0)
    self.assertGreater(row['eightbit']['branch_counts']['PC0_nonzero'],0)
    self.assertGreater(row['eightbit']['branch_counts']['PC2_zero'],0)
    self.assertGreater(row['eightbit']['branch_counts']['PC2_nonzero'],0)
    if arm=='jointly_tuned' or row['seed']!=19:
     self.assertEqual(row['eightbit']['correct'],4217)
     self.assertEqual(row['status'],'PASS')
    else:
     self.assertEqual(row['eightbit']['correct'],4216)
     self.assertEqual(row['status'],'FAIL')
     self.assertEqual(row['eightbit']['failure_count'],1)
     self.assertEqual((row['eightbit']['failures'][0]['a'],row['eightbit']['failures'][0]['b']),(177,77))
