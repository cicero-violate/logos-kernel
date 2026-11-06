#!/usr/bin/env python3
"""Verify matched *executed training-graph shape* per curriculum arm.

Dependencies: Python3 standard library, PyTorch, NC005 checkpoint packages,
training_report.json, curriculum.py. No training, no checkpoint edits.
Writes compute_budget.json. Equal forward/backward path and batch sizes do
not guarantee equal wall-time or equal learning exposure.
"""
import hashlib
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as fd:
  for b in iter(lambda:fd.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def run():
 d=json.loads((HERE/'training_report.json').read_text())
 stages=d['stages']; seeds=d['seeds'];arms=d['arms'];batch=d['batch']
 assert len(stages)==7 and batch==256
 graphs=[]
 for i,(stage,opset,updates) in enumerate(stages):
  bit=stage not in ('PC_BRANCH',)
  zero=stage in ('PC_BRANCH','LOOPS')
  graphs.append({'stage':stage,'updates':updates,'batch':batch,
     'bit_and_meaning_samples':updates*batch if bit else 0,
     'zero_samples':updates*batch if zero else 0,
     'graph_type':('both' if bit and zero else 'bit_and_meaning' if bit else 'zero_only'),
     'same_for_both_arms':True})
 samples={'bit_and_meaning':sum(x['bit_and_meaning_samples'] for x in graphs),
          'zero':sum(x['zero_samples'] for x in graphs)}
 assert d['stage_updates']==2250
 assert samples['bit_and_meaning']==1850*256
 assert samples['zero']==750*256
 observations={}
 for seed in seeds:
  observations[str(seed)]={}
  for arm in arms:
   r=d['results'][f'{arm}_seed{seed}']
   assert len(r['stages'])==7
   for i,x in enumerate(r['stages']):
    assert x['updates_this_stage']==graphs[i]['updates']
    assert x['updates_cumulative']==sum(g['updates'] for g in graphs[:i+1])
   pre=ROOT/r['stages'][3]['checkpoint']['path']
   post=ROOT/r['stages'][4]['checkpoint']['path']
   assert sha(pre)==r['stages'][3]['checkpoint']['sha256']
   assert sha(post)==r['stages'][4]['checkpoint']['sha256']
   before=torch.load(pre,map_location='cpu',weights_only=True)
   after=torch.load(post,map_location='cpu',weights_only=True)
   unchanged=[];changed=[]
   for k,v in before['model'].items():
    assert v.shape==after['model'][k].shape
    if torch.equal(v,after['model'][k]):unchanged.append(k)
    else:changed.append(k)
   assert changed and all(k.startswith('is_zero.') for k in changed),(seed,arm,changed)
   assert any(k.startswith('bit.') for k in unchanged)
   assert any(k.startswith('meaning.') for k in unchanged)
   observations[str(seed)][arm]={'PC_BRANCH_previous_stage':r['stages'][3]['checkpoint'],
    'PC_BRANCH_end_stage':r['stages'][4]['checkpoint'],
    'only_zero_predicate_tensors_updated':True,
    'zero_tensors_changed':changed,
    'bit_meaning_tensors_unchanged':len([k for k in unchanged if k.startswith(('bit.','meaning.'))]),
    'stage_updates':graphs[4]['updates']}
 doc={'task':'NC-005','result':'IMPLEMENTER_MATCHED_COMPUTE_SCHEDULE_VERIFIED',
      'scope':'Matched per-stage graph types, batch sizes, update counts, optimizer and source data pools; different example distributions by intended replay intervention.',
      'stages':graphs,'samples_per_model':samples,'models':observations,
      'seeds':seeds,'arms':arms,
      'checkpoint_provenance':{f'{arm}_seed{seed}':d['results'][f'{arm}_seed{seed}']['source_checkpoint_sha256']
          for arm in arms for seed in seeds},
      'limitations':['Graph/sample counts are not measured FLOPs, wall-time, latency or energy.',
       'Arms are not identically sample-matched: old-task rehearsal is the independent variable.',
       'Only local truth-table and supplied program-trace training; not arbitrary neural program invention.',
       'The earlier f7f544d revision had asymmetric branch-stage compute and is superseded by this corrected run.',
       'No independent agent certification here.']}
 out=HERE/'compute_budget.json';out.write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'result':doc['result'],'stages':len(graphs),'per_model':samples,
  'branch_stage_bit_network_frozen_six_models':True,
  'sha256':sha(out)},indent=2))
if __name__=='__main__':run()
