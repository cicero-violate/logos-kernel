import hashlib, itertools, json, random, subprocess, sys
from pathlib import Path
import torch

ROOT=Path('/workspace/experiments/logos-kernel'); HERE=ROOT/'artifacts/NC-005'
sys.path.insert(0,str(HERE)); import curriculum as c
torch.set_num_threads(1)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def state_hash(x):
 h=hashlib.sha256()
 def add(v):
  if torch.is_tensor(v):
   q=v.detach().cpu().contiguous(); h.update(b'T');h.update(str(q.dtype).encode());h.update(str(tuple(q.shape)).encode());h.update(q.numpy().tobytes())
  elif isinstance(v,dict):
   h.update(b'D')
   for k in sorted(v,key=lambda z:repr(z)): add(k);add(v[k])
  elif isinstance(v,(list,tuple)): h.update(b'L');[add(z) for z in v]
  else: h.update(b'P');h.update(repr(v).encode())
 add(x);return h.hexdigest()
def model(path):
 p=torch.load(path,map_location='cpu',weights_only=True);m=c.ExpandedISA();m.load_state_dict(p['model']);m.eval();return m,p
def independent_local(m,ops):
 rows=[];truth=[];ids=[]
 for oi,op in enumerate(c.OPS):
  if op not in ops: continue
  for a,b,ci,pr,nx,fl in itertools.product((0,1),repeat=6):
   rows.append([float(i==oi) for i in range(9)]+[a,b,ci,pr,nx,fl]);ids.append(oi)
   if op=='ADD': z=a+b+ci; y=(z&1,int(z>=2))
   elif op=='SUB': z=a+(1-b)+ci;y=(z&1,int(z>=2))
   elif op=='AND':y=(a&b,0)
   elif op=='OR':y=(a|b,0)
   elif op=='XOR':y=(a^b,0)
   elif op=='AND1':y=(a&fl,0)
   elif op=='SHL':y=(pr,0)
   elif op=='SHR':y=(nx,0)
   elif op=='SAR':y=(a if fl else nx,0)
   truth.append(y)
 x=torch.tensor(rows,dtype=torch.float32); y=torch.tensor(truth,dtype=torch.bool); ii=torch.tensor(ids)
 with torch.no_grad(): bp=m.bit_from_features(x).ge(0);mp=m.meaning(x[:,:9]).argmax(1)
 bad=(~bp.eq(y).all(1)).nonzero().flatten().tolist();mb=(mp!=ii).nonzero().flatten().tolist()
 return {'bit_correct':len(rows)-len(bad),'bit_tested':len(rows),'meaning_correct':len(rows)-len(mb),'meaning_tested':len(rows),'bit_failed_indices':bad[:10],'meaning_failed_indices':mb[:10]}
def pairset(seed,n=256):
 edge=[0,1,255,32768,65535]; p=[(a,b) for a in edge for b in edge];r=random.Random(260000+seed)
 while len(p)<n: p.append((r.randrange(65536),r.randrange(65536)))
 return p
def arithmetic(m,seed):
 pairs=pairset(seed); x=c.bits([a for a,b in pairs]);y=c.bits([b for a,b in pairs]);out={}
 for op in c.OPS:
  with torch.no_grad(): z,carry=c.step_numeric(m,op,x,y);pred=c.words(z)
  # Reference is deliberately computed only after all predictions.
  def ref(a,b):
   return {'ADD':(a+b)&65535,'SUB':(a-b)&65535,'AND':a&b,'OR':a|b,'XOR':a^b,'AND1':a&1,'SHL':(a<<1)&65535,'SHR':a>>1,'SAR':(a>>1)|(a&32768)}[op]
  refs=[ref(a,b) for a,b in pairs]; bad=[{'a':a,'b':b,'pred':p,'ref':q} for (a,b),p,q in zip(pairs,pred,refs) if p!=q]
  d={'correct':len(pairs)-len(bad),'tested':len(pairs),'failures':bad[:8]}
  if op=='SUB':
   cp=[int(v>=.5) for v in carry.tolist()];cr=[int(a>=b) for a,b in pairs]; cb=[{'a':a,'b':b,'pred':p,'ref':q} for (a,b),p,q in zip(pairs,cp,cr) if p!=q]
   d['no_borrow']={'correct':len(pairs)-len(cb),'tested':len(pairs),'failures':cb[:8]}
  out[op]=d
 return out
def zero(m):
 bad=[];correct=0
 for off in range(0,65536,4096):
  vals=list(range(off,off+4096))
  with torch.no_grad(): pred=m.zero_logits(c.bits(vals)).ge(0).tolist()
  # truth formed only after prediction
  for v,p in zip(vals,pred):
   q=(v==0);correct+=p==q
   if p!=q and len(bad)<8:bad.append({'value':v,'pred':p,'ref':q})
 return {'correct':correct,'tested':65536,'failures':bad}
def sequences(m,seed):
 rng=random.Random(370000+seed);pairs=[(rng.randrange(65536),rng.randrange(65536)) for _ in range(256)]
 progs={'subtract_then_xor':[('SUB','X','Y'),('XOR','R','X')], 'xor_then_double':[('XOR','X','Y'),('SHL','R',None)]}
 out={}
 for name,prog in progs.items():
  pred=c.run_sequence(m,prog,pairs)
  refs=[]
  for a,b in pairs:
   if name=='subtract_then_xor': refs.append(((a-b)&65535)^a)
   else: refs.append(((a^b)<<1)&65535)
  bad=[{'a':a,'b':b,'pred':p,'ref':q} for (a,b),p,q in zip(pairs,pred,refs) if p!=q]
  out[name]={'correct':256-len(bad),'tested':256,'failures':bad[:8]}
 return out
def loops(m,seed):
 four=[(a,b) for a in range(16) for b in range(16)];rng=random.Random(480000+seed);eight=[(rng.randrange(256),rng.randrange(256)) for _ in range(256)]
 return {'four_bit':c.run_fixed_mul(m,four),'eight_bit_distinct_sample':c.run_fixed_mul(m,eight)}

report=json.loads((HERE/'training_report.json').read_text()); inv=json.loads((ROOT/'artifacts/NC-000/inventory.json').read_text())
results={'identity':{'runtime_thread_started_session_id':'01a1aeea-7525-7463-834f-7a0d78080be7','event_lab_owner':'1db9edb983ff4df9933612e7c826e55f','distinct':True},'final':{},'stages':{},'stage03_04':{},'integrity':{}}
for arm in c.ARMS:
 for seed in c.SEEDS:
  key=f'{arm}_seed{seed}'; path=HERE/f'checkpoints/{key}_stage06.pt';m,p=model(path)
  results['final'][key]={'local':independent_local(m,set(c.OPS)),'arithmetic':arithmetic(m,seed),'package_sha256':sha(path),'model_state_sha256':state_hash(p['model']),'optimizer_state_sha256':state_hash(p['optimizer']),'rng_state_sha256':state_hash({'torch':p['rng_torch'],'python':p['rng_python'],'numpy':p['numpy_rng']})}
  if arm=='rehearsal':results['final'][key].update({'zero':zero(m),'sequences':sequences(m,seed),'loops':loops(m,seed)})
  rows=[]
  for i,(stage,newops,_) in enumerate(c.STAGES):
   sp=HERE/f'checkpoints/{arm}_seed{seed}_stage{i:02d}.pt';sm,pay=model(sp);prior=set(c.OLD)
   for j in range(i+1):prior.update(c.STAGES[j][1])
   rows.append({'stage':stage,'local_prior':independent_local(sm,prior),'package_sha256':sha(sp),'model_state_sha256':state_hash(pay['model']),'optimizer_state_sha256':state_hash(pay['optimizer']),'rng_state_sha256':state_hash({'torch':pay['rng_torch'],'python':pay['rng_python'],'numpy':pay['numpy_rng']})})
  results['stages'][key]=rows
  b=torch.load(HERE/f'checkpoints/{key}_stage03.pt',map_location='cpu',weights_only=True)['model'];a=torch.load(HERE/f'checkpoints/{key}_stage04.pt',map_location='cpu',weights_only=True)['model']
  changed=[k for k in b if not torch.equal(b[k],a[k])];results['stage03_04'][key]={'changed':changed,'only_is_zero':bool(changed) and all(k.startswith('is_zero.') for k in changed),'bit_meaning_all_equal':all(torch.equal(b[k],a[k]) for k in b if k.startswith(('bit.','meaning.')))}
orig=[]
for q in inv['files']:
 p=Path(inv['roots'][q['repo']])/q['path'];orig.append({'path':str(p),'sha256':sha(p),'expected':q['sha256'],'match':sha(p)==q['sha256'],'bytes_match':p.stat().st_size==q['bytes']})
champ=Path(inv['roots']['rlt_champion'])/'snapshots/current_unified_v3.pt'
src=[]
for seed in c.SEEDS:
 p=ROOT/f'artifacts/NC-004/coupled_seed{seed}.pt';src.append({'seed':seed,'path':str(p),'sha256':sha(p)})
code_names=['README.md','curriculum.py','trace_training.py','analyze_retention.py','make_manifest.py','verify_fresh.py','test_curriculum.py','audit_budget.py','curriculum_receipts.jsonl','training_report.json','retention_matrix.json','fresh_reload.json','compute_budget.json','checkpoint_manifest.json']
tracked=[]
for name in code_names:
 p=HERE/name; blob=subprocess.check_output(['git','show',f'cf5c4893:{p.relative_to(ROOT)}']);tracked.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'commit_blob_sha256':hashlib.sha256(blob).hexdigest(),'match':p.read_bytes()==blob})
repos={}
for label,path in inv['roots'].items():
 if label in ('rlt_champion','standalone_isa'):
  repos[label]={'path':path,'head':subprocess.check_output(['git','-C',path,'rev-parse','HEAD'],text=True).strip(),'tracked_worktree_diff':subprocess.check_output(['git','-C',path,'diff','--name-only'],text=True).splitlines(),'tracked_index_diff':subprocess.check_output(['git','-C',path,'diff','--cached','--name-only'],text=True).splitlines()}
results['integrity']={'original_60':orig,'original_all_match':all(x['match'] and x['bytes_match'] for x in orig),'champion':{'path':str(champ),'sha256':sha(champ),'expected':inv['champion']['selected_bundle_sha256'],'match':sha(champ)==inv['champion']['selected_bundle_sha256']},'nc004_sources':src,'tracked_at_audited_commit':tracked,'all_tracked_match':all(x['match'] for x in tracked),'original_repositories':repos,'checkpoint_manifest_sha256':sha(HERE/'checkpoint_manifest.json')}
print(json.dumps(results,sort_keys=True))
