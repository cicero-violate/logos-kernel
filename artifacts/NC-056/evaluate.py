"""Trusted NC-056 evaluator. Python stdlib + Linux bubblewrap.

Fresh seeds are drawn only after frozen source checks. Targets stay outside the
participant mount/PID/network namespaces. Seed disclosure follows evaluation.
Local sandbox separation is not a cryptographic independent-review attestation.
"""
from collections import defaultdict
from decimal import Decimal, localcontext
from fractions import Fraction as F
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import secrets
import shutil
import subprocess
import tempfile
import time

HERE=Path(__file__).resolve().parent
FAMILIES=("constant","polynomial","geometric","periodic","multi_frequency",
          "nonlinear_modular","logistic_chaos","switching","coupled_vector",
          "nonuniform_polynomial","random_noise","aliasing","noisy_polynomial")
ARMS=("mdl_search","autoregression","persistence","interpolation","random_candidates","no_invariant")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(obj): return json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False)


def limits():
    resource.setrlimit(resource.RLIMIT_CPU,(30,31))
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE,(16*1024**2,16*1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))


def isolated(source, public):
    bwrap=shutil.which("bwrap")
    if not bwrap: raise RuntimeError("bubblewrap required; no unsandboxed fallback")
    if len(source)>65536: raise ValueError("source cap")
    with tempfile.TemporaryDirectory(prefix="nc056-") as td:
        candidate=Path(td,"candidate.py");candidate.write_bytes(source)
        out=Path(td,"out");err=Path(td,"err")
        cmd=[bwrap,"--unshare-all","--die-with-parent","--new-session","--clearenv",
             "--ro-bind","/usr","/usr","--ro-bind","/lib","/lib","--ro-bind","/lib64","/lib64",
             "--dev","/dev","--proc","/proc","--tmpfs","/tmp","--dir","/app",
             "--ro-bind",str(candidate),"/app/candidate.py","--chdir","/app",
             "--setenv","HOME","/tmp","--setenv","PYTHONDONTWRITEBYTECODE","1",
             "--","/usr/bin/python3","-I","/app/candidate.py"]
        with out.open("wb") as stdout,err.open("wb") as stderr:
            completed=subprocess.run(cmd,input=("".join(canonical(p)+"\n" for p in public)).encode(),
                stdout=stdout,stderr=stderr,env={"PATH":"/usr/bin:/bin"},cwd="/",
                timeout=45,preexec_fn=limits)
        if completed.returncode: raise RuntimeError(err.read_text()[:1000])
        raw=out.read_bytes()
        if len(raw)>16*1024**2: raise ValueError("output cap")
        result=[json.loads(line) for line in raw.decode().splitlines()]
    if len(result)!=len(public): raise ValueError("one response per case required")
    if any(r.get("case_id")!=p["case_id"] for r,p in zip(result,public)): raise ValueError("ID mismatch")
    return result


def generate(seed):
    rng=random.Random(seed)
    cases=[]
    for family in FAMILIES:
        for regime,n,h in (("short",16,8),("long",30,24)):
            p=rng.randrange(2,20);degree=rng.randrange(1,5)
            coeff=[rng.randrange(-5,6) for _ in range(degree)]+[rng.choice((-3,-2,1,2,3))]
            rate=rng.choice((F(2),F(3),F(3,2)))
            period=rng.randrange(3,7);cycle=[rng.randrange(-9,10) for _ in range(period)]
            modulus=rng.choice((31,43,59,73,101,113));offset=rng.randrange(1,8)
            r=Decimal("4") if regime=="short" else Decimal("3.91")
            switch=n//2 if regime=="short" else n+3
            times=[i*(i+1)//2 for i in range(n+h)] if family=="nonuniform_polynomial" else list(range(n+h))
            if family=="aliasing": times=[4*i for i in range(n)]+[4*(n-1)+j for j in range(1,h+1)]
            series=[]
            if family in ("nonlinear_modular","logistic_chaos","coupled_vector"):
                value=F(p%modulus) if family=="nonlinear_modular" else (F(p),F(3)) if family=="coupled_vector" else Decimal(1)/Decimal(p+3)
                with localcontext() as ctx:
                    ctx.prec=90
                    for t in range(n+h):
                        series.append(value)
                        if family=="nonlinear_modular": value=(value*value+offset)%modulus
                        elif family=="coupled_vector": value=(value[1],value[0]+value[1])
                        else:value=r*value*(1-value)
            else:
                for t in times:
                    if family=="constant": v=F(p)
                    elif family in ("polynomial","noisy_polynomial"): v=F(sum(c*t**i for i,c in enumerate(coeff)))
                    elif family=="nonuniform_polynomial":v=F(p-3*t+2*t*t)
                    elif family=="geometric":v=p*rate**t
                    elif family=="periodic":v=F(cycle[t%period])
                    elif family=="multi_frequency":v=F((2,0,-2,0)[t%4]+(0,1,-1)[(t+p)%3])
                    elif family=="switching":v=F(p+2*min(t,switch)-3*max(0,t-switch))
                    elif family=="random_noise":v=F(rng.randrange(-50,51))
                    elif family=="aliasing":v=F((1,0,-1,0)[t%4]*p)
                    else:raise ValueError(family)
                    series.append(v)
            observed=series[:n]
            if family=="noisy_polynomial":observed=[v+F(rng.choice((-1,1)),1000) for v in observed]
            encode=lambda v:[str(x) for x in v] if isinstance(v,tuple) else str(v)
            public={"case_id":secrets.token_hex(16),"shape":"vector2" if family=="coupled_vector" else "scalar",
                    "observed_times":times[:n],"observed_values":[encode(v) for v in observed],"query_times":times[n:]}
            private={"family":family,"regime":regime,"seed":seed,"reference":[encode(v) for v in series[n:]],
                     "degree":2 if family=="nonuniform_polynomial" else degree,
                     "coefficients":[p,-3,2] if family=="nonuniform_polynomial" else coeff,
                     "rate":str(rate),"period":period,"modulus":modulus,"offset":offset,"logistic_rate":str(r)}
            cases.append((public,private))
    rng.shuffle(cases)
    return cases


def model_matches(models, private):
    if any(m is None for m in models):return False
    m=models[0];fam=private["family"]
    if fam in ("polynomial","nonuniform_polynomial"):
        if m["kind"]!="polynomial" or m["order"]!=private["degree"]:return False
        # Exact polynomial identity: d+1 distinct evaluations suffice when both
        # degree bounds are known. This proves a candidate formula, not induction
        # from finitely many unknown observations to a universal physical law.
        for t in range(private["degree"]+1):
            value=F(m["coeff"][-1])
            for i in range(len(m["coeff"])-2,-1,-1):value=F(m["coeff"][i])+(t-F(m["nodes"][i]))*value
            if value!=sum(c*t**i for i,c in enumerate(private["coefficients"])):return False
        return True
    if fam=="geometric":return m["kind"]=="ar" and m["coeff"]==[private["rate"]]
    if fam=="nonlinear_modular":return m["kind"]=="quadratic_mod" and m["modulus"]==private["modulus"] and m["offset"]==private["offset"]
    if fam=="logistic_chaos":return m["kind"]=="logistic" and abs(m["rate"]-float(private["logistic_rate"]))<1e-10
    if fam=="coupled_vector":return all(m["kind"]=="ar" and m["coeff"]==["1","1"] for m in models)
    if fam=="periodic":return m["kind"]=="periodic" and m["period"]==private["period"]
    if fam=="multi_frequency":return m["kind"]=="periodic" and m["period"]==12
    return False


def grade(public, private, submission):
    prediction=submission["forecast"]
    truth=private["reference"]
    if len(prediction)!=len(truth):raise ValueError("horizon mismatch")
    errors=[];tolerances=[];exact=[]
    for predicted,target in zip(prediction,truth):
        a=predicted if isinstance(predicted,list) else [predicted]
        b=target if isinstance(target,list) else [target]
        if len(a)!=len(b):raise ValueError("shape mismatch")
        for x,y in zip(a,b):
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):raise ValueError("invalid number")
            error=abs(x-float(F(y)));errors.append(error);tolerances.append(1e-6+1e-8*abs(float(F(y))))
            exact.append(F(str(x))==F(y))
    valid=all(e<=t for e,t in zip(errors,tolerances))
    nonconstant=any(row!=truth[0] for row in truth[1:])
    structural=model_matches(submission["models"],private)
    baseline=submission["arm"] in ("persistence","interpolation","no_invariant")
    return {"forecast_pass":valid,"exact_output_values":all(exact),"maximum_error":max(errors),
            "mean_error":sum(errors)/len(errors),"structural_signature_match":structural,
            "bounded_law_recovery":bool(valid and nonconstant and structural and not baseline and not any(submission["failures"])),
            "horizon":len(truth),"first_failed_scalar":next((i for i,(e,t) in enumerate(zip(errors,tolerances)) if e>t),None)}


def verify_freeze():
    contract=json.loads((HERE/"minimum_order_discovery_contract.json").read_text())
    for rel,expected in contract["source_sha256"].items():
        if sha(HERE/rel)!=expected:raise ValueError("source changed after preregistration: "+rel)
    root=HERE.parents[1]
    for rel in ["minimum_order_discovery_contract.json",*contract["source_sha256"]]:
        path=HERE/rel
        blob=subprocess.check_output(["git","-C",str(root),"show","HEAD:"+str(path.relative_to(root))])
        if blob!=path.read_bytes():raise ValueError("commit source and contract before drawing heldout seeds")
    return contract


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--replay-seeds",type=Path)
    parser.add_argument("--output",type=Path,default=HERE/"heldout_law_recovery.json")
    args=parser.parse_args()
    contract=verify_freeze()
    if args.output.exists():raise ValueError("refuse evidence overwrite; choose a new output")
    seeds=json.loads(args.replay_seeds.read_text())["seeds"] if args.replay_seeds else [secrets.randbits(64) for _ in range(3)]
    cases=[case for seed in seeds for case in generate(seed)]
    source=(HERE/"discover.py").read_bytes()
    records=[];metrics=defaultdict(lambda:defaultdict(lambda:dict(cases=0,forecast_pass=0,bounded_law_recovery=0,
        candidate_checks=0,cpu_seconds=0.0,maximum_error=0.0,budget_exhaustions=0)))
    started=time.perf_counter()
    for start in range(0,len(cases),12):
        batch=cases[start:start+12]
        outputs=isolated(source,[public for public,_ in batch])
        for (public,private),response in zip(batch,outputs):
            if [a["arm"] for a in response["arms"]]!=list(ARMS):raise ValueError("arms mismatch")
            graded=[]
            for arm in response["arms"]:
                if not 0<=arm["candidate_checks"]<=128:raise ValueError("candidate cap violated")
                score=grade(public,private,arm)
                summary=metrics[private["family"]][arm["arm"]]
                summary["cases"]+=1
                for key in ("forecast_pass","bounded_law_recovery"):summary[key]+=score[key]
                summary["candidate_checks"]+=arm["candidate_checks"]
                summary["cpu_seconds"]+=arm["cpu_seconds"]
                summary["maximum_error"]=max(summary["maximum_error"],score["maximum_error"])
                summary["budget_exhaustions"]+=arm["budget_exhausted"]
                graded.append({"submission":arm,"score":score})
            records.append({"public":public,"private_disclosed_after_submission":private,"arms":graded})
        print(f"evaluated {min(start+12,len(cases))}/{len(cases)} cases",flush=True)
    # Receipt falsification controls: corrupt every future answer, spoof constant
    # answers with the correct signature, and assert the judge rejects discovery.
    controls={"corrupted_forecast_rejected":0,"constant_spoof_rejected":0,"cases":len(records)}
    for record in records:
        public=record["public"];private=record["private_disclosed_after_submission"]
        genuine=record["arms"][0]["submission"]
        changed=json.loads(json.dumps(genuine))
        changed["forecast"]=[[(float(F(v))+abs(float(F(v)))+1000) for v in t] if isinstance(t,list)
                               else float(F(t))+abs(float(F(t)))+1000 for t in private["reference"]]
        assert not grade(public,private,changed)["forecast_pass"]
        controls["corrupted_forecast_rejected"]+=1
        changed["forecast"]=[genuine["forecast"][0]]*len(public["query_times"])
        assert not grade(public,private,changed)["bounded_law_recovery"]
        controls["constant_spoof_rejected"]+=1
    totals={arm:{key:sum(f[arm][key] for f in metrics.values()) for key in
                    ("cases","forecast_pass","bounded_law_recovery","candidate_checks","cpu_seconds","budget_exhaustions")}
            for arm in ARMS}
    by_seed={str(seed):{arm:{"cases":0,"forecast_pass":0,"bounded_law_recovery":0} for arm in ARMS} for seed in seeds}
    for record in records:
        seed=str(record["private_disclosed_after_submission"]["seed"])
        for arm in record["arms"]:
            row=by_seed[seed][arm["submission"]["arm"]];row["cases"]+=1
            for key in ("forecast_pass","bounded_law_recovery"):row[key]+=arm["score"][key]
    trace_path=args.output.with_name(args.output.stem+"_traces.jsonl")
    if trace_path.exists():raise ValueError("trace exists")
    trace_path.write_text("".join(canonical(r)+"\n" for r in records))
    result={"task":"NC-056","status":"IMPLEMENTER_CANDIDATE_NOT_INDEPENDENTLY_VERIFIED",
            "seeds":seeds,"source_sha256":contract["source_sha256"],"contract_sha256":sha(HERE/"minimum_order_discovery_contract.json"),
            "cases":len(records),"per_family":dict(metrics),"per_seed":by_seed,"totals":totals,
            "controls":controls,"trace_path":str(trace_path.relative_to(HERE)),"trace_sha256":sha(trace_path),
            "end_to_end_wall_seconds":time.perf_counter()-started,
            "comparison":"equal candidate-check and CPU stopping ceilings, not equal consumed CPU; compare errors and actual CPU together",
            "training_cpu_seconds":0,"new_model_weights":False,"claim":"bounded symbolic candidate discovery, not RSI or AGI",
            "holdout_scope":"three fresh post-freeze evaluator seeds, unseen parameters and 16/30-prefix, 8/24-horizon trajectories; exact rational observations except Decimal90 logistic; oracle excluded from participant sandbox",
            "limitations":contract["limitations"]}
    verify_freeze()
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"output":str(args.output),"totals":totals,"controls":controls},indent=2))


if __name__=="__main__":main()
