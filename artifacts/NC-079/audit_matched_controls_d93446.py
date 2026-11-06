"""NC-079 limited real-search audit for per-generation equal-size random controls.

Runs real search for library and random_library arms (not every experimental arm).
Checks both library count *and exact expression-size multiset* for 3 worlds plus
curriculum across seeds 11/19/29. Does not establish full campaign acceptance.
Requires the pre-existing, uncommitted NC-079 control-wiring changes to be present.
"""
import argparse
import hashlib
import json
import pathlib
import sys
from unittest.mock import patch

HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import stack as s
import run_compounding as comp
import run_curriculum as curr

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generations",type=int,default=3)
    p.add_argument("--budget",type=int,default=1500)
    p.add_argument("--train-goals",type=int,default=3)
    p.add_argument("--sealed-goals",type=int,default=2)
    p.add_argument("--retention-goals",type=int,default=2)
    p.add_argument("--seeds",nargs="+",type=int,default=[11,19,29])
    p.add_argument("--out",type=pathlib.Path,required=True)
    a=p.parse_args()
    xs,ys=s.probes()
    reports=[]
    for world in (*comp.WORLDS,"curriculum"):
        for seed in a.seeds:
            world_name=world if isinstance(world,str) else world.name
            if world_name=="curriculum":
                real,grown=curr.run_arm("library",seed,a,xs,ys)
            else:
                real,grown=comp.run_arm(world,"library",seed,a,xs,ys)
            matched=[]
            original=s.random_library
            def checked_random(rng,reference):
                fake=original(rng,reference)
                expected=sorted(s.size(e) for e in reference)
                observed=sorted(s.size(e) for e in fake)
                assert expected==observed,(world_name,seed,len(matched),expected,observed)
                matched.append({"generation_added":len(matched),
                                "reference_sizes":expected,
                                "random_sizes":observed,
                                "same_exact_shape_multiset":True})
                return fake
            with patch.object(s,"random_library",checked_random):
                if world_name=="curriculum":
                    control,_=curr.run_arm("random_library",seed,a,xs,ys,grown)
                else:
                    control,_=comp.run_arm(world,"random_library",seed,a,xs,ys,grown)
            assert len(real)==len(control)==len(matched)==a.generations
            for generation,(r,c) in enumerate(zip(real,control)):
                if generation==0:
                    assert r["library_size"]==c["library_size"]==0
                else:
                    n=len(grown[generation-1])
                    assert r["library_size"]==c["library_size"]==n,(world_name,seed,generation)
            report={"world":world_name,"seed":seed,"generations":a.generations,
                    "matched_count_and_size_multiset_all_generations":True,
                    "library_size_by_generation":[r["library_size"] for r in real],
                    "random_library_size_by_generation":[r["library_size"] for r in control],
                    "matched_shapes":matched,
                    "library_solved":[r["solved"] for r in real],
                    "random_solved":[r["solved"] for r in control]}
            reports.append(report)
            print("CHECK",world_name,seed,"counts",report["library_size_by_generation"],
                  "final_ref_sizes",matched[-1]["reference_sizes"],flush=True)
    sources={n:sha(HERE/n) for n in
             ["stack.py","run_compounding.py","run_curriculum.py",
              "test_stack.py","test_matched_library_controls_d93446.py",
              "audit_matched_controls_d93446.py"]}
    result={"schema":"logos.nc079.matched-control-audit/1",
            "scope":"bounded real-search 3 seeds x 4 worlds: library/random-library only, small budget; NOT full NC-079 acceptance",
            "config":{"generations":a.generations,"budget":a.budget,
                      "train_goals":a.train_goals,"sealed_goals":a.sealed_goals,
                      "retention_goals":a.retention_goals,"seeds":a.seeds},
            "source_sha256_at_run":sources,"matches":reports,
            "all_pass":all(v["matched_count_and_size_multiset_all_generations"] for v in reports)}
    assert result["all_pass"] and len(reports)==len(a.seeds)*4
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("SUCCESS",len(reports),"pairs",flush=True)

if __name__=="__main__":
    main()
