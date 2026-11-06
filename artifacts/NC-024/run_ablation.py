"""NC-024: preregistered sealed-family ranking ablation.
Dependencies: Python 3, torch, sibling search_policy.py, NC-020/023/001.
"""
import collections
import hashlib
import json
import sys
import time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import search_policy as p

SEEDS = (11, 19, 29)
ARMS = ("neural", "untrained", "heuristic", "random")
PROOF_BUDGET = 6
POOL_CAP = 40
TRAIN_N = 6
EVAL_N = 3

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load():
    src = ROOT / "artifacts/NC-020/equivalence_benchmark.jsonl"
    meta = json.loads((ROOT/"artifacts/NC-020/splits_and_semantics.json").read_text())
    cases = [json.loads(line) for line in src.read_text().splitlines()]
    assert len(cases)==274
    assert p.ss.digest([{k:v for k,v in c.items() if k!="record_sha256"} for c in cases]) == meta["dataset_sha256"]
    assert all(p.ss.digest({k:v for k,v in c.items() if k!="record_sha256"}) == c["record_sha256"] for c in cases)
    grouped = collections.defaultdict(list)
    for case in cases:
        grouped[(case["split"],case["family"],case["label"])].append(case)
    def select(split, n):
        return [c for (sp,f,label),rs in sorted(grouped.items())
                if sp == split and label=="equivalent" for c in rs[:n]]
    train, val, test = select("train",TRAIN_N),select("validation",EVAL_N),select("test",EVAL_N)
    assert len(train)==36 and len(val)==12 and len(test)==15
    assert not (set(c["family"] for c in train) & set(c["family"] for c in val+test))
    negatives=[c for c in cases if c["split"] in ("validation","test") and c["label"]!="equivalent"]
    return train,val,test,negatives,sha(src),meta["dataset_sha256"]

def run():
    torch.set_num_threads(1)
    train,val,test,negatives,data_sha,semantic_sha=load()
    rows=[]; traces=[]; checkpoints={}
    start=time.perf_counter()
    for seed in SEEDS:
        model,training=p.fit(train,seed,epochs=2)
        torch.manual_seed(seed)
        untrained=p.RecurrentRewriteRanker()
        model.eval();untrained.eval()
        checkpoint=HERE/f"ranker_seed{seed}.pt"
        torch.save({"seed":seed,"state_dict":model.state_dict(),"train_ids":[c["id"] for c in train],
                    "training":training,"model":"new_shared_recurrent_attention"},checkpoint)
        checkpoints[str(seed)]={"path":str(checkpoint.relative_to(ROOT)),"sha256":sha(checkpoint),"training":training}
        for split,examples in (("validation",val),("test",test)):
            for case in examples:
                for arm in ARMS:
                    ranker=model if arm=="neural" else untrained if arm=="untrained" else None
                    outcome=p.ranked_search(case,arm,policy=ranker,seed=seed,budget=PROOF_BUDGET,pool_cap=POOL_CAP)
                    ok,proof=p.ss.equivalence(case["lhs"],outcome["optimized"],case)
                    assert ok and proof["method"]==outcome["proof"]["method"] and outcome["explored_proofs"]<=PROOF_BUDGET
                    traces.append({"seed":seed,"split":split,"family":case["family"],"id":case["id"],
                        "arm":arm,"source_sha256":p.ss.digest(case["lhs"]),
                        "selected_sha256":outcome["optimized_sha256"],
                        "proof":outcome["proof"],"proposals":outcome["trace"]})
                    row={k:outcome[k] for k in ("source_dag_ops","optimized_dag_ops","explored_proofs",
                        "verified_proposals","generated_proposals","ranker_cpu_s","full_cpu_s","full_wall_s")}
                    row.update({"seed":seed,"split":split,"family":case["family"],
                                "id":case["id"],"arm":arm,"optimized_sha256":outcome["optimized_sha256"],"proved":True})
                    rows.append(row)
    controls=[]
    for case in negatives:
        response=p.ss.negative_control(case)
        assert not response["accepted"],case["id"]
        controls.append({"id":case["id"],"split":case["split"],"family":case["family"],
                         "accepted":False,"reason":response.get("reason","inequivalent")})
    summary={}
    for arm in ARMS:
        summary[arm]={}
        for split in ("validation","test"):
            rs=[r for r in rows if r["arm"]==arm and r["split"]==split]
            summary[arm][split]={"cases":len(rs),"improved":sum(r["optimized_dag_ops"]<r["source_dag_ops"] for r in rs),
                "source_dag_ops":sum(r["source_dag_ops"] for r in rs),
                "optimized_dag_ops":sum(r["optimized_dag_ops"] for r in rs),
                "verified_proposals":sum(r["verified_proposals"] for r in rs),
                "proof_evaluations":sum(r["explored_proofs"] for r in rs),
                "generated_proposals":sum(r["generated_proposals"] for r in rs),
                "full_search_cpu_s":round(sum(r["full_cpu_s"] for r in rs),5),
                "full_search_wall_s":round(sum(r["full_wall_s"] for r in rs),5),
                "ranker_cpu_s":round(sum(r["ranker_cpu_s"] for r in rs),5)}
    train_cpu=sum(c["training"]["training_cpu_seconds"] for c in checkpoints.values())
    neural=summary["neural"]["test"]
    comparators=[summary[a]["test"] for a in ARMS if a!="neural"]
    gain=(all(neural["optimized_dag_ops"]<c["optimized_dag_ops"] and neural["improved"]>c["improved"] for c in comparators)
          and neural["full_search_cpu_s"]+train_cpu <= min(c["full_search_cpu_s"] for c in comparators))
    report={"task":"NC-024","schema":"logos.nc024.ablation/1",
        "status":"CANDIDATE_NOT_INDEPENDENTLY_VERIFIED","comparison_verdict":"GAIN" if gain else "NO_GAIN",
        "seeds":SEEDS,"arms":ARMS,
        "split_counts":{"train":len(train),"validation":len(val),"test":len(test),"negative_controls":len(negatives)},
        "train_ids":[c["id"] for c in train],"validation_ids":[c["id"] for c in val],
        "test_ids":[c["id"] for c in test],
        "budgets":{"proof_evaluations_per_case":PROOF_BUDGET,"frontier_pool_cap":POOL_CAP,
                   "epochs":2,"training_per_family":TRAIN_N,"evaluation_per_family":EVAL_N},
        "dataset_file_sha256":data_sha,"dataset_semantic_sha256":semantic_sha,
        "checkpoints":checkpoints,"total_training_cpu_s":round(train_cpu,6),
        "summary":summary,"rows":rows,"negative_controls":controls,
        "source_sha256":{"search_policy.py":sha(HERE/"search_policy.py"),
                        "run_ablation.py":sha(HERE/"run_ablation.py"),
                        "symbolic_search.py":sha(ROOT/"artifacts/NC-023/symbolic_search.py"),
                        "benchmark.py":sha(ROOT/"artifacts/NC-020/benchmark.py"),
                        "canonical_isa.py":sha(ROOT/"artifacts/NC-001/canonical_isa.py")},
        "wall_elapsed_s":round(time.perf_counter()-start,4),
        "limits":["New RLT-style candidate ranker, not original champion or reused weights.",
                  "Training imitates a deterministic static-cost ranker, not invented laws.",
                  "Training and inference CPU overhead counted separately; equal proof budgets do not imply equal total compute.",
                  "Ideal DAG cost is not hardware runtime.","No target RHS exposed; no historical pretraining isolation guarantee.",
                  "Finite polynomial or 3/4/5-bit output proof only; no stateful machine, floating, unrestricted math or AGI."]}
    (HERE/"neural_proposal_traces.jsonl").write_text("".join(p.ss.canonical(t)+"\n" for t in traces))
    (HERE/"search_policy_ablation.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"verdict":report["comparison_verdict"],
        "test":{a:summary[a]["test"] for a in ARMS},
        "training_cpu_s":report["total_training_cpu_s"],
        "rows":len(rows),"negative_controls":len(controls),"wall_s":report["wall_elapsed_s"]},sort_keys=True))

if __name__=="__main__":
    run()
