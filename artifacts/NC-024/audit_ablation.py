"""NC-024 candidate-side replay; NOT independent reviewer approval.
Dependencies: Python 3, torch, search_policy.py, NC-023/020/001 and frozen outputs.
"""
import collections
import hashlib
import json
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import search_policy as p

def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()

def audit():
    report=json.loads((HERE/"search_policy_ablation.json").read_text())
    traces=[json.loads(x) for x in (HERE/"neural_proposal_traces.jsonl").read_text().splitlines()]
    records=[json.loads(x) for x in (ROOT/"artifacts/NC-020/equivalence_benchmark.jsonl").read_text().splitlines()]
    cases={c["id"]:c for c in records}
    assert report["task"]=="NC-024" and report["comparison_verdict"]=="NO_GAIN"
    assert len(traces)==len(report["rows"])==324
    assert report["seeds"]==[11,19,29]
    train=set(report["train_ids"])
    heldout=set(report["test_ids"])|set(report["validation_ids"])
    assert len(train)==36 and len(heldout)==27 and train.isdisjoint(heldout)
    assert all(cases[id]["split"]=="train" for id in train)
    assert all(cases[id]["split"]!="train" for id in heldout)
    for path,hashval in report["source_sha256"].items():
        src=HERE/path if path in ("search_policy.py","run_ablation.py") else (
            ROOT/"artifacts"/("NC-023" if path=="symbolic_search.py" else
                              "NC-020" if path=="benchmark.py" else "NC-001")/path)
        assert sha(src)==hashval,("source drift",path)
    for seed,meta in report["checkpoints"].items():
        assert sha(ROOT/meta["path"])==meta["sha256"]
        assert meta["training"]["gradient_updates"]>0
    seen=collections.Counter()
    proposals=0
    for row, trace in zip(report["rows"],traces):
        key=(row["seed"],row["split"],row["id"],row["arm"])
        assert key==(trace["seed"],trace["split"],trace["id"],trace["arm"])
        seen[key]+=1
        case=cases[row["id"]]
        assert row["proved"] and row["split"]==case["split"]
        lhs=case["lhs"]
        origins={p.ss.digest(lhs): lhs}
        best=lhs
        assert trace["source_sha256"]==p.ss.digest(lhs)
        for step in trace["proposals"]:
            parent=origins[step["parent_sha256"]]
            choices=[node for rule,node in p.ss.candidates(parent)
                     if rule==step["rule"] and p.ss.digest(node)==step["candidate_sha256"]]
            assert choices,("missing generated candidate",key)
            candidate=choices[0]
            try:
                ok,proof=p.ss.equivalence(lhs,candidate,case)
            except (TypeError,ValueError,KeyError,p.ss.ref.isa.ISAError):
                ok,proof=False,{"method":"rejected_invalid_candidate"}
            assert bool(ok)==step["accepted"]
            assert proof["method"]==step["proof_method"]
            if ok:
                origins[step["candidate_sha256"]]=candidate
                if p.ss.score(candidate)<p.ss.score(best):
                    best=candidate
            proposals+=1
        assert len(trace["proposals"])==row["explored_proofs"]<=6
        assert p.ss.digest(best)==row["optimized_sha256"]==trace["selected_sha256"]
        assert p.ss.cost(best)["dag_ops"]==row["optimized_dag_ops"]
        ok,proof=p.ss.equivalence(lhs,best,case)
        assert ok and proof==trace["proof"]
    assert all(v==1 for v in seen.values())
    for neg in report["negative_controls"]:
        assert not p.ss.negative_control(cases[neg["id"]])["accepted"]
    return {"status":"CANDIDATE_REPLAY_PASS_NOT_INDEPENDENT_REVIEW",
            "rows_replayed":len(traces),"proposals_replayed":proposals,
            "negative_controls_rejected":len(report["negative_controls"]),
            "checkpoint_hashes_checked":len(report["checkpoints"]),
            "source_hashes_checked":len(report["source_sha256"]),
            "trained_advantage":"NO_GAIN"}

if __name__=="__main__":
    print(json.dumps(audit(),sort_keys=True))
