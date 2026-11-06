"""Independent-use deterministic replay of all frozen NC-070 heldout cases.

Reconstructs ISA oracle answers and policy decisions from source. Compares every
non-timing evaluation row, every intervention trace and CI against frozen data.
CPU times are measured anew and intentionally excluded from bytewise parity.

The script is a reproducibility aid, not an independent verification receipt.
"""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import discovery as d


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable(row):
    return {key:value for key,value in row.items() if key!="cpu_seconds"}


def check(directory):
    root=Path(directory)
    report=json.loads((root/"discovery_results.json").read_text())
    assert report["schema"]=="logos.nc070.discovery/1"
    seeds=report["seeds"]
    rounds=report["rounds"]
    assert len(seeds)>=3 and len(set(seeds))==len(seeds)
    assert report["heldout_ids"]==list(d.HELDOUT_IDS)
    assert report["hypothesis_count"]==64
    assert report["heldout_graph_count"]==len(d.HELDOUT_IDS)
    expected_rows=report["evaluation_rows"]
    expected_traces=[json.loads(x) for x in (root/"experiment_traces.jsonl").read_text().splitlines()]
    actual_rows,actual_traces=d.run_trials(seeds,rounds)
    # JSON-encode to convert tuples in the freshly generated data into lists.
    actual_rows=json.loads(json.dumps(actual_rows))
    actual_traces=json.loads(json.dumps(actual_traces))
    assert len(actual_rows)==len(expected_rows)==len(seeds)*len(d.HELDOUT_IDS)*3
    assert len(actual_traces)==len(expected_traces)==len(actual_rows)*rounds
    for i,(actual,expected) in enumerate(zip(actual_rows,expected_rows)):
        if stable(actual)!=stable(expected):
            raise AssertionError(f"evaluation row mismatch {i}: {stable(actual)} != {stable(expected)}")
        graph=d.ALL_GRAPHS[expected["truth_graph_id"]]
        assert expected["truth_edges"] == json.loads(json.dumps(d.canonical_edges(graph)))
        assert expected["exact"]==int(expected["predicted_graph_id"]==expected["truth_graph_id"])
    for i,(actual,expected) in enumerate(zip(actual_traces,expected_traces)):
        if actual!=expected:
            raise AssertionError(f"trace mismatch {i}: {actual} != {expected}")
    summary=d.cluster_bootstrap(actual_rows,seeds,rounds)
    for arm in d.ARMS:
        for metric in ("exact","total","uniquely_identified","oracle_experiments","accuracy",
                       "graph_cluster_bootstrap_accuracy_ci95"):
            if metric in summary[arm]:
                assert summary[arm][metric]==report["summary"][arm][metric], (arm,metric)
    assert summary["paired_active_minus_random"]==report["summary"]["paired_active_minus_random"]
    assert json.loads(json.dumps(d.observational_equivalence()))==report["observational_equivalence"]
    return {"status":"REPRODUCED","rows":len(actual_rows),"traces":len(actual_traces),
            "heldout_graphs":len(d.HELDOUT_IDS),"seeds":seeds,
            "exact":{arm:summary[arm]["exact"] for arm in d.ARMS},
            "result_sha256":sha(root/"discovery_results.json"),
            "trace_sha256":sha(root/"experiment_traces.jsonl")}


if __name__=="__main__":
    folder=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
    print(json.dumps(check(folder),sort_keys=True,indent=2))
