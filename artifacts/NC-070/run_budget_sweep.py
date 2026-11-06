"""Recompute recovery-vs-experiment curves with matched interventions and CPU.

Outputs one summary for budgets 0..3, with 5 fixed seeds and 14 SHA-split
heldout graphs. Uses no training, GPU, network, or external model.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import discovery as d

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--seeds",nargs="+",type=int,default=[11,19,29,31,43])
    args=ap.parse_args()
    results=[]
    for budget in range(4):
        rows,_=d.run_trials(args.seeds,budget)
        summary=d.cluster_bootstrap(rows,args.seeds,budget)
        results.append({"interventions_per_case":budget,
                        "per_arm":{arm:summary[arm] for arm in d.ARMS},
                        "paired_active_minus_random":summary["paired_active_minus_random"],
                        "cpu_accounting":summary["cpu_accounting"]})
        print("BUDGET",budget,"N",summary["active"]["total"],"ACTIVE",summary["active"]["exact"],
              "RANDOM",summary["random"]["exact"],"PASSIVE",summary["passive"]["exact"],
              "CPU_MATCH",summary["cpu_accounting"]["matched_measured_cpu_20pct"],flush=True)
    args.out.write_text(json.dumps({
        "schema":"logos.nc070.budget-sweep/1",
        "graph_family":"exactly 64 four-register rooted-tree copy SCMs compiled to canonical ISA",
        "seeds":args.seeds,"heldout_graphs":len(d.HELDOUT_IDS),
        "all_policy_inputs_exclude_true_graph":True,
        "curve":results,
        "limits":"Graph clustered 95% bootstrap CIs may include zero; no robust active-over-random benefit claimed in those cases.",
    },indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    main()
