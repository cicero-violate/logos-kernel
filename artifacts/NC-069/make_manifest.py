"""Build intervention_benchmark.json, the results summary and CANDIDATE_MANIFEST.json after training ends."""
import hashlib
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import causal_benchmark as cb  # noqa: E402

ARMS = ("interventional", "passive_tokens", "passive_rewrite", "shuffled_control", "leaked_control")
BUCKETS = ("heldout_combo", "heldout_program", "factual")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(results):
    out = {}
    for arm in ARMS:
        runs = [r for r in results["runs"] if r["arm"] == arm]
        row = {"seeds": [r["seed"] for r in runs], "cpu_seconds_mean": round(statistics.mean(r["cpu_seconds"] for r in runs), 1)}
        for bucket in BUCKETS:
            acc = [r["buckets"][bucket]["accuracy"] for r in runs]
            effect = [r["buckets"][bucket]["effect_accuracy"] for r in runs if r["buckets"][bucket]["effect_accuracy"] is not None]
            row[bucket] = {"accuracy_mean": round(statistics.mean(acc), 4), "accuracy_min": min(acc), "accuracy_max": max(acc),
                           "effect_accuracy_mean": round(statistics.mean(effect), 4) if effect else None}
        out[arm] = row
    return out


def verdict(summary, baselines):
    """A preregistered reading of the comparison. It decides nothing about review."""
    def combo(arm):
        return summary[arm]["heldout_combo"]
    interventional = combo("interventional")["effect_accuracy_mean"]
    return {
        "metric": "mean exact four-register accuracy where the intervention changed the outcome, on unseen combinations",
        "interventional": interventional,
        "passive_tokens": combo("passive_tokens")["effect_accuracy_mean"],
        "passive_rewrite": combo("passive_rewrite")["effect_accuracy_mean"],
        "shuffled_control": combo("shuffled_control")["effect_accuracy_mean"],
        "leaked_control": combo("leaked_control")["effect_accuracy_mean"],
        "best_fixed_baseline": max(v["effect_accuracy"] or 0 for v in baselines["heldout_combo"].values()),
        "beats_passive_tokens": interventional > combo("passive_tokens")["effect_accuracy_mean"],
        "beats_passive_rewrite": interventional > combo("passive_rewrite")["effect_accuracy_mean"],
        "beats_shuffled_control": interventional > combo("shuffled_control")["effect_accuracy_mean"],
        "leak_control_detects_leakage": combo("leaked_control")["effect_accuracy_mean"] > 0.9}


def main(owner):
    results = json.loads((HERE / "results.json").read_text())
    if len(results["runs"]) != len(ARMS) * len(results["config"]["seeds"]):
        raise SystemExit("training has not finished")
    tests = {b: cb.generate(1000, results["config"]["test_samples"], b) for b in BUCKETS}
    benchmark = {"schema": "logos.nc069.benchmark/1", "isa_version": cb.isa.VERSION, "width": cb.WIDTH, "signedness": cb.SIGN,
                 "program_body_length": [cb.MIN_BODY, cb.MAX_BODY], "branch_rate": cb.BRANCH_RATE, "split_salt": cb.SPLIT_SALT,
                 "kinds": list(cb.KINDS), "test_generation_seed": 1000,
                 "held_out_combinations": results["held_out_combinations"],
                 "buckets": {b: {"samples": len(s), "outcome_changed": sum(x["target"] != x["factual"] for x in s),
                                 "by_kind": {k: sum(x["intervention"]["kind"] == k for x in s) for k in cb.KINDS}}
                             for b, s in tests.items()},
                 "test_sets_sha256": {b: hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest() for b, s in tests.items()}}
    (HERE / "intervention_benchmark.json").write_text(json.dumps(benchmark, indent=1))
    summary = summarize(results)
    comparison = verdict(summary, results["baselines"])
    (HERE / "results_summary.json").write_text(json.dumps({"summary": summary, "comparison": comparison,
        "baselines_heldout_combo": results["baselines"]["heldout_combo"]}, indent=1))
    sources = ["causal_benchmark.py", "train_eval.py", "test_causal_benchmark.py", "make_manifest.py", "recompute_baselines.py", "README.md"]
    evidence = ["results.json", "results_summary.json", "intervention_benchmark.json"]
    manifest = {"schema_version": 1, "task": "NC-069", "owner": owner,
        "source_sha256": {f: sha(HERE / f) for f in sources},
        "canonical_isa_sha256": sha(HERE.parent / "NC-001" / "canonical_isa.py"),
        "evidence_sha256": {f: sha(HERE / f) for f in evidence},
        "checkpoint_sha256": {r["checkpoint"]: r["checkpoint_sha256"] for r in results["runs"]},
        "config": results["config"], "comparison": comparison,
        "reproduction": ["PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-069/test_causal_benchmark.py",
                         "PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-069/train_eval.py --steps 2500 --out /tmp/nc069-replay"],
        "negative_controls": ["shuffled_control trains with intervention tokens shuffled across each batch",
                              "leaked_control receives the answer as input and should score near 100%",
                              "copy_input, factual_oracle and majority baselines",
                              "independent_run in test_causal_benchmark.py applies interventions during execution, not by rewriting"],
        "dependencies": ["Python 3", "PyTorch 2.12 CPU", "artifacts/NC-001/canonical_isa.py"],
        "limitations": ["4-bit twos-complement programs of 3 to 6 instructions with forward jumps only, so no loops",
                        "The machine is fully deterministic and observable; this says nothing about noisy or real-world causal inference",
                        "CPU is measured under shared load with tabs running, so CPU seconds are approximate, while steps and samples are exactly equal",
                        "Finite sampled held-out sets, not every program or intervention",
                        "The passive_rewrite arm is given the rewriting rule, so it does not test discovering that rule"]}
    (HERE / "CANDIDATE_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(comparison, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
