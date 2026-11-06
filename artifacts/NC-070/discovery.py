"""NC-070: bounded active discovery of ISA-register causal graphs.

A learner infers one of 64 four-node directed rooted trees from outputs of
chosen do(register := value) interventions. ALL candidate trees yield identical
passive observations. The true graph is never an argument to the policy.

The oracle evaluates real canonical-ISA programs via the independently passed
NC-069 intervention adapter. The learner uses a separately coded graph simulator
to predict candidate responses, and updates its version space only from oracle
responses. No graph labels, hidden program or held-out answer enter planning.

This is finite-hypothesis active system identification, NOT a learned world
model, general causal discovery or novel ISA operator invention.
"""
import argparse
from collections import deque
import hashlib
from itertools import combinations
import json
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "NC-069"))
import causal_benchmark as nc069

REGS = tuple(nc069.isa.REGS)
N = len(REGS)
assert N == 4, "NC-070 is defined on the four canonical registers"
SALT = "logos/NC-070/heldout-graph/v1"
ARMS = ("active", "random", "passive")
MAX_EXPERIMENTS = N - 1


def directed_trees():
    """Exactly 64 distinct rooted labelled trees on four nodes (Cayley)."""
    undirected = list(combinations(range(N), 2))
    trees = set()
    for picked in combinations(undirected, N-1):
        graph = [[] for _ in range(N)]
        for a, b in picked:
            graph[a].append(b)
            graph[b].append(a)
        for root in range(N):
            parents = [None] * N
            parents[root] = -1
            queue = deque([root])
            while queue:
                node = queue.popleft()
                for child in sorted(graph[node]):
                    if parents[child] is None:
                        parents[child] = node
                        queue.append(child)
            if None not in parents:
                trees.add(tuple(parents))
    return tuple(sorted(trees))


ALL_GRAPHS = directed_trees()
assert len(ALL_GRAPHS) == 64


def canonical_edges(parents):
    return tuple((parent, child) for child, parent in enumerate(parents) if parent >= 0)


def topological(parents):
    root = parents.index(-1)
    queue = deque([root])
    ordered = [root]
    while queue:
        node = queue.popleft()
        for child, parent in enumerate(parents):
            if parent == node:
                queue.append(child)
                ordered.append(child)
    if len(ordered) != N:
        raise ValueError("not a tree")
    return ordered


def canonical_program(parents):
    order = topological(parents)
    body = [{"op": "MOV", "dst": REGS[child], "src": REGS[parents[child]]}
            for child in order[1:]]
    program = nc069.isa.validate({
        "version": nc069.isa.VERSION, "width": 4,
        "signedness": "twos_complement", "program": body + [{"op": "HALT"}]})
    return program, {child: index for index, child in enumerate(order[1:])}


def make_oracle(parents):
    """Return an opaque query callable; no graph, program or label exposed."""
    program, steps = canonical_program(parents)
    root = parents.index(-1)

    def answer(node, value, exogenous_bit):
        inputs = {reg: exogenous_bit for reg in REGS}
        if node is None:
            intervention = {"kind": "none"}
        elif node == root:
            intervention = {"kind": "change_input", "register": REGS[node],
                            "value": value}
        else:
            intervention = {
                "kind": "replace_instruction", "step": steps[node],
                "instruction": {"op": "LOADI", "dst": REGS[node], "imm": value},
            }
        return tuple(nc069.outcome(program, inputs, intervention))
    return answer


def simulate(parents, node, value, exogenous_bit):
    """Separate scalar predictor used by policy; does not call NC-069 or ISA."""
    state = [exogenous_bit] * N
    for current in topological(parents):
        if current == node:
            state[current] = value
        elif parents[current] >= 0:
            state[current] = state[parents[current]]
    return tuple(state)


def test_graph_ids():
    return tuple(i for i, g in enumerate(ALL_GRAPHS)
                 if int(hashlib.sha256((SALT + "|" + repr(g)).encode()).hexdigest()[:8], 16) % 5 == 0)


HELDOUT_IDS = test_graph_ids()
assert 4 <= len(HELDOUT_IDS) < len(ALL_GRAPHS)


def choose_active(hypotheses, used, bit):
    """Minimum expected survivor count. Uses HYPOTHESES only; no hidden graph."""
    candidates = [node for node in range(N) if node not in used]
    if not candidates:
        raise ValueError("no unused nodes")
    score = {}
    # Do identical 64*4 candidate simulations for every arm/round,
    # independent of the size of the shrinking version space.
    all_predictions = {
        node: [simulate(graph, node, 1-bit, bit) for graph in ALL_GRAPHS]
        for node in range(N)
    }
    for node in candidates:
        buckets = {}
        for idx in hypotheses:
            observed = all_predictions[node][idx]
            buckets[observed] = buckets.get(observed, 0) + 1
        score[node] = sum(size * size for size in buckets.values())
    return min(candidates, key=lambda node: (score[node], node)), score


def run_episode(query, seed, episode_nonce, arm, rounds):
    """Calls query only. The true graph is not an input or accessible field."""
    if arm not in ARMS or rounds < 0 or rounds > MAX_EXPERIMENTS:
        raise ValueError("invalid arm or budget")
    random_nodes = list(range(N))
    random.Random(f"NC-070/random-query/{seed}/{episode_nonce}").shuffle(random_nodes)
    bits = [random.Random(f"NC-070/exo/{seed}/{episode_nonce}/{t}").randrange(2)
            for t in range(rounds)]
    hypotheses = tuple(range(len(ALL_GRAPHS)))
    traces = []
    used = set()
    start_cpu = time.process_time_ns()
    for t, bit in enumerate(bits):
        suggested, scores = choose_active(hypotheses, used, bit)
        # All arms spend the same counterfactual-planning work for fair
        # algorithmic-compute comparisons; only active USES the score.
        if arm == "active":
            node = suggested
        elif arm == "random":
            node = random_nodes[t]
        else:
            node = None
        value = None if node is None else 1-bit
        observed = query(node, value, bit)
        if node is not None:
            hypotheses = tuple(idx for idx in hypotheses if
                               simulate(ALL_GRAPHS[idx], node, value, bit) == observed)
            if not hypotheses:
                raise AssertionError("oracle observation excludes every graph")
            used.add(node)
        else:
            # Passive observations (bit,bit,bit,bit) never shrink the space.
            assert observed == (bit,) * N
        traces.append({
            "step": t, "exogenous_bit": bit, "selected_register": (
                "none" if node is None else REGS[node]),
            "do_value": value, "observed_register_values": observed,
            "active_suggestion": REGS[suggested],
            "expected_survivor_size_scores": {REGS[k]: v for k, v in scores.items()},
            "surviving_hypotheses": len(hypotheses),
        })
    cpu_seconds = (time.process_time_ns()-start_cpu)/1e9
    # Identical tie RNG for all arms, with no access to the true graph.
    guessed = random.Random(f"NC-070/tie/{seed}/{episode_nonce}").choice(hypotheses)
    return {
        "predicted_graph_id": guessed, "remaining_graphs": len(hypotheses),
        "unique_identification": len(hypotheses) == 1,
        "cpu_seconds": cpu_seconds,
        "oracle_calls": rounds, "traces": traces,
    }


def edge_f1(true_graph, inferred_graph):
    gold, pred = set(canonical_edges(true_graph)), set(canonical_edges(inferred_graph))
    overlap = len(gold & pred)
    return 2 * overlap / (len(gold) + len(pred))


def run_trials(seeds, rounds):
    """Only this outer scorer knows the held-out true graphs."""
    evaluations = []
    traces = []
    for seed in seeds:
        nonce_rng = random.Random(f"NC-070/independent-episode-nonce/{seed}")
        for case_index, graph_index in enumerate(HELDOUT_IDS):
            episode_nonce = nonce_rng.getrandbits(128)
            truth = ALL_GRAPHS[graph_index]
            # Create a query closure. Policy receives no truth/graph_index.
            query = make_oracle(truth)
            for arm in ARMS:
                episode = run_episode(query, seed, episode_nonce, arm, rounds)
                guessed = episode["predicted_graph_id"]
                evaluations.append({
                    "seed": seed, "case_index": case_index,
                    "truth_graph_id": graph_index,
                    "truth_edges": canonical_edges(truth),
                    "arm": arm, "experiments": rounds,
                    "predicted_graph_id": guessed,
                    "predicted_edges": canonical_edges(ALL_GRAPHS[guessed]),
                    "exact": int(guessed == graph_index),
                    "unique": int(episode["unique_identification"]),
                    "remaining_graphs": episode["remaining_graphs"],
                    "edge_f1": edge_f1(truth, ALL_GRAPHS[guessed]),
                    "oracle_calls": episode["oracle_calls"],
                    "cpu_seconds": episode["cpu_seconds"],
                })
                for step in episode["traces"]:
                    traces.append({
                        "seed": seed, "case_index": case_index,
                        "arm": arm, **step,
                        # Ground truth is never in a per-step trace or a
                        # callable available to run_episode.
                    })
    return evaluations, traces


def cluster_bootstrap(evaluations, seeds, rounds, samples=4000):
    """Graph-clustered paired bootstrap: seeds are repeated within graph."""
    by_key = {(r["seed"], r["case_index"], r["arm"]): r for r in evaluations}
    cases = sorted({r["case_index"] for r in evaluations})
    n = len(cases)
    rng = random.Random("NC-070/CI/bootstrap/v1")
    output = {}
    for arm in ARMS:
        numer = sum(r["exact"] for r in evaluations if r["arm"] == arm)
        denom = sum(1 for r in evaluations if r["arm"] == arm)
        unique = sum(r["unique"] for r in evaluations if r["arm"] == arm)
        cpu = sum(r["cpu_seconds"] for r in evaluations if r["arm"] == arm)
        oracle = sum(r["oracle_calls"] for r in evaluations if r["arm"] == arm)
        output[arm] = {
            "exact": numer, "total": denom, "accuracy": numer/denom,
            "uniquely_identified": unique, "oracle_experiments": oracle,
            "cpu_seconds": round(cpu, 5),
            "correct_per_cpu_second": round(numer / cpu, 5) if cpu else None,
            "correct_per_oracle_experiment": round(numer/oracle, 5) if oracle else None,
            "mean_edge_f1": round(statistics.mean(r["edge_f1"] for r in evaluations if r["arm"] == arm), 5),
        }
    diffs = []
    active_cis, random_cis = [], []
    for _ in range(samples):
        chosen = [cases[rng.randrange(n)] for __ in cases]
        means = {}
        for arm in ("active", "random"):
            mean = sum(by_key[(seed, c, arm)]["exact"] for c in chosen for seed in seeds) / (n*len(seeds))
            means[arm] = mean
        active_cis.append(means["active"])
        random_cis.append(means["random"])
        diffs.append(means["active"]-means["random"])
    def interval(values):
        ordered = sorted(values)
        return [round(ordered[int(0.025 * samples)], 5),
                round(ordered[min(samples-1, int(0.975*samples))], 5)]
    output["active"]["graph_cluster_bootstrap_accuracy_ci95"] = interval(active_cis)
    output["random"]["graph_cluster_bootstrap_accuracy_ci95"] = interval(random_cis)
    output["paired_active_minus_random"] = {
        "accuracy_difference": round(output["active"]["accuracy"]-output["random"]["accuracy"], 5),
        "graph_cluster_bootstrap_ci95": interval(diffs),
        "sampling_unit": "heldout graph, retaining all seed replicates",
    }
    c = output["active"]["cpu_seconds"]
    output["cpu_accounting"] = {
        "same_oracle_calls_per_arm": len(seeds)*len(cases)*rounds,
        "same_full_hypothesis_counterfactual_simulations_per_round": len(ALL_GRAPHS)*N,
        "active_vs_random_cpu_ratio": round(c/max(output["random"]["cpu_seconds"],1e-9), 4),
        "active_vs_passive_cpu_ratio": round(c/max(output["passive"]["cpu_seconds"],1e-9), 4),
        "matched_measured_cpu_20pct": all(
            abs(c-output[arm]["cpu_seconds"])<=0.20*c for arm in ("random", "passive")),
        "note": "Planner calls and oracle-call counts are matched; observed CPU may diverge as version spaces shrink. If 20% tolerance fails, do not claim measured CPU parity.",
    }
    return output


def observational_equivalence():
    """Two distinct programs have identical passive distributions but different do outputs."""
    first = ALL_GRAPHS[0]
    for second in ALL_GRAPHS[1:]:
        for node in range(N):
            a = simulate(first, node, 1, 0)
            b = simulate(second, node, 1, 0)
            if a != b:
                return {
                    "graph_a": canonical_edges(first),
                    "graph_b": canonical_edges(second),
                    "passive_distribution": [
                        {"prob": 0.5, "outcome": [0]*N},
                        {"prob": 0.5, "outcome": [1]*N}],
                    "distinguishing_intervention": {"register": REGS[node], "value": 1,
                                                    "exogenous_bit": 0},
                    "interventional_outcome_a": a,
                    "interventional_outcome_b": b,
                    "all_64_hypotheses_passively_equivalent": True,
                }
    raise AssertionError("there must be observational-equivalent distinguishable cases")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=[11,19,29,31,43])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--out", type=Path, default=HERE)
    a = p.parse_args()
    if len(set(a.seeds)) != len(a.seeds) or len(a.seeds) < 3:
        raise ValueError("at least three distinct seeds required")
    if a.rounds not in range(MAX_EXPERIMENTS+1):
        raise ValueError("rounds outside bounded range")
    a.out.mkdir(parents=True, exist_ok=True)
    evaluations, traces = run_trials(a.seeds, a.rounds)
    summary = cluster_bootstrap(evaluations, a.seeds, a.rounds)
    report = {
        "schema":"logos.nc070.discovery/1",
        "algorithm": "finite-version-space greedy expected-survivors (not neural)",
        "graph_family": "all 64 labeled rooted 4-register trees, identity-copy structural equations",
        "true_graphs_visible_to_policy": False,
        "heldout_definition":"SHA-256 split of 64 program/graph keys, salt NC-070/v1, graphs unused for fit",
        "hypothesis_count": len(ALL_GRAPHS),
        "heldout_graph_count":len(HELDOUT_IDS),
        "heldout_ids": HELDOUT_IDS,
        "seeds":a.seeds,
        "rounds":a.rounds,
        "per_seed": {
            str(seed): {
                arm: {
                    "exact":sum(r["exact"] for r in evaluations if r["seed"]==seed and r["arm"]==arm),
                    "denominator":len(HELDOUT_IDS),
                    "unique":sum(r["unique"] for r in evaluations if r["seed"]==seed and r["arm"]==arm),
                } for arm in ARMS
            } for seed in a.seeds
        },
        "summary":summary,
        "observational_equivalence":observational_equivalence(),
        "evaluation_rows":evaluations,
        "limitations":[
            "All 64 graph hypotheses and copy-equation family are supplied, so inference is finite hypothesis elimination, not discovery of a new graph family",
            "Each graph receives repeated seeds; confidence intervals resample graphs, not individual rows",
            "Exogenous inputs are binary 0/1 within the canonical 4-bit ISA, and the policy can intervene on any register",
            "Passive observations are designed to be identical for all graphs; active superiority is narrow and by construction",
            "Candidate graphs may include heldout graph structures in the model's hypothesis set; heldout means true environment was not used for fitting or decisions",
            "Matched planning-work/oracle calls do not automatically mean measured CPU parity; measured ratios and an explicit gate are reported",
            "No graph-identification guarantee beyond this finite family; no AGI, new mathematical discovery or champion promotion",
        ],
    }
    (a.out/"discovery_results.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    with (a.out/"experiment_traces.jsonl").open("w") as f:
        for row in traces:
            f.write(json.dumps(row,sort_keys=True)+"\n")
    print(json.dumps({
        "heldout_graphs":len(HELDOUT_IDS),
        "trials_per_arm":len(a.seeds)*len(HELDOUT_IDS),
        "summary":summary,
        "traces":len(traces)
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
