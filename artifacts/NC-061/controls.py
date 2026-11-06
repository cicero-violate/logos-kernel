"""Control candidates that the NC-061 scoreboard must score correctly. Seeds here are public, unlike a real run."""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scoreboard as sb  # noqa: E402


def control_reveal(seed, goals):
    return {"generator": sb.GENERATOR, "seed": f"control-{seed}", "nonce": "control",
            "thresholds": {**sb.THRESHOLDS, "goals": goals}}


def matches(body, examples):
    program = sb.full_program(body)
    for e in examples:
        got = sb.isa.reference(program, {"X": e["X"], "Y": e["Y"], "R": 0, "T": 0}, max_steps=len(body) + 2)
        if got["registers"]["R"] != e["R"]:
            return False
    return True


def known_good(goals):
    """Holds the answers, so it should solve everything. It stands in for a strong candidate."""
    answers = {sb.canonical(g["examples"]): g["answer"] for g in goals}
    return lambda examples, budget: (answers[sb.canonical(examples)], 1)


def constant(examples, budget):
    return [{"op": "LOADI", "dst": "R", "imm": 0}], 1


def random_search(seed):
    rng = random.Random(f"control-random|{seed}")
    def run(examples, budget):
        for used in range(1, budget + 1):
            body = [sb._instruction(rng) for _ in range(rng.randint(1, 3))]
            if matches(body, examples):
                return body, used
        return None, budget
    return run


def memorizer(training):
    """Perfect on its training functions and unable to generalize to sealed goals."""
    tables = [(t, b) for t, b in training]
    def run(examples, budget):
        for used, (table, body) in enumerate(tables[:budget], 1):
            if all(table[sb.DOMAIN.index((e["X"], e["Y"]))] == e["R"] for e in examples):
                return body, used
        return None, min(budget, len(tables))
    return run


def budget_cheat(goals):
    good = known_good(goals)
    return lambda examples, budget: (good(examples, budget)[0], budget * 10)


def run(goals=10, seeds=(11, 19, 29), budget=64):
    per = {name: [] for name in ("constant", "known_good", "random_search", "leaked_answer", "regressed", "memorizer", "budget_cheat")}
    for seed in seeds:
        goal_set = sb.sealed_goals(control_reveal(seed, goals))
        rng = random.Random(f"control-train|{seed}")
        sealed = {sb.table_key(g["table"]) for g in goal_set}
        training = []
        while len(training) < 30:
            body = [sb._instruction(rng) for _ in range(rng.randint(1, 3))]
            table = sb.truth_table(body)
            if sb.table_key(table) not in sealed and len(set(table)) > 1:
                training.append((table, body))
        train_keys = [sb.table_key(t) for t, _ in training]
        good = known_good(goal_set)
        per["constant"].append(sb.score_candidate(constant, goal_set, budget, retention=1.0))
        per["known_good"].append(sb.score_candidate(good, goal_set, budget, retention=1.0))
        per["random_search"].append(sb.score_candidate(random_search(seed), goal_set, budget, retention=1.0))
        per["leaked_answer"].append(sb.score_candidate(good, goal_set, budget, training_table_keys=sealed, retention=1.0))
        per["regressed"].append(sb.score_candidate(good, goal_set, budget, retention=0.5))
        per["memorizer"].append(sb.score_candidate(memorizer(training), goal_set, budget, training_table_keys=train_keys, retention=1.0))
        per["budget_cheat"].append(sb.score_candidate(budget_cheat(goal_set), goal_set, budget, retention=1.0))
    baseline = per["constant"]
    verdicts = {name: sb.compare(baseline, rows) for name, rows in per.items()}
    return {"schema": "logos.nc061.controls/1", "goals_per_seed": goals, "seeds": list(seeds), "budget": budget,
            "scores": per, "verdicts_against_constant_baseline": verdicts}


if __name__ == "__main__":
    report = run()
    (HERE / "control_results.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: [v["verdict"], v.get("gain")] for k, v in report["verdicts_against_constant_baseline"].items()}, indent=0))
    print({k: [r["solved"] for r in v] for k, v in report["scores"].items()})
