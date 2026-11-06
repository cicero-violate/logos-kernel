"""Append 500 user-requested experiments through dag.py, never rewriting history."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "plans/agi_progression_160/roadmap.jsonl"


def run(*args):
    return subprocess.run(["rtk", "proxy", "python", "dag.py", *args],
                          cwd=ROOT, text=True, capture_output=True)


def events():
    return [json.loads(line)["event"] for line in
            (ROOT / "instructions.jsonl").read_text().splitlines()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build():
    original = [json.loads(line) for line in BASE.read_text().splitlines()]
    assert len(original) == 160
    tracks = [line.split("|") for line in (HERE / "tracks.txt").read_text().splitlines() if line]
    assert len(tracks) == 50, len(tracks)
    current = {e["id"]: e for e in events() if e["kind"] == "task"}
    tasks = []
    rows = list(original)
    for track_index, (track, anchor, baseline, metric, experiments) in enumerate(tracks, 1):
        objectives = experiments.split(";")
        assert len(objectives) == 10, (track, len(objectives))
        assert anchor in current, anchor
        for step, objective in enumerate(objectives, 1):
            number = 91 + len(tasks) + 1
            task_id = f"AGI-{number:03}"
            deps = [anchor] if step == 1 else [f"AGI-{number-1:03}"]
            # The final integrated audit also depends on every preceding track's
            # independent replication; other tracks can progress independently.
            if track_index == 50 and step == 1:
                deps += [f"AGI-{91+i*10:03}" for i in range(1, 50)]
            stage = 160 + len(tasks) + 1
            experiment = f"{objective}. Use bounded, evaluator-generated local tasks in the {track.lower()} track."
            task = {
                "kind": "task", "id": task_id,
                "title": f"{track}: {objective[0].lower() + objective[1:]}",
                "status": "planned", "priority": "P3", "execution": "agent_requested",
                "depends_on": deps,
                "instruction": [experiment,
                    f"Compare against {baseline}. Measure {metric}; define units, denominators, practical effect thresholds and failure strata before executing.",
                    "Preregister a concrete generator, disjoint development and hidden evaluation families, seeds, CPU/wall/memory limits, and stopping rules in protocol.json. Pin code and inputs before revealing evaluation outcomes.",
                    "Reuse predecessor evidence and its limitations. A predecessor's NO_GAIN or administrative closure is not a qualified capability: record unmet capability prerequisites and keep dependent gain claims blocked.",
                    "Run only within the assigned ticket and explicit resource/authority limits. Preserve other workers' claims and frozen candidates. Record unsuccessful trials and total development, training, search and verification costs."],
                "acceptance": [experiment,
                    f"Report {metric} for the candidate and {baseline}, including worst-family outcomes, raw trial denominators, costs, and independently replayable failure examples.",
                    "Any empirical gain claim needs at least three independent seeds or independently sampled task sets, uncertainty estimates, equal allocated comparison budgets, and preregistered practical improvement thresholds; report NO_GAIN or NOT_ESTABLISHED when unsupported.",
                    "Include a deliberately broken candidate or invalid-input control that the evaluation rejects, plus an appropriate positive reference or oracle upper bound with its privileged information disclosed.",
                    "Retest relevant qualified predecessor capabilities against their preregistered retention floors; no average score may conceal a required floor violation.",
                    "A genuinely distinct reviewer independently checks pinned artifacts and fresh hidden cases. Store the reviewer receipt separately from the frozen candidate manifest; no self-approval or automatic champion promotion."],
                "outputs": [f"artifacts/{task_id}/{name}" for name in
                    ("protocol.json", "results.json", "trials.jsonl", "CANDIDATE_MANIFEST.json", "independent_verification.json")],
                "avoid": ["No claim that a roadmap entry is an implemented capability or a guaranteed route to AGI.",
                    "No hidden evaluation access, invented results, unbounded resource use, unauthorized external actions, permission expansion, or modification of evaluator authority.",
                    "Do not rewrite ledger history, alter another agent's active work, or treat a passed experiment with negative results as a positive capability certificate."],
                "roadmap": {"id": "agi-progression-660-v1", "extension_index": len(tasks)+1,
                    "progression_stage": stage, "track": track, "track_index": track_index,
                    "track_step": step, "basis": "User-requested proposed experiments; research hypotheses, not established capability claims."}
            }
            if task_id in current and current[task_id] != task:
                raise RuntimeError("Existing task ID collision: " + task_id)
            tasks.append(task)
            rows.append({"stage": stage, "task_id": task_id, "title": task["title"],
                         "depends_on": deps, "reused_existing_task": False,
                         "track": track, "acceptance": task["acceptance"], "outputs": task["outputs"]})
    assert len(tasks) == 500 and len(rows) == 660
    seen = set()
    for row in rows:
        assert row["task_id"] not in seen
        assert set(row["depends_on"]) <= seen, (row["task_id"], row["depends_on"])
        seen.add(row["task_id"])
    assert len({task["title"] for task in tasks}) == 500
    for name, data in (("new_tasks.jsonl", tasks), ("roadmap.jsonl", rows)):
        (HERE / name).write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in data))
    text = ["# Proposed 660-stage AGI research progression", "",
        "This extension adds 500 concrete experiments in 50 tracks to the existing 160-stage roadmap. All 500 are planned tasks, not claims of implemented capabilities or a proven optimal route to AGI.", "",
        "Canonical task definitions: [instructions.jsonl](../../instructions.jsonl). Full machine-readable progression: [roadmap.jsonl](roadmap.jsonl). Extension tasks: [new_tasks.jsonl](new_tasks.jsonl). Original roadmap: [160 stages](../agi_progression_160/README.md).", "",
        "The original 160 stages remain unchanged. Numbering is a valid dependency order, not a claim that every earlier branch must finish before a later branch starts. Each new track anchors to an existing capability and progresses through ten experiments; the final integrated audit depends on all 49 preceding track replications. Existing runtime priorities and claims are preserved.", "",
        "Every task requires a concrete preregistered protocol, named comparison baselines, task-specific metrics, negative controls, bounded resources, retention checks and a separate independent receipt. Failed hypotheses remain visible; NO_GAIN does not unlock unsupported capability claims.", "",
        "## New tracks", "", "| Track | Task IDs | Prerequisite | Measurement |", "|---|---|---|---|"]
    for i, (name, anchor, baseline, metric, _) in enumerate(tracks):
        text.append(f"| {name} | AGI-{92+10*i:03}–AGI-{101+10*i:03} | {anchor} | {metric} |")
    text += ["", "## All 660 stages", "", "| Stage | Task | Milestone | Prerequisites |", "|---:|---|---|---|"]
    text += [f"| {r['stage']} | {r['task_id']} | {r['title']} | {', '.join(r['depends_on']) or '—'} |" for r in rows]
    (HERE / "README.md").write_text("\n".join(text)+"\n")
    return tasks, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    tasks, rows = build()
    print(json.dumps({"validated_extension": len(tasks), "total_stages": len(rows),
                      "first": tasks[0]["id"], "last": tasks[-1]["id"]}), flush=True)
    if not args.apply:
        return
    before = (ROOT / "instructions.jsonl").read_bytes()
    existing = {e["id"]: e for e in events() if e["kind"] == "task"}
    added = 0
    for i, task in enumerate(tasks, 1):
        if task["id"] in existing:
            assert existing[task["id"]] == task
            continue
        temporary = HERE / "next_task.json"
        write_json(temporary, task)
        result = run("add-task", "--file", str(temporary))
        if result.returncode:
            # An interrupted commit may have persisted the event. Never replay it
            # automatically; the canonical writer's explicit sync is required.
            raise RuntimeError(result.stdout + result.stderr)
        added += 1
        if i % 25 == 0:
            print(f"Appended {i}/500 extension tasks", flush=True)
    new_ids = {task["id"] for task in tasks}
    for _ in range(5):
        result = run("list")
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        opened = [t for t in json.loads(result.stdout) if t["status"] in ("planned", "working")]
        old = [t for t in opened if t["id"] not in new_ids]
        new = sorted((t for t in opened if t["id"] in new_ids), key=lambda t: t["id"])
        priority = {"kind": "priorities", "reason": "User requested 500 MORE AGI roadmap experiments. Preserve existing relative order, effective priority and ownership; append AGI-092 through AGI-591. Proposed research, not capability certification. Full progression: plans/agi_progression_660/roadmap.jsonl.",
                    "assignments": [{"id": t["id"], "priority": t["priority"], "rank": rank}
                                    for rank, t in enumerate(old+new, 1)]}
        path = HERE / "priority_snapshot.json"
        write_json(path, priority)
        result = run("priority-plan", "--file", str(path))
        if result.returncode == 0:
            break
        if "priority plan must cover every open task exactly once" not in result.stdout+result.stderr:
            raise RuntimeError(result.stdout+result.stderr)
    else:
        raise RuntimeError("Concurrent state changes prevented priority snapshot; tasks remain appended safely")
    after = (ROOT / "instructions.jsonl").read_bytes()
    assert after.startswith(before), "Ledger prefix changed"
    current = {e["id"]: e for e in events() if e["kind"] == "task"}
    assert all(current[t["id"]] == t for t in tasks)
    assert all(current[task_id] == task for task_id, task in existing.items())
    result = run("validate")
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    receipt = {"extension_tasks": 500, "appended_this_run": added, "total_roadmap_stages": 660,
               "unique_titles": len({t["title"] for t in tasks}), "dependency_order": "PASS",
               "original_ledger_prefix_preserved": True, "existing_task_definitions_preserved": True,
               "roadmap_sha256": hashlib.sha256((HERE / "roadmap.jsonl").read_bytes()).hexdigest(),
               "ledger_validation": json.loads(result.stdout),
               "scope": "Structural roadmap checks only; no experiment performed and no capability certified."}
    write_json(HERE / "validation.json", receipt)
    (HERE / "next_task.json").unlink(missing_ok=True)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
