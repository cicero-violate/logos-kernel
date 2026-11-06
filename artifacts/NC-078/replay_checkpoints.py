"""NC-078 deterministic, crash-resumable replay from pinned weights.

Replays the entire neural/shuffled/untrained candidate stream in the exact original
order; every accepted program passes the canonical exhaustive verifier.
No training, model promotion, network access, or alteration of frozen artifacts.
A partial replay is NOT evidence for a full held-out count match.

Usage:
  OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-078/replay_checkpoints.py --seed 11 --out /tmp/nc078-review-11 --max-goals 4
  OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-078/replay_checkpoints.py --seed 11 --out /tmp/nc078-review-11
The second invocation resumes at the fifth goal. An independent verifier must run
all seeds 11,19,29 to completion and check report.status=COMPLETE, with no mismatch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synth
import train_synth as ts

ARMS = ("neural", "shuffled_examples", "untrained")
ZERO = "0" * 64


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    with tmp.open("w") as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def chain(records):
    head = ZERO
    for r in records:
        payload = json.dumps(r, sort_keys=True, separators=(",", ":"))
        head = hashlib.sha256((head + payload).encode()).hexdigest()
    return head


def expected_tallies(result, arm):
    return result["arms"][arm]


def run(seed, out, max_goals=None):
    torch.set_num_threads(2)
    manifest = json.loads((HERE / "CANDIDATE_MANIFEST.json").read_text())
    report = json.loads((HERE / "results.json").read_text())
    candidates = {r["info"]["seed"]: r for r in report["seeds"]}
    if seed not in candidates:
        raise ValueError("seed not in frozen results")
    for filename, want in manifest["source_sha256"].items():
        if filename != "replay_checkpoints.py" and digest(HERE / filename) != want:
            raise ValueError("source hash mismatch: " + filename)
    for filename, want in manifest["checkpoint_sha256"].items():
        if digest(HERE / filename) != want:
            raise ValueError("checkpoint hash mismatch: " + filename)
    if digest(HERE / "results.json") != manifest["evidence_sha256"]["results.json"]:
        raise ValueError("result hash mismatch")
    if digest(HERE.parent / "NC-001/canonical_isa.py") != manifest["canonical_isa_sha256"]:
        raise ValueError("canonical verifier hash mismatch")
    # Recreate the exact torch RNG state after the original train(seed) and
    # untrained Model() constructor. Gradient updates use no torch RNG:
    # optimizer batches/examples use NumPy RNG and model dropout is 0.
    torch.manual_seed(seed)
    model = ts.Model()
    state = torch.load(HERE / f"model_seed{seed}.pt", map_location="cpu", weights_only=True)
    if state["seed"] != seed:
        raise ValueError("checkpoint seed mismatch")
    model.load_state_dict(state["state"])
    model.eval()
    untrained = ts.Model()
    untrained.eval()
    goals = {**synth.named_tables(), **synth.unseen_goals(report["config"]["unseen"])}
    names = list(goals)
    goal_hash = hashlib.sha256(b"".join(
        n.encode() + b"\0" + goals[n].tobytes() for n in names
    )).hexdigest()
    assignments = [(arm, n) for arm in ARMS for n in names]
    if max_goals is not None and not 0 <= max_goals <= len(assignments):
        raise ValueError("bad --max-goals")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "replay.json"
    identity = {
        "schema": "logos.nc078.checkpoint-replay/1",
        "seed": seed,
        "model_sha256": digest(HERE / f"model_seed{seed}.pt"),
        "results_sha256": digest(HERE / "results.json"),
        "goals_sha256": goal_hash,
        "source_sha256": {name: digest(HERE / name) for name in
            ["synth.py", "train_synth.py", "replay_checkpoints.py"]},
        "canonical_isa_sha256": digest(HERE.parent / "NC-001/canonical_isa.py"),
        "torch_version": str(torch.__version__),
    }
    if path.exists():
        data = json.loads(path.read_text())
        for k, v in identity.items():
            if data.get(k) != v:
                raise ValueError("resume identity mismatch: " + k)
        records = data["records"]
        if data.get("chain_sha256") != chain(records):
            raise ValueError("replay chain corrupt")
        if len(records) > len(assignments):
            raise ValueError("resume length overflow")
        for i, r in enumerate(records):
            if r["arm"] != assignments[i][0] or r["goal"] != assignments[i][1]:
                raise ValueError("resume order mismatch")
        torch.set_rng_state(torch.tensor(list(bytes.fromhex(data["rng_state_hex"])), dtype=torch.uint8))
    else:
        records = []
        data = {**identity, "records": [], "rng_state_hex": "", "chain_sha256": ZERO}
    limit = len(assignments) if max_goals is None else max_goals
    for index in range(len(records), limit):
        arm, name = assignments[index]
        chosen = untrained if arm == "untrained" else model
        found = ts.neural_search(chosen, goals[name], seed, name, shuffled=(arm == "shuffled_examples"))
        if found["solved"]:
            assert found["program"] is not None
            assert synth.verify_with_reference(found["program"], goals[name])
        records.append({
            "arm": arm, "goal": name, "solved": bool(found["solved"]),
            "candidates": found["candidates"], "revisions": found["revisions"],
            "program": found["program"],
        })
        data.update(records=records, chain_sha256=chain(records),
                    rng_state_hex=bytes(torch.get_rng_state().tolist()).hex(),
                    completed_goals=len(records), total_goals=len(assignments),
                    status="PARTIAL", complete_arms={})
        for a in ARMS:
            group = [r for r in records if r["arm"] == a]
            if len(group) != len(names):
                continue
            named_names = list(synth.named_tables())
            def tally(kinds):
                rows = {r["goal"]: r for r in group if r["goal"] in kinds}
                return ts.tally(rows)
            actual = {
                "named": tally(named_names),
                "unseen": tally(names[len(named_names):]),
                "named_solved": sorted(r["goal"] for r in group
                                       if r["goal"] in named_names and r["solved"]),
                "revisions": sum(r["revisions"] for r in group),
            }
            expected = expected_tallies(candidates[seed], a)
            exact = all(actual[k] == expected[k] for k in actual)
            data["complete_arms"][a] = {"matches_pinned_counts": exact, "actual": actual}
            if not exact:
                atomic_json(path, data)
                raise AssertionError("checkpoint count mismatch: " + a)
        if len(records) == len(assignments):
            data["status"] = "COMPLETE"
        atomic_json(path, data)
        if (index + 1) % 4 == 0 or data["status"] == "COMPLETE":
            print(f"seed={seed} goals={index + 1}/{len(assignments)} arm={arm} "
                  f"status={data['status']} matched={list(data['complete_arms'])}", flush=True)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(11, 19, 29))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-goals", type=int)
    args = parser.parse_args()
    result = run(args.seed, args.out, args.max_goals)
    print(json.dumps({"status": result["status"], "seed": args.seed,
                      "completed_goals": result["completed_goals"],
                      "total_goals": result["total_goals"],
                      "chain_sha256": result["chain_sha256"],
                      "complete_arms": list(result["complete_arms"])}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
