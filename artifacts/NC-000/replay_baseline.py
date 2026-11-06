#!/usr/bin/env python3
"""Read-only NC-000 baseline replay. This is NOT independent-agent verification.

Dependencies:
  Python >= 3.10, Git, PyTorch
  /workspace/experiments/rlt_split_algorithms_ablation_v1 (run.py, task_conditioned_rlt,
    neural_isa_transfer_v1/verify_hybrid_independent.py, unified_shared_v1/
    verify_shared.py, all original configs and saved .pt checkpoints)
  /workspace/experiments/latticeforge_neural_isa_test (train_causal.py,
    transformer_controller/train_controller.py, v1/v2 weights)
  artifacts/NC-000/inventory.json (immutable SHA-256 inventory).
Writes ONLY a selected JSON output under artifacts/NC-000/. Never modifies model checkpoints.
"""
import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "artifacts/NC-000/inventory.json"
RESULT = ROOT / "artifacts/NC-000/replay_result.json"


def sha(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def inventory_check(record):
    manifest = json.loads(INVENTORY.read_text())
    roots = {name: Path(path) for name, path in manifest["roots"].items()}
    observed = []
    errors = []
    for entry in manifest["files"]:
        path = roots[entry["repo"]] / entry["path"]
        try:
            digest = sha(path)
            length = path.stat().st_size
            match = digest == entry["sha256"] and length == entry["bytes"]
            if entry.get("expected_sha256") is not None:
                match = match and digest == entry["expected_sha256"]
        except OSError as exc:
            match = False
            digest = "UNREADABLE:" + type(exc).__name__
        observed.append((entry["repo"], entry["path"]))
        if not match:
            errors.append({"path": str(path), "observed_sha256": digest})
    for name in ("rlt_champion", "standalone_isa"):
        actual = git(roots[name], "rev-parse", "HEAD")
        expected = manifest["repository_commits"][name]
        if actual != expected:
            errors.append({"repository": name, "unexpected_commit": actual, "expected": expected})
        state = git(roots[name], "status", "--porcelain")
        if state:
            errors.append({"repository": name, "dirty_worktree": state})
    if len(set(observed)) != len(observed):
        errors.append({"duplicate_paths": True})
    record["inventory"] = {
        "sha256": sha(INVENTORY),
        "files_checked": len(observed),
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    if errors:
        raise RuntimeError("inventory mismatch or source worktree modified")
    return roots


def champion_smoke(roots, record):
    r = roots["rlt_champion"]
    commands = [
        (["add", "7", "9", "--width", "8"], "addition", 16),
        (["--seed", "19", "core", "sub", "7", "9", "--width", "8"], "general_core:sub", 254),
    ]
    samples = []
    for args, expected_task, expected_value in commands:
        p = subprocess.run(
            [sys.executable, "run.py", *args], cwd=r, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        data = json.loads(p.stdout)
        passed = data["task"] == expected_task and data["value"] == expected_value
        samples.append({"command": "python run.py " + " ".join(args),
                        "expected": expected_value, "predicted": data["value"],
                        "model_route": data.get("algorithm"), "passed": passed})
    record["champion_smoke"] = {"status": "PASS" if all(x["passed"] for x in samples) else "FAIL",
                                "samples": samples}
    if record["champion_smoke"]["status"] != "PASS":
        raise RuntimeError("champion smoke mismatch")


def standalone(roots, record):
    import torch
    isa = roots["standalone_isa"]
    sys.path[:0] = [str(isa), str(isa / "transformer_controller")]
    from train_causal import NeuralISA, neural_multiply
    from train_controller import TransformerPC, neural_rollout
    arith = NeuralISA()
    arith.load_state_dict(torch.load(isa / "causal/weights.pt", map_location="cpu",
                                     weights_only=True)["model"])
    arith.eval()
    pairs = [(13, 11), (255, 255), (0, 123), (1, 255), (127, 128),
             (255, 128), (16, 16), (7, 9), (128, 128), (37, 63)]
    with torch.no_grad():
        predicted = neural_multiply(arith, [a for a, b in pairs], [b for a, b in pairs])
    mlp_hits = sum(int(value == a * b) for value, (a, b) in zip(predicted, pairs))
    tests = [(13, 11), (255, 255), (37, 63), (128, 128), (16, 16), (7, 9),
             (127, 128), (3, 2), (0, 0), (1, 0), (0, 255), (255, 1),
             (2, 3), (15, 15)]
    controllers = []
    for version, rel in (("v1", "transformer_controller/controller_weights.pt"),
                         ("v2", "transformer_controller/v2/controller_weights.pt")):
        model = TransformerPC()
        model.load_state_dict(torch.load(isa / rel, map_location="cpu",
                                         weights_only=True)["model"])
        model.eval()
        with torch.no_grad():
            report = neural_rollout(model, arith, tests, batchsize=32)
        controllers.append({"version": version, "checkpoint_sha256": sha(isa / rel),
                            "correct": report["correct"], "tested": report["tested"],
                            "failed_cases": report["failures"],
                            "mode": "Transformer controller + frozen MLP arithmetic"})
    status = "PASS" if mlp_hits == len(pairs) and [c["correct"] for c in controllers] == [6, 14] else "FAIL"
    record["standalone_isa"] = {
        "mode": "standalone causal NeuralISA MLP + separate v1/v2 Transformer controllers",
        "mlp": {"correct": mlp_hits, "tested": len(pairs),
                "checkpoint_sha256": sha(isa / "causal/weights.pt")},
        "controllers": controllers,
        "status": status,
        "historical_v1_failure_is_expected": True,
    }
    if status != "PASS":
        raise RuntimeError("standalone ISA regression")


def shared_seed_replay(roots, record, seeds, random8):
    import torch
    r = roots["rlt_champion"]
    shared_dir = r / "neural_isa_transfer_v1/unified_shared_v1"
    sys.path.insert(0, str(shared_dir))
    from verify_shared import load, local_oracle
    from verify_hybrid_independent import independent
    from train_controller import SEED
    ids, x, y = local_oracle()
    all4 = [(a, b) for a in range(16) for b in range(16)]
    rng = random.Random(SEED + 991122)
    random_pairs = [(rng.randrange(256), rng.randrange(256)) for _ in range(4096)]
    edges = (0, 1, 2, 3, 7, 15, 31, 63, 127, 128, 255)
    sample8 = random_pairs[:random8] + [(a, b) for a in edges for b in edges]
    values = []
    for seed in seeds:
        model, trained = load(seed, "jointly_tuned")
        with torch.no_grad():
            bit, syntax = model.local_logits(ids, x)
            local_hits = int(((bit >= 0).float() == y).all(-1).sum())
            syntax_hits = int((syntax.argmax(-1) == ids).sum())
            four = [independent(all4[k:k + 64], model, model)
                    for k in range(0, len(all4), 64)]
            eight = [independent(sample8[k:k + 128], model, model)
                     for k in range(0, len(sample8), 128)]
        n4 = sum(z["tested"] for z in four)
        c4 = sum(z["correct"] for z in four)
        n8 = sum(z["tested"] for z in eight)
        c8 = sum(z["correct"] for z in eight)
        values.append({
            "seed": seed, "arm": "jointly_tuned",
            "checkpoint_sha256": trained["checkpoint_sha256"],
            "local_truth": {"correct": local_hits, "tested": len(ids)},
            "syntax": {"correct": syntax_hits, "tested": len(ids)},
            "fourbit_exhaustive": {"correct": c4, "tested": n4},
            "eightbit_sampled": {"correct": c8, "tested": n8,
                                  "random_pairs": random8, "edge_pairs": len(edges) ** 2},
            "pass": local_hits == syntax_hits == len(ids) == 320 and
                    c4 == n4 == 256 and c8 == n8 == len(sample8),
        })
    record["shared_isa"] = {
        "status": "PASS" if all(row["pass"] for row in values) else "FAIL",
        "mode": "one shared RLT causal-attention Transformer, task heads and fixed ROM host",
        "source_program_learned": False,
        "seed_results": values,
        "same_seeded_random_pairs": True,
        "full_8bit_space_exhaustive": False,
        "no_older_pretraining_disjointness_proof": True,
    }
    if record["shared_isa"]["status"] != "PASS":
        raise RuntimeError("shared ISA regression")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="11,19,29",
                        help="Comma-delimited known seed IDs. Default: 11,19,29")
    parser.add_argument("--random8", type=int, default=128,
                        help="First N old deterministic random 8-bit pairs + 121 edge pairs (0-4096)")
    parser.add_argument("--output", default="artifacts/NC-000/replay_result.json",
                        help="JSON output under artifacts/NC-000/")
    args = parser.parse_args()
    output = Path(args.output)
    output = (ROOT / output).resolve() if not output.is_absolute() else output.resolve()
    if output.parent != (ROOT / "artifacts/NC-000").resolve():
        parser.error("output must be directly under artifacts/NC-000/")
    seeds = [int(z) for z in args.seeds.split(",")]
    if not seeds or any(seed not in (11, 19, 29) for seed in seeds) or len(seeds) != len(set(seeds)):
        parser.error("seeds must be a unique subset of 11,19,29")
    if not 0 <= args.random8 <= 4096:
        parser.error("random8 must be between 0 and 4096")
    import torch
    torch.set_num_threads(1)
    record = {
        "task": "NC-000", "status": "INCOMPLETE",
        "role": "IMPLEMENTER_REPLAY_NOT_INDEPENDENT_AGENT",
        "is_independent_agent_verification": False,
        "pass_event_authorized": False,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "cuda": torch.cuda.is_available(), "device": "cpu"},
        "arguments": {"seeds": seeds, "random8": args.random8},
        "limits": [
            "Source verifier imports original model and independent host replay; it is not a separate-agent proof.",
            "Fixed 8-op ROM and Python state control; no neural invention of multiplication.",
            "Only exhaustive over 4-bit pairs; 8-bit samples do not establish all 65,536 pairs.",
            "Original seven-operation RLT champion was not jointly trained with ISA specialist.",
            "Historical v1 standalone controller fails 8/14 cases; v2 passes.",
            "Independent agent must inspect code, rerun tests, and issue a separate baseline_receipt.json.",
        ],
    }
    try:
        roots = inventory_check(record)
        champion_smoke(roots, record)
        standalone(roots, record)
        shared_seed_replay(roots, record, seeds, args.random8)
        record["status"] = "IMPLEMENTER_REPLAY_PASS"
    except Exception as exc:
        record["status"] = "IMPLEMENTER_REPLAY_FAIL"
        record["error"] = str(exc)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": record["status"], "result_path": str(output),
                      "inventory": record.get("inventory", {}),
                      "champion": record.get("champion_smoke", {}).get("status"),
                      "standalone": record.get("standalone_isa", {}).get("status"),
                      "shared": record.get("shared_isa", {})}, indent=2))
    if record["status"] != "IMPLEMENTER_REPLAY_PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
