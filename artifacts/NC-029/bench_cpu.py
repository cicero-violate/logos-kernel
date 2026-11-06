#!/usr/bin/env python3
"""NC-029 read-only CPU benchmark; NEVER trains or changes a champion.

Dependencies: Python 3.10+, Linux resource module, PyTorch, Git,
  /workspace/experiments/rlt_split_algorithms_ablation_v1/
    config.json, rlt_core.py, task_conditioned_rlt.py,
    algorithms/carry_recurrence/module.py, frozen seed-11 checkpoints,
    snapshots/current_unified_v3.pt,
  artifacts/NC-000/inventory.json. No numpy, GPU, network or model API.
  Three local controls are EXACT PROGRAMMED carry circuits, NOT trained models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
KERNEL = HERE.parents[1]
RLT = Path("/workspace/experiments/rlt_split_algorithms_ablation_v1")
ARMS = ("dense_gemm_control", "sparse_local_control", "scalar_recurrence_control",
        "rlt_trained_seed11", "carry_gru_champion_seed11")
CONTROL = set(ARMS[:3])
WIDTH = 8
TRIALS = 3
REPEATS = 5
THREADS = 1
SEED = 29062026
EXPECTED_BUNDLE = "4c42d13fd46f57ac0db6eccbaef35e99c4cefcec393a3a3a4faf146645e8f683"
CHECKPOINTS = {
    "rlt_trained_seed11": "task_conditioned_rlt_v1/multitask_train_v1/outputs/seed11/seed11_conditioned_step640.pt",
    "carry_gru_champion_seed11": "outputs/new16/carry_recurrence_seed11.pt",
}
SOURCES = ("config.json", "rlt_core.py", "task_conditioned_rlt.py",
           "algorithms/carry_recurrence/module.py", "snapshots/current_unified_v3.pt")
OUT = HERE / "cpu_baseline.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def pairs():
    edges = (0, 1, 7, 15, 31, 63, 127, 255)
    p = [(a, b) for a in edges for b in edges]
    rng = random.Random(SEED)
    p.extend((rng.randrange(256), rng.randrange(256)) for _ in range(64))
    assert len(p) == 128
    return p


def task_hash(p):
    return hashlib.sha256(json.dumps(p, separators=(",", ":")).encode()).hexdigest()


def oracle(p):
    return [(a + b) % (1 << WIDTH) for a, b in p]


def provenance():
    config = json.loads((RLT / "config.json").read_text())
    assert config["task_routes"]["addition"] == "carry_recurrence"
    assert config["seed"] == 11
    assert config["general_core"]["trained_multitask"] is True
    assert config["runtime"]["symbolic_fallback"] is False
    inventory = json.loads((KERNEL / "artifacts/NC-000/inventory.json").read_text())
    assert inventory["champion"]["selected_bundle_sha256"] == EXPECTED_BUNDLE
    assert digest(RLT / "snapshots/current_unified_v3.pt") == EXPECTED_BUNDLE
    hashes = {p: {"sha256": digest(RLT / p), "bytes": (RLT / p).stat().st_size}
              for p in SOURCES}
    cp = {}
    for arm, rel in CHECKPOINTS.items():
        f = RLT / rel
        cp[arm] = {"path": str(f), "sha256": digest(f), "bytes": f.stat().st_size}
    assert cp["rlt_trained_seed11"]["sha256"] == config["general_core"]["trained_checkpoint_sha256_by_seed"]["11"]
    assert cp["carry_gru_champion_seed11"]["path"].endswith(
        config["addition_modules"]["carry_recurrence"]["weights"].format(seed=11))
    rev = subprocess.check_output(["git", "-C", str(RLT), "rev-parse", "HEAD"], text=True).strip()
    assert rev == inventory["repository_commits"]["rlt_champion"], "RLT_SOURCE_REVISION_CHANGED"
    return {"original_rlt_revision": rev, "sources": hashes, "checkpoints": cp,
            "inventory_sha256": digest(KERNEL / "artifacts/NC-000/inventory.json")}


def controls(arm, a, b):
    import torch
    width = WIDTH
    result = torch.zeros_like(a)
    if arm == "dense_gemm_control":
        # 16-state homogeneous dense transition; 8 aliases per carry value.
        mats = torch.zeros((4, 16, 16), dtype=torch.float32)
        for pair in range(4):
            x, y = pair // 2, pair % 2
            for source in range(16):
                carry = int(source >= 8)
                dest = 8 if x + y + carry >= 2 else 0
                mats[pair, source, dest] = 1
        state = torch.zeros((len(a), 16), dtype=torch.float32)
        state[:, 0] = 1
    elif arm == "sparse_local_control":
        c0, c1 = torch.ones_like(a), torch.zeros_like(a)
    elif arm == "scalar_recurrence_control":
        carry = torch.zeros_like(a)
    else:
        raise ValueError(arm)
    for i in range(width):
        x, y = (a >> i) & 1, (b >> i) & 1
        if arm == "dense_gemm_control":
            carry = (state[:, 8:].sum(-1) > .5).long()
            z = x ^ y ^ carry
            state = torch.bmm(state.unsqueeze(1), mats[2 * x + y]).squeeze(1)
        elif arm == "sparse_local_control":
            carry = c1
            z = x ^ y ^ carry
            both_zero, both_one, mixed = (1 - x) * (1 - y), x * y, x ^ y
            c0, c1 = (both_zero * (c0 + c1) + mixed * c0,
                      both_one * (c0 + c1) + mixed * c1)
        else:
            z = x ^ y ^ carry
            carry = (x & y) | ((x ^ y) & carry)
        result = result | (z << i)
    return result


def model_for(arm):
    import torch
    sys.path.insert(0, str(RLT))
    if arm == "rlt_trained_seed11":
        from task_conditioned_rlt import TaskConditionedRLT
        return TaskConditionedRLT.from_combined_checkpoint(RLT / CHECKPOINTS[arm]).eval()
    if arm == "carry_gru_champion_seed11":
        from algorithms.carry_recurrence.module import LearnedCarryGRU
        model = LearnedCarryGRU()
        model.load_state_dict(torch.load(RLT / CHECKPOINTS[arm],
                                         map_location="cpu", weights_only=True), strict=True)
        return model.eval()
    return None


def run_forward(arm, model, a, b):
    import torch
    if arm in CONTROL:
        return controls(arm, a, b)
    from rlt_core import decode_logits
    with torch.no_grad():
        logits = model(a, b, "add", width=WIDTH) if arm.startswith("rlt_") else model(a, b, width=WIDTH)
        return decode_logits(logits).long()


def worker(arm, trial):
    # Separate processes remove per-model cross-arm memory accounting.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    data = pairs()
    a = torch.tensor([x for x, _ in data], dtype=torch.long)
    b = torch.tensor([y for _, y in data], dtype=torch.long)
    truth = oracle(data)   # computed outside model, AFTER predictions in timed loops
    t0, c0 = time.perf_counter(), time.process_time()
    model = model_for(arm)
    load_wall = time.perf_counter() - t0
    loaded_cpu = time.process_time() - c0
    with torch.no_grad():
        t = time.perf_counter()
        first = run_forward(arm, model, a, b)
        cold_forward = time.perf_counter() - t
        for _ in range(2):
            run_forward(arm, model, a, b)
        ts, cs = time.perf_counter(), time.process_time()
        predicted = None
        verified = 0
        for _ in range(REPEATS):
            predicted = [int(x) for x in run_forward(arm, model, a, b).tolist()]
            verified += sum(int(x == y) for x, y in zip(predicted, truth))
        cpu = time.process_time() - cs
        wall = time.perf_counter() - ts
    assert [int(x) for x in first.tolist()] == predicted, "NONDETERMINISTIC_FORWARD"
    mistakes = [{"a": a, "b": b, "predicted": pred, "truth": target}
                for (a, b), pred, target in zip(data, predicted, truth) if pred != target][:12]
    expected = len(data) * REPEATS
    assert cpu > 0
    if arm in CONTROL:
        assert verified == expected, "EXACT_CONTROL_FAILED"
    return {
        "arm": arm, "trial": trial, "class": "fixed_symbolic_carry_circuit" if arm in CONTROL else "frozen_trained_neural_model",
        "task_sha256": task_hash(data), "width_bits": WIDTH,
        "cases": len(data), "timed_repetitions": REPEATS,
        "verified": verified, "evaluated": expected,
        "accuracy": verified / expected,
        "counterexamples_first_12": mistakes,
        "cold_first_forward_wall_s": cold_forward,
        "checkpoint_load_wall_s": load_wall,
        "checkpoint_load_cpu_s": loaded_cpu,
        "warm_timed_wall_s": wall, "warm_timed_cpu_s": cpu,
        "verified_answers_per_cpu_s": verified / cpu,
        "inferences_per_cpu_s": expected / cpu,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "checkpoint_disk_bytes": (RLT / CHECKPOINTS[arm]).stat().st_size if arm in CHECKPOINTS else 0,
        "trainable_parameters": sum(p.numel() for p in model.parameters()) if model is not None else 0,
        "trained_during_benchmark": False,
        "energy_joules": None,
    }


def protocol():
    return {
        "task": "NC-029", "version": 1, "status": "PREREGISTERED_WITH_BENCHMARK_SOURCE",
        "scientific_scope": "8-bit unsigned modular addition, 128 fixed operand pairs",
        "controls_are_not_neural_models": True,
        "dense_gemm": "16-state dense carry-transition matrices, exactly specified from binary addition rules",
        "sparse_local": "two carry lanes with input-local sparse state transition, exact symbolic rules",
        "scalar": "single binary carry using boolean recurrence, exact symbolic rules",
        "neural": "frozen trained 7-task RLT/addition and frozen active carry GRU seed11",
        "avoid_comparing_symbolic_exactness_to_neural_learning": True,
        "width_bits": WIDTH, "dataset_seed": SEED, "task_sha256": task_hash(pairs()),
        "task_cases": len(pairs()), "task_generation": "64 fixed ordered edge pairs + 64 seeded random ordered pairs; repeats are timings, not independent novel examples",
        "trials": TRIALS, "repetitions_per_trial": REPEATS,
        "measurement": {
            "load_wall_s": "within fresh worker after importing torch until model ready",
            "cold_first_forward_wall_s": "first model call before warmups",
            "warm_timed": "2 warmup calls then 5 full-batch forward + postprediction exact checks",
            "cpu_s": "time.process_time() includes execution and reference verification, excludes startup/load",
            "peak_rss_kib": "Linux resource.getrusage(RUSAGE_SELF).ru_maxrss; includes Python/PyTorch runtime",
            "accuracy": "correct predictions / 128 same tasks; timed repeats share labels, not independent samples",
            "energy_joules": "null; no calibrated RAPL/external sensor was used",
            "cold_process": "fresh Python worker per arm/trial; outer worker_launch_wall_s includes startup plus full run",
        },
        "environment": "CPU-only, threads=1, CUDA_VISIBLE_DEVICES blank, PYTHONDONTWRITEBYTECODE=1",
        "source_inputs": ["artifacts/NC-000/inventory.json", *SOURCES, *CHECKPOINTS.values()],
        "limitations": [
            "Exact symbolic controls contain the binary addition law; they are not learned, trained, or parameter-matched with RLT.",
            "Dense 16-state matrix control is not a benchmark of a full dense Transformer; compare kernels only, no generalized architectural efficiency claim.",
            "Only 128 8-bit tasks; not 16/32-bit transfer, not a statistical power/performance claim.",
            "Repeated timing trials on one host can be affected by system load/cache/turbo and are not independent learning seeds.",
            "CUDA disabled but no OS-enforced CPU isolation or calibrated power meter.",
            "No new neural model trained; training energy, training time, and training data cost not measured.",
        ],
    }


def run_all(out):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    src = provenance()
    records = []
    for trial in range(TRIALS):
        order = ARMS[trial:] + ARMS[:trial]
        for arm in order:
            t = time.perf_counter()
            p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", arm,
                                "--trial", str(trial)], cwd=KERNEL, env=env,
                               capture_output=True, text=True, timeout=120)
            if p.returncode:
                raise RuntimeError(f"WORKER_FAIL {arm} trial{trial}: {p.stderr[-2500:]}")
            row = json.loads(p.stdout)
            row["worker_launch_wall_s"] = time.perf_counter() - t
            assert row["task_sha256"] == task_hash(pairs())
            records.append(row)
    end = provenance()
    assert src == end, "FROZEN_SOURCE_OR_CHECKPOINT_CHANGED"
    groups = {}
    for arm in ARMS:
        rows = [r for r in records if r["arm"] == arm]
        assert len(rows) == TRIALS
        groups[arm] = {
            "class": rows[0]["class"], "trials": TRIALS,
            "accuracy_numerator_each": [r["verified"] for r in rows],
            "accuracy_denominator_each": [r["evaluated"] for r in rows],
            "accuracy": rows[0]["accuracy"],
            "cpu_seconds_median": statistics.median(r["warm_timed_cpu_s"] for r in rows),
            "wall_seconds_median": statistics.median(r["warm_timed_wall_s"] for r in rows),
            "cold_forward_seconds_median": statistics.median(r["cold_first_forward_wall_s"] for r in rows),
            "launch_wall_seconds_median": statistics.median(r["worker_launch_wall_s"] for r in rows),
            "peak_rss_kib_max": max(r["peak_rss_kib"] for r in rows),
            "model_disk_bytes": rows[0]["checkpoint_disk_bytes"],
            "model_parameter_count": rows[0]["trainable_parameters"],
            "verified_answers_per_cpu_s_total": sum(r["verified"] for r in rows) /
                                                 sum(r["warm_timed_cpu_s"] for r in rows),
            "counterexamples_first_12": rows[0]["counterexamples_first_12"],
        }
    pareto = {}
    for kind in ("fixed_symbolic_carry_circuit", "frozen_trained_neural_model"):
        sub = {k: v for k, v in groups.items() if v["class"] == kind}
        pareto[kind] = [k for k,v in sub.items() if not any(
            j != k and z["accuracy"] >= v["accuracy"] and
            z["verified_answers_per_cpu_s_total"] >= v["verified_answers_per_cpu_s_total"] and
            z["peak_rss_kib_max"] <= v["peak_rss_kib_max"] and
            (z["accuracy"] > v["accuracy"] or
             z["verified_answers_per_cpu_s_total"] > v["verified_answers_per_cpu_s_total"] or
             z["peak_rss_kib_max"] < v["peak_rss_kib_max"])
            for j,z in sub.items())]
    result = {
        "task": "NC-029", "status": "IMPLEMENTER_MEASURED_NOT_INDEPENDENTLY_VERIFIED",
        "task_sha256": task_hash(pairs()), "source_provenance_before_and_after_identical": True,
        "provenance": src, "system": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "torch": subprocess.check_output([sys.executable, "-c", "import torch;print(torch.__version__)"],
                                             env=env, text=True).strip(),
            "cpu_count_visible": os.cpu_count(), "torch_threads": THREADS,
            "seed": SEED, "energy_instrumentation": "NOT_AVAILABLE_OR_NOT_CALIBRATED",
        },
        "protocol_sha256": digest(HERE / "measurement_protocol.json"),
        "trials": records, "summary": groups, "within_class_pareto_frontier": pareto,
        "training_cpu_seconds": None, "training_energy_joules": None,
        "no_training_performed": True, "energy_joules": None,
        "limitations": protocol()["limitations"],
        "independent_verification": "PENDING",
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "path": str(out),
                      "summary": groups, "pareto": pareto}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=ARMS)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--write-protocol", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.trial), sort_keys=True))
    elif args.write_protocol:
        (HERE / "measurement_protocol.json").write_text(
            json.dumps(protocol(), indent=2, sort_keys=True) + "\n")
        print("WROTE measurement_protocol.json")
    elif args.run:
        out = Path(args.output).resolve()
        if out.parent != HERE.resolve():
            parser.error("--output must reside directly under artifacts/NC-029")
        if not (HERE / "measurement_protocol.json").is_file():
            parser.error("preregister measurement_protocol.json before run")
        assert json.loads((HERE / "measurement_protocol.json").read_text()) == protocol()
        run_all(out)
    else:
        parser.error("choose --write-protocol, --run, or --worker")


if __name__ == "__main__":
    main()
