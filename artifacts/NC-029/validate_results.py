#!/usr/bin/env python3
"""NC-029 data-integrity validation, not independent scientific verification.

Dependencies: Python 3 standard library, bench_cpu.py, measurement_protocol.json,
cpu_baseline.json, host_environment.json, frozen original RLT files.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import statistics

import bench_cpu as b

HERE = Path(__file__).resolve().parent


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def validate(data):
    require(data["task"] == "NC-029", "WRONG_TASK")
    require(data["status"] == "IMPLEMENTER_MEASURED_NOT_INDEPENDENTLY_VERIFIED", "FALSE_VERDICT")
    require(data["task_sha256"] == b.task_hash(b.pairs()), "TASK_HASH_MISMATCH")
    require(data["protocol_sha256"] == b.digest(HERE / "measurement_protocol.json"), "PROTOCOL_HASH_MISMATCH")
    require(data["provenance"] == b.provenance(), "SOURCE_DRIFT")
    require(data["no_training_performed"] is True, "TRAINING_CLAIM")
    require(data["energy_joules"] is None, "ENERGY_INFERENCE")
    rows = data["trials"]
    require(len(rows) == len(b.ARMS) * b.TRIALS, "TRIAL_COUNT")
    require({(r["arm"], r["trial"]) for r in rows} ==
            {(arm, t) for arm in b.ARMS for t in range(b.TRIALS)}, "MISSING_OR_DUPLICATE_TRIAL")
    for arm in b.ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        corrects = [r["verified"] for r in sub]
        totals = [r["evaluated"] for r in sub]
        s = data["summary"][arm]
        require(all(r["task_sha256"] == data["task_sha256"] for r in sub), "MIXED_TASKS")
        require(all(r["width_bits"] == b.WIDTH and r["cases"] == 128 and
                    r["timed_repetitions"] == b.REPEATS for r in sub), "MIXED_DENOMINATORS")
        require(all(r["energy_joules"] is None for r in sub), "UNMEASURED_ENERGY_CLAIM")
        require(all(0 <= r["verified"] <= r["evaluated"] for r in sub), "INVALID_CORRECTNESS")
        require(all(math.isfinite(r["warm_timed_cpu_s"]) and r["warm_timed_cpu_s"] > 0 for r in sub),
                "INVALID_CPU_SECONDS")
        require(s["accuracy_numerator_each"] == corrects and
                s["accuracy_denominator_each"] == totals, "WRONG_DENOMINATORS")
        require(all(v == s["accuracy"] * n for v, n in zip(corrects, totals)), "BAD_ACCURACY")
        require(math.isclose(s["verified_answers_per_cpu_s_total"],
                             sum(corrects) / sum(r["warm_timed_cpu_s"] for r in sub),
                             rel_tol=1e-10), "BAD_CPU_THROUGHPUT")
        require(math.isclose(s["cpu_seconds_median"],
                             statistics.median(r["warm_timed_cpu_s"] for r in sub),
                             rel_tol=1e-10), "BAD_CPU_MEDIAN")
        require(s["model_disk_bytes"] == sub[0]["checkpoint_disk_bytes"], "WRONG_MODEL_BYTES")
        require(s["peak_rss_kib_max"] == max(r["peak_rss_kib"] for r in sub), "WRONG_PEAK_RSS")
        if arm in b.CONTROL:
            require(all(v == n for v, n in zip(corrects, totals)), "EXACT_CONTROL_FAILED")
            require(s["class"] == "fixed_symbolic_carry_circuit", "SYMBOLIC_MISLABEL")
            require(s["model_parameter_count"] == 0, "IMAGINARY_TRAINED_MODEL")
        else:
            require(s["class"] == "frozen_trained_neural_model", "NEURAL_MISLABEL")
            require(s["model_parameter_count"] > 0, "MISSING_MODEL")
    assert len(json.loads((HERE / "host_environment.json").read_text())["cpu_model"]) > 0
    require(data["independent_verification"] == "PENDING", "FALSE_APPROVAL")
    return {"task": "NC-029", "integrity": "PASS", "rows": len(rows),
            "cases": 128, "same_cases_each_run": True, "independent_verification": False}


if __name__ == "__main__":
    original = json.loads((HERE / "cpu_baseline.json").read_text())
    print(json.dumps(validate(original), sort_keys=True))
    # Negative control: 1 wrong metric must fail closed, with original data untouched.
    altered = json.loads(json.dumps(original))
    altered["summary"][b.ARMS[0]]["accuracy_numerator_each"][0] -= 1
    try:
        validate(altered)
    except ValueError:
        print("NEGATIVE_CONTROL:PASS")
    else:
        raise AssertionError("INTEGRITY_NEGATIVE_CONTROL_NOT_DETECTED")
