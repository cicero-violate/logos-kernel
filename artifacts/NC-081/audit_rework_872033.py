"""Bounded evidence assembler/checker for NC-081 corrected three-seed replay.

This is an IMPLEMENTER artifact, NOT an independent review. It refuses to
assemble an acceptance summary if any seed's measured CPU control is unmatched.
Only reads isolated replay files and pinned, original case evidence.

Dependencies: Python 3 standard library, source/evidence in artifacts/NC-081.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (11, 19, 29)
ARMS = ("all_concepts", "model", "random_pick", "frequent",
        "other_goal_examples", "no_library")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(condition, message):
    if not condition:
        raise ValueError(message)


def audit(root):
    original = json.loads((HERE / "independent_verification.json").read_text())
    old_manifest = json.loads((HERE / "CANDIDATE_MANIFEST.json").read_text())
    library_sha = digest(HERE / "la015_library_pinned.py")
    checked(library_sha == "0c876a9e8d90e9401323b0068a66acc9b6d1e9f3051f22a771fe9d2a2d098b8a",
            "pinned dependency changed")
    results = {}
    source = {name: digest(HERE / name) for name in
              ("selector.py", "test_selector.py", "la015_library_pinned.py", "README.md")}
    evidence = {}
    valid_cpu_matches = []
    for seed in SEEDS:
        sd = root / f"seed{seed}"
        report_path = sd / "selector_results.json"
        checked(report_path.is_file(), f"seed {seed}: missing replay")
        report = json.loads(report_path.read_text())
        checked(report.get("schema") == "logos.nc081.results/1", f"seed {seed}: schema")
        checked(tuple(report["seeds"]) == (str(seed),), f"seed {seed}: contaminated multiseed output")
        cfg = report["config"]
        for key, expected in (
            ("dreams", 30000), ("steps", 3000), ("exam_goals", 300),
            ("top_k", 12), ("budget", 8000)):
            checked(cfg.get(key) == expected, f"seed {seed}: config {key}: {cfg.get(key)}")
        checked(report["la015_library_py_sha256"] == library_sha,
                f"seed {seed}: loaded different dependency")
        data = report["seeds"][str(seed)]
        checked(data["exam_goals"] == 300, f"seed {seed}: wrong exam goal count")
        checked(data["exam_sha256"] == original["three_seed_audit"][str(seed)]["sealed_goal_digest"],
                f"seed {seed}: changed sealed exam")
        checked(data["training"]["dreams"] == 30000 and data["training"]["sealed_side_skipped"] > 0,
                f"seed {seed}: training mismatch")
        checked(data["library_sha256"] == old_manifest["evidence_sha256"][f"library_seed{seed}.jsonl"],
                f"seed {seed}: wrong library snapshot")
        checked(digest(sd / f"library_seed{seed}.jsonl") == data["library_sha256"],
                f"seed {seed}: copied library mismatch")
        checked(digest(sd / f"selector_seed{seed}.pt") == data["checkpoint_sha256"],
                f"seed {seed}: checkpoint mismatch")
        counts = {arm: data["arms"][arm]["solved"] for arm in ARMS}
        old = original["three_seed_audit"][str(seed)]["fresh_solve_counts"]
        checked(counts == old, f"seed {seed}: solve counts changed: {counts} vs {old}")
        target = (data["training"]["train_cpu_seconds"] + data["inference_cpu_seconds"]
                  + data["arms"]["model"]["cpu_seconds"])
        eq = data["arms"]["all_concepts_equal_total_cpu"]
        actual = eq["cpu_seconds"]
        diff = (actual - target) / target
        checked(eq["target_cpu_seconds"] > 0 and eq["tolerance"] == 0.1,
                f"seed {seed}: CPU matching metadata")
        checked(abs(target - eq["target_cpu_seconds"]) <= 0.02,
                f"seed {seed}: target not inclusive")
        checked(math.isfinite(diff) and len(eq["attempts"]) <= 5 and eq["attempts"],
                f"seed {seed}: invalid calibration")
        matches = eq["matched"] and abs(diff) <= 0.10
        valid_cpu_matches.append(matches)
        results[str(seed)] = {
            "sealed_exam_sha256": data["exam_sha256"],
            "library_snapshot_sha256": data["library_sha256"],
            "library_head": data["library_head"],
            "checkpoint_sha256": data["checkpoint_sha256"],
            "train_cpu_seconds": data["training"]["train_cpu_seconds"],
            "inference_cpu_seconds": data["inference_cpu_seconds"],
            "model_search_cpu_seconds": data["arms"]["model"]["cpu_seconds"],
            "model_total_cpu_seconds": round(target, 4),
            "calibrated_all_concepts_cpu_seconds": actual,
            "relative_cpu_difference_percent": round(diff * 100, 3),
            "equal_cpu_matched_within_10_percent": bool(matches),
            "baseline_calibration_attempts": eq["attempts"],
            "solved_counts": {**counts, "all_concepts_equal_total_cpu": eq["solved"]},
        }
        for name in (f"library_seed{seed}.jsonl", f"selector_seed{seed}.pt", "selector_results.json"):
            evidence[f"seed{seed}/{name}"] = digest(sd / name)
    outcome = ("NO_GAIN" if all(r["solved_counts"]["model"] <=
                               r["solved_counts"]["all_concepts"] for r in results.values())
               else "MIXED_NEEDS_INTERPRETATION")
    return {
        "schema": "logos.nc081.corrected-replay-audit/1",
        "task": "NC-081",
        "role": "implementer evidence, not independent verification",
        "source_sha256": source,
        "pinned_external_library": {
            "file": "artifacts/NC-081/la015_library_pinned.py",
            "sha256": library_sha,
        },
        "observed": results,
        "all_three_equal_cpu_controls_matched": all(valid_cpu_matches),
        "scientific_result": outcome,
        "reproduction": "Three independent full 300-sealed-exam runs with 30000 accepted training dreams, 3000 steps, 8000 per-goal candidate budget, top-12 selection, seeds 11/19/29",
        "isolated_evidence_sha256": evidence,
        "limitations": [
            "CPU matching tolerance fixed at 10%, measured under shared host load",
            "Calibration attempts incur additional experimental CPU not included in final model versus baseline comparison",
            "Only 300 sealed goals per seed in bounded 8-bit families; no broad optimality",
            "The label components may contain sealed-side expressions, while accepted dream target functions are on training side",
            "The train/test behavior relies on inherited concept library snapshots, not a globally certified novelty proof",
            "No champion promotion or claim of general intelligence",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    result = audit(a.root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "all_cpu_matched": result["all_three_equal_cpu_controls_matched"],
        "outcome": result["scientific_result"],
        "results": {seed: {
            "cpu_diff_percent": v["relative_cpu_difference_percent"],
            "solved": v["solved_counts"]}
            for seed, v in result["observed"].items()},
        "audit_sha256": digest(a.out),
    }, sort_keys=True))
    if not result["all_three_equal_cpu_controls_matched"]:
        raise SystemExit("UNMATCHED_CPU: not ready to request review")


if __name__ == "__main__":
    main()
