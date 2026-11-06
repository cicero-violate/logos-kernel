"""Independent NC-029 review: read frozen candidate; write reviewer evidence only.

Dependencies: stdlib, installed CPU PyTorch, candidate's pinned local inputs.
Run from the kernel root with PYTHONDONTWRITEBYTECODE=1 and one BLAS thread.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REV = "bb3c8149bd6e6a4e42f4c94df8f06cc12b599d82"
MANIFEST_SHA = "d6a0656685e96f7f164d2d9caa9e98b3ce79d53e62e6db33da4bd98ee96561f1"
REVIEWER = "turn-723503c3aca84553afdf88269a2cedae"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest_path = HERE / "CANDIDATE_MANIFEST.json"
    assert sha(manifest_path.read_bytes()) == MANIFEST_SHA
    manifest = json.loads(manifest_path.read_text())
    checked = {}
    files = dict(manifest["source_files_and_evidence_sha256"])
    files[str(manifest_path.relative_to(ROOT))] = {"sha256": MANIFEST_SHA}
    for rel, entry in files.items():
        raw = (ROOT / rel).read_bytes()
        committed = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{REV}:{rel}"])
        assert raw == committed, rel
        assert sha(raw) == entry["sha256"], rel
        checked[rel] = sha(raw)
    upstream = Path(manifest["dependencies"]["original_repo"])
    provenance = manifest["source_dependency_provenance"]
    for rel, entry in provenance["sources"].items():
        assert sha((upstream / rel).read_bytes()) == entry["sha256"], rel
    for entry in provenance["checkpoints"].values():
        assert sha(Path(entry["path"]).read_bytes()) == entry["sha256"]
    assert sha((ROOT / "artifacts/NC-000/inventory.json").read_bytes()) == provenance["inventory_sha256"]

    import torch
    import bench_cpu as b
    import host_probe
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert b.provenance() == provenance
    assert len(set(b.pairs())) == 128
    assert json.loads((HERE / "measurement_protocol.json").read_text()) == b.protocol()
    a = torch.arange(256).repeat_interleave(256)
    c = torch.arange(256).repeat(256)
    expected = (a + c) & 255
    controls = {}
    for arm in b.ARMS[:3]:
        output = b.controls(arm, a, c)
        correct = int((output == expected).sum())
        corrupted_correct = int(((output ^ 1) == expected).sum())
        assert correct == 65536 and corrupted_correct == 0
        controls[arm] = {"correct": correct, "total": 65536, "xor1_corrupted_correct": corrupted_correct}

    replay_path = HERE / "reviewer_replay_723503c3.json"
    replay = json.loads(replay_path.read_text())
    original = json.loads((HERE / "cpu_baseline.json").read_text())
    assert replay["provenance"] == provenance == original["provenance"]
    assert replay["task_sha256"] == manifest["tasks"]["pair_sha256"]
    assert len(replay["trials"]) == 15
    assert {(r["arm"], r["trial"]) for r in replay["trials"]} == {(arm, t) for arm in b.ARMS for t in range(3)}
    for row in replay["trials"]:
        assert row["evaluated"] == row["verified"] == 640
        assert row["accuracy"] == 1 and not row["counterexamples_first_12"]
        assert row["warm_timed_cpu_s"] > 0 and row["energy_joules"] is None
    summary = {}
    for arm, s in replay["summary"].items():
        rows = [r for r in replay["trials"] if r["arm"] == arm]
        rate = sum(r["verified"] for r in rows) / sum(r["warm_timed_cpu_s"] for r in rows)
        assert rate == s["verified_answers_per_cpu_s_total"]
        summary[arm] = {"unique_cases_correct": 128, "unique_cases_total": 128,
                        "timing_trials": 3, "verified_answers_per_cpu_s": rate,
                        "peak_rss_kib": s["peak_rss_kib_max"]}
    frontiers = {}
    for kind in {s["class"] for s in replay["summary"].values()}:
        members = {arm: (s["accuracy"], s["verified_answers_per_cpu_s_total"], -s["peak_rss_kib_max"])
                   for arm, s in replay["summary"].items() if s["class"] == kind}
        frontiers[kind] = sorted(arm for arm, point in members.items() if not any(
            all(x >= y for x, y in zip(other, point)) and any(x > y for x, y in zip(other, point))
            for other in members.values()))
        assert frontiers[kind] == sorted(replay["within_class_pareto_frontier"][kind])

    host = host_probe.probe()
    recorded_host = json.loads((HERE / "host_environment.json").read_text())
    for key in ("python_executable_sha256", "torch_init_sha256", "torch", "cpu_model", "cpu_affinity"):
        assert host[key] == recorded_host[key], key
    events = [json.loads(line)["event"] for line in (ROOT / "instructions.jsonl").read_text().splitlines()]
    task = next(e for e in events if e.get("kind") == "task" and e.get("id") == "NC-029")
    assert "artifacts/NC-029/independent_verification.json" not in task["outputs"]
    # Both declared outputs are pinned implementer inputs: replacing either would
    # change the frozen candidate that this reviewer is required to preserve.
    assert all(path in checked for path in task["outputs"])
    assert b.provenance() == provenance
    for rel, digest in checked.items():
        assert sha((ROOT / rel).read_bytes()) == digest
    receipt = {
        "task": "NC-029", "owner": manifest["owner"], "verifier": REVIEWER,
        "verdict": "HOLD", "scientific_reproduction": "PASS_WITH_SCOPED_LIMITS",
        "candidate_sha256": MANIFEST_SHA, "candidate_revision": REV,
        "checked_candidate_files": checked,
        "tests": {"kernel_dispatch_tests": 36, "candidate_tests": 5,
                  "candidate_integrity_and_corrupted_metric_control": "PASS",
                  "independent_exhaustive_symbolic_controls": controls},
        "replay_sha256": sha(replay_path.read_bytes()), "replay_summary": summary,
        "replay_pareto_frontiers": frontiers,
        "host": host, "frozen_provenance_preserved": True,
        "blocker": "No separate declared verifier receipt output. dag.py requires PASS evidence in task outputs; both declared outputs are pinned candidate inputs. Reviewer cannot overwrite those and approve the resulting changed candidate.",
        "next_action": "Owner must publish a candidate that reserves one existing declared output for independent receipt while retaining raw benchmark evidence at another pinned path, or obtain an append-only protocol amendment supporting a separate receipt. Do not rewrite task history or bypass the PASS gate.",
        "proof_scope": "Five arms independently rerun in 15 fresh CPU processes on the same 128 unique 8-bit modular-addition cases. Each trial repeats that set five times; repeats are not novel cases. All three programmed controls additionally checked on all 65,536 pairs. Neural correctness remains sampled, seed11 only. Timing is host/load dependent. No training, power measurement, general architectural superiority, held-out transfer, or recursive self-improvement demonstrated. Memory is observed via RSS, not a fixed enforced limit. Python executable and torch init hashes/version checked; full shared-library closure is not pinned. Reviewer identity is a distinct turn label, not cryptographic authentication.",
    }
    output = HERE / "independent_verification.json"
    assert not output.exists(), "Do not overwrite existing reviewer evidence"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(output), "sha256": sha(output.read_bytes()), "verdict": receipt["verdict"], "replay_summary": summary}, indent=2))


if __name__ == "__main__":
    main()
