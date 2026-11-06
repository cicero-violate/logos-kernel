"""Deterministic sandbox canary; not an RLT training implementation."""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

OUTPUT = Path("/output")
CKPT = OUTPUT / "checkpoint.json"
FORBIDDEN = [
    "/inputs/champion.pt",
    "/inputs/worker.py",
    "/workspace/experiments/logos-kernel/instructions.jsonl",
    "/workspace/experiments/logos-kernel/artifacts/NC-044/independent_verification.json",
    "/workspace/experiments/logos-kernel/artifacts/NC-022/independent_receipt.json",
    "/workspace/experiments/rlt_split_algorithms_ablation_v1/snapshots/current_unified_v3.pt",
    "/etc/passwd",
    "/home/sandbox-cross-boundary-probe"
]

def atomic_json(path, value):
    temp = path.with_suffix(".new")
    with open(temp, "w") as f:
        json.dump(value, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)

def restore_rng(data):
    state = data["rng_state"]
    rng = random.Random()
    rng.setstate((state[0], tuple(state[1]), state[2]))
    return rng

def main():
    seen = []
    for path in FORBIDDEN:
        try:
            with open(path, "ab") as f:
                f.write(b"DENIED")
            seen.append([path, "WRITABLE"])
        except OSError:
            seen.append([path, "DENIED"])
    if any(x[1] != "DENIED" for x in seen):
        return 4
    if Path("/etc/passwd").exists() or Path("/workspace").exists() or Path("/home").exists():
        return 5
    with open("/inputs/champion.pt", "rb") as f:
        prefix = hashlib.sha256(f.read(1024)).hexdigest()
    if CKPT.exists():
        state = json.loads(CKPT.read_text())
        rng = restore_rng(state)
    else:
        rng = random.Random(2026)
        state = {"step": 0, "optimizer": {"momentum": 0}, "rng_state": rng.getstate()}
    for step in range(state["step"], 8):
        state["optimizer"]["momentum"] = (state["optimizer"]["momentum"] * 7 + rng.randrange(1000)) % 65537
        state["step"] = step + 1
        state["rng_state"] = rng.getstate()
        state["champion_prefix_sha256"] = prefix
        state["worker_id"] = os.environ["TRAINER_WORKER_ID"]
        atomic_json(CKPT, state)
        if (OUTPUT / "cancel.request").exists():
            return 76
        if (OUTPUT / "pause.request").exists():
            return 75
        time.sleep(.09)
    atomic_json(OUTPUT / "denials.json", {"attempts": seen, "host_write_denied": True})
    return 0

if __name__ == "__main__":
    sys.exit(main())
