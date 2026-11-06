#!/usr/bin/env python3
"""NC-044: explicit, CPU-only, unprivileged candidate-trainer sandbox.

No global service, agent authentication, champion promotion, or auto-claim.
Python standard library + Linux bubblewrap are required.
"""
import argparse
import hashlib
import fcntl
import json
import os
from pathlib import Path
import resource
import secrets
import signal
import subprocess
import sys
import time

REPO = Path("/workspace/experiments/logos-kernel")
CHAMPION = Path("/workspace/experiments/rlt_split_algorithms_ablation_v1/snapshots/current_unified_v3.pt")
CHAMPION_SHA256 = "4c42d13fd46f57ac0db6eccbaef35e99c4cefcec393a3a3a4faf146645e8f683"
NC022 = REPO / "artifacts/NC-022/independent_receipt.json"
DEFAULT_ROOT = Path("/workspace/.local/state/logos-kernel-trainer")
SCHEMA = "logos-trainer-isolation-v1"
PAUSED = 75
CANCELLED = 76

class BoundaryError(ValueError):
    pass

def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def safe_path(value, existing=True, file=False):
    """Reject symlink at any path component, including the terminal."""
    p = Path(value)
    if not p.is_absolute() or ".." in p.parts:
        raise BoundaryError("absolute paths without traversal required")
    for q in (p, *p.parents):
        if q.is_symlink():
            raise BoundaryError("symlink component rejected: " + str(q))
    if existing and not p.exists():
        raise BoundaryError("missing path: " + str(p))
    if file and not p.is_file():
        raise BoundaryError("not a regular file: " + str(p))
    return p

def disjoint(a, b):
    return not (a == b or a in b.parents or b in a.parents)

def validate_root(root):
    root = safe_path(root, existing=False)
    for critical in (REPO, CHAMPION, REPO.parent / "rlt_split_algorithms_ablation_v1",
                     Path("/usr"), Path("/etc"), Path("/home"), Path("/tmp")):
        if not disjoint(root, critical):
            raise BoundaryError("state root overlaps protected path: " + str(critical))
    if not (root == DEFAULT_ROOT or DEFAULT_ROOT in root.parents):
        raise BoundaryError("state root must remain below " + str(DEFAULT_ROOT))
    return root

def write_json_atomic(p, value):
    p = safe_path(p, existing=False)
    temporary = p.with_name(p.name + ".tmp-" + secrets.token_hex(8))
    with open(temporary, "x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, p)

def read_run(root, run_id):
    if not run_id.startswith("run-") or not all(c in "0123456789abcdef" for c in run_id[4:]) or len(run_id) != 36:
        raise BoundaryError("invalid generated run ID")
    root = validate_root(root)
    run = safe_path(root / run_id)
    info = safe_path(run / "run.json", file=True)
    x = json.loads(info.read_text())
    if x.get("schema") != SCHEMA or x.get("run_id") != run_id:
        raise BoundaryError("invalid run manifest")
    return run, x

def checks(x):
    files = [("champion", CHAMPION, CHAMPION_SHA256),
             ("nc022_receipt", NC022, x["nc022_sha256"]),
             ("source", Path(x["source"]), x["source_sha256"])]
    for label, path, expected in files:
        safe_path(path, file=True)
        if digest(path) != expected:
            raise BoundaryError("changed pinned input: " + label)
    if x["champion_sha256"] != CHAMPION_SHA256:
        raise BoundaryError("wrong champion pin")
    for item in x["inputs"]:
        safe_path(item["path"], file=True)
        if digest(item["path"]) != item["sha256"]:
            raise BoundaryError("changed pinned dataset: " + item["name"])

def make_run(root, source, inputs, wall, cpu, memory_mb, disk_mb):
    root = validate_root(root)
    safe_path(CHAMPION, file=True)
    if digest(CHAMPION) != CHAMPION_SHA256:
        raise BoundaryError("CHAMPION HASH MISMATCH")
    source = safe_path(source, file=True)
    safe_path(NC022, file=True)
    if not 1 <= cpu <= 3600 or not 1 <= wall <= 3600 or not 64 <= memory_mb <= 8192 or not 1 <= disk_mb <= 1024:
        raise BoundaryError("quota out of accepted range")
    items = []
    used_names = set()
    for inp in inputs:
        # name:absolute_path:sha256 (avoid ambiguous colon in Linux paths)
        name, filename, sha = inp.split(":", 2)
        if not name.replace("_", "").isalnum() or name in used_names or not name:
            raise BoundaryError("duplicate or invalid input name")
        used_names.add(name)
        p = safe_path(filename, file=True)
        if p == CHAMPION or p == source or REPO / "instructions.jsonl" == p:
            raise BoundaryError("input is protected control data")
        if digest(p) != sha:
            raise BoundaryError("unverified input checksum")
        items.append({"name": name, "path": str(p), "sha256": sha})
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_id = "run-" + secrets.token_hex(16)
    run = root / run_id
    run.mkdir(mode=0o700)
    (run / "output").mkdir(mode=0o700)
    worker_id = "worker-" + secrets.token_hex(16)
    x = {"schema": SCHEMA, "run_id": run_id, "worker_id": worker_id,
         "creator_pid": os.getpid(), "creator_uid": os.getuid(),
         "source": str(source), "source_sha256": digest(source),
         "champion_path": str(CHAMPION), "champion_sha256": CHAMPION_SHA256,
         "nc022_path": str(NC022), "nc022_sha256": digest(NC022),
         "inputs": items, "quota": {"wall_seconds": wall, "cpu_seconds": cpu,
         "memory_mb": memory_mb, "disk_mb": disk_mb},
         "role": "TRAINER", "outputs": "/output",
         "auth_limit": "Same Unix UID can modify host files outside the sandbox; worker label is not authenticated.",
         "source_limit": "Input hashes prove integrity only, not independent approval."}
    write_json_atomic(run / "run.json", x)
    return run, x

def size_bytes(folder):
    total = 0
    for parent, _, files in os.walk(folder, followlinks=False):
        for fn in files:
            p = Path(parent) / fn
            if not p.is_symlink() and p.is_file():
                total += p.stat().st_size
    return total

def limits(cpu, memory_bytes, disk_bytes):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (disk_bytes, disk_bytes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
        try:
            os.nice(15)
        except OSError:
            pass
    return apply

def execute(root, run_id, resume=False):
    run, _ = read_run(root, run_id)
    with open(run / ".lock", "a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BoundaryError("already running: " + run_id) from error
        return execute_locked(root, run_id, resume=resume)

def execute_locked(root, run_id, resume=False):
    run, x = read_run(root, run_id)
    checks(x)
    if (run / "output" / "cancel.request").exists():
        raise BoundaryError("cancelled run cannot restart")
    if (run / "output" / "pause.request").exists():
        if not resume:
            raise BoundaryError("paused run: use resume")
        (run / "output" / "pause.request").unlink()
    checkpoint = run / "output" / "checkpoint.json"
    previous_report = run / "last_execution.json"
    if previous_report.exists():
        previous = json.loads(previous_report.read_text())
        old_hash = previous.get("checkpoint_sha256")
        if old_hash is not None:
            if not checkpoint.is_file() or checkpoint.is_symlink() or digest(checkpoint) != old_hash:
                raise BoundaryError("checkpoint changed since last verified process exit")
        elif checkpoint.exists():
            raise BoundaryError("untracked checkpoint after prior run")
    elif checkpoint.exists():
        raise BoundaryError("orphan checkpoint; fail closed on interrupted host launch")
    q = x["quota"]
    if size_bytes(run / "output") > q["disk_mb"] * 1024 * 1024:
        raise BoundaryError("disk quota exhausted before launch")
    # Minimal mount namespace: no host /workspace, /home, /etc, or other
    # agent worktrees. Only versioned sources and champion are visible read-only.
    cmd = ["bwrap", "--tmpfs", "/", "--ro-bind", "/usr", "/usr",
           "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
           "--symlink", "usr/lib", "/lib64",
           "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
           "--dir", "/inputs", "--dir", "/output",
           "--bind", str(run / "output"), "/output",
           "--ro-bind", str(CHAMPION), "/inputs/champion.pt",
           "--ro-bind", str(x["source"]), "/inputs/worker.py",
           "--unshare-net", "--unshare-pid", "--die-with-parent", "--new-session",
           "--clearenv", "--setenv", "PATH", "/usr/bin",
           "--setenv", "HOME", "/output", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
           "--setenv", "TRAINER_RUN_ID", run_id,
           "--setenv", "TRAINER_WORKER_ID", x["worker_id"]]
    for item in x["inputs"]:
        cmd.extend(["--ro-bind", item["path"], "/inputs/" + item["name"]])
    cmd.extend(["/usr/bin/python3", "/inputs/worker.py"])
    started = time.monotonic()
    p = subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, preexec_fn=limits(q["cpu_seconds"],
                         q["memory_mb"] * 1024 * 1024, q["disk_mb"] * 1024 * 1024))
    outcome = "unknown"
    try:
        while p.poll() is None:
            if time.monotonic() - started > q["wall_seconds"]:
                outcome = "wall_timeout"
                os.killpg(p.pid, signal.SIGKILL)
                break
            if size_bytes(run / "output") > q["disk_mb"] * 1024 * 1024:
                outcome = "disk_quota"
                os.killpg(p.pid, signal.SIGKILL)
                break
            time.sleep(0.1)
        out, err = p.communicate(timeout=3)
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
            p.communicate()
    if outcome == "unknown":
        outcome = {0: "completed", PAUSED: "paused", CANCELLED: "cancelled"}.get(p.returncode, "failed")
    report = {"run_id": run_id, "worker_id": x["worker_id"], "sandbox_pid": p.pid,
              "launcher_uid": os.getuid(), "returncode": p.returncode,
              "outcome": outcome, "wall_elapsed_seconds": round(time.monotonic() - started, 3),
              "output_bytes": size_bytes(run / "output"),
              "checkpoint_sha256": digest(checkpoint) if checkpoint.is_file() and not checkpoint.is_symlink() else None,
              "source_sha256": x["source_sha256"],
              "champion_sha256": CHAMPION_SHA256,
              "stdout": out.decode("utf-8", "replace")[-4096:],
              "stderr": err.decode("utf-8", "replace")[-4096:]}
    # Parent writes report only in the uniquely owned run directory.
    write_json_atomic(run / "last_execution.json", report)
    return report

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "run", "resume", "pause", "cancel", "status"])
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--run-id")
    parser.add_argument("--source")
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--wall", type=int, default=15)
    parser.add_argument("--cpu", type=int, default=10)
    parser.add_argument("--memory-mb", type=int, default=512)
    parser.add_argument("--disk-mb", type=int, default=16)
    a = parser.parse_args(argv)
    try:
        if a.command == "init":
            if not a.source: raise BoundaryError("--source required")
            run, value = make_run(a.root, a.source, a.input, a.wall, a.cpu, a.memory_mb, a.disk_mb)
            result = {"run": str(run), "run_id": value["run_id"],
                      "worker_id": value["worker_id"], "champion_sha256": CHAMPION_SHA256}
        else:
            if not a.run_id: raise BoundaryError("--run-id required")
            run, value = read_run(a.root, a.run_id)
            if a.command in ("pause", "cancel"):
                (run / "output" / (a.command + ".request")).touch(exist_ok=True)
                result = {"run_id": a.run_id, "requested": a.command,
                          "acknowledged": False}
            elif a.command == "status":
                result = {"run_id": a.run_id, "worker_id": value["worker_id"],
                          "last": json.loads((run / "last_execution.json").read_text())
                          if (run / "last_execution.json").exists() else None,
                          "paused_request": (run / "output" / "pause.request").exists(),
                          "cancel_request": (run / "output" / "cancel.request").exists()}
            else:
                result = execute(a.root, a.run_id, resume=a.command == "resume")
        print(json.dumps(result, sort_keys=True))
    except (BoundaryError, OSError, ValueError, KeyError) as error:
        print(json.dumps({"error": str(error), "status": "REFUSED"}))
        return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())
