#!/usr/bin/env python3
"""Capture CPU/software fingerprint; stdlib + optional installed torch.
Dependencies: Python 3, Linux /proc, original RLT source already in bench_cpu.py.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def probe():
    import torch
    cpu = Path("/proc/cpuinfo").read_text()
    first = cpu.split("\n\n")[0]
    cpu_meta = {}
    for line in first.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key.strip() in ("model name", "vendor_id", "cpu family", "model", "stepping", "microcode", "flags"):
                cpu_meta[key.strip()] = value.strip()
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    quota_file = Path("/sys/fs/cgroup/cpu.max")
    mem_file = Path("/sys/fs/cgroup/memory.max")
    return {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_executable_sha256": sha(Path(sys.executable).resolve()),
        "torch": torch.__version__,
        "torch_init_sha256": sha(torch.__file__),
        "torch_compilation": torch.__config__.show(),
        "uname": list(platform.uname()),
        "cpu_model": cpu_meta,
        "cpu_info_sha256": hashlib.sha256(cpu.encode()).hexdigest(),
        "cpu_affinity": affinity,
        "visible_cpu_count": os.cpu_count(),
        "cpu_cgroup_quota": quota_file.read_text().strip() if quota_file.is_file() else None,
        "memory_cgroup_limit": mem_file.read_text().strip() if mem_file.is_file() else None,
        "source": "measured on benchmark host after trials; CPU frequency/load not pinned",
    }


if __name__ == "__main__":
    out = HERE / "host_environment.json"
    out.write_text(json.dumps(probe(), indent=2, sort_keys=True) + "\n")
    d = json.loads(out.read_text())
    print(json.dumps({"cpu": d["cpu_model"].get("model name"), "torch": d["torch"], "python": d["python"].split()[0], "affinity_cpus": len(d["cpu_affinity"] or [])}))
