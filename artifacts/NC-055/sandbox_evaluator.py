"""Trusted, bounded Linux evaluation adapter for NC-055.

The candidate is launched under bubblewrap with only its own Python source,
Python's read-only runtime, and public JSONL on stdin. Oracle, reference targets
and HMAC key stay outside the mount and process namespaces. Without functional
bubblewrap this FAILS CLOSED; do not substitute normal subprocess execution.

Boundary: local Linux user/mount/PID/net namespaces, not a proof of kernel
security; the published 99-case fixture can still be learned/memorized from
its visible observations. For unrestricted new-code evaluation add independent
fresh-seed holdouts and external, authenticated evaluator operations.

Dependencies: Python 3 stdlib, /usr/bin/python3, bubblewrap, oracle.py,
benchmark.py. No filesystem writes except an ephemeral scratch directory.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import secrets
import shutil
import subprocess
import sys
import tempfile

import benchmark
import oracle

MAX_SOURCE = 65536
MAX_OUTPUT = 1048576
CPU_SECONDS = 5
WALL_SECONDS = 15


def _bounds():
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS + 1))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT, MAX_OUTPUT))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def evaluate(program, split="test"):
    if split not in oracle.SPLITS:
        raise ValueError("unknown split")
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("isolation unavailable: bubblewrap required; refusing candidate execution")
    src = Path(program).read_bytes()
    if not src or len(src) > MAX_SOURCE:
        raise ValueError("candidate source is empty or too large")
    cases = list(oracle.cases(split))
    # Blind the published generator's family-major iteration order.
    secrets.SystemRandom().shuffle(cases)
    encoding = {"train":"float64", "validation":"decimal_text", "test":"first_delta"}[split]
    public_cases = [benchmark.encode_view(pub, encoding) for pub, _ in cases]
    payload = "".join(json.dumps(p, separators=(",", ":")) + "\n" for p in public_cases)
    with tempfile.TemporaryDirectory(prefix="nc055-participant-") as temp:
        # No data or source files from this repository are mounted into the guest.
        candidate = Path(temp, "candidate.py")
        candidate.write_bytes(src)
        candidate.chmod(0o444)
        out_path, err_path = Path(temp, "stdout"), Path(temp, "stderr")
        cmd = [bwrap, "--unshare-all", "--die-with-parent", "--new-session",
               "--clearenv", "--ro-bind", "/usr", "/usr",
               "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
               "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
               "--dir", "/app", "--ro-bind", str(candidate), "/app/candidate.py",
               "--chdir", "/app", "--setenv", "HOME", "/tmp",
               "--setenv", "PYTHONPATH", "/app", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
               "--", "/usr/bin/python3", "-I", "/app/candidate.py"]
        with out_path.open("wb") as stdout, err_path.open("wb") as stderr:
            try:
                child = subprocess.run(cmd, input=payload.encode(), stdout=stdout, stderr=stderr,
                                       env={"PATH":"/usr/bin:/bin"}, cwd="/",
                                       timeout=WALL_SECONDS, preexec_fn=_bounds, check=False)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("candidate exceeded wall limit") from exc
        if child.returncode != 0:
            raise RuntimeError(f"isolated candidate exited {child.returncode}: "
                               + err_path.read_text(errors="replace")[:500])
        stdout_data = out_path.read_bytes()
        if len(stdout_data) > MAX_OUTPUT:
            raise RuntimeError("candidate exceeded output limit")
    lines = stdout_data.decode("utf-8").splitlines()
    if len(lines) != len(cases):
        raise ValueError(f"expected {len(cases)} JSONL submissions; got {len(lines)}")
    scores = []
    for line, (public, private) in zip(lines, cases):
        submitted = json.loads(line)
        if not isinstance(submitted, dict) or submitted.get("case_id") != public["case_id"]:
            raise ValueError("candidate returned missing/mismatched case ID")
        scores.append(benchmark.grade(public, private, submitted))
    return {"split":split, "denominator":len(scores),
            "forecast_pass":sum(s["forecast_pass"] for s in scores),
            "bounded_discovery_pass":sum(s["bounded_discovery_pass"] for s in scores),
            "proof_scope":"bubblewrap-isolated candidate for a public finite regression fixture; not unseen-seed generalization"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("program", help="path to a single candidate Python source file")
    p.add_argument("--split", choices=oracle.SPLITS.keys(), default="test")
    args = p.parse_args()
    print(json.dumps(evaluate(args.program, args.split),sort_keys=True,indent=2))


if __name__ == "__main__":
    main()
