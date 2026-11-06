# NC-028 — fail-closed mathematical discovery pilot

**State:** Implementer pilot / independent verification pending. This is **not** an
autonomous discovery release or a newly trained general-purpose solver.

## Mechanism and boundaries

- `isolation.py`: Linux Landlock ABI >=5 restricts filesystem reads to trusted
  dependencies; writable paths are limited to private `scratch/` and
  `/dev/null`. libseccomp blocks socket, network, ptrace, inode-metadata mutation, and several namespace
  syscalls. RLIMIT caps wall-clock, CPU, file size, address space, and open FDs.
  No GPU. This is neither a VM nor a separate UID/network namespace. Full
  adversarial security is **not independently established**.
- `propose_worker.py`: immutable NC-007 controller weights produce action
  tokens autoregressively. The worker contains **no target recipes** or
  evaluation oracle. Unknown goal strings are rejected, not mapped to a
  hand-coded solution.
- `execute_worker.py`: after a proposal is frozen, a *separate* contained
  process executes typed HALT-bounded actions through existing NC-006 frozen
  neural primitives. NC-007 execution code still includes historical
  supervised target recipes but **not** the proposing controller.
- `arena.py`: preregistered fixed seed and 8 unseen-in-this-pilot numeric
  pairs per (seed, task), three frozen controller seeds, plus fixed
  deterministic and random host-controlled baselines. Each arm has one
  proposed program, max five action tokens, same eight pairs. Wall and CPU
  costs are **not identical** across arms. Referee numeric formulas stay in
  the parent host, outside the proposal process.

## Acceptance restrictions

The model was previously trained on **all eleven recognized goal IDs and their
exact action recipes**. Its structural unseen-goal generalization is **zero by
interface**. Report `NO_DISCOVERY` irrespective of known-goal numeric accuracy.
A known recipe recalled on new operands is NOT a discovered new operator.
No neural weight, original RLT champion, registry, or global default is changed.
Proof scope is finite sampled unsigned 16-bit arithmetic, not all 2^32 pairs.

The sandbox and scientific evidence require a **distinct real independent
verifier** before NC-028 can pass. No self-approval, no external network, no
self-modification, no promotion. Treat this pilot as a reproducible
failure/feasibility study only.

## Reproduce

```bash
cd /workspace/experiments/logos-kernel
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -q -s artifacts/NC-028 -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-028/arena.py
python dag.py validate
```

**Code/file dependencies:** Python 3.14 stdlib, Linux 6.18 Landlock,
`libseccomp.so.2`, `libmkl*.so.3`, NumPy, PyTorch, files under
`artifacts/NC-000/001/004/005/006/007`, the original immutable
`/workspace/experiments/latticeforge_neural_isa_test`, and the RLT checkpoint
`/workspace/experiments/rlt_split_algorithms_ablation_v1/snapshots/current_unified_v3.pt`.
The `isolation.py` runner currently depends on these exact host paths; it is
not portable without updating and verifying the allowlist.
