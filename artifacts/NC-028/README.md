# NC-028 — fail-closed mathematical discovery pilot

**State:** Implementer pilot / independent verification pending. This is **not** an
autonomous discovery release or a newly trained general-purpose solver.

## Mechanism and boundaries

- `isolation.py`: Linux Landlock ABI >=5 restricts filesystem reads to trusted
  dependencies; writable paths are limited to private `scratch/` and
  `/dev/null`. libseccomp blocks socket, network, ptrace, inode-metadata mutation, and several namespace
  syscalls. RLIMIT caps wall-clock, CPU, file size, address space, and open FDs.
  Process creation is denied through fork/vfork, process-form clone, and
  clone3; pthread-style CLONE_THREAD is permitted. RLIMIT_NPROC=2048 is a
  **per-real-UID host ceiling**, not an isolated per-sandbox quota. stdout
  and stderr use regular scratch-backed files so RLIMIT_FSIZE=4 MiB applies,
  rather than memory-unbounded pipes. This does **not** establish cgroup
  containment, per-sandbox task limits, or trusted-UID separation. No GPU.
  This is neither a VM nor a separate UID/network namespace. Full adversarial
  security is **not independently established**.
- `propose_worker.py`: immutable NC-007 controller weights produce action
  tokens autoregressively, with optional bounded rejection-only reranking. The worker contains **no target recipes** or
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
  costs are **not identical** across arms; results do **not** establish
  matched-CPU superiority. Referee numeric formulas stay in
  the parent host, outside the proposal process.

## Acceptance restrictions

The model was previously trained on **all eleven recognized goal IDs and their
exact action recipes**. Its structural unseen-goal generalization is **zero by
interface**. Report `NO_DISCOVERY` irrespective of known-goal numeric accuracy.
A known recipe recalled on new operands is NOT a discovered new operator.
No neural weight, original RLT champion, registry, or global default is changed.
Proof scope is finite sampled unsigned 16-bit arithmetic, not all 2^32 pairs.

The process/FSIZE containment changes have only local regression coverage.
A bounded rejection-only **frozen-controller reranker** is now implemented
in `propose_worker.py`: with prior rejected complete action sequences it
enumerates alternative neural-logit-scored sequences (max 5 actions, beam 32),
without giving the proposer expected answers. `feedback_trial.py` records
three isolated probes: a genuine finite sampled known-goal check, refusal of
a structurally unseen goal, and a **clearly labeled synthetic observation-bit-flip**
negative control forcing one alternative. This is NOT a trained
feedback-conditioned model: reranking does not update weights and the
counterexample values do not condition logits. It does NOT constitute novel
task generalization or an effective natural counterexample-revision loop.
Any synthetic-fault case is excluded from performance claims.

The old frozen-controller reranker has no learned correction; the separate
lexical controller below has numeric-counterexample-weighted online negative
updates. This is a causal dependence on observed numeric errors, **not** proof
of counterexample-guided synthesis or useful repair. Matched consumed-CPU
comparisons, if within the preregistered tolerance on a fresh reproduction,
remain an empirical budget comparison rather than exact wall/CPU equality.
Independent acceptance and discovery remain unproven.
The frozen known-goal training coverage makes neural invention unsupported.

The sandbox and scientific evidence require a **distinct real independent
verifier** before NC-028 can pass. No self-approval, no external network, no
self-modification, no promotion. Treat this pilot as a reproducible
failure/feasibility study only.

## Reproduce

```bash
cd /workspace/experiments/logos-kernel
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -q -s artifacts/NC-028 -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-028/arena.py
PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-028/feedback_trial.py
python dag.py validate
```

**Code/file dependencies:** Python 3.14 stdlib, Linux 6.18 Landlock,
`libseccomp.so.2`, `libmkl*.so.3`, NumPy, PyTorch, files under
`artifacts/NC-000/001/004/005/006/007`, the original immutable
`/workspace/experiments/latticeforge_neural_isa_test`, and the RLT checkpoint
`/workspace/experiments/rlt_split_algorithms_ablation_v1/snapshots/current_unified_v3.pt`.
The `isolation.py` runner currently depends on these exact host paths; it is
not portable without updating and verifying the allowlist.

## Additional unseen-goal neural benchmark (rework, implementer-only)

`train_lexical.py` trains a character/ngram neural controller on **11 known
supervised goal/program labels**, explicitly excluding the target goal
`UNSEEN_XOR_DOUBLE_PLUS_Y`. This is supervised textual transfer, not proof
of neural algorithm invention. `lexical_model.py` runs a frozen checkpoint
copy in an isolated process; its bounded online update changes only the
in-memory model of a single proposal call.

After execution, the host selects a **real numeric failure from the
8-example feedback split** (`pair_index,x,y,expected,observed`) and rejects
the failed action sequence. The second lexical proposal uses 12 bounded SGD
steps penalizing the rejected sequence, weighted by the numeric error.
An actual counterexample changes the policy gradient even when the aggregate
counts remain identical. **It does not differentiate through the algorithm
or infer which replacement actions will be correct.** No target action
sequence, held-out oracle labels or solution recipe reaches the proposer.
The disjoint eight-case held-out split is used for scoring only.

The four predeclared benchmark arms are:

- `lexical_neural`: two neural propose+execute attempts with genuine feedback.
- `lexical_frozen`: two neural propose+execute attempts, no feedback update.
- `fixed_host`: four deterministic host-chosen programs, no trained proposer.
- `random_host`: four seeded host-chosen programs, no trained proposer.

Every arm has the same **12 process-CPU-second and 20 wall-second allocated
caps**, the same disjoint feedback/heldout operand sets and the same execution
worker. Baseline arms receive more attempts because they avoid neural-proposal
CPU cost. Actual consumed CPU and wall are recorded; the run is called
`cpu_matched`/`wall_matched` only if **every arm** differs from the neural
reference by at most 15%. Otherwise comparability is reported as NOT MATCHED.
This tolerance, attempt schedule and stopping rule are fixed in source before
execution, independent of scores. This is *approximate* measured-compute
matching, not identical effort or identical proposal counts. Repeated held-out
testing is not independent sample accumulation; the feedback-best candidate is
selected strictly from feedback counts. Claims of neural superiority are
prohibited without an actual matched multiseed comparison and independent
review.

`NO_DISCOVERY` remains the scientific result unless a distinct verifier
reproduces evidence and accepts it. Single-seed finite arithmetic samples do
not prove generalization, autonomous discovery, model-based causal reasoning,
or safe deployment. Frozen original RLT champion and pretrained checkpoints
remain unchanged.

### Reproduction and dependencies

```bash
cd /workspace/experiments/logos-kernel
PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-028/novel_trial.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -q -s artifacts/NC-028 -p 'test_*.py'
python dag.py validate
```

Files: `novel_trial.py` depends on `arena.py`, `isolation.py`,
`lexical_model.py`, `lexical_controller.pt`, `execute_worker.py`,
NC-001/005/006/007 frozen sources, manifests and checkpoint weights,
Linux Landlock ABI >=5, libseccomp.so.2, Python 3.14, NumPy and PyTorch CPU.
The tests additionally use `test_lexical.py` and pinned benchmark evidence.
No other repository or champion is modified by the benchmark.
