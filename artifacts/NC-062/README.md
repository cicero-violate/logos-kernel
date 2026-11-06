# NC-062 — bounded generation-over-generation local learning canary

**Boundary:** This is a four-bit, four-template **NO_DISCOVERY**-or-negative-evidence experiment, not a general self-improving agent.

### Contamination and prospective clean rerun

The original NC-061 seed sets were randomly generated and **eight of nine**
seed/generation evaluations overlapped the four known training-library
functions (one overlapping function among ten). All three original generation
comparisons are correctly `CONTAMINATED`; **they cannot establish gain.**
The original `scoreboard_results.json`, `commitments.json`, and
`sealed_reveals.json` are retained immutably.

`evaluate_uncontaminated.py` starts a *new*, prospectively committed evaluation
after all candidate checkpoints are frozen, without further training. Before
seeing fresh sealed goals, it fixes a deterministic 40-goal pool and keeps
the first ten functions excluding all four initial training keys and
previously exposed discovery-function keys. It writes new
`eval_v2_protocol.json`, `eval_v2_commitments.json`,
`eval_v2_candidate_freeze.json`, `eval_v2_results.json`, and
`eval_v2_reveals.json`. The fixed candidate checkpoints, labels, and
prior contaminated evidence are unchanged; no seed is hand-picked for its
measured candidate success. `audit_clean_eval.py` independently recomputes
the eligibility and all 90 finite-scored results from the retained model
choices, and can rerun every sandboxed model inference with `--full`.
The second evaluation is a *mitigation experiment*, not a replacement
for the original preregistration; both are visible to reviewers.
 Neither the global NC-009 registry nor the upstream RLT champion is altered.

## Components

- `worker_source.py`: 24→24→4 CPU MLP. The worker executes only as a `python -I -B -c` program under NC-028 `isolation.run` (Landlock, libseccomp, no sockets/fork, bounded CPU/memory/file/wall). Parent sends exact, typed public 8-example training/evaluation rows, never sealed answer programs or full tables. `init`, `train`, `infer` are strictly separated; inference never receives training labels. Three frozen initial states (seeds 11/19/29), each successively updated through three candidate generations.
- `run_loop.py`: trusted parent/oracle binds the NC-009 independently verified library, validates the 16→4 bit compiler by exhaustive 256-pair equivalence, freezes the experiment protocol and hashes of all sealed reveals before candidate checkpoint training. A separate sealed evaluation uses NC-061 `score_candidate` and `compare` on fresh seed/threshold commitments, three seeds per generation.
- `generation_trace.jsonl`: all six training-only discovery goals per generation and seed; actual model-ranked candidates; exact four-bit truth-table verdicts; first actual failed input and feedback-excluded later candidates. It never records a discovered primitive merely because a known template passes.
- `library_versions.json`: hash-linked local versions of the original four accepted NC-009 operators. New programs require independent 256-pair verification and NC-009-style canonical dedup; the global accepted registry cannot be written by the contained worker.
- `checkpoint_g{0,1,2,3}_seed{11,19,29}.json`: each candidate MLP's complete finite CPU tensor state, parent hash, seed, training input hash, source SHA, gradient-step count. No checkpoint was promoted or swapped into the RLT champion.
- `scoreboard_results.json`: generation results, full numerators/denominators, worker process CPU, measured costs, no-library-training ablation (same initial random model and same output vocabulary), NC-061 verdicts, limits, counterexamples, and scientific claim.
- `commitments.json`: public hashes, written before candidate training; `sealed_reveals.json`: reviewer-reproducible reveals written only after corresponding candidate scoring, not sent to the subprocess. Shared-UID parent access still precludes a cryptographic secrecy claim.
- `secondary_controls.py` / `secondary_controls.json`: fixed, constant,
  seeded-random, and public-example-filtered symbolic baselines on the same
  uncontaminated 90 held-out functions. All host controls are marked
  **NOT PHYSICALLY CPU MATCHED** to sandboxed neural inference; symbolic
  additionally uses four proposals versus the neural budget of one.
- `training_replay.py` / `training_replay.json`: independently reruns
  each of the three initializations and all nine SGD successors inside the
  same NC-028 sandbox and matches all 12 model tensor hashes exactly (72
  gradient steps).
- `test_loop.py`: typed translation, exact proof, tampered chain, score denominator, reveal tamper, leakage interface and model-ranking controls.
- `audit_results.py` and `replay_audit.json`: candidate-side provenance and honest replay audit; these **are not an independent reviewer**. The independent reviewer must separately inspect model weights, rejection feedback, checkpoint lineage, sealed goal overlap, compute matching and architecture.

## Dependencies

- Python 3.10+ standard library; Git.
- PyTorch CPU, NumPy and system libraries already permitted by NC-028's pinned sandbox allowlist.
- Linux Landlock ABI ≥5 and `libseccomp.so.2`; POSIX process metering via `resource.RUSAGE_CHILDREN`.
- `artifacts/NC-028/isolation.py`; `artifacts/NC-009/promote.py` and `operator_promotions.jsonl`; `artifacts/NC-001/canonical_isa.py`; `artifacts/NC-061/scoreboard.py`.
- Read-only provenance of `/workspace/experiments/rlt_split_algorithms_ablation_v1/snapshots/current_unified_v3.pt`.
- **No network, external LLM/ChatGPT API, service installation, system-wide setting, global accepted-library edit, or champion write.**

## Scientific restrictions

The training library contains four **previously known** operator semantics. The MLP predicts a class among those four; it does not synthesize unseen opcodes or symbolic expressions. The host compiles generic tokens into a canonical four-bit ISA and independently verifies exhaustively over all 256 pairs. Counterexample-filtered model rankings do **not** constitute gradient-based CEGIS repair. There is no proof of mathematical invention or unrestricted generalization.

The "no-library training" control uses the exact frozen random starting model but retains the same output-token architecture; it measures effects of supervised library training, **not** an isolated causal ablation of a fully autonomous system. Retention in the scoreboard is fixed catalog preservation, not independently measured pretrained neural retention. External child CPU is reported as process `RUSAGE_CHILDREN`; NC-061's in-process cached-call CPU cannot be used as model inference cost. Tiny timing noise and shared-host load still restrict matched-budget inference. No positive improvement claim is permitted if CPU matching fails, an evaluation overlaps declared training functions, or a new program is unsupported.

A fresh rerun must occur in a **clean, separately cloned checkout** or with outputs removed by an authorized owner. Do not overwrite committed evidence or re-use formerly revealed sealed goals for a new scientific comparison.

## Reproduce and review

```bash
cd /workspace/experiments/logos-kernel
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -q -s artifacts/NC-062 -p 'test_loop.py'
PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-062/audit_results.py
PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-062/audit_clean_eval.py
PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-062/audit_clean_eval.py --full
python dag.py validate
python -m unittest -q test_dag
python -m unittest -q test_pull
```

The pilot executes one `run_loop.py` from a fresh checkout with its declared finite budgets.
The training reproduction script writes an immutable `training_replay.json` and
must only be re-run in a **fresh isolated copy**, never overwrite a pinned
checkpoint or evidence file. The full inference replay re-verifies original
model predictions from frozen checkpoints, but does not claim bit-identical
process CPU timing. An independent verifier must reproduce the three-generation evidence before recording a review result; implementer tests are not independent acceptance.
