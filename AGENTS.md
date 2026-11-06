# AGENTS.md — Logos Kernel Agent Protocol

## Mandatory entry point

You are an agent working on Logos Kernel, the neural computer learning program. **Read this file first. Then obey the canonical append-only task ledger.**

- **Repository:** `/workspace/experiments/logos-kernel`
- **Source of truth:** `instructions.jsonl` (task definitions, dependencies, status events, hash chain)
- **Only approved ledger writer:** `python dag.py ...`
- **Specification and CLI help:** `README.md`, `python dag.py --help`
- **Primary goal:** progressive small-to-big neural learning, verified reusable operations, autonomous instruction composition, one qualified recurrent-Transformer champion.

**Do not interpret the task file as permission to execute all tasks immediately.** Work only on a ready task, respect dependencies, preserve the current champion, and require independent evidence before marking completion.

## First command sequence — every new agent session

```bash
cd /workspace/experiments/logos-kernel
python dag.py validate
python -m unittest -q test_dag
python dag.py ready
python dag.py list
```

1. **STOP on validation failure.** Do not edit, truncate, reset, reorder, or regenerate `instructions.jsonl` to make validation pass. Escalate the original error and its ledger head hash.
2. Read the full `event` for your selected task ID in `instructions.jsonl`: `instruction`, `depends_on`, `acceptance`, `outputs`, and `avoid`. The list command is an index, not the full specification.
3. Choose only a `planned` task shown by `python dag.py ready`. Respect its priority and dependency order. Do not start a downstream task based on another agent's promise or an unverified PASS claim.
4. Coordinate ownership with other agents. The ledger offers an atomic status transition, **not** an external distributed scheduling or lease service. If a start is rejected because another worker took the task, select another ready task or stop.

## Claim a task

```bash
python dag.py start NC-000 --agent YOUR_AGENT_ID
python dag.py validate
```

Replace `NC-000` with an actually ready task from the current ledger; replace `YOUR_AGENT_ID` with a stable, distinct agent identity. **Never start a task already `working` or `passed`.**

- Treat the task's listed outputs as required deliverable targets; paths are not proof that a file exists.
- For work spanning other repos, record exact source repository, file paths, Git commit, and checkpoint SHA-256.
- Implement in the relevant model/experiment repository. Keep this repository focused on task instructions, status events, verifier receipts, and concise metadata.
- Keep the original champion and all qualified checkpoints immutable. New candidates belong in new paths/configs; never silently switch defaults.
- Do not bypass operator execution authorization or run unauthorized commands.

## Required implementation sequence

1. **Observe:** inventory the current state and existing work. Do not rebuild a capability already verified.
2. **Lock context:** state precise task scope, input/output contracts, architecture, numerical width, seeds, data splits, budgets, artifact paths, and acceptance criteria.
3. **Implement:** execute only the selected task's bounded instructions. Save source changes and full neural checkpoints, including model topology and weights, tokenizer/vocabulary, optimizer/RNG state when resuming training, and configuration.
4. **Verify:** run exact task acceptance tests, fresh-process checkpoint reload, held-out and adversarial inputs, regressions, and relevant ablations. Track full inputs and failures, not only summary percentages.
5. **Independent verification:** a verifier **different from the implementing agent** reruns or audits the specific candidate and produces a machine-readable evidence file tied to exact source and checkpoint hashes.
6. **Record:** write the receipt in this repository at the task's declared relative artifact path, then append a status event with `dag.py`.
7. **Review/promotion:** qualified research work is not automatically promoted into the shared RLT or default model. Promotion occurs only under the acceptance gates of the relevant DAG task.

### Verification receipt minimum

For each task, include in a JSON receipt:
- Task ID; implementing agent; independent verifier identity and finding.
- Exact source repository and commit; source-file hashes; model/config/checkpoint SHA-256.
- Exact reproduction commands, environment versions, training regime, fixed seeds, and distinct train/test splits.
- Expected versus observed results, number of inputs, exhaustive versus sampled scope, negative cases, counterexamples and limitations.
- PASS/FAIL/INCONCLUSIVE, timestamp, and whether previous capabilities were retained.
- Exact neural-versus-symbolic execution boundary, including any host-side ROM, dispatch, arithmetic or external interpreter.

**A self-authored PASS or existence of a JSON file is not independent verification.** `dag.py` checks an evidence path exists and that a verifier name is provided; it cannot authenticate the verifier, check the file content, or prove scientific correctness. Agents must enforce these stricter requirements.

## Append-only status commands

```bash
# After a working task fails an acceptance gate:
python dag.py fail NC-000 --agent YOUR_AGENT_ID --note "measured failure and receipt path"

# If a planned/working task is blocked:
python dag.py block NC-000 --agent YOUR_AGENT_ID --note "specific blocker"

# After independent verifier supplies the stated receipt file:
python dag.py pass NC-000 \
  --agent YOUR_AGENT_ID \
  --verifier INDEPENDENT_VERIFIER_ID \
  --evidence artifacts/NC-000/baseline_receipt.json

python dag.py validate
python dag.py ready
```

These are **examples of syntax**, not instructions to mark `NC-000` passed. The receipt path must already exist inside this repository. The ledger transitions are finite: `planned → working/blocked/archived`; `working → passed/failed/blocked`; `blocked/failed → working` where the validator permits; `passed` is terminal.

If task instructions need correction, do **not** edit old JSONL records. Prepare a new task event with a unique ID and dependencies referring to tasks already in the ledger, then:

```bash
python dag.py add-task --file /path/to/new_task_event.json
python dag.py validate
```

The new-task JSON must match a `kind: "task"` event found in the ledger. It must include `id`, `priority`, `title`, `depends_on`, non-empty `instruction`, `acceptance`, `outputs`, `avoid`, and `status: "planned"`. Explicitly document supersession in the new task's instructions; the current ledger does not implement deletion, cancellation, or automatic supersession.

## Non-negotiable technical boundaries

- **Primitive accuracy ≠ whole-program proof.** A 320/320 local truth-table check is not an exhaustive test of every reachable program state.
- **Sampled ≠ exhaustive.** A 4,217-case 8-bit multiplication result is not the full 65,536-pair space. Record the actual coverage and any duplicates.
- **MLP ≠ Transformer.** State which architecture actually performs arithmetic; distinguish a Transformer containing feedforward sublayers from an external arithmetic MLP specialist.
- **Execution ≠ invention.** Running a predefined shift-and-add ROM does not show the model invented multiplication. For invention, the model must propose the instruction sequence without the target program being supplied in evaluation.
- **Translation ≠ execution.** A predicted human-readable math template is not sufficient; verify that learned semantics causally affect neural execution where claimed.
- **Verified symbolic operator ≠ verified neural weights.** Record implementation mode, finite input domain, numerical semantics and proof strength.
- **Two Transformers ≠ one shared recurrent Transformer.** Actual weight sharing and joint training require independent evidence.
- **Attention ablation at inference ≠ matched training ablation.**
- **More parameters ≠ better.** Optimize verified correctness, retention, reusable task coverage and critical-path compute cost; measure FLOPs/latency and seed variance.
- No speculative AGI claim; no unmeasured arbitrary-width guarantee; no unverifiable success claims.

## Git and concurrency protocol

- `instructions.jsonl` is append-only via `dag.py`. **Never** use editors, manual `echo >>`, scripts rewriting rows, `git checkout -- instructions.jsonl`, `git reset --hard`, force-push or history rewriting.
- The writer uses advisory filesystem locking and a hash chain. This protects against ordinary concurrent CLI appends and accidental corruption, **not** malicious edits or multiple machines with incompatible filesystem locking.
- Other agents may have uncommitted work. Inspect `git status` before changes; stage only files you own. Do not commit or discard another worker's uncommitted ledger edits without coordinating serialization.
- Prefer one coordinated Git commit owner for the shared ledger. Git operations are not protected by `dag.py`'s file lock; simultaneous `git add/commit` can conflict even when JSONL appends are serialized.
- Preserve append-only history across merges. If ledger branches diverge, do not merge JSONL text mechanically; reconcile by re-appending events through `dag.py` against one chosen canonical history, with verifier review.
- Hashes and Git history provide integrity metadata, not correctness. Use a protected remote/off-host backup for stronger durability.

## Related source repositories and files

- `/workspace/experiments/latticeforge_neural_isa_test/`: prior neural ISA arithmetic, Transformer control scripts, checkpoints and evidence.
- `/workspace/experiments/rlt_split_algorithms_ablation_v1/`: unified RLT and reported qualified `rlt_attention_isa` capability. Verify current paths/configs under NC-000 before relying on them.
- Within this repository: `instructions.jsonl` (sole task ledger), `dag.py` (writer/validator), `test_dag.py` (ledger tests), `README.md` (protocol details), `artifacts/NC-XXX/` (intended task receipts).

**Dependencies:** Python 3 standard library and Git for this instruction repository. Downstream neural training depends on the external experiment code, PyTorch and its recorded training dependencies. Do not assume a GPU exists.

## Completion report format

Report the task ID, exact repository and commit/checkpoint hashes, changed files, test commands, numeric results with denominators, independent verifier identity, known failure modes, new ledger event/hash, and next *ready* task. If blocked or unverified, state that explicitly. Do not claim completion when only setup or partial implementation succeeded.

**Start at `python dag.py ready`. Follow the ledger, not conversational assumptions.**
