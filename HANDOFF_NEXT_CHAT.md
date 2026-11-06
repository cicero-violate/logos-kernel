# Logos Kernel — stateless new-chat handoff
Date: 2026-11-06 (America/Toronto)
Scope: construction DAG and mathematical-discovery research. This is an informational snapshot, **not** a new DAG event, claim, verifier approval, or evidence of an active worker.

## Canonical source of truth

- Repository: `/workspace/experiments/logos-kernel/`
- Protocol: `AGENTS.md`; architecture/context: `README.md`
- Immutable event history: `instructions.jsonl`; **only** `python dag.py` may append ledger events. Never hand-edit or reconstruct JSONL.
- At this audit: **55 tasks, 91 events, 11 passed, 44 planned**; SHA-256 ledger head `6617ef655e7f182f6456a29b98f0e5d8f53135195c0c4ed3914ddf5a026d10e6`. Main Git commit before this handoff: `4d4aafd75ac0` (DAG event 91).
- Audit: `python dag.py validate` PASS; `python -m unittest -q test_dag` **19/19 PASS**.
- Only known exclusive claim: `NC-028`, real owner `1db9edb983ff4df9933612e7c826e55f`. **Do not take over or force-release.** A claim alone does not establish running progress.
- Eight unclaimed ready tasks at audit: `NC-044` (rank 2, P0), `NC-048` (3, P0), `NC-008` (12, P0), `NC-023` (14, P0), `NC-029` (23, P0), `NC-021` (27, P1), `NC-016` (41, P3), `NC-018` (43, P3). Recompute readiness on every new tab.
- There are 16 **pre-existing** staged/untracked files under `artifacts/NC-000`, `NC-001`, `NC-005`, `NC-006`, `NC-019`. They belong to other work; do not stage all, reset, delete, or claim them. Some historical PASS evidence (especially NC-000) lives only in the original checkout and is not yet part of Git HEAD, so isolated worktrees may not pass `dag.py validate` unless the authorized owner resolves that.

## Major construction tracks and actual state

1. **Discovery sandbox:** `NC-028`, rank 1, P0, **claimed**. Preserve owner; do not infer completed autonomous deployment.
2. **Separate candidate trainer:** `NC-044` rank 2 isolation **READY**; `NC-045` rank 4 separate append-only training JSONL / stateless tab recovery **BLOCKED on NC-044**; `NC-046` rank 9 bounded CPU training worker **BLOCKED**; `NC-047` rank 11 independent champion-promotion gate **BLOCKED**. No dedicated trainer running or distinct training JSONL verified created by this audit.
3. **Algorithmic automated teacher / massive bootstrap:** `NC-048` rank 3 research-source/reference audit **READY**; `NC-049` rank 5 proof-backed synthetic mathematics; `NC-050` rank 6 adaptive TSCL/ALP-GMM and prerequisites; `NC-051` rank 7 counterexample/CEGIS/adversarial lessons; `NC-052` rank 8 DreamCoder/Stitch-style abstraction library; `NC-053` rank 10 teacher–trainer ablations; `NC-054` rank 22 later meta-teacher. Tasks after NC-048 are dependency-blocked; references are specified, not necessarily downloaded.
4. **Mathematical discovery:** `NC-008` rank 12 neural operator invention **READY**; `NC-023` rank 14 formal rewrite/equality-saturation baseline **READY**; `NC-024`–`NC-026` dependency-blocked search policy and verified reusable abstractions; `NC-040`–`NC-043` dependency-blocked self-generated math questions, interestingness/utility, bounded research loop and strategy meta-learning.
5. **Independent Logos Autonomy project:** `/workspace/experiments/logos-autonomy/` is a separate local CPU discovery-runtime repo and ledger, not the trainer ledger. Follow its own `AGENTS.md`; never conflate ownership.

## User's newest scientific hypothesis — NOT YET A DEDICATED LEDGER TASK

**Find stable laws beneath change.** Search for the **lowest sufficient order**, not the highest arbitrary finite difference, at which a recurrent relation or invariant explains observations across unseen sequences/domains.

- `Δx_t = x_{t+1} - x_t`; `Δ^{k+1}x_t = Δ^kx_{t+1} - Δ^kx_t`.
- For polynomial sequences, constant `Δ^k` identifies a degree-at-most-`k` structure under exact uniform sampling; do not generalize this to all dynamics.
- Periodic and nonlinear systems may have stable recurrence **operators** even when no low-order finite difference becomes constant (e.g. `x_{t+2}=2cos(ω)x_{t+1}-x_t`).
- Candidate discovery objective: jointly minimize **held-out prediction error, law description length, invariance violation and total CPU cost**, subject to mathematical validity, no leakage and prior-skill retention. Do not minimize raw delta alone: constant-output collapse, noisy high-order differencing, interpolation of finite samples and nonphysical smoothness are failure modes.
- Preference: a *small repeatedly applied shared nonlinear/recurrent kernel*; dot-product composition plus nonlinearities/state; compress many representations into reusable stable patterns; measure genuine generalization and reuse per compute unit.
- **No claim** of AGI, aesthetic experience or spontaneous new theorem without independent proof. Preserve user time: avoid tangent methods that do not answer exact hypothesis. Do not add an NC-055 task or reprioritize merely because this handoff describes the idea; request/obey explicit user instruction before altering DAG.

## Prior natural-scene visual pilot — negative controls

- Separate worktree `/workspace/experiments/logos-temporal-beauty/`.
- Branch `experiment/global-temporal-vision-v2`, latest Git commit `73a231d80956`.
- Report `EXPERIMENT_REPORT_GLOBAL_V2.md` and metrics `outputs/global_temporal_v2/metrics.json`. Original videos stay out of Git; SHA-256 manifests and licensed provenance recorded.
- 12,339-parameter 64-token global-attention + 3-frame GRU predictor trained on solar and ocean footage, evaluated on held-out clouds/waterfall. **NO_GAIN** from global attention or temporal history vs matched controls. 7/7 experiment tests PASS; checkpoint recomputation PASS; independent scientific verification **not** performed.
- User's research goal is **mathematical understanding, abstraction, invariant discovery, creative conjecture**, not visual aesthetics or multimodal demos. Treat vision as optional evidence for learning stable laws, not as an objective by itself.
- Immutable original RLT champion: `/workspace/experiments/rlt_split_algorithms_ablation_v1/snapshots/current_unified_v3.pt`, SHA-256 `4c42d13fd46f57ac0db6eccbaef35e99c4cefcec393a3a3a4faf146645e8f683`. Do not edit, retrain in place, or promote candidates without independent verifier and explicit approval.

## New stateless tab: instructions

1. Activate `$agent-event-lab`. Use disk/Git, **not** this snapshot as authority. Read `AGENTS.md`, `README.md`, `HANDOFF_NEXT_CHAT.md`, `instructions.jsonl` and the exact selected task record.
2. Execute in Logos Kernel: `python dag.py validate`; `python -m unittest -q test_dag`; `python dag.py ready`; `python dag.py list`. Stop/report if any check fails. Inspect Git and running sessions/claim ownership.
3. If asked to advance construction, select the **highest-ranked eligible unclaimed** task (expected `NC-044`, or `NC-048` if trainer work is already owned). Claim only with the **actual assigned agent identity**, never invented identity. Preserve `NC-028`.
4. Implement in isolated candidate branch/worktree; log source files and code dependencies, data hashes, exact seed/budgets, negative controls, complete optimizer/RNG state when needed; never touch champion. Request a genuinely different agent for independent verification.
5. If asked to pursue the delta-invariant hypothesis, first inventory existing task overlap and then define falsifiable matched baselines (polynomial vs periodic vs nonlinear/noisy/mixed dynamics, trivial constant predictor, memorization, finite-difference order overfit) with held-out transfer. Append new tasks **only on explicit user instruction** via `dag.py add-task` and complete `priority-plan` snapshot.

## Copy/paste starter prompt for a fresh ChatGPT tab

```text
$agent-event-lab
Project: /workspace/experiments/logos-kernel/
This tab is stateless. Recover all context from disk and Git.
Read AGENTS.md, README.md, HANDOFF_NEXT_CHAT.md and instructions.jsonl.
Validate DAG/hash chain, tests, Git status, owner claims, task readiness and independent evidence.
Do not overwrite champion or pre-existing artifacts. Don't invent agent identities.
Explain the first eligible task and any blockers. If authorized to implement, claim only with the real runtime identity.
Scientific direction: smallest stable mathematical laws beneath changing observations; recurrent abstraction and verified out-of-sample transfer. Do not confuse small delta or smoothness with correctness.
Report exact changes, commits, tests, independent verification and remaining limitations.
```

## File and tooling dependencies

- Construction DAG and this handoff: Python 3 standard library, Git, and `dag.py` / `test_dag.py`; no Python model packages required for DAG checks.
- Training/visual pilots: their own scripts and checkpoints plus PyTorch, NumPy, FFmpeg, and hash-verified data; inspect each project's own README before execution.
- This handoff itself does not install packages, start background work, create a third agent, or schedule research.
