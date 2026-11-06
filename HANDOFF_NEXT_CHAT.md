# Logos Kernel — stateless new-chat handoff
Date: 2026-11-06 (America/Toronto), updated after NC-055–060 (events 92–98)
Scope: construction DAG and mathematical-discovery research. This is an informational snapshot, **not** a new DAG event, claim, verifier approval, or evidence of an active worker.

## Canonical source of truth

- Repository: `/workspace/experiments/logos-kernel/`
- Protocol: `AGENTS.md`; architecture/context: `README.md`
- Immutable event history: `instructions.jsonl`; **only** `python dag.py` may append ledger events. Never hand-edit or reconstruct JSONL.
- At this audit: **61 tasks, 98 events, 11 passed, 50 planned**; SHA-256 ledger head `5e692f3c216a6f1b926a26d91b811112c04f9a0274353c0465995f2b45d424ec`. Current DAG priority snapshot: `abb1f13d4e8f` (DAG event 98).
- Audit: `python dag.py validate` PASS; `python -m unittest -q test_dag` **19/19 PASS**.
- Only known exclusive claim: `NC-028`, real owner `1db9edb983ff4df9933612e7c826e55f`. **Do not take over or force-release.** A claim alone does not establish running progress.
- Nine unclaimed ready tasks at audit: `NC-044` (rank 2, P0), `NC-048` (rank 3, P0), `NC-055` (rank 4, P0), `NC-008` (rank 16, P0), `NC-023` (rank 18, P0), `NC-029` (rank 27, P0), `NC-021` (rank 33, P1), `NC-016` (rank 47, P3), `NC-018` (rank 49, P3). Recompute readiness on every new tab.
- There are 16 **pre-existing** staged/untracked files under `artifacts/NC-000`, `NC-001`, `NC-005`, `NC-006`, `NC-019`. They belong to other work; do not stage all, reset, delete, or claim them. Some historical PASS evidence (especially NC-000) lives only in the original checkout and is not yet part of Git HEAD, so isolated worktrees may not pass `dag.py validate` unless the authorized owner resolves that.

## Major construction tracks and actual state

1. **Discovery sandbox:** `NC-028`, rank 1, P0, **claimed**. Preserve owner; do not infer completed autonomous deployment.
2. **Separate candidate trainer:** `NC-044` rank 2 isolation **READY**; `NC-045` rank 5 separate append-only training JSONL / stateless tab recovery **BLOCKED on NC-044**; `NC-046` rank 13 bounded CPU training worker **BLOCKED**; `NC-047` rank 15 independent champion-promotion gate **BLOCKED**. No dedicated trainer running or distinct training JSONL verified created by this audit.
3. **Algorithmic automated teacher / massive bootstrap:** `NC-048` rank 3 research-source/reference audit **READY**; `NC-049` rank 9 proof-backed synthetic mathematics; `NC-050` rank 10 adaptive TSCL/ALP-GMM and prerequisites; `NC-051` rank 11 counterexample/CEGIS/adversarial lessons; `NC-052` rank 12 DreamCoder/Stitch-style abstraction library; `NC-053` rank 14 teacher–trainer ablations; `NC-054` rank 26 later meta-teacher. Tasks after NC-048 are dependency-blocked; references are specified, not necessarily downloaded.
4. **Mathematical discovery:** `NC-008` rank 16 neural operator invention **READY**; `NC-023` rank 18 formal rewrite/equality-saturation baseline **READY**; `NC-024`–`NC-026` dependency-blocked search policy and verified reusable abstractions; `NC-040`–`NC-043` dependency-blocked self-generated math questions, interestingness/utility, bounded research loop and strategy meta-learning.
5. **Independent Logos Autonomy project:** `/workspace/experiments/logos-autonomy/` is a separate local CPU discovery-runtime repo and ledger, not the trainer ledger. Follow its own `AGENTS.md`; never conflate ownership.

## Operator-authorized invariant-first discovery — now in the DAG

The user explicitly authorized the six proposed invariant-first tasks. They are
already appended via dag.py and priority-ranked. DO NOT duplicate them.

- `NC-055`, rank 4, P0, READY: Falsifiable mathematical invariant benchmark; deps NC-019/NC-020 already PASS.
- `NC-056`, rank 6, P0, BLOCKED: Find minimum sufficient finite-difference order or recurrence law; deps NC-055/NC-020.
- `NC-057`, rank 7, P0, BLOCKED: Search optimized for verified stability, compression, heldout transfer per CPU; deps NC-055/056.
- `NC-058`, rank 8, P0, BLOCKED: Shared nonlinear recurrent kernel, weight/depth controls and matched CPU; deps NC-055/056/022.
- `NC-059`, rank 29, P1, BLOCKED: Transfer verified invariants among representations; deps NC-055/056/030/058.
- `NC-060`, rank 30, P1, BLOCKED: Automated teacher of increasingly higher-order invariants; deps NC-055/056/057/050/053/059.

Scientific hypothesis: find stable laws beneath change. Search for the LOWEST
sufficient order, not the highest arbitrary finite difference, at which a
recurrence or invariant explains observations across unseen domains.

- Delta: Δx_t = x_(t+1) - x_t, and Δ^(k+1)x_t = Δ^kx_(t+1) - Δ^kx_t.
- Polynomial streams can have constant finite difference. Oscillatory,
  discontinuous, chaotic or piecewise systems need stable recurrence operators
  even when no small finite difference stays constant.
- Proposed objective: balance heldout error, law description length, genuine
  invariance and total CPU while preserving correctness/retention as hard gates.
- Control for constant-state collapse, noise amplification, short-sequence
  high-order interpolation and unearned proof from finite observations.
- Hypothesis requires multi-seed matched-CPU experimental verification.
  Never claim AGI or mathematical discovery merely from smaller deltas.
- No task is claimed or implemented by adding its instruction record.
- Preserve original champion; all new work requires actual agent ownership
  and independent verification for PASS.

## Prior natural-scene visual pilot — negative controls

- Separate worktree `/workspace/experiments/logos-temporal-beauty/`.
- Branch `experiment/global-temporal-vision-v2`, latest Git commit `73a231d80956`.
- Report `EXPERIMENT_REPORT_GLOBAL_V2.md` and metrics `outputs/global_temporal_v2/metrics.json`. Original videos stay out of Git; SHA-256 manifests and licensed provenance recorded.
- 12,339-parameter 64-token global-attention + 3-frame GRU predictor trained on solar and ocean footage, evaluated on held-out clouds/waterfall. **NO_GAIN** from global attention or temporal history vs matched controls. 7/7 experiment tests PASS; checkpoint recomputation PASS; independent scientific verification **not** performed.
- User's research goal is **mathematical understanding, abstraction, invariant discovery, creative conjecture**, not visual aesthetics or multimodal demos. Treat vision as optional evidence for learning stable laws, not as an objective by itself.
- Immutable original RLT champion: `/workspace/experiments/rlt_split_algorithms_ablation_v1/snapshots/current_unified_v3.pt`, SHA-256 `4c42d13fd46f57ac0db6eccbaef35e99c4cefcec393a3a3a4faf146645e8f683`. Do not edit, retrain in place, or promote candidates without independent verifier and explicit approval.

## New stateless tab: verifier-first instructions

This is only a snapshot. The active ledger, candidate hashes and runtime identity must be rechecked every turn; earlier task rankings below may now be stale.

1. Activate Event Lab. Read AGENTS.md, README.md, TAB_PROMPT.md, HANDOFF_NEXT_CHAT.md, instructions.jsonl, then the exact task's contract and artifacts.
2. Run `python dag.py validate`; `python -m unittest -q test_dag`; `python dag.py list`; `python dag.py ready`; inspect Git, owner identities and unchanged champion.
3. **Search for pending independent audits across all non-passed tasks first** (including claimed/blocked). A frozen candidate awaiting a truly separate verifier takes precedence over claiming a new task. `dag.py ready` is implementation-only.
4. If reviewer is genuinely independent, act as VERIFIER without changing implementer ownership. Reproduce and attack the pinned revision; place the attributed receipt at the task's declared output path. Reviewer does not append PASS. If not independent, report HOLD and optionally implement a **different** eligible ready task.
5. If no audit eligible for this real session, resume its legitimately owned task or claim the first ready unclaimed task. Never impersonate owners, release their claims, overwrite pre-existing artifacts or promote a champion.
6. For invariant discovery, NC-055–NC-060 remain a dependency-ordered research track, not proof of completed discovery.

## Copy/paste static prompt for a fresh ChatGPT tab

See `TAB_PROMPT.md` for the maintained full prompt. Its role dispatch is **VERIFIER FIRST → IMPLEMENTER SECOND → HOLD**. Do not use an older prompt that starts by claiming the highest ready task.

```text
$agent-event-lab
Project: /workspace/experiments/logos-kernel/
Stateless: recover all context from disk/Git every turn.
Read AGENTS.md, README.md, TAB_PROMPT.md, HANDOFF_NEXT_CHAT.md, instructions.jsonl.
Validate DAG and tests; inspect Git, claims, results and actual runtime identity.
FIRST find frozen candidates awaiting independent audit across CLAIMED/BLOCKED/PLANNED tasks; dag.py ready only shows implementation candidates.
If truly independent, act as VERIFIER and reproduce the exact candidate; write an attributed receipt at the declared task path. Do not claim or approve your own code.
Otherwise resume a legitimately owned task or claim the highest dependency-ready unclaimed task as IMPLEMENTER.
If neither role legitimately possible, HOLD with exact blocker; never fabricate verifier identities or release other owners.
Preserve RLT champion, existing artifacts and ledger; report exact changed files, commit, tests and proof scope.
```

## File and tooling dependencies

- Construction DAG and this handoff: Python 3 standard library, Git, and `dag.py` / `test_dag.py`; no Python model packages required for DAG checks.
- Training/visual pilots: their own scripts and checkpoints plus PyTorch, NumPy, FFmpeg, and hash-verified data; inspect each project's own README before execution.
- This handoff itself does not install packages, start background work, create a third agent, or schedule research.
