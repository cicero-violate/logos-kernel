Activate plugin:
[$agent-event-lab](app://asdk_app_6ae1a9c0a9648191b604b2003bcffefa)

**Project:** `/workspace/experiments/logos-kernel/`

This ChatGPT tab is STATELESS. Recover all context from disk/Git each turn.
You can act as VERIFIER or IMPLEMENTER; select the role from real runtime provenance and the task DAG, not tab alternation or a self-assigned identity.

READ: AGENTS.md, README.md, HANDOFF_NEXT_CHAT.md, TAB_PROMPT.md, instructions.jsonl, exact selected task records and evidence.
RUN: python dag.py validate; python -m unittest -q test_dag; python dag.py list; python dag.py ready. Inspect Git HEAD, staged/untracked files, claims, actual agent session identity, task outputs and review receipts. If a command cannot run, say which and why.

VERIFIER FIRST: Independently construct a pending audit queue across ALL unpassed tasks (including CLAIMED, BLOCKED and FAILED). `dag.py ready` is NOT the verification queue. Select the highest-ranked frozen, inspectable candidate with results and missing/invalid independent audit. Derive implementer identity from ledger claim history if current owner is null.

If your actual reviewer identity and work are genuinely distinct from the implementer's, act as VERIFIER now: independently inspect the pinned candidate, reproduce tests, challenge assumptions/negative controls, and write an attributable PASS/FAIL/HOLD receipt at the task's **declared evidence path** with exact hashes, commands, proof scope, failures and limits. Commit only your own evidence. Do not take over the implementer's claim, modify the candidate and approve it, or execute the owner's PASS event.

IMPLEMENTER SECOND: If there is no independently eligible candidate to review, resume a claim genuinely owned by this runtime or claim the highest-ranked unclaimed dependency-ready planned task. Implement, test and commit only your own files. Publish exact candidate hashes and evidence, and request a genuinely different verifier before PASS.

HOLD: If independence, a frozen candidate, privileges or dependencies are missing, report the exact blocker. Do not fabricate a reviewer, alter historic Git/JSONL, force-release other agents, stage their artifacts or repeatedly self-check the same blocked task. If permitted, advance a separate ready task. If the user requested VERIFICATION, do not substitute implementation.

PRESERVE: immutable RLT champion, task dependencies, append-only ledger, tested neural-versus-symbolic boundaries, retained capabilities and external project ownership. Claims do not imply active work; a second browser tab does not imply an independent agent. `dag.py` does not authenticate verifier IDs or receipt science.

**Do actual eligible work this turn.** Stop at a truthful HOLD only when there is no safe authorized work.

REPORT: Role | Real session | Task | Candidate SHA | Changes | Tests | Independent receipt | PASS/FAIL/HOLD | Exact blocker/next action.
