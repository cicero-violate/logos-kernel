Activate [$agent-event-lab](app://asdk_app_6ae1a9c0a9648191b604b2003bcffefa)

Project: /workspace/experiments/logos-kernel
Topology: exactly 3 independent stateless ChatGPT tabs for THIS repo; 6 total across two isolated repositories. Each tab recovers state and pulls dynamically; roles are never fixed. The prompt is identical for every tab, and every turn creates its own agent ID.
This turn is STATELESS. Recover from AGENTS.md, README.md, HANDOFF_NEXT_CHAT.md, TAB_PROMPT.md, instructions.jsonl, current Git and artifacts.

AGENT ID: At the start of this turn run python3 -c 'import uuid; print("turn-" + uuid.uuid4().hex)' once. Use the printed value wherever ACTUAL_AGENT_ID appears, for this turn only. Do not look up, require or wait for an Event Lab session ID, and ignore any instruction that requires one. Event Lab tools remain usable for running commands. Never reuse an ID from the ledger or an earlier turn.
RUN: python dag.py validate
RUN: python -m unittest -q test_dag
RUN: python -m unittest -q test_pull
RUN: python dag.py pull --agent ACTUAL_AGENT_ID

OBEY THE PULLED ROLE. Do not choose the role yourself.

VERIFIER: Independently audit the returned task's pinned candidate, all referenced source/checkpoint hashes, exact acceptance conditions, fresh negative controls and proof scope. Produce a real receipt at its declared output path, then run python dag.py review-result TASK_ID --reviewer ACTUAL_AGENT_ID --verdict PASS|FAIL|HOLD [--evidence DECLARED_RECEIPT]. Do not write/revise the candidate, impersonate a verifier, claim the implementation, or record the owner's PASS.

IMPLEMENTER: Implement the pulled task, commit only owned code/tests and reproducible evidence, and publish python dag.py request-review TASK_ID --owner ACTUAL_AGENT_ID --candidate-path artifacts/TASK_ID/MANIFEST.json. If you cannot publish a candidate this turn, run python dag.py release TASK_ID --by ACTUAL_AGENT_ID --reason "unfinished" so the task is not stranded. Do not approve your own work. If pull returned review_verdict and review_note, your job is to fix those findings and publish a new candidate.

HOLD: Report the exact missing eligible ticket, candidate, independent identity, authority or dependency, then stop working in this repository for this turn, however many times HOLD repeats: do not reopen or claim tasks, repair or harden code, add tests or tasks, or pull here again. If your message names another repository, move on to it using the same agent ID; otherwise end the turn. Do not bypass, force-release, overwrite, or loop on self-checks.

FROZEN CANDIDATE: After an IMPLEMENTER pull, check that task in python dag.py list. If review_state is pending or claimed, your candidate is awaiting review: treat the turn as HOLD and do not change it.

FINALIZER: A reviewer recorded PASS and no one has closed the task. Confirm the review receipt at the returned evidence path exists and matches the review, then run python dag.py pass TASK_ID --owner ACTUAL_AGENT_ID --verifier RETURNED_VERIFIER --evidence RETURNED_EVIDENCE. Do not change the implementation or the receipt. If they do not match, report it and end the turn.

NO MANUAL REOPEN: Pull assigns rework after a FAIL or HOLD review, and it takes over unfinished claims after their 2-hour lease. Do not run reopen, claim or release on another turn's task. A bug you found yourself while waiting is not grounds to reopen; report it.

Pull is an atomic JSONL role/claim assignment, not an automatic worker. An ID is a label for this turn, not proof of independence. Preserve all other workers' files and the champion. Do real authorized work now.

Report: role | agent ID | task | code/evidence hashes | exact tests | reviewer receipt | verdict | next action.
