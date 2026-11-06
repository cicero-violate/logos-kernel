Activate [$agent-event-lab](app://asdk_app_6ae1a9c0a9648191b604b2003bcffefa)

Project: /workspace/experiments/logos-kernel
Topology: exactly 3 independent stateless ChatGPT tabs for THIS repo; 6 total across two isolated repositories. Each tab recovers state and pulls dynamically; roles are never fixed. Each tab must use its own actual Event Lab session identity.
This turn is STATELESS. Recover from AGENTS.md, README.md, HANDOFF_NEXT_CHAT.md, TAB_PROMPT.md, instructions.jsonl, current Git and artifacts.

Use the Event Lab runtime's REAL agent/session identity. Never invent one.
RUN: python dag.py validate
RUN: python -m unittest -q test_dag
RUN: python -m unittest -q test_pull
RUN: python dag.py pull --agent ACTUAL_AGENT_ID

OBEY THE PULLED ROLE. Do not choose the role yourself.

VERIFIER: Independently audit the returned task's pinned candidate, all referenced source/checkpoint hashes, exact acceptance conditions, fresh negative controls and proof scope. Produce a real receipt at its declared output path, then run python dag.py review-result TASK_ID --reviewer ACTUAL_AGENT_ID --verdict PASS|FAIL|HOLD [--evidence DECLARED_RECEIPT]. Do not write/revise the candidate, impersonate a verifier, claim the implementation, or record the owner's PASS.

IMPLEMENTER: Implement the pulled task, commit only owned code/tests and reproducible evidence, and publish python dag.py request-review TASK_ID --owner ACTUAL_AGENT_ID --candidate-path artifacts/TASK_ID/MANIFEST.json. Do not approve your own work. The owner can record final PASS only after a distinct reviewer recorded PASS for that candidate.

HOLD: Report the exact missing eligible ticket, candidate, independent identity, authority or dependency. Do not bypass, force-release, overwrite, or loop on self-checks.

Pull is an atomic JSONL role/claim assignment, not an automatic worker. A new tab does not mean a different identity. Preserve all other workers' files and the champion. Do real authorized work now.

Report: role | actual session | task | code/evidence hashes | exact tests | reviewer receipt | verdict | next action.
