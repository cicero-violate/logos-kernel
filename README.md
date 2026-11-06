# Neural Computer - append-only task DAG

Canonical path: /workspace/experiments/neural_computer_dag_v1/

## Source of truth

- instructions.jsonl: hash-chained append-only event log, one JSON object per line.
- Task events are immutable DAG nodes; status events append transitions without rewriting nodes.
- dag.py: standard-library CLI; flock, O_APPEND behavior, fsync, SHA-256 hash validation and dependency gates.
- The task DAG is acyclic. Runtime instruction control-flow graphs may have loops.
- Agents must use the CLI rather than editing the JSONL.

## Agent workflow

    cd /workspace/experiments/neural_computer_dag_v1
    python dag.py validate
    python dag.py ready
    python dag.py list
    python dag.py start NC-000 --agent worker-1
    # After an independent verifier has created the cited evidence file:
    python dag.py pass NC-000 --agent worker-1 --verifier verifier-2 --evidence artifacts/NC-000/baseline_receipt.json
    python dag.py block NC-000 --agent worker-1 --note missing_dependency
    python dag.py fail NC-000 --agent worker-1 --note acceptance_failed
    python dag.py add-task --file new_task.json

Tasks are not launched by the JSONL ledger. Agents should only take ready tasks; record actual results with append-only status events, commit the ledger after events, and attach verified receipts. To correct instructions, append a new task that supersedes the old task instead of editing history.

## Non-negotiable boundaries

1. Freeze original champion and qualified ISA checkpoints before changing anything.
2. Save topology, complete weights, optimizer, vocab, seeds, architecture, hashes and reload receipts; individual neuron values alone are insufficient.
3. Distinguish neural arithmetic, Transformer controller, fixed ROM, symbolic verifier and genuinely generated programs.
4. Local 320/320, exhaustive four-bit 256/256, sampled 8-bit 4,217 cases and exhaustive eight-bit 65,536 cases are different levels of evidence.
5. Independent verification, counterexamples and all older-skill retention gates precede any promotion.
6. Train small-to-big: primitives, math semantics, composition, invented operations, generated program CFG, observer feedback, shared recurrent Transformer, architecture optimization.
7. Preserve source-checkpoint provenance and require matched data/compute ablations.
8. No silent default champion switch; verify rollback.

## Related repositories

- /workspace/experiments/latticeforge_neural_isa_test/ : standalone neural ISA experiment.
- /workspace/experiments/rlt_split_algorithms_ablation_v1/ : current RLT integration and reported rlt_attention_isa specialist.
- Verify current source paths in NC-000. These are references only; not copied code or Git submodules.

## Dependencies and limitations

Instruction repo dependencies: Python 3 standard library and Git. Neural training code lives in external projects and depends on their own model/data dependencies including PyTorch.
Append-only hash chains and Git detect accidental alteration but not malicious full-chain rewrites by privileged writers. A protected offsite remote and signed commits provide stronger integrity. Hash integrity does not prove correctness. Passing tasks requires independent verifier evidence.
