# NC-005 — progressive curriculum with explicit retention

Model: a new 5,981-parameter expanded neural ISA meaning-to-bit MLP and a neural 16-bit zero-predicate head. Each seed warm-starts five operations from qualified NC-004 coupled checkpoints. The expanded network adds four local opcodes AND, XOR, SUB, SAR; the complete trained local set is ADD, SHL, SHR, AND1, OR, AND, XOR, SUB, SAR. This is NOT the selected 25,089-param RLT Transformer. It does not overwrite or promote the champion.

## Seven stages

| Stage | Trained and evaluated capability | Boundary |
|---|---|---|
| BOOLEAN | Boolean bit transitions AND1 OR AND XOR | All local patterns are training-seen |
| CARRY_ADD | Local carry and 16-bit ADD | Host iterates bit positions |
| SHIFTS | SHL SHR and signed SAR | Host performs bit placement |
| COMPARE_SUB | SUB via learned result+no-borrow carry | Host interprets final borrow as unsigned comparison |
| PC_BRANCH | Learned is-zero predicate | Host supplies opcode, PC target, jump inversion and HALT |
| SEQUENCES | Training on actual intermediate bit-state traces from 3 supplied two-op programs | Program choice and routing remain host-defined |
| LOOPS | Training on actual intermediate bit-state traces from a fixed shift-and-add multiplication loop | Loop control and ROM are host-provided |

Matched three seeds: 11, 19 and 29. Two arms: rehearsal (replays previous qualified truth-table rows with current examples) and new_only (current-stage training examples without old-task replay). Per-seed initial tensors, optimizer type and learning rate, batch size, stage order, stage-local sampling seeds, forward/backward paths and step budgets are matched. The PC_BRANCH stage trains the neural zero predicate only for both arms; the earlier committed implementation had an asymmetric extra arithmetic update and has been superseded without rewriting Git history. The difference in example exposure is intentional: the curricula are not sample-identical. New optimizer moments are initialized after expanding the NC-004 model; this is NOT exact continuation of its old Adam history.

## Files and dependencies

Training: curriculum.py and trace_training.py; Python 3, PyTorch, NumPy, NC-004 causal_semantics.py, its original coupled_seed11/19/29.pt weights, NC-001 canonical_isa.py, original standalone train_causal.py, NC-000 inventory.json, frozen RLT snapshots/current_unified_v3.pt for hash validation.

Evidence: training_report.json, curriculum_receipts.jsonl (42 model/stage rows); checkpoints/[arm]_seed[seed]_stage[00..06].pt (42 full tensors/optimizer/RNG packages); analyze_retention.py and retention_matrix.json; verify_fresh.py and fresh_reload.json; audit_budget.py and compute_budget.json; test_curriculum.py; pilot/ with 42 archived earlier checkpoint files whose sequence/loop phases replayed truth tables only and are NOT accepted as final trajectory-trained evidence.

Rehearsal and fresh-only comparisons use the same generated source pools and matched update counts but not identical training sample exposure. In-domain local truth tables are complete (64 patterns/op). Prior five NC-004 operations are pretrained on all their local patterns. New sequence and loop stages use HOST-SUPPLIED instruction traces for training. Two held-out program-order tests are subtract_then_xor and xor_then_double, but all underlying bit rules are supervised. Training loop operand examples are 0..31; 8-bit evaluation samples are wider; no pristine data-disjoint proof for the old NC-004 source pretraining.

## Commands

    cd /workspace/experiments/logos-kernel
    python dag.py validate
    python -m unittest -q test_dag
    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -q -s artifacts/NC-005 -p test_curriculum.py
    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-005/analyze_retention.py
    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-005/verify_fresh.py

Training source command (DO NOT rerun after evidence verification; replaces task-local checkpoints):

    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-005/curriculum.py train

## Acceptance and limitations

Each rehearsal stage must retain every prior local bit pattern and meaning label, and must pass the relevant declared heldout arithmetic, learned zero-predicate, supplied two-op sequence and fixed-loop evaluations. Any observed regression is recorded, never silently hidden. The new-only arm is an unqualified forgetting control; failures of that arm do not establish a successful universal optimizer.

Numeric bit transitions and zero predicate are NEURAL. The fixed program ROM, opcode selection, PC-target updates, register routing, flags, HALT, and reference oracle are HOST mechanisms. These experiments do not establish autonomous instruction composition, all 15 canonical ISA opcodes, a Transformer reasoning breakthrough, or general AGI. Any exhaustive test of a 16-bit zero predicate validates that predicate only, not all machine states or loops.

Independent scientific audit by a distinct real agent is required before recording DAG PASS. No champion/default promotion is authorized.

Variables: W=16 bits; s=training stage; P_s=previously qualified local operations, each with 64 bit-pattern inputs; R_s=sum over P_s correct / (64*|P_s|). Stage retention gate requires R_s=1; other heldout accuracy gates are independent.
