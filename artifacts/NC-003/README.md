# NC-003 — independent frozen neural ISA baseline verification

## Results and gates

This task audits the existing **single shared-attention RLT ISA specialist**, not the seven-operation general RLT default. Two historical arms have three saved model seeds each.

| Frozen checkpoint | seed 11 | seed 19 | seed 29 |
|---|---:|---:|---:|
| jointly_tuned local primitive / 320 | 320 | 320 | 320 |
| jointly_tuned learned meaning / 320 | 320 | 320 | 320 |
| jointly_tuned exhaustive 4-bit MUL / 256 | 256 | 256 | 256 |
| jointly_tuned sampled 8-bit MUL / 4217 | 4217 | 4217 | 4217 |
| frozen_shared exhaustive 4-bit MUL / 256 | 256 | 256 | 256 |
| frozen_shared sampled 8-bit MUL / 4217 | 4217 | **4216** | 4217 |

The **frozen_shared seed 19 fails** on 177 × 77: predicted 19293, reference 13629, HALT true. The failed control arm remains explicitly disqualified. The qualified experimental arm is jointly_tuned; no promotion to seven-operation RLT default.

The 4217-case set contains 4096 deterministic pseudorandom draws + 121 edge cross-products, with 133 duplicate evaluations, hence **4084 unique pairs**. Coverage is only 4084/65536 = approximately 6.23% of 8-bit operand space. The operand pairs are identical across seeds, not three independent test sets.

## Files and dependencies

- replay_independent.py: distinct implementation of frozen SharedRLTISA model loading, primitive truth table, predicted-opcode/PC/host register loop, both arms and three seeds; writes verification.json.
- verification.json: per-arm/seed exact results, failure cases, source/weights hashes, HALT counts, branch-state and PC agreement, arithmetic and PC interventions. Implementer evidence, NOT a separate-agent receipt.
- host_leakage_audit.py: source inspection, evaluation duplicates, teacher-supervision boundaries and exhaustive plan; writes host_leakage_audit.json.
- test_replay.py: synthetic host-control correctness tests and saved-result invariants. Synthetic oracle stub is not evidence of trained neural behavior.
- independent_receipt.json: reserved for a real distinct agent to generate only after independent review.

Runtime dependencies: Python 3 standard library, PyTorch; replay_independent.py imports original frozen neural_isa_transfer_v1/unified_shared_v1/model.py, rlt_micro_v2.py, rlt_micro.py, rlt_core.py, task_conditioned_rlt.py, source seven-task pretrained weights, seed11/19/29 native_v2 weights and both arms' seed11/19/29 saved checkpoints. NC-000 inventory pins selected champion. Original RLT and standalone ISA repos/checkpoints are never written.

host_leakage_audit.py additionally reads original training code, teacher-state generator, historical production runtime and historical verifier for audit; NONE of those trainers/verifiers are imported by the new frozen inference loop.

## Reproduce

    cd /workspace/experiments/logos-kernel
    python dag.py validate
    python -m unittest -q test_dag
    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-003/host_leakage_audit.py
    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-003/replay_independent.py
    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -q -s artifacts/NC-003 -p test_replay.py

## Mechanism and limitations

SharedRLTISA.forward consumes [fixed ROM, X/Y/R/T registers, PC query] to predict opcode, next PC and HALT. The same causal attention encoder is used for six local bit features and opcode query to produce numeric bit outputs. A **Python host** dispatches decoded neural opcode to learned bit transition, performs bit indexing/register routing, and supplies the fixed 8-op shift-and-add ROM. Reference multiplication is computed only AFTER neural outputs for score comparison, not substituted into inference.

Training used all 320 local truth patterns and teacher-forced controller states. Original seven-task pretrained RLT already had MUL training. Therefore this audit cannot establish neural invention of multiplication, unseen-rule generalization, no pretraining contamination, free program synthesis, or semantics in all machine states. This old controller ROM is not automatically a model trained on the canonical NC-001 ISA.

Eight-bit exhaustive follow-up is PLANNED ONLY: enumerate all 65,536 ordered pairs as 64 shards of 1,024 pairs; preserve full failed/unhalted traces; separately probe every PC, JZ_B/JZ_T taken/not taken, PC6 backedges and PC7 HALT using novel state combinations; measure any overlap with training states when recoverable. This does not prove arbitrary registers or longer input widths.

Never promote the unqualified frozen_shared arm. Distinct real verifier must independently reproduce before NC-003 PASS. Preserve original champion and old checkpoints.
