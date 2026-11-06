# NC-001 — Logos canonical ISA/math contract

This task defines a *designed*, deterministic, **symbolic reference ISA**. It does **not** add a learned instruction model, arithmetic checkpoint, program synthesis, or an English language model.

## Outputs

- isa_spec.json — machine-readable, versioned opcode, state, signedness, AST, PC, and HALT semantics
- canonical_isa.py — pure Python validation, canonical JSON, strict typed AST lowering/evaluation, bounded reference interpreter
- test_isa.py — conformance tests covering all 15 opcodes, all fields, bad inputs, flags, signedness, branches and loops
- conformance.json — implementer test denominators, reference benchmarks, SHA-256s, preserved champion
- make_conformance.py — reproducible producer of conformance.json

## Fixed interface

Version: logos.isa/1.

Program must contain width (1..64 bits), signedness (unsigned or twos_complement), and a list of typed instructions. The final instruction **must** be HALT. Registers X, Y, R, T all have the same width, and every initial value must be supplied as a raw unsigned bit pattern. Jumps use explicit absolute target indices. No implicit branch operand, flags, register initialization, overflow behavior, or program fall-through.

Supported opcodes: LOADI, MOV, ADD, SUB, AND, OR, XOR, AND1, SHL, SHR, SAR, JZ, JNZ, JMP, HALT.

ALU flags are C (ADD carry-out / SUB no-borrow / shift discarded bit), V (two's-complement overflow for addition/subtraction), Z (zero), N (sign bit). MOV/LOADI/control opcodes preserve flags. Branches test an explicit register rather than hidden flags. SHL/SHR are logical; SAR is only valid with twos_complement. PC on HALT stays at the HALT instruction. Without HALT execution, step_limit is a failure, never a success.

The mathematical AST has typed bitvector reads/constants and explicit operator nodes; strict AST validation rejects wrong widths, signedness, arity, extra keys, boolean/bitvector conflation, invalid operands and deep nesting. Evaluating the reference AST affects only **symbolic** arithmetic; this is not evidence of learned neural semantic grounding.

## Exact commands

    cd /workspace/experiments/logos-kernel
    python dag.py validate
    python -m unittest -q test_dag
    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -v -s artifacts/NC-001 -p 'test_isa.py'
    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-001/make_conformance.py

### Python usage

    import sys
    sys.path.insert(0, "/workspace/experiments/logos-kernel/artifacts/NC-001")
    import canonical_isa as isa
    p = {"version":"logos.isa/1","width":16,"signedness":"unsigned",
         "program":[{"op":"ADD","dst":"R","a":"X","b":"Y"},{"op":"HALT"}]}
    value = isa.reference(p, {"X":7,"Y":9,"R":0,"T":0})
    assert value["status"] == "halted" and value["registers"]["R"] == 16

## Source dependencies

- ISA implementation: canonical_isa.py + Python 3 standard library only
- Tests: test_isa.py + canonical_isa.py + Python unittest/random/json
- Receipt generation: make_conformance.py + test_isa.py + canonical_isa.py + isa_spec.json + artifacts/NC-000/inventory.json + selected original RLT bundle (hash-only)
- DAG: dag.py + instructions.jsonl + test_dag.py + Python standard library and Git
- **No weights or model code changed** in /workspace/experiments/rlt_split_algorithms_ablation_v1 or /workspace/experiments/latticeforge_neural_isa_test.

## Boundaries and failure modes

- A predefined multiplication ROM composed of these ISA operations is not a neural discovery or learned neural multiplication.
- Conformance includes exhaustive 4-bit operand pairs only; sampled 8-bit tests do not prove all 65,536 pairs.
- Zero tensor/weight updates or training were performed; this is the *interface* for later neural experiments.
- Canonical semantics must not be silently substituted for a neural prediction. Only score the separately predicted AST or state against the reference.
- A **different real agent** must audit implementation, hashes, malformed cases and independent reference/oracle comparisons, then produce a real verifier receipt before marking NC-001 passed.
