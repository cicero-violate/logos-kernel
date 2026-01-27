// Scope: Canonical Stage Semantic Contract (documentation-only)
//
// This file defines the semantic obligations of any Stage implementation
// executed by the kernel.
//
// These are NOT currently mechanically enforced.
// They define the correctness boundary expected by the kernel.
//
// ---------------------------------------------------------------------------
// CORE SEMANTIC INVARIANTS
// ---------------------------------------------------------------------------
//
// 1. Purity
//    Stage output must depend ONLY on:
//      - input value
//      - provided TypeSig
//
//    No reads from:
//      - global mutable state
//      - environment variables
//      - filesystem
//      - network
//      - clocks
//
// ---------------------------------------------------------------------------
//
// 2. Determinism
//    Given identical input + TypeSig:
//      -> logic() must return identical Delta + Proof
//      -> differential() must return identical Delta
//
//    No randomness.
//    No time-based behavior.
//
// ---------------------------------------------------------------------------
//
// 3. Totality (Within Domain)
//    For valid domain inputs:
//      - Stage must not panic.
//      - Errors must be returned as PipelineError.
//
// ---------------------------------------------------------------------------
//
// 4. No Hidden Side Effects
//    Stage must not:
//      - spawn threads
//      - perform I/O
//      - mutate shared memory
//
// ---------------------------------------------------------------------------
//
// 5. Delta Honesty
//    Delta.before must equal the input value.
//    Delta.after must be the produced output.
//
// ---------------------------------------------------------------------------
//
// 6. Proof Optionality
//    Proof may be empty.
//    Proof presence must not affect Delta.
//
// ---------------------------------------------------------------------------
// MODE CONSISTENCY
// ---------------------------------------------------------------------------
//
// For a given input:
//   logic().delta.after
//     MUST equal
//   differential().delta.after
//
// Proof may differ.
//
// ---------------------------------------------------------------------------
// NON-GOALS (NOT STAGE RESPONSIBILITY)
// ---------------------------------------------------------------------------
//
// - Scheduling
// - Parallelism
// - Caching
// - Memoization
// - Replay
// - Authorization
//
// ---------------------------------------------------------------------------
// STABILITY GUARANTEE
// ---------------------------------------------------------------------------
//
// Any change to this contract MUST be:
//   1) Intentional
//   2) Reviewed
//   3) Reflected in tests or future mechanical enforcement
//
// This file is the human-readable mirror of stage truth.
//
