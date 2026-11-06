# NC-019 — Semantic-first research objective and immutable audit boundary

This is a preregistered evaluation contract plus local symbolic reference checks. It is NOT an empirical model improvement, neural training experiment, universal measure of meaning or a new RLT champion. Independent verification by a distinct actual agent is required for closure.

## Precedence: NC-019 and NC-018

| Scope | Objective | Priority |
|---|---|---|
| Research: what makes semantic learning better? | NC-019: semantic correctness, unseen-family transfer, independently verified new capability, previous-skill retention, then compute/search cost | PRIMARY |
| Execution: how cheaply can a verified answer be computed? | NC-018: minimize FULL measured end-to-end execution cost under the SAME verified correctness/HALT/retention contract | SUBORDINATE |
| Candidate model promotion | NC-019 semantic and retention gates, independent proof, negative controls | NO cost-only promotion |
| Deployment implementation | NC-018 may optimize latency, energy and memory for verified equivalent operations | Must preserve NC-019 proof scope |
| Conflict | A faster but incorrect, unverified, non-halting or forgetting candidate is ineligible | Fail closed |

NC-018 remains intact with its original task, dependencies and outputs. Its historical append-only events are not rewritten, invalidated or closed. NC-019 establishes evaluation precedence only.

The objective is a measured vector, not a universal scalar meaning score:

    Q = (semantic_equivalence, unseen_composition, verified_new_capability, retention)
    C = (training, inference, proposal, search, verification, end_to_end_wall,
         memory, energy_if_measured, expansions, calls, model_bytes)

Only after passing semantic and retention gates can systems be compared on a Pareto frontier of capability AND total measured cost. No neural or host/verification work can be silently excluded from that cost.

## Preregistered input contract

- Contract version: logos.semantic-objective/1.
- Immutable roster and split commitment: the SHA-256 in benchmark_contract.json. Changing a split/family/seed/budget requires a NEW version and append-only DAG task; never rewrite the existing JSON under a committed task.
- Types: NC-001 fixed-width bitvectors. Widths 4, 8, 16. Raw unsigned bitpatterns; two signedness modes unsigned and twos_complement. ADD/SUB/SHL wrap modulo 2^width, SHR is logical, SAR is sign-preserving and only valid in two's complement.
- AST operators: typed bv.read, bv.const, bv.add, bv.sub, bv.and, bv.or, bv.xor, bv.and1, bv.shl, bv.shr, bv.sar. Explicit arity, width and signedness. No implicit real-number arithmetic, floats, unbounded arithmetic, cast or boolean coercion.
- Families: train 5 positive; validation 2 positive; test 6 positive and 7 negative. Structural fingerprints do not overlap across splits. Raw operands can overlap (4-bit space is exhaustively reused); NO claim of exact operand/pretraining disjointness.
- Deterministic inputs: width4 enumerates every 16^arity ordered assignment. Width8 and width16 use 256 deterministic pseudorandom draws WITH replacement per family/seed/mode. Seeds 11,19,29; derived generator seed = first eight bytes of SHA256 over canonical JSON (version, family, split, width, mode, seed).
- Source pinning: SHA256 of NC-001, NC-004, NC-005 research receipts and frozen RLT champion weight checkpoint is recorded in benchmark_contract.json.
- Prospective budgets: per-arm and seed 2,250 optimizer steps, batch 256; per family at most 1,024 proposals, 1,024 search expansions, 1,024 verifier calls, 900 seconds elapsed, 8,192 inference calls. THESE ARE CEILINGS, not observed costs and not claims that trained systems satisfy them.

NC-020 later implements the expanded typed equivalence benchmark and independent proof receipts. NC-019 preregisters families BEFORE training new candidate systems.

## Exact score definitions and negative controls

| Dimension | Numerator | Denominator and hard condition |
|---|---|---|
| Equivalence | TP and TN, separately | TP+FN+abstain_positive > 0 AND TN+FP+abstain_negative > 0; balanced accuracy = 0.5*[TP/P + TN/N] |
| Unseen composition | Correct, well-typed and HALTED neural output on structural test families | All preregistered held-out cases >0; timeout, nonexecution, wrong types, substituted host/oracle answers incorrect |
| Verified gain | Distinct new versioned operators/families absent from the frozen baseline with independent evidence and executable result | Preregistered goals >0; not examples, cached programs or an externally supplied ROM |
| Retention | Correct verified old-skill cases after each learning stage | Frozen independent old-skill denominator, same before and after; every regression disclosed |
| Efficiency | Independently verified new family count | Total real positive measured end-to-end seconds AND nonzero proposal and verification work; full cost/memory vector |

The executable scorer in score_contract.py enforces these denominators and rejects inconsistent/missing records. For scientific use the evaluator must bind all equivalence case IDs and oracle labels, all held-out composition case IDs and trusted execution traces, and frozen retention case-set/skill-contract SHA256 for each old skill. Any positive efficiency/gain numerator requires an externally authenticated proof callback. It defaults to zero transfer/verified-gain credit for all claimant-provided evidence unless an evaluator-only trusted callback independently validates execution/proof. Such a callback is an external trust assumption; the scorer cannot authenticate agent identity by parsing strings. Always-YES and always-NO each score 50% balanced accuracy on the reference fixture of 13 positive + 7 negative families; all-abstain scores zero. A symbolic host answer never counts as neural transfer. No proposal or verification activity cannot earn a zero-cost victory. Identity strings/self-reports alone do not authenticate independent proof; a real verifier must attest independently.

The reference computation in reference_contract.py uses TWO separately written symbolic evaluators, NC-001 canonical_isa and independent bitvector Python, and checks matching values. Seed-11 wider samples plus all four-bit inputs produced 89,712 comparisons: 39 exhaustive finite four-bit family/mode rows and 78 sampled eight- or sixteen-bit rows. Every negative family has a concrete four-bit counterexample, including SAR-vs-SHR for twos-complement negative values.

These are local CODE checks, NOT evidence that a distinct agent verified anything, NOT an empirical neural semantic score, and NOT exhaustive eight/sixteen-bit equivalence proofs. A syntactically legal bitvector family with sampled agreement is labelled sampled evidence only.

## Scientific boundary and failure modes

- Training on all 64 local bit patterns is in-domain retention, not unseen rule learning.
- A causally coupled meaning classifier is a genuine tested pathway, not proof of improved general reasoning.
- Supplied instruction sequences/ROMs, host register routing, flags, PC updates, HALT and symbolic proofs are NOT neural program invention.
- The selected original RLT checkpoint was pretrained with arithmetic including multiplication, so pretraining-disjoint novelty claims require additional proof.
- Semantic errors or forgotten previously qualified skills disqualify cost advantages.
- A one-number "system goodness" or meaning score would hide incompatible proof scopes and measurement units; all axes remain visible.
- Generalization across raw operand values does not imply structural formula-family holdout.
- Cold/warm timing, proposal/search/verifier overhead, memory/data movement and amortized training/storage must all be disclosed before asserting execution optimization.
- Missing energy should be null/NOT_MEASURED, never silently zero.

## Independent verifier responsibilities

The real Event Lab owner has claimed NC-019 through the append-only ledger. A DIFFERENT actual agent must inspect the committed implementation and contract, validate the SHA256 source/weights and frozen family roster, run the full DAG test suite and NC-019 test suite, and independently generate reference counterexamples and balanced-score controls. It must verify width/signedness and overflow rules, nonempty test denominators, no shared structural fingerprints, no training leakage claimed without evidence and task-precedence nonmutation. It records exact Git commit, hardware/software environment, seed/split/case counts, source hashes, verifier runtime identity, failures and scientific limits. Only after that separate audit may the task owner append PASS.

No champion/default promotion is authorized by NC-019. NC-020, NC-021 and NC-022 each have independent acceptance gates.

## Reproduction and dependencies

Commands, from /workspace/experiments/logos-kernel:

    PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-019/reference_contract.py
    PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -v -s artifacts/NC-019 -p test_semantic_objective.py
    python dag.py validate
    python -m unittest -q test_dag

- reference_contract.py requires Python3 standard library, NC-001 canonical_isa.py, NC-004 causal_semantics.py/checkpoint.json, NC-005 training_report.json/independent_receipt.json, NC-000 inventory.json and the original frozen RLT champion .pt for SHA256 only. No model/PyTorch is loaded.
- score_contract.py requires Python3 standard library only.
- test_semantic_objective.py requires unittest + both NC-019 modules and generated objective/contract/receipt files.
- benchmark_contract.json and semantic_objective.json are machine-readable committed evaluation contracts; reference_receipt.json is implementer symbolic evidence.
- independent_receipt.json is reserved for a DIFFERENT REAL AGENT and must NOT be self-issued or fabricated.

Avoid calling a faster computation better understanding, sampled equivalence an exact full-width proof or synthetic/oracle-derived programs algorithms genuinely extracted from frozen neural weights.
