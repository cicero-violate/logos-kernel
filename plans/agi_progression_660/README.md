# Proposed 660-stage AGI research progression

This extension adds 500 concrete experiments in 50 tracks to the existing 160-stage roadmap. All 500 are planned tasks, not claims of implemented capabilities or a proven optimal route to AGI.

Canonical task definitions: [instructions.jsonl](../../instructions.jsonl). Full machine-readable progression: [roadmap.jsonl](roadmap.jsonl). Extension tasks: [new_tasks.jsonl](new_tasks.jsonl). Original roadmap: [160 stages](../agi_progression_160/README.md).

The original 160 stages remain unchanged. Numbering is a valid dependency order, not a claim that every earlier branch must finish before a later branch starts. Each new track anchors to an existing capability and progresses through ten experiments; the final integrated audit depends on all 49 preceding track replications. Existing runtime priorities and claims are preserved.

Every task requires a concrete preregistered protocol, named comparison baselines, task-specific metrics, negative controls, bounded resources, retention checks and a separate independent receipt. Failed hypotheses remain visible; NO_GAIN does not unlock unsupported capability claims.

## New tracks

| Track | Task IDs | Prerequisite | Measurement |
|---|---|---|---|
| Evaluation integrity | AGI-092–AGI-101 | AGI-005 | leak detection recall and false positives |
| Statistical reliability | AGI-102–AGI-111 | AGI-006 | coverage and false discovery rate |
| Resource accounting | AGI-112–AGI-121 | AGI-007 | unaccounted cost and quality per allocated budget |
| Runtime recovery | AGI-122–AGI-131 | AGI-009 | lost work duplicate effects and recovery latency |
| Reproducible experiments | AGI-132–AGI-141 | AGI-089 | replay agreement and missing dependency rate |
| Typed program semantics | AGI-142–AGI-151 | AGI-034 | validity and semantic equivalence rate |
| Numerical generalization | AGI-152–AGI-161 | AGI-016 | absolute relative and worst-case error |
| Variable binding | AGI-162–AGI-171 | AGI-012 | binding accuracy under renaming |
| Object permanence | AGI-172–AGI-181 | AGI-013 | identity switches and trajectory consistency |
| Temporal abstraction | AGI-182–AGI-191 | AGI-014 | boundary quality and downstream prediction |
| Spatial reasoning | AGI-192–AGI-201 | AGI-015 | relational accuracy and transformation consistency |
| Multimodal grounding | AGI-202–AGI-211 | AGI-019 | grounded task accuracy and causal modality contribution |
| Compositional language | AGI-212–AGI-221 | AGI-017 | execution accuracy on held-out constructions |
| Working memory | AGI-222–AGI-231 | AGI-021 | accuracy per byte and distractor sensitivity |
| Episodic memory | AGI-232–AGI-241 | AGI-022 | reconstruction accuracy and provenance integrity |
| Semantic consolidation | AGI-242–AGI-251 | AGI-023 | transfer benefit and compression distortion |
| Memory revision | AGI-252–AGI-261 | AGI-025 | revision correctness and stale belief rate |
| Continual learning | AGI-262–AGI-271 | AGI-027 | forward transfer backward transfer and forgetting |
| Modular representations | AGI-272–AGI-281 | AGI-029 | retention capacity and routing overhead |
| Replay and compression | AGI-282–AGI-291 | AGI-030 | retention per stored byte |
| Uncertainty calibration | AGI-292–AGI-301 | AGI-031 | Brier score coverage and calibration error |
| Abstention and assistance | AGI-302–AGI-311 | AGI-032 | risk coverage and assistance cost |
| Distribution shift | AGI-312–AGI-321 | AGI-033 | shift detection delay false alarms and recovery |
| Formal reasoning | AGI-322–AGI-331 | AGI-034 | valid proof rate and verified solution cost |
| Search allocation | AGI-332–AGI-341 | AGI-035 | verified solutions per budget |
| Causal identification | AGI-342–AGI-351 | AGI-036 | intervention accuracy and identification uncertainty |
| Analogical transfer | AGI-352–AGI-361 | AGI-037 | structural transfer accuracy |
| Counterexample learning | AGI-362–AGI-371 | AGI-038 | repair generalization and regressions |
| Depth generalization | AGI-372–AGI-381 | AGI-039 | success probability by reasoning depth |
| Metacognitive control | AGI-382–AGI-391 | AGI-040 | quality cost frontier and allocation regret |
| State-space planning | AGI-392–AGI-401 | AGI-041 | goal attainment plan validity and cost |
| Partial observability | AGI-402–AGI-411 | AGI-043 | belief calibration and achieved utility |
| Experimental design | AGI-412–AGI-421 | AGI-044 | information gain and downstream prediction |
| Hierarchical skills | AGI-422–AGI-431 | AGI-045 | completion cost and compositional transfer |
| Tool interoperability | AGI-432–AGI-441 | AGI-047 | argument correctness and verified effects |
| Failure recovery | AGI-442–AGI-451 | AGI-048 | recovery success duplicate effects and cost |
| Long-horizon execution | AGI-452–AGI-461 | AGI-049 | completion by action count and duration |
| Few-shot acquisition | AGI-462–AGI-471 | AGI-051 | learning curve and total experience cost |
| Curriculum learning | AGI-472–AGI-481 | AGI-055 | time to criterion and retained coverage |
| Cross-domain transfer | AGI-482–AGI-491 | AGI-056 | net transfer and worst-domain regression |
| Sparse-feedback learning | AGI-492–AGI-501 | AGI-057 | success per interaction and reward delay sensitivity |
| Goal clarification | AGI-502–AGI-511 | AGI-062 | completion benefit per clarification |
| Preference and constraint learning | AGI-512–AGI-521 | AGI-064 | constraint satisfaction and preference regret |
| Authority and oversight | AGI-522–AGI-531 | AGI-070 | unauthorized effect rate and correction recovery |
| Cooperative coordination | AGI-532–AGI-541 | AGI-066 | joint utility communication cost and conflict rate |
| Truthful reporting | AGI-542–AGI-551 | AGI-067 | claim support rate and omission severity |
| Faithful explanation | AGI-552–AGI-561 | AGI-068 | intervention faithfulness and support precision |
| Self-update attribution | AGI-562–AGI-571 | AGI-071 | causal gain per total update cost |
| Reversible improvement | AGI-572–AGI-581 | AGI-076 | verified net gain retention and rollback success |
| Integrated generality audit | AGI-582–AGI-591 | AGI-090 | breadth depth reliability and acquisition efficiency |

## All 660 stages

| Stage | Task | Milestone | Prerequisites |
|---:|---|---|---|
| 1 | NC-000 | Freeze and inventory working champion and ISA specialist | — |
| 2 | NC-001 | Define canonical ISA, register state and typed math syntax | NC-000 |
| 3 | NC-002 | Persist full neurons and trainable state | NC-000 |
| 4 | NC-003 | Independent baseline verification | NC-000, NC-001, NC-002 |
| 5 | NC-004 | Teach ISA syntax to mathematical semantics causally | NC-001, NC-002 |
| 6 | NC-005 | Progressive learning small to big with retention | NC-003, NC-004 |
| 7 | NC-006 | Create verified neural circuit/operator library | NC-003, NC-005 |
| 8 | NC-007 | Learn composition and reusable operation macros | NC-005, NC-006 |
| 9 | NC-008 | Neural operation invention from task specification | NC-007 |
| 10 | NC-009 | Verify, deduplicate, promote and save invented operators | NC-006, NC-008 |
| 11 | NC-010 | Neurally generate programs with branches, loops and HALT | NC-008, NC-009 |
| 12 | NC-011 | Neural self-observation, verified feedback and replay | NC-005, NC-010 |
| 13 | NC-012 | Unify controller and arithmetic into one recurrent Transformer | NC-003, NC-010, NC-011 |
| 14 | NC-013 | Measure best Transformer architecture via ablation | NC-012 |
| 15 | NC-014 | Exhaustive bounded correctness and new-task transfer | NC-009, NC-012, NC-013 |
| 16 | NC-015 | Promote one champion with single interface and reversible rollback | NC-003, NC-009, NC-013, NC-014 |
| 17 | NC-018 | Minimize verified computation required to produce an answer | NC-007 |
| 18 | NC-019 | Define semantic-first learning and discovery objective | NC-001, NC-004 |
| 19 | NC-020 | Build typed mathematical semantic-equivalence benchmark | NC-001, NC-019 |
| 20 | NC-021 | Ablate causal meaning against detached semantics for transfer | NC-004, NC-005, NC-019, NC-020 |
| 21 | NC-022 | Transfer progressive rehearsal and retention into selected RLT Transformer | NC-003, NC-005, NC-019 |
| 22 | NC-023 | Establish a verified mathematical rewrite and equality-saturation baseline | NC-001, NC-019, NC-020 |
| 23 | NC-024 | Evaluate neural-guided formula rewrite search against deterministic search | NC-008, NC-020, NC-023 |
| 24 | NC-025 | Train a search policy that improves future algorithm discovery | NC-005, NC-024 |
| 25 | NC-026 | Discover verified reusable mathematical abstractions across tasks | NC-009, NC-020, NC-025 |
| 26 | NC-027 | Consolidate and compact semantic capability without forgetting | NC-002, NC-021, NC-022, NC-026 |
| 27 | NC-028 | Release Logos Kernel into bounded autonomous mathematical-discovery sandbox | NC-007, NC-019, NC-020 |
| 28 | NC-029 | CPU-only compute and correctness baseline | NC-000, NC-003 |
| 29 | NC-030 | Typed mappings between mathematical representation spaces | NC-001, NC-020, NC-023 |
| 30 | NC-031 | Learn cost-aware valid transformation routing | NC-029, NC-030, NC-023 |
| 31 | NC-040 | Generate autonomous mathematical research questions | NC-025, NC-026, NC-020 |
| 32 | NC-041 | Learn mathematical interestingness and question utility | NC-040, NC-025 |
| 33 | NC-042 | Close bounded self-directed mathematical research loop | NC-040, NC-041, NC-026 |
| 34 | NC-043 | Demonstrate meta-learning of self-directed research strategy | NC-042, NC-025, NC-026 |
| 35 | NC-044 | Isolate third-agent trainer identity and immutable champion boundary | NC-002, NC-003, NC-022 |
| 36 | NC-045 | Bootstrap independent append-only trainer DAG and stateless tab recovery | NC-044 |
| 37 | NC-046 | Run bounded continuous trainer candidate checkpoints with verified replay | NC-045, NC-022 |
| 38 | NC-047 | Independently gate promotion of trained successors and preserve rollback | NC-046 |
| 39 | NC-048 | Audit, pin and stage research references for CPU-only automated mathematical teaching | NC-001, NC-003, NC-022 |
| 40 | NC-049 | Construct proof-backed synthetic mathematical lesson generator | NC-048, NC-001, NC-006, NC-020 |
| 41 | NC-050 | Implement adaptive learning-progress and prerequisite-aware automated teacher | NC-048, NC-005, NC-022 |
| 42 | NC-051 | Add counterexample-guided and adversarial mathematical lesson generation | NC-049, NC-050, NC-006 |
| 43 | NC-052 | Bootstrap reusable program abstractions via Stitch and DreamCoder-style library learning | NC-048, NC-049, NC-007 |
| 44 | NC-053 | Integrate automated teacher with isolated trainer and benchmark massive bootstrap | NC-045, NC-046, NC-049, NC-050, NC-051, NC-052 |
| 45 | NC-054 | Adapt teacher to self-generated mathematical questions and meta-curriculum | NC-053, NC-040, NC-041, NC-043 |
| 46 | NC-055 | Construct falsifiable invariant-discovery benchmark across dynamical families | NC-019, NC-020 |
| 47 | NC-056 | Discover minimum sufficient finite-difference order or invariant recurrence law | NC-055, NC-020 |
| 48 | NC-057 | Evaluate stability-first invariant-guided mathematical search objective | NC-055, NC-056 |
| 49 | NC-058 | Test repeated shared nonlinear kernel for recursive invariant refinement | NC-055, NC-056, NC-022 |
| 50 | NC-059 | Verify transfer of learned invariants across mathematical representations | NC-055, NC-056, NC-030, NC-058 |
| 51 | NC-060 | Teach progressively higher-order verified invariants with adaptive curriculum | NC-055, NC-056, NC-057, NC-050, NC-053, NC-059 |
| 52 | NC-061 | Freeze a tamper-evident self-improvement scoreboard before any improvement run | NC-020, NC-029 |
| 53 | NC-062 | Close the smallest verified generation-over-generation improvement loop | NC-008, NC-009, NC-028, NC-061 |
| 54 | NC-063 | Run a bounded unattended self-improvement campaign without ChatGPT | NC-047, NC-062 |
| 55 | NC-064 | Audit genuine multi-branch tree and evolutionary mathematical search against symbolic controls | NC-008, NC-023, NC-024, NC-029 |
| 56 | NC-065 | Falsify cross-generation nested algorithm-library compounding against flat and cold memory | NC-026, NC-052, NC-029, NC-062, NC-064 |
| 57 | NC-066 | Evaluate failure-generated adversarial mathematical worlds and transferable repair | NC-051, NC-055, NC-026 |
| 58 | NC-067 | Test receipt-conditioned evolutionary optimizer and developmental budget allocation | NC-043, NC-050, NC-064, NC-065, NC-066 |
| 59 | NC-068 | Test strictly isolated resource-bounded architecture mutations and source-rewrite recursion | NC-013, NC-047, NC-067 |
| 60 | NC-069 | Causal stage 1: predict interventions on the ISA machine from an exact oracle | NC-001, NC-003, NC-004, NC-020 |
| 61 | NC-070 | Causal stage 2: discover causal structure by choosing the agent's own experiments | NC-069 |
| 62 | NC-071 | Causal stage 3: reuse learned mechanisms in unseen combinations | NC-070, NC-006 |
| 63 | NC-072 | Causal stage 4: localize a changed mechanism and relearn only that part | NC-071 |
| 64 | NC-073 | Causal stage 5: answer counterfactuals against the exact machine | NC-071 |
| 65 | NC-074 | Recursive causal representation: compress variables into verified higher levels | NC-071, NC-007 |
| 66 | NC-075 | Causal stage 6: detect hidden variables and avoid false causal links | NC-072, NC-073 |
| 67 | NC-076 | Causal stage 7: roll a causal world model forward over time | NC-074, NC-075 |
| 68 | NC-077 | Test whether causal knowledge improves the generational improvement loop | NC-062, NC-071 |
| 69 | NC-078 | Meta-learn ISA program synthesis and test it on functions excluded from training | NC-001, NC-007 |
| 70 | AGI-001 | Operationally define broad general intelligence | NC-019, NC-020 |
| 71 | AGI-002 | Audit contamination across the complete task lifecycle | AGI-001, NC-061 |
| 72 | AGI-003 | Establish calibrated human reference distributions | AGI-001 |
| 73 | AGI-004 | Measure one checkpoint across a capability matrix | AGI-001, NC-015 |
| 74 | AGI-005 | Build an independent blind evaluation interface | AGI-002, NC-047 |
| 75 | AGI-006 | Preregister statistical power and stopping rules | AGI-003, NC-029 |
| 76 | AGI-007 | Account for total experience and compute | AGI-006 |
| 77 | AGI-008 | Connect kernel evidence to the local autonomy runtime | AGI-002, NC-063 |
| 78 | AGI-009 | Recover integrated experiments after interruption | AGI-008 |
| 79 | AGI-010 | Demonstrate a bounded running research campaign | AGI-009 |
| 80 | AGI-011 | Learn explicit relational role binding | NC-030, NC-026 |
| 81 | AGI-012 | Generalize under alpha-renaming and symbol replacement | AGI-011 |
| 82 | AGI-013 | Track object identity under transformations | AGI-011 |
| 83 | AGI-014 | Segment temporal events into reusable transitions | AGI-013, NC-076 |
| 84 | AGI-015 | Reason across spatial reference frames | AGI-013 |
| 85 | AGI-016 | Preserve units and dimensional types | AGI-012, NC-030 |
| 86 | AGI-017 | Ground language into typed task contracts | AGI-011, NC-078 |
| 87 | AGI-018 | Execute grounded instructions with explicit semantics | AGI-017, NC-010 |
| 88 | AGI-019 | Align symbolic and perceptual representations | AGI-013, AGI-017 |
| 89 | AGI-020 | Audit compositional representation transfer | AGI-012, AGI-014, AGI-015, AGI-016, AGI-018, AGI-019 |
| 90 | AGI-021 | Build bounded task-relevant working memory | AGI-020, NC-027 |
| 91 | AGI-022 | Store replayable episodic experience | AGI-021, NC-028 |
| 92 | AGI-023 | Consolidate episodes into reusable semantic knowledge | AGI-022, NC-026 |
| 93 | AGI-024 | Retrieve evidence with provenance and uncertainty | AGI-023, AGI-002 |
| 94 | AGI-025 | Revise memory when evidence contradicts it | AGI-024 |
| 95 | AGI-026 | Measure amortized benefit from memory reuse | AGI-025, NC-065 |
| 96 | AGI-027 | Learn a stream of changing task distributions | AGI-020, NC-022 |
| 97 | AGI-028 | Establish a persistent forgetting regression suite | AGI-027 |
| 98 | AGI-029 | Reduce destructive interference between skills | AGI-028, NC-027 |
| 99 | AGI-030 | Compress replay without losing rare capabilities | AGI-026, AGI-029 |
| 100 | AGI-031 | Calibrate uncertainty across tasks and shifts | NC-061, AGI-006 |
| 101 | AGI-032 | Learn when to abstain or ask for assistance | AGI-031 |
| 102 | AGI-033 | Detect unfamiliar task distributions | AGI-020, AGI-031 |
| 103 | AGI-034 | Combine learned reasoning with checkable proof obligations | AGI-032, NC-023 |
| 104 | AGI-035 | Allocate search using expected information value | AGI-031, NC-064 |
| 105 | AGI-036 | Separate causal heuristics from spurious shortcuts | AGI-033, NC-077 |
| 106 | AGI-037 | Transfer abstractions through structural analogy | AGI-020, NC-059 |
| 107 | AGI-038 | Revise reasoning from independent counterexamples | AGI-034, NC-051 |
| 108 | AGI-039 | Generalize reasoning to greater depth and length | AGI-037, NC-010 |
| 109 | AGI-040 | Control reasoning effort with calibrated metacognition | AGI-035, AGI-038, AGI-039 |
| 110 | AGI-041 | Plan with explicit state and goal models | AGI-018, NC-076 |
| 111 | AGI-042 | Detect divergence between plans and execution | AGI-041 |
| 112 | AGI-043 | Maintain beliefs under partial observability | AGI-042, NC-075 |
| 113 | AGI-044 | Choose actions that resolve uncertainty | AGI-043, NC-070 |
| 114 | AGI-045 | Learn reusable hierarchical skills and options | AGI-044, NC-074 |
| 115 | AGI-046 | Use tools through reversible capability boundaries | AGI-018, NC-047 |
| 116 | AGI-047 | Generalize to unfamiliar tool schemas | AGI-046 |
| 117 | AGI-048 | Recover from stale observations and tool failures | AGI-047, AGI-042 |
| 118 | AGI-049 | Extend reliable multi-step task completion | AGI-045, AGI-048 |
| 119 | AGI-050 | Stop and reallocate bounded campaigns reliably | AGI-049, AGI-010 |
| 120 | AGI-051 | Acquire new primitives from few demonstrations | AGI-020, AGI-007, NC-078 |
| 121 | AGI-052 | Compose newly acquired skills without retraining | AGI-051, NC-071 |
| 122 | AGI-053 | Generate procedural curricula with coverage guarantees | AGI-051, NC-053 |
| 123 | AGI-054 | Hold out entire generative task families | AGI-053, AGI-002 |
| 124 | AGI-055 | Optimize prerequisite-aware skill acquisition | AGI-054, NC-050 |
| 125 | AGI-056 | Distill transferable skills across domains | AGI-055, AGI-030 |
| 126 | AGI-057 | Explore tasks with sparse or delayed feedback | AGI-054, AGI-044 |
| 127 | AGI-058 | Balance new learning with old capability retention | AGI-056, AGI-028 |
| 128 | AGI-059 | Transfer causal mechanisms between environments | AGI-052, NC-077 |
| 129 | AGI-060 | Audit skill-acquisition efficiency across domains | AGI-058, AGI-059, AGI-007 |
| 130 | AGI-061 | Recognize ambiguous or underspecified user goals | AGI-017, AGI-032 |
| 131 | AGI-062 | Ask clarification questions with measurable value | AGI-061, AGI-044 |
| 132 | AGI-063 | Maintain versioned user preferences and corrections | AGI-025, AGI-062 |
| 133 | AGI-064 | Satisfy interacting goals and hard constraints | AGI-018, AGI-063 |
| 134 | AGI-065 | Track roles, authority and information access | AGI-046, AGI-064 |
| 135 | AGI-066 | Coordinate bounded cooperative problem solving | AGI-065, AGI-045 |
| 136 | AGI-067 | Report epistemic and execution status truthfully | AGI-031, AGI-025 |
| 137 | AGI-068 | Produce explanations faithful to causal computation | AGI-067, NC-073 |
| 138 | AGI-069 | Expose and negotiate conflicting preferences | AGI-064, AGI-067 |
| 139 | AGI-070 | Preserve effective human oversight and interruption | AGI-050, AGI-065, AGI-069 |
| 140 | AGI-071 | Attribute improvement to the proposed self-update | AGI-060, NC-062, NC-077 |
| 141 | AGI-072 | Keep updater and evaluator authority disjoint | AGI-071, AGI-005, NC-047 |
| 142 | AGI-073 | Meta-learn optimization on unseen task families | AGI-072, NC-067 |
| 143 | AGI-074 | Propose small reversible source or architecture changes | AGI-073, NC-068 |
| 144 | AGI-075 | Reject corrupt or harmful candidate updates | AGI-074 |
| 145 | AGI-076 | Retain capabilities across successive self-updates | AGI-075, AGI-058 |
| 146 | AGI-077 | Preserve task objectives through self-modification | AGI-076, AGI-064 |
| 147 | AGI-078 | Measure net improvement after all learning costs | AGI-077, AGI-007 |
| 148 | AGI-079 | Detect plateaus and falsify unproductive hypotheses | AGI-078, AGI-006 |
| 149 | AGI-080 | Demonstrate compounding transfer across generations | AGI-079, NC-065 |
| 150 | AGI-081 | Solve unfamiliar abstraction puzzles under controlled priors | AGI-060, AGI-039 |
| 151 | AGI-082 | Repair unfamiliar software from executable specifications | AGI-081, AGI-048 |
| 152 | AGI-083 | Conduct bounded quantitative scientific investigations | AGI-081, AGI-059 |
| 153 | AGI-084 | Synthesize evidence from long conflicting documents | AGI-024, AGI-068 |
| 154 | AGI-085 | Solve unfamiliar resource-allocation and scheduling problems | AGI-081, AGI-045 |
| 155 | AGI-086 | Compare breadth and depth against human references | AGI-082, AGI-083, AGI-084, AGI-085, AGI-003 |
| 156 | AGI-087 | Stress generalization under adversarial distribution change | AGI-086, AGI-033 |
| 157 | AGI-088 | Measure robust long-horizon general task performance | AGI-087, AGI-049 |
| 158 | AGI-089 | Obtain independent replication on fresh task generators | AGI-088, AGI-005 |
| 159 | AGI-090 | Produce an auditable general-capability readiness dossier | AGI-089, AGI-070, AGI-080 |
| 160 | AGI-091 | Assess AGI evidence against the preregistered definition | AGI-090, AGI-001 |
| 161 | AGI-092 | Evaluation integrity: detect task text leaked into training | AGI-005 |
| 162 | AGI-093 | Evaluation integrity: detect answer keys hidden in retrieval | AGI-092 |
| 163 | AGI-094 | Evaluation integrity: detect near-duplicate evaluation examples | AGI-093 |
| 164 | AGI-095 | Evaluation integrity: detect generator seeds exposed to candidates | AGI-094 |
| 165 | AGI-096 | Evaluation integrity: detect scoring code exposed through errors | AGI-095 |
| 166 | AGI-097 | Evaluation integrity: detect adaptive overfitting through repeated queries | AGI-096 |
| 167 | AGI-098 | Evaluation integrity: separate tuning feedback from final evaluation | AGI-097 |
| 168 | AGI-099 | Evaluation integrity: rotate hidden generators without changing the rubric | AGI-098 |
| 169 | AGI-100 | Evaluation integrity: audit evaluator access under concurrent runs | AGI-099 |
| 170 | AGI-101 | Evaluation integrity: replicate contamination resistance with fresh canaries | AGI-100 |
| 171 | AGI-102 | Statistical reliability: calibrate intervals on known Bernoulli outcomes | AGI-006 |
| 172 | AGI-103 | Statistical reliability: estimate variance across seeds and task families | AGI-102 |
| 173 | AGI-104 | Statistical reliability: handle paired correlated candidate outcomes | AGI-103 |
| 174 | AGI-105 | Statistical reliability: correct comparisons across candidate populations | AGI-104 |
| 175 | AGI-106 | Statistical reliability: validate sequential stopping under null gains | AGI-105 |
| 176 | AGI-107 | Statistical reliability: estimate power for small practical improvements | AGI-106 |
| 177 | AGI-108 | Statistical reliability: audit missing and censored experiment outcomes | AGI-107 |
| 178 | AGI-109 | Statistical reliability: distinguish equivalence from insufficient evidence | AGI-108 |
| 179 | AGI-110 | Statistical reliability: check subgroup claims under imbalance | AGI-109 |
| 180 | AGI-111 | Statistical reliability: replicate the complete decision rule on simulated null campaigns | AGI-110 |
| 181 | AGI-112 | Resource accounting: count failed search and compilation work | AGI-007 |
| 182 | AGI-113 | Resource accounting: charge teacher-generated demonstrations | AGI-112 |
| 183 | AGI-114 | Resource accounting: account for retrieval and indexing costs | AGI-113 |
| 184 | AGI-115 | Resource accounting: measure memory and checkpoint storage peaks | AGI-114 |
| 185 | AGI-116 | Resource accounting: record evaluator and verifier expenditure | AGI-115 |
| 186 | AGI-117 | Resource accounting: compare allocated versus consumed budgets | AGI-116 |
| 187 | AGI-118 | Resource accounting: attribute shared cache costs across tasks | AGI-117 |
| 188 | AGI-119 | Resource accounting: measure startup and amortization break-even | AGI-118 |
| 189 | AGI-120 | Resource accounting: detect hidden retries and orphan processes | AGI-119 |
| 190 | AGI-121 | Resource accounting: reconcile independent process accounting with campaign receipts | AGI-120 |
| 191 | AGI-122 | Runtime recovery: recover after interruption before proposal persistence | AGI-009 |
| 192 | AGI-123 | Runtime recovery: recover after interruption during evaluation | AGI-122 |
| 193 | AGI-124 | Runtime recovery: recover after result persistence before review | AGI-123 |
| 194 | AGI-125 | Runtime recovery: recover after review before promotion | AGI-124 |
| 195 | AGI-126 | Runtime recovery: reject partial checkpoint writes | AGI-125 |
| 196 | AGI-127 | Runtime recovery: deduplicate repeated completion messages | AGI-126 |
| 197 | AGI-128 | Runtime recovery: recover from full temporary storage | AGI-127 |
| 198 | AGI-129 | Runtime recovery: recover from missing artifact dependencies | AGI-128 |
| 199 | AGI-130 | Runtime recovery: reconstruct state from committed event history | AGI-129 |
| 200 | AGI-131 | Runtime recovery: reproduce recovery under randomized fault schedules | AGI-130 |
| 201 | AGI-132 | Reproducible experiments: pin source and configuration identity | AGI-089 |
| 202 | AGI-133 | Reproducible experiments: record numeric precision and backend choices | AGI-132 |
| 203 | AGI-134 | Reproducible experiments: detect undeclared file reads | AGI-133 |
| 204 | AGI-135 | Reproducible experiments: detect environment-dependent behavior | AGI-134 |
| 205 | AGI-136 | Reproducible experiments: reproduce CPU-only reference computations | AGI-135 |
| 206 | AGI-137 | Reproducible experiments: bound nondeterministic outcome variation | AGI-136 |
| 207 | AGI-138 | Reproducible experiments: rebuild artifacts from clean local directories | AGI-137 |
| 208 | AGI-139 | Reproducible experiments: verify dataset split and generator identity | AGI-138 |
| 209 | AGI-140 | Reproducible experiments: produce minimal independently replayable bundles | AGI-139 |
| 210 | AGI-141 | Reproducible experiments: replicate a bundle with independently chosen environment perturbations | AGI-140 |
| 211 | AGI-142 | Typed program semantics: infer scalar and vector argument types | AGI-034 |
| 212 | AGI-143 | Typed program semantics: reject incompatible shape compositions | AGI-142 |
| 213 | AGI-144 | Typed program semantics: preserve signedness across conversions | AGI-143 |
| 214 | AGI-145 | Typed program semantics: model overflow and saturation explicitly | AGI-144 |
| 215 | AGI-146 | Typed program semantics: check aliasing and mutable state effects | AGI-145 |
| 216 | AGI-147 | Typed program semantics: infer loop invariants for bounded loops | AGI-146 |
| 217 | AGI-148 | Typed program semantics: track exceptions as typed outcomes | AGI-147 |
| 218 | AGI-149 | Typed program semantics: validate preconditions before operator composition | AGI-148 |
| 219 | AGI-150 | Typed program semantics: check postconditions after composed execution | AGI-149 |
| 220 | AGI-151 | Typed program semantics: replicate type-guided transfer on unseen operator families | AGI-150 |
| 221 | AGI-152 | Numerical generalization: generalize arithmetic beyond training magnitudes | AGI-016 |
| 222 | AGI-153 | Numerical generalization: preserve cancellation-sensitive calculations | AGI-152 |
| 223 | AGI-154 | Numerical generalization: handle underflow and overflow boundaries | AGI-153 |
| 224 | AGI-155 | Numerical generalization: recognize ill-conditioned linear systems | AGI-154 |
| 225 | AGI-156 | Numerical generalization: propagate interval uncertainty through expressions | AGI-155 |
| 226 | AGI-157 | Numerical generalization: enforce dimensional consistency with affine units | AGI-156 |
| 227 | AGI-158 | Numerical generalization: compare exact rational and floating execution | AGI-157 |
| 228 | AGI-159 | Numerical generalization: select precision from an error budget | AGI-158 |
| 229 | AGI-160 | Numerical generalization: reject unstable extrapolation with calibrated uncertainty | AGI-159 |
| 230 | AGI-161 | Numerical generalization: replicate numerical reliability on fresh adversarial instances | AGI-160 |
| 231 | AGI-162 | Variable binding: bind nested lexical scopes | AGI-012 |
| 232 | AGI-163 | Variable binding: distinguish shadowed variable names | AGI-162 |
| 233 | AGI-164 | Variable binding: track capture-avoiding substitution | AGI-163 |
| 234 | AGI-165 | Variable binding: bind roles under argument permutation | AGI-164 |
| 235 | AGI-166 | Variable binding: resolve references across nested records | AGI-165 |
| 236 | AGI-167 | Variable binding: compose higher-order function arguments | AGI-166 |
| 237 | AGI-168 | Variable binding: preserve binding through serialization | AGI-167 |
| 238 | AGI-169 | Variable binding: generalize to unseen symbol alphabets | AGI-168 |
| 239 | AGI-170 | Variable binding: diagnose interference from repeated distractor names | AGI-169 |
| 240 | AGI-171 | Variable binding: replicate binding transfer on unseen grammar productions | AGI-170 |
| 241 | AGI-172 | Object permanence: track identical objects through crossings | AGI-013 |
| 242 | AGI-173 | Object permanence: maintain identity through full occlusion | AGI-172 |
| 243 | AGI-174 | Object permanence: distinguish object splits from observation noise | AGI-173 |
| 244 | AGI-175 | Object permanence: handle object merges with uncertain identity | AGI-174 |
| 245 | AGI-176 | Object permanence: track objects under camera movement | AGI-175 |
| 246 | AGI-177 | Object permanence: separate appearance changes from identity changes | AGI-176 |
| 247 | AGI-178 | Object permanence: infer containment through hidden transitions | AGI-177 |
| 248 | AGI-179 | Object permanence: maintain identity after temporary sensor loss | AGI-178 |
| 249 | AGI-180 | Object permanence: calibrate ambiguous association hypotheses | AGI-179 |
| 250 | AGI-181 | Object permanence: replicate identity tracking on fresh simulated worlds | AGI-180 |
| 251 | AGI-182 | Temporal abstraction: segment variable-duration events | AGI-014 |
| 252 | AGI-183 | Temporal abstraction: distinguish repeated events from prolonged states | AGI-182 |
| 253 | AGI-184 | Temporal abstraction: represent nested temporal intervals | AGI-183 |
| 254 | AGI-185 | Temporal abstraction: infer event order under missing observations | AGI-184 |
| 255 | AGI-186 | Temporal abstraction: align asynchronous event streams | AGI-185 |
| 256 | AGI-187 | Temporal abstraction: compress repeated transitions into reusable events | AGI-186 |
| 257 | AGI-188 | Temporal abstraction: detect changes in transition duration | AGI-187 |
| 258 | AGI-189 | Temporal abstraction: compose events across different time scales | AGI-188 |
| 259 | AGI-190 | Temporal abstraction: predict outcomes after time-warped sequences | AGI-189 |
| 260 | AGI-191 | Temporal abstraction: replicate event abstractions on held-out event combinations | AGI-190 |
| 261 | AGI-192 | Spatial reasoning: transform between local and global frames | AGI-015 |
| 262 | AGI-193 | Spatial reasoning: infer relative position from partial constraints | AGI-192 |
| 263 | AGI-194 | Spatial reasoning: preserve topology under geometric deformation | AGI-193 |
| 264 | AGI-195 | Spatial reasoning: resolve mirror and rotation ambiguities | AGI-194 |
| 265 | AGI-196 | Spatial reasoning: compose uncertain spatial transformations | AGI-195 |
| 266 | AGI-197 | Spatial reasoning: plan around unseen obstacle arrangements | AGI-196 |
| 267 | AGI-198 | Spatial reasoning: reason about containment and connectivity | AGI-197 |
| 268 | AGI-199 | Spatial reasoning: track moving reference frames | AGI-198 |
| 269 | AGI-200 | Spatial reasoning: detect inconsistent spatial descriptions | AGI-199 |
| 270 | AGI-201 | Spatial reasoning: replicate spatial transfer under novel coordinate conventions | AGI-200 |
| 271 | AGI-202 | Multimodal grounding: ground synthetic visual attributes in symbols | AGI-019 |
| 272 | AGI-203 | Multimodal grounding: ground synthetic sounds in event labels | AGI-202 |
| 273 | AGI-204 | Multimodal grounding: align action descriptions with observed state changes | AGI-203 |
| 274 | AGI-205 | Multimodal grounding: resolve conflicting text and visual evidence | AGI-204 |
| 275 | AGI-206 | Multimodal grounding: detect absent referents in descriptions | AGI-205 |
| 276 | AGI-207 | Multimodal grounding: transfer grounding across rendering styles | AGI-206 |
| 277 | AGI-208 | Multimodal grounding: combine incomplete complementary modalities | AGI-207 |
| 278 | AGI-209 | Multimodal grounding: calibrate uncertainty under missing modalities | AGI-208 |
| 279 | AGI-210 | Multimodal grounding: reject modality-specific shortcut cues | AGI-209 |
| 280 | AGI-211 | Multimodal grounding: replicate grounding with independently generated scenes | AGI-210 |
| 281 | AGI-212 | Compositional language: interpret nested quantifiers in local tasks | AGI-017 |
| 282 | AGI-213 | Compositional language: distinguish negation scope | AGI-212 |
| 283 | AGI-214 | Compositional language: resolve conditional instructions | AGI-213 |
| 284 | AGI-215 | Compositional language: preserve exception clauses during execution | AGI-214 |
| 285 | AGI-216 | Compositional language: interpret comparative and superlative constraints | AGI-215 |
| 286 | AGI-217 | Compositional language: bind pronouns to typed task entities | AGI-216 |
| 287 | AGI-218 | Compositional language: map paraphrases to equivalent contracts | AGI-217 |
| 288 | AGI-219 | Compositional language: detect contradictions between instruction clauses | AGI-218 |
| 289 | AGI-220 | Compositional language: generalize to new combinations of known grammar rules | AGI-219 |
| 290 | AGI-221 | Compositional language: replicate language grounding through executable outcome checks | AGI-220 |
| 291 | AGI-222 | Working memory: retain task state across irrelevant messages | AGI-021 |
| 292 | AGI-223 | Working memory: update slots without overwriting unrelated facts | AGI-222 |
| 293 | AGI-224 | Working memory: represent multiple competing hypotheses | AGI-223 |
| 294 | AGI-225 | Working memory: track delayed obligations across subtasks | AGI-224 |
| 295 | AGI-226 | Working memory: compress state while preserving decision variables | AGI-225 |
| 296 | AGI-227 | Working memory: allocate memory by measured future relevance | AGI-226 |
| 297 | AGI-228 | Working memory: recover task state after context truncation | AGI-227 |
| 298 | AGI-229 | Working memory: detect stale working-memory entries | AGI-228 |
| 299 | AGI-230 | Working memory: measure memory capacity across dependency depth | AGI-229 |
| 300 | AGI-231 | Working memory: replicate working-memory gains under new distractor distributions | AGI-230 |
| 301 | AGI-232 | Episodic memory: index episodes by causal outcome | AGI-022 |
| 302 | AGI-233 | Episodic memory: retrieve temporally distant relevant failures | AGI-232 |
| 303 | AGI-234 | Episodic memory: separate similar episodes with different contexts | AGI-233 |
| 304 | AGI-235 | Episodic memory: record counterfactual alternatives without inventing events | AGI-234 |
| 305 | AGI-236 | Episodic memory: recover episodes after storage interruption | AGI-235 |
| 306 | AGI-237 | Episodic memory: detect duplicated episodes across imports | AGI-236 |
| 307 | AGI-238 | Episodic memory: link action outcomes to exact observations | AGI-237 |
| 308 | AGI-239 | Episodic memory: apply explicit episode retention limits | AGI-238 |
| 309 | AGI-240 | Episodic memory: retrieve rare failures without losing common cases | AGI-239 |
| 310 | AGI-241 | Episodic memory: replicate episodic transfer on fresh sequential tasks | AGI-240 |
| 311 | AGI-242 | Semantic consolidation: extract reusable relations from episodes | AGI-023 |
| 312 | AGI-243 | Semantic consolidation: distinguish universal rules from local regularities | AGI-242 |
| 313 | AGI-244 | Semantic consolidation: preserve exceptions during rule compression | AGI-243 |
| 314 | AGI-245 | Semantic consolidation: merge equivalent abstractions with different names | AGI-244 |
| 315 | AGI-246 | Semantic consolidation: separate incompatible contextual rules | AGI-245 |
| 316 | AGI-247 | Semantic consolidation: track evidence supporting each abstraction | AGI-246 |
| 317 | AGI-248 | Semantic consolidation: revise abstractions after counterexamples | AGI-247 |
| 318 | AGI-249 | Semantic consolidation: measure consolidation cost against retrieval savings | AGI-248 |
| 319 | AGI-250 | Semantic consolidation: prevent unsupported facts during summarization | AGI-249 |
| 320 | AGI-251 | Semantic consolidation: replicate abstraction usefulness on unseen task families | AGI-250 |
| 321 | AGI-252 | Memory revision: resolve newer evidence with weaker reliability | AGI-025 |
| 322 | AGI-253 | Memory revision: represent unresolved contradictory sources | AGI-252 |
| 323 | AGI-254 | Memory revision: propagate retractions to dependent conclusions | AGI-253 |
| 324 | AGI-255 | Memory revision: restore previous beliefs after invalid corrections | AGI-254 |
| 325 | AGI-256 | Memory revision: track time-bounded validity of facts | AGI-255 |
| 326 | AGI-257 | Memory revision: separate user corrections from inferred preferences | AGI-256 |
| 327 | AGI-258 | Memory revision: detect circular evidence dependencies | AGI-257 |
| 328 | AGI-259 | Memory revision: quarantine corrupted memory records | AGI-258 |
| 329 | AGI-260 | Memory revision: rebuild derived knowledge after source removal | AGI-259 |
| 330 | AGI-261 | Memory revision: replicate revision behavior on adversarial correction sequences | AGI-260 |
| 331 | AGI-262 | Continual learning: learn smoothly drifting task distributions | AGI-027 |
| 332 | AGI-263 | Continual learning: adapt to abrupt mechanism switches | AGI-262 |
| 333 | AGI-264 | Continual learning: handle recurring tasks after long absence | AGI-263 |
| 334 | AGI-265 | Continual learning: learn imbalanced rare and frequent skills | AGI-264 |
| 335 | AGI-266 | Continual learning: distinguish new tasks from noisy old tasks | AGI-265 |
| 336 | AGI-267 | Continual learning: measure interference under reordered curricula | AGI-266 |
| 337 | AGI-268 | Continual learning: compare online and buffered updates | AGI-267 |
| 338 | AGI-269 | Continual learning: detect catastrophic forgetting before promotion | AGI-268 |
| 339 | AGI-270 | Continual learning: evaluate retention after extended inactivity | AGI-269 |
| 340 | AGI-271 | Continual learning: replicate continual learning across fresh task orders | AGI-270 |
| 341 | AGI-272 | Modular representations: discover modules from measured interference | AGI-029 |
| 342 | AGI-273 | Modular representations: route unseen combinations to known modules | AGI-272 |
| 343 | AGI-274 | Modular representations: share subskills without merging incompatible rules | AGI-273 |
| 344 | AGI-275 | Modular representations: detect redundant modules | AGI-274 |
| 345 | AGI-276 | Modular representations: split overloaded modules using held-out evidence | AGI-275 |
| 346 | AGI-277 | Modular representations: merge modules with equivalent behavior | AGI-276 |
| 347 | AGI-278 | Modular representations: measure routing errors independently of expert errors | AGI-277 |
| 348 | AGI-279 | Modular representations: preserve module interfaces through updates | AGI-278 |
| 349 | AGI-280 | Modular representations: compare modularity at matched parameter budgets | AGI-279 |
| 350 | AGI-281 | Modular representations: replicate modular transfer under new task mixtures | AGI-280 |
| 351 | AGI-282 | Replay and compression: select replay examples by measured forgetting | AGI-030 |
| 352 | AGI-283 | Replay and compression: preserve rare edge cases during compression | AGI-282 |
| 353 | AGI-284 | Replay and compression: compare generative and stored replay | AGI-283 |
| 354 | AGI-285 | Replay and compression: detect synthetic replay drift | AGI-284 |
| 355 | AGI-286 | Replay and compression: allocate replay across conflicting skills | AGI-285 |
| 356 | AGI-287 | Replay and compression: compress demonstrations into verified programs | AGI-286 |
| 357 | AGI-288 | Replay and compression: retain provenance through replay transformation | AGI-287 |
| 358 | AGI-289 | Replay and compression: measure privacy-sensitive synthetic record leakage locally | AGI-288 |
| 359 | AGI-290 | Replay and compression: reconstruct deleted replay indices from durable metadata | AGI-289 |
| 360 | AGI-291 | Replay and compression: replicate compression tradeoffs on unseen skill streams | AGI-290 |
| 361 | AGI-292 | Uncertainty calibration: calibrate confidence across difficulty strata | AGI-031 |
| 362 | AGI-293 | Uncertainty calibration: separate aleatoric and model uncertainty | AGI-292 |
| 363 | AGI-294 | Uncertainty calibration: track calibration after distribution changes | AGI-293 |
| 364 | AGI-295 | Uncertainty calibration: calibrate structured multi-step answers | AGI-294 |
| 365 | AGI-296 | Uncertainty calibration: propagate uncertainty through dependent subclaims | AGI-295 |
| 366 | AGI-297 | Uncertainty calibration: detect overconfidence after repeated failed attempts | AGI-296 |
| 367 | AGI-298 | Uncertainty calibration: calibrate confidence with limited observations | AGI-297 |
| 368 | AGI-299 | Uncertainty calibration: compare calibration across domains without pooling away failures | AGI-298 |
| 369 | AGI-300 | Uncertainty calibration: evaluate confidence when tools return conflicting results | AGI-299 |
| 370 | AGI-301 | Uncertainty calibration: replicate calibration on evaluator-owned task families | AGI-300 |
| 371 | AGI-302 | Abstention and assistance: abstain on unsupported factual claims | AGI-032 |
| 372 | AGI-303 | Abstention and assistance: request evidence for ambiguous observations | AGI-302 |
| 373 | AGI-304 | Abstention and assistance: escalate failed verification without fabricating success | AGI-303 |
| 374 | AGI-305 | Abstention and assistance: choose between more computation and assistance | AGI-304 |
| 375 | AGI-306 | Abstention and assistance: calibrate abstention under asymmetric error costs | AGI-305 |
| 376 | AGI-307 | Abstention and assistance: avoid excessive abstention on solvable tasks | AGI-306 |
| 377 | AGI-308 | Abstention and assistance: explain the missing information needed to proceed | AGI-307 |
| 378 | AGI-309 | Abstention and assistance: resume after receiving minimal clarification | AGI-308 |
| 379 | AGI-310 | Abstention and assistance: measure assistance burden over long tasks | AGI-309 |
| 380 | AGI-311 | Abstention and assistance: replicate cost-sensitive abstention on fresh scenarios | AGI-310 |
| 381 | AGI-312 | Distribution shift: detect changed observation encodings | AGI-033 |
| 382 | AGI-313 | Distribution shift: detect changed action consequences | AGI-312 |
| 383 | AGI-314 | Distribution shift: detect changed task objectives | AGI-313 |
| 384 | AGI-315 | Distribution shift: distinguish noise increases from mechanism changes | AGI-314 |
| 385 | AGI-316 | Distribution shift: identify novel combinations of known factors | AGI-315 |
| 386 | AGI-317 | Distribution shift: detect changes in available tools | AGI-316 |
| 387 | AGI-318 | Distribution shift: measure uncertainty on unfamiliar causal graphs | AGI-317 |
| 388 | AGI-319 | Distribution shift: localize which assumptions stopped holding | AGI-318 |
| 389 | AGI-320 | Distribution shift: recover without training on final evaluation labels | AGI-319 |
| 390 | AGI-321 | Distribution shift: replicate shift handling under mixed simultaneous changes | AGI-320 |
| 391 | AGI-322 | Formal reasoning: construct witnesses for existential claims | AGI-034 |
| 392 | AGI-323 | Formal reasoning: find counterexamples to universal claims | AGI-322 |
| 393 | AGI-324 | Formal reasoning: check induction base and step separately | AGI-323 |
| 394 | AGI-325 | Formal reasoning: translate finite constraints into executable checks | AGI-324 |
| 395 | AGI-326 | Formal reasoning: verify algebraic rewrites under explicit domains | AGI-325 |
| 396 | AGI-327 | Formal reasoning: detect vacuous proofs from inconsistent assumptions | AGI-326 |
| 397 | AGI-328 | Formal reasoning: compose proofs with tracked premises | AGI-327 |
| 398 | AGI-329 | Formal reasoning: identify unsupported lemma dependencies | AGI-328 |
| 399 | AGI-330 | Formal reasoning: distinguish bounded model checking from general proof | AGI-329 |
| 400 | AGI-331 | Formal reasoning: replicate proof construction on unseen theorem templates | AGI-330 |
| 401 | AGI-332 | Search allocation: estimate branch promise without target leakage | AGI-035 |
| 402 | AGI-333 | Search allocation: allocate search between diverse hypotheses | AGI-332 |
| 403 | AGI-334 | Search allocation: detect repeated equivalent search states | AGI-333 |
| 404 | AGI-335 | Search allocation: stop low-value branches using calibrated bounds | AGI-334 |
| 405 | AGI-336 | Search allocation: reuse partial solutions across related problems | AGI-335 |
| 406 | AGI-337 | Search allocation: compare breadth and depth under equal budgets | AGI-336 |
| 407 | AGI-338 | Search allocation: account for heuristic training costs | AGI-337 |
| 408 | AGI-339 | Search allocation: recover useful partial work after interruption | AGI-338 |
| 409 | AGI-340 | Search allocation: detect search policy overfitting to tuning tasks | AGI-339 |
| 410 | AGI-341 | Search allocation: replicate allocation gains on new branching structures | AGI-340 |
| 411 | AGI-342 | Causal identification: distinguish confounding from direct causation | AGI-036 |
| 412 | AGI-343 | Causal identification: identify collider-induced associations | AGI-342 |
| 413 | AGI-344 | Causal identification: infer causal direction with allowed interventions | AGI-343 |
| 414 | AGI-345 | Causal identification: detect latent confounder ambiguity | AGI-344 |
| 415 | AGI-346 | Causal identification: separate measurement error from mechanism noise | AGI-345 |
| 416 | AGI-347 | Causal identification: recover mechanisms under selection bias | AGI-346 |
| 417 | AGI-348 | Causal identification: represent nonidentifiable causal alternatives | AGI-347 |
| 418 | AGI-349 | Causal identification: choose interventions that distinguish rival graphs | AGI-348 |
| 419 | AGI-350 | Causal identification: test transport assumptions before transfer | AGI-349 |
| 420 | AGI-351 | Causal identification: replicate identification under fresh generated causal structures | AGI-350 |
| 421 | AGI-352 | Analogical transfer: map relational roles across renamed domains | AGI-037 |
| 422 | AGI-353 | Analogical transfer: transfer recursive structures across representations | AGI-352 |
| 423 | AGI-354 | Analogical transfer: distinguish useful analogies from misleading resemblance | AGI-353 |
| 424 | AGI-355 | Analogical transfer: infer partial mappings with missing elements | AGI-354 |
| 425 | AGI-356 | Analogical transfer: preserve constraints during analogical substitution | AGI-355 |
| 426 | AGI-357 | Analogical transfer: use multiple analogies without conflating premises | AGI-356 |
| 427 | AGI-358 | Analogical transfer: detect negative transfer before execution | AGI-357 |
| 428 | AGI-359 | Analogical transfer: charge analogy discovery and verification cost | AGI-358 |
| 429 | AGI-360 | Analogical transfer: learn when direct solving beats analogy | AGI-359 |
| 430 | AGI-361 | Analogical transfer: replicate transfer across independently held-out domain pairs | AGI-360 |
| 431 | AGI-362 | Counterexample learning: localize the minimal failing premise | AGI-038 |
| 432 | AGI-363 | Counterexample learning: generate counterexamples within explicit bounds | AGI-362 |
| 433 | AGI-364 | Counterexample learning: distinguish implementation bugs from wrong specifications | AGI-363 |
| 434 | AGI-365 | Counterexample learning: repair rules without memorizing individual failures | AGI-364 |
| 435 | AGI-366 | Counterexample learning: prioritize counterexamples by expected coverage | AGI-365 |
| 436 | AGI-367 | Counterexample learning: maintain a versioned failure regression set | AGI-366 |
| 437 | AGI-368 | Counterexample learning: detect contradictory repair requests | AGI-367 |
| 438 | AGI-369 | Counterexample learning: evaluate repairs on disjoint hidden cases | AGI-368 |
| 439 | AGI-370 | Counterexample learning: measure repair accumulation across generations | AGI-369 |
| 440 | AGI-371 | Counterexample learning: replicate counterexample-driven learning with independent failure generators | AGI-370 |
| 441 | AGI-372 | Depth generalization: extend nested expression depth beyond training | AGI-039 |
| 442 | AGI-373 | Depth generalization: execute longer loops with unseen trip counts | AGI-372 |
| 443 | AGI-374 | Depth generalization: solve deeper dependency graphs | AGI-373 |
| 444 | AGI-375 | Depth generalization: maintain invariants across recursive calls | AGI-374 |
| 445 | AGI-376 | Depth generalization: detect stack and memory limits explicitly | AGI-375 |
| 446 | AGI-377 | Depth generalization: separate arithmetic errors from control errors | AGI-376 |
| 447 | AGI-378 | Depth generalization: test extrapolation to longer symbol sequences | AGI-377 |
| 448 | AGI-379 | Depth generalization: measure error accumulation across composed operators | AGI-378 |
| 449 | AGI-380 | Depth generalization: compare learned iteration with unrolled computation | AGI-379 |
| 450 | AGI-381 | Depth generalization: replicate depth scaling across fresh recursive families | AGI-380 |
| 451 | AGI-382 | Metacognitive control: predict when extra reasoning is useful | AGI-040 |
| 452 | AGI-383 | Metacognitive control: choose between retrieval and fresh derivation | AGI-382 |
| 453 | AGI-384 | Metacognitive control: choose when to call a verifier | AGI-383 |
| 454 | AGI-385 | Metacognitive control: detect unproductive repeated reasoning | AGI-384 |
| 455 | AGI-386 | Metacognitive control: allocate effort across dependent subproblems | AGI-385 |
| 456 | AGI-387 | Metacognitive control: revise confidence after failed checks | AGI-386 |
| 457 | AGI-388 | Metacognitive control: stop at a preregistered sufficiency threshold | AGI-387 |
| 458 | AGI-389 | Metacognitive control: track opportunity cost across queued tasks | AGI-388 |
| 459 | AGI-390 | Metacognitive control: audit effort allocation under misleading confidence cues | AGI-389 |
| 460 | AGI-391 | Metacognitive control: replicate metacognitive gains on unseen workload mixtures | AGI-390 |
| 461 | AGI-392 | State-space planning: plan with interacting preconditions | AGI-041 |
| 462 | AGI-393 | State-space planning: handle irreversible simulated state transitions | AGI-392 |
| 463 | AGI-394 | State-space planning: detect unreachable goals | AGI-393 |
| 464 | AGI-395 | State-space planning: optimize plans under resource constraints | AGI-394 |
| 465 | AGI-396 | State-space planning: replan when intermediate goals become infeasible | AGI-395 |
| 466 | AGI-397 | State-space planning: compare heuristic admissibility and empirical usefulness | AGI-396 |
| 467 | AGI-398 | State-space planning: use abstractions without losing required state | AGI-397 |
| 468 | AGI-399 | State-space planning: verify plans against exact transition semantics | AGI-398 |
| 469 | AGI-400 | State-space planning: transfer plans across renamed action schemas | AGI-399 |
| 470 | AGI-401 | State-space planning: replicate planning on new generator families | AGI-400 |
| 471 | AGI-402 | Partial observability: maintain multimodal state beliefs | AGI-043 |
| 472 | AGI-403 | Partial observability: update beliefs after ambiguous observations | AGI-402 |
| 473 | AGI-404 | Partial observability: distinguish observation noise from hidden changes | AGI-403 |
| 474 | AGI-405 | Partial observability: plan actions with delayed observations | AGI-404 |
| 475 | AGI-406 | Partial observability: value information under action costs | AGI-405 |
| 476 | AGI-407 | Partial observability: avoid premature collapse to one hypothesis | AGI-406 |
| 477 | AGI-408 | Partial observability: recover after misleading sensor evidence | AGI-407 |
| 478 | AGI-409 | Partial observability: represent unobservable state equivalence | AGI-408 |
| 479 | AGI-410 | Partial observability: bound belief memory with measured loss | AGI-409 |
| 480 | AGI-411 | Partial observability: replicate partial-observation planning on fresh hidden-state systems | AGI-410 |
| 481 | AGI-412 | Experimental design: select tests distinguishing two mechanisms | AGI-044 |
| 482 | AGI-413 | Experimental design: design experiments under limited resets | AGI-412 |
| 483 | AGI-414 | Experimental design: choose measurement precision under cost limits | AGI-413 |
| 484 | AGI-415 | Experimental design: handle delayed experimental outcomes | AGI-414 |
| 485 | AGI-416 | Experimental design: detect when experiments are uninformative | AGI-415 |
| 486 | AGI-417 | Experimental design: balance replication against novel interventions | AGI-416 |
| 487 | AGI-418 | Experimental design: maintain preregistered stopping rules | AGI-417 |
| 488 | AGI-419 | Experimental design: update experiment plans after null results | AGI-418 |
| 489 | AGI-420 | Experimental design: separate discovery data from confirmation data | AGI-419 |
| 490 | AGI-421 | Experimental design: replicate active learning on evaluator-generated mechanisms | AGI-420 |
| 491 | AGI-422 | Hierarchical skills: discover reusable subgoal boundaries | AGI-045 |
| 492 | AGI-423 | Hierarchical skills: infer skill initiation conditions | AGI-422 |
| 493 | AGI-424 | Hierarchical skills: infer reliable skill termination conditions | AGI-423 |
| 494 | AGI-425 | Hierarchical skills: compose skills with compatible state contracts | AGI-424 |
| 495 | AGI-426 | Hierarchical skills: detect incompatible skill effects | AGI-425 |
| 496 | AGI-427 | Hierarchical skills: recover when a skill partially executes | AGI-426 |
| 497 | AGI-428 | Hierarchical skills: reuse skills at different temporal scales | AGI-427 |
| 498 | AGI-429 | Hierarchical skills: prune redundant or unreliable skills | AGI-428 |
| 499 | AGI-430 | Hierarchical skills: charge discovery cost to downstream benefit | AGI-429 |
| 500 | AGI-431 | Hierarchical skills: replicate hierarchy gains on unseen goal combinations | AGI-430 |
| 501 | AGI-432 | Tool interoperability: interpret unfamiliar typed local API schemas | AGI-047 |
| 502 | AGI-433 | Tool interoperability: adapt to reordered optional parameters | AGI-432 |
| 503 | AGI-434 | Tool interoperability: handle pagination and partial result contracts | AGI-433 |
| 504 | AGI-435 | Tool interoperability: distinguish read operations from state changes | AGI-434 |
| 505 | AGI-436 | Tool interoperability: validate returned values against declared types | AGI-435 |
| 506 | AGI-437 | Tool interoperability: handle versioned schema migrations | AGI-436 |
| 507 | AGI-438 | Tool interoperability: detect contradictory tool documentation | AGI-437 |
| 508 | AGI-439 | Tool interoperability: compose tools with incompatible data formats | AGI-438 |
| 509 | AGI-440 | Tool interoperability: learn a new tool from bounded examples | AGI-439 |
| 510 | AGI-441 | Tool interoperability: replicate tool transfer with evaluator-authored local APIs | AGI-440 |
| 511 | AGI-442 | Failure recovery: recover from transient timeouts | AGI-048 |
| 512 | AGI-443 | Failure recovery: handle success responses with missing effects | AGI-442 |
| 513 | AGI-444 | Failure recovery: handle effects committed before timeout | AGI-443 |
| 514 | AGI-445 | Failure recovery: detect stale cached observations | AGI-444 |
| 515 | AGI-446 | Failure recovery: retry idempotent operations safely | AGI-445 |
| 516 | AGI-447 | Failure recovery: compensate reversible partial transactions | AGI-446 |
| 517 | AGI-448 | Failure recovery: respect explicit terminal errors | AGI-447 |
| 518 | AGI-449 | Failure recovery: preserve evidence across interrupted tool calls | AGI-448 |
| 519 | AGI-450 | Failure recovery: bound retry storms under correlated failures | AGI-449 |
| 520 | AGI-451 | Failure recovery: replicate recovery under fresh injected fault combinations | AGI-450 |
| 521 | AGI-452 | Long-horizon execution: maintain goals across many subtasks | AGI-049 |
| 522 | AGI-453 | Long-horizon execution: track dependency changes during execution | AGI-452 |
| 523 | AGI-454 | Long-horizon execution: resume after delayed external observations in simulation | AGI-453 |
| 524 | AGI-455 | Long-horizon execution: detect cumulative small state errors | AGI-454 |
| 525 | AGI-456 | Long-horizon execution: manage limited resources over long runs | AGI-455 |
| 526 | AGI-457 | Long-horizon execution: schedule verification checkpoints by risk | AGI-456 |
| 527 | AGI-458 | Long-horizon execution: avoid drifting from the original task contract | AGI-457 |
| 528 | AGI-459 | Long-horizon execution: measure reliability at fixed task horizons | AGI-458 |
| 529 | AGI-460 | Long-horizon execution: estimate completion horizons with uncertainty | AGI-459 |
| 530 | AGI-461 | Long-horizon execution: replicate long-horizon performance on independent task compositions | AGI-460 |
| 531 | AGI-462 | Few-shot acquisition: learn unfamiliar arithmetic primitives from examples | AGI-051 |
| 532 | AGI-463 | Few-shot acquisition: learn unfamiliar string transformations | AGI-462 |
| 533 | AGI-464 | Few-shot acquisition: infer new finite-state transition rules | AGI-463 |
| 534 | AGI-465 | Few-shot acquisition: acquire relational predicates from sparse labels | AGI-464 |
| 535 | AGI-466 | Few-shot acquisition: learn functions with ambiguous demonstrations | AGI-465 |
| 536 | AGI-467 | Few-shot acquisition: select discriminating follow-up examples | AGI-466 |
| 537 | AGI-468 | Few-shot acquisition: generalize acquired skills to unseen inputs | AGI-467 |
| 538 | AGI-469 | Few-shot acquisition: transfer priors without leaking target definitions | AGI-468 |
| 539 | AGI-470 | Few-shot acquisition: separate adaptation gains from extra inference search | AGI-469 |
| 540 | AGI-471 | Few-shot acquisition: replicate acquisition efficiency across fresh primitive families | AGI-470 |
| 541 | AGI-472 | Curriculum learning: infer prerequisite relations from interventions | AGI-055 |
| 542 | AGI-473 | Curriculum learning: distinguish difficulty from novelty | AGI-472 |
| 543 | AGI-474 | Curriculum learning: select tasks near the learning frontier | AGI-473 |
| 544 | AGI-475 | Curriculum learning: avoid curricula dominated by easy reward | AGI-474 |
| 545 | AGI-476 | Curriculum learning: include rare prerequisites with delayed benefit | AGI-475 |
| 546 | AGI-477 | Curriculum learning: adapt curriculum after stalled learning | AGI-476 |
| 547 | AGI-478 | Curriculum learning: compare curriculum strategies with equal teaching budgets | AGI-477 |
| 548 | AGI-479 | Curriculum learning: retain previously learned families during progression | AGI-478 |
| 549 | AGI-480 | Curriculum learning: test curricula on new learners and task generators | AGI-479 |
| 550 | AGI-481 | Curriculum learning: replicate prerequisite-aware learning with independently sampled orders | AGI-480 |
| 551 | AGI-482 | Cross-domain transfer: transfer arithmetic abstractions into planning | AGI-056 |
| 552 | AGI-483 | Cross-domain transfer: transfer relational bindings into language execution | AGI-482 |
| 553 | AGI-484 | Cross-domain transfer: transfer causal reasoning into diagnosis | AGI-483 |
| 554 | AGI-485 | Cross-domain transfer: transfer memory policies into tool workflows | AGI-484 |
| 555 | AGI-486 | Cross-domain transfer: transfer uncertainty policies across modalities | AGI-485 |
| 556 | AGI-487 | Cross-domain transfer: detect conflicting domain-specific conventions | AGI-486 |
| 557 | AGI-488 | Cross-domain transfer: separate shared structure from superficial overlap | AGI-487 |
| 558 | AGI-489 | Cross-domain transfer: measure negative transfer before consolidation | AGI-488 |
| 559 | AGI-490 | Cross-domain transfer: compare transfer at matched total experience | AGI-489 |
| 560 | AGI-491 | Cross-domain transfer: replicate broad transfer on previously excluded domains | AGI-490 |
| 561 | AGI-492 | Sparse-feedback learning: explore when successful outcomes are rare | AGI-057 |
| 562 | AGI-493 | Sparse-feedback learning: learn with delayed terminal feedback | AGI-492 |
| 563 | AGI-494 | Sparse-feedback learning: assign credit across long action sequences | AGI-493 |
| 564 | AGI-495 | Sparse-feedback learning: distinguish accidental success from reliable skill | AGI-494 |
| 565 | AGI-496 | Sparse-feedback learning: use intrinsic signals without reward proxy gaming | AGI-495 |
| 566 | AGI-497 | Sparse-feedback learning: compare exploration under equal reset budgets | AGI-496 |
| 567 | AGI-498 | Sparse-feedback learning: recover from misleading sparse rewards | AGI-497 |
| 568 | AGI-499 | Sparse-feedback learning: learn from informative failures without hidden labels | AGI-498 |
| 569 | AGI-500 | Sparse-feedback learning: retain successful behavior after reward changes | AGI-499 |
| 570 | AGI-501 | Sparse-feedback learning: replicate sparse-feedback gains on fresh local environments | AGI-500 |
| 571 | AGI-502 | Goal clarification: identify missing task success criteria | AGI-062 |
| 572 | AGI-503 | Goal clarification: identify ambiguous referents | AGI-502 |
| 573 | AGI-504 | Goal clarification: identify conflicting resource constraints | AGI-503 |
| 574 | AGI-505 | Goal clarification: choose questions separating plausible interpretations | AGI-504 |
| 575 | AGI-506 | Goal clarification: avoid asking for already available information | AGI-505 |
| 576 | AGI-507 | Goal clarification: incorporate partial user answers | AGI-506 |
| 577 | AGI-508 | Goal clarification: proceed on reversible independent subtasks | AGI-507 |
| 578 | AGI-509 | Goal clarification: preserve unresolved assumptions explicitly | AGI-508 |
| 579 | AGI-510 | Goal clarification: measure user effort against avoided rework | AGI-509 |
| 580 | AGI-511 | Goal clarification: replicate clarification utility with held-out task ambiguities | AGI-510 |
| 581 | AGI-512 | Preference and constraint learning: distinguish hard constraints from soft preferences | AGI-064 |
| 582 | AGI-513 | Preference and constraint learning: learn preferences from explicit pairwise choices | AGI-512 |
| 583 | AGI-514 | Preference and constraint learning: represent context-dependent preferences | AGI-513 |
| 584 | AGI-515 | Preference and constraint learning: update preferences after user corrections | AGI-514 |
| 585 | AGI-516 | Preference and constraint learning: handle inconsistent preference observations | AGI-515 |
| 586 | AGI-517 | Preference and constraint learning: detect impossible combinations of constraints | AGI-516 |
| 587 | AGI-518 | Preference and constraint learning: explain concrete tradeoffs before consequential choices | AGI-517 |
| 588 | AGI-519 | Preference and constraint learning: avoid inventing preferences from demographic proxies | AGI-518 |
| 589 | AGI-520 | Preference and constraint learning: retain preference provenance across tasks | AGI-519 |
| 590 | AGI-521 | Preference and constraint learning: replicate preference handling on unseen constraint combinations | AGI-520 |
| 591 | AGI-522 | Authority and oversight: enforce scoped local tool permissions | AGI-070 |
| 592 | AGI-523 | Authority and oversight: reject instructions embedded in untrusted task data | AGI-522 |
| 593 | AGI-524 | Authority and oversight: separate proposal authority from approval authority | AGI-523 |
| 594 | AGI-525 | Authority and oversight: honor cancellation during long-running evaluation | AGI-524 |
| 595 | AGI-526 | Authority and oversight: stop queued dependent actions after revocation | AGI-525 |
| 596 | AGI-527 | Authority and oversight: preserve audit trails after denied operations | AGI-526 |
| 597 | AGI-528 | Authority and oversight: resume only within updated authorization | AGI-527 |
| 598 | AGI-529 | Authority and oversight: test least-privilege access to local artifacts | AGI-528 |
| 599 | AGI-530 | Authority and oversight: measure operator visibility into pending effects | AGI-529 |
| 600 | AGI-531 | Authority and oversight: replicate oversight under adversarial local instruction mixtures | AGI-530 |
| 601 | AGI-532 | Cooperative coordination: share task-relevant evidence with simulated partners | AGI-066 |
| 602 | AGI-533 | Cooperative coordination: allocate complementary subtasks without duplicate work | AGI-532 |
| 603 | AGI-534 | Cooperative coordination: resolve conflicting observations through provenance | AGI-533 |
| 604 | AGI-535 | Cooperative coordination: coordinate on shared bounded resources | AGI-534 |
| 605 | AGI-536 | Cooperative coordination: recover from a simulated partner interruption | AGI-535 |
| 606 | AGI-537 | Cooperative coordination: track commitments without assuming partner success | AGI-536 |
| 607 | AGI-538 | Cooperative coordination: detect stale shared plans | AGI-537 |
| 608 | AGI-539 | Cooperative coordination: compare centralized and decentralized coordination | AGI-538 |
| 609 | AGI-540 | Cooperative coordination: measure gains after charging communication overhead | AGI-539 |
| 610 | AGI-541 | Cooperative coordination: replicate cooperation with independently varied partner policies | AGI-540 |
| 611 | AGI-542 | Truthful reporting: ground completion claims in durable artifacts | AGI-067 |
| 612 | AGI-543 | Truthful reporting: separate attempts from successful effects | AGI-542 |
| 613 | AGI-544 | Truthful reporting: report failed tests alongside passing tests | AGI-543 |
| 614 | AGI-545 | Truthful reporting: distinguish estimates from measurements | AGI-544 |
| 615 | AGI-546 | Truthful reporting: expose unresolved uncertainty in task status | AGI-545 |
| 616 | AGI-547 | Truthful reporting: detect fabricated citations in synthetic documents | AGI-546 |
| 617 | AGI-548 | Truthful reporting: correct earlier reports after new evidence | AGI-547 |
| 618 | AGI-549 | Truthful reporting: avoid overstating bounded verification scope | AGI-548 |
| 619 | AGI-550 | Truthful reporting: align progress summaries with actual process state | AGI-549 |
| 620 | AGI-551 | Truthful reporting: replicate reporting accuracy under pressure and conflicting logs | AGI-550 |
| 621 | AGI-552 | Faithful explanation: identify evidence that changed a decision | AGI-068 |
| 622 | AGI-553 | Faithful explanation: distinguish correlated features from causal reasons | AGI-552 |
| 623 | AGI-554 | Faithful explanation: explain counterfactual decision changes | AGI-553 |
| 624 | AGI-555 | Faithful explanation: track premise dependence through composed reasoning | AGI-554 |
| 625 | AGI-556 | Faithful explanation: expose uncertainty in explanatory claims | AGI-555 |
| 626 | AGI-557 | Faithful explanation: test explanations after hidden premise interventions | AGI-556 |
| 627 | AGI-558 | Faithful explanation: detect plausible but unsupported causal stories | AGI-557 |
| 628 | AGI-559 | Faithful explanation: generate minimal sufficient executable witnesses | AGI-558 |
| 629 | AGI-560 | Faithful explanation: adapt explanation detail without changing factual claims | AGI-559 |
| 630 | AGI-561 | Faithful explanation: replicate explanation faithfulness on unseen reasoning mechanisms | AGI-560 |
| 631 | AGI-562 | Self-update attribution: separate update effects from additional search | AGI-071 |
| 632 | AGI-563 | Self-update attribution: separate learning effects from extra data | AGI-562 |
| 633 | AGI-564 | Self-update attribution: pair candidates on identical held-out tasks | AGI-563 |
| 634 | AGI-565 | Self-update attribution: randomize update assignment across independent runs | AGI-564 |
| 635 | AGI-566 | Self-update attribution: measure variance in update outcomes | AGI-565 |
| 636 | AGI-567 | Self-update attribution: detect selective reporting of successful generations | AGI-566 |
| 637 | AGI-568 | Self-update attribution: attribute gains to individual changed components | AGI-567 |
| 638 | AGI-569 | Self-update attribution: track regressions caused by coupled updates | AGI-568 |
| 639 | AGI-570 | Self-update attribution: estimate update benefit beyond tuning families | AGI-569 |
| 640 | AGI-571 | Self-update attribution: replicate causal update gains with independent evaluator seeds | AGI-570 |
| 641 | AGI-572 | Reversible improvement: constrain source mutations to declared interfaces | AGI-076 |
| 642 | AGI-573 | Reversible improvement: pin parent and child artifact lineage | AGI-572 |
| 643 | AGI-574 | Reversible improvement: verify equivalence for intended behavior-preserving changes | AGI-573 |
| 644 | AGI-575 | Reversible improvement: reject modifications to evaluator authority | AGI-574 |
| 645 | AGI-576 | Reversible improvement: inject regressions to test promotion gates | AGI-575 |
| 646 | AGI-577 | Reversible improvement: restore parent state after failed promotion | AGI-576 |
| 647 | AGI-578 | Reversible improvement: retest retained capabilities after every generation | AGI-577 |
| 648 | AGI-579 | Reversible improvement: detect objective drift across repeated updates | AGI-578 |
| 649 | AGI-580 | Reversible improvement: charge failed proposals and verification costs | AGI-579 |
| 650 | AGI-581 | Reversible improvement: replicate a bounded multi-generation improvement campaign | AGI-580 |
| 651 | AGI-582 | Integrated generality audit: evaluate one pinned system across all declared tracks | AGI-090, AGI-101, AGI-111, AGI-121, AGI-131, AGI-141, AGI-151, AGI-161, AGI-171, AGI-181, AGI-191, AGI-201, AGI-211, AGI-221, AGI-231, AGI-241, AGI-251, AGI-261, AGI-271, AGI-281, AGI-291, AGI-301, AGI-311, AGI-321, AGI-331, AGI-341, AGI-351, AGI-361, AGI-371, AGI-381, AGI-391, AGI-401, AGI-411, AGI-421, AGI-431, AGI-441, AGI-451, AGI-461, AGI-471, AGI-481, AGI-491, AGI-501, AGI-511, AGI-521, AGI-531, AGI-541, AGI-551, AGI-561, AGI-571, AGI-581 |
| 652 | AGI-583 | Integrated generality audit: audit hidden specialist selection and routing costs | AGI-582 |
| 653 | AGI-584 | Integrated generality audit: test unfamiliar combinations spanning multiple tracks | AGI-583 |
| 654 | AGI-585 | Integrated generality audit: measure new-skill acquisition after candidate freeze | AGI-584 |
| 655 | AGI-586 | Integrated generality audit: reconcile total training and evaluation experience | AGI-585 |
| 656 | AGI-587 | Integrated generality audit: apply worst-domain retention and reliability floors | AGI-586 |
| 657 | AGI-588 | Integrated generality audit: repeat adversarial shifts without revising success criteria | AGI-587 |
| 658 | AGI-589 | Integrated generality audit: obtain independent fresh-generator replication | AGI-588 |
| 659 | AGI-590 | Integrated generality audit: assemble unresolved failures and calibrated readiness evidence | AGI-589 |
| 660 | AGI-591 | Integrated generality audit: assess against the original rubric and report NOT_ESTABLISHED for missing gates | AGI-590 |
