# Proposed 160-stage progression toward AGI

69 existing kernel tasks are reused; 91 additional milestones extend them. This is a dependency-ordered research roadmap, not a guarantee, a global optimum, or an assertion that AGI has been achieved.

The immediate spine remains: exact baselines → sealed scoreboard → verified invention and reusable operators → causal intervention/transfer → measured generation-over-generation gain → bounded unattended campaigns. Later branches develop generality, memory, uncertainty, planning, new-skill learning, human oversight and independent evaluation.

Canonical task definitions live in `../../instructions.jsonl`. This readable index does not assign workers or mark tasks complete. Runtime priority keeps existing tasks in their current relative order; new milestones follow until evidence justifies a future priority snapshot.

Optional existing lexicon and speculative hardware branches remain in the kernel but are not prerequisites for this core 160-stage progression.

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

Research basis: [On the Measure of Intelligence](https://arxiv.org/abs/1911.01547v2); [Levels of AGI](https://arxiv.org/abs/2311.02462v5); [Measuring AI Ability to Complete Long Software Tasks](https://arxiv.org/abs/2503.14499v4); [DreamCoder](https://arxiv.org/abs/2006.08381v1); [Towards Causal Representation Learning](https://arxiv.org/abs/2102.11107v1).

Every new task declares a separate independent review receipt. A scientific NO_GAIN result may close an experiment, but dependent capability claims still require the stated evidence; a passed task is not automatic proof of AGI.
