# NC-024 preregistered neural-guided rewrite ranking

NC-020 fixed benchmark only. Train: first 6 equivalent examples per train family (36).
Evaluation: first 3 equivalent examples per validation/test family (12/15).
No heldout target RHS or optimized code fed into training. Three seeds 11,19,29.
Two epochs AdamW lr .004; shared recurrent z_H/z_L two-state attention, 32 hidden,
4 heads. Training labels are deterministic one-step minimum static cost,
not novel algorithm discovery.
Controls: trained recurrent attention, untrained same architecture, deterministic
lowest static cost, seeded uniform random. Identical NC-023 candidate generator
and proof oracle; 6 proof proposals per case; 40 frontier slots.
Record critical-path wall/CPU, full training CPU, train split, checkpoint hashes,
candidate traces, rejected invalid/non-equivalent validation and test controls.
NO_GAIN unless neural strictly exceeds all three controls in improved cases
and total DAG cost with no greater full CPU inclusive of training.
Static cost is ideal DAG operations, not measured hardware speed.
RLT-style is a newly initialized model, NOT adapted original RLT champion.
No new rewrite rules, no champion modification, no service, no AGI claim.
Dependencies: Python 3.10+, PyTorch, NC-001, NC-020, NC-023, Git.
