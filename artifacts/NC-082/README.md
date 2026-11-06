# NC-082: a Transformer controller that runs programs it has never seen

The earlier controller took a fixed ROM. Here the program is an input that varies per example. A 3-layer Transformer (289,564 parameters) reads program tokens, PC and registers, and predicts the opcode, operands, destination, immediate and next PC. This is implementer evidence, not independent verification.

- **Training:** random programs with exact one-step targets, half of them on a JZ with its register balanced between zero and nonzero. No full program is ever rolled out in training and no multiplication answer appears.
- **Testing:** no teacher forcing. Each predicted next PC feeds the next step.
- **What does what:** the Transformer predicts each instruction's fields and next PC. NC-080's learned operators compute ALU bits. The host keeps registers, feeds predictions forward, iterates bits and dispatches on the predicted opcode.
- **Not claimed:** this runs programs it is given. It does not invent them; NC-010 covers that.

    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-082/test_controller.py
    OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-082/controller.py --library artifacts/NC-082/library_snapshot.jsonl --seeds 11 --out /tmp/nc082-replay
    OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-082/extended_eval.py
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python artifacts/NC-082/audit_by_category.py

Dependencies: Python 3, NumPy, PyTorch CPU, and `artifacts/NC-080` source and checkpoints.

`first_run_results_no_branch_curriculum.json` records seed 11 before the zero-test curriculum: 120 of 120 concepts and 7 of 256 multiplications, with every one-step error on JZ next PC.
