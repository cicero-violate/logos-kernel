"""Train a small compositional-name neural baseline on known NC-007 macros only.

Dependencies: Python 3 stdlib, PyTorch, lexical_model.py, and pinned
artifacts/NC-007/composition_receipt.json. No held-out program or oracle label.
"""
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from lexical_model import Controller, ACTIONS, START, goal_features

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent/"NC-007"/"composition_receipt.json"
HELDOUT = "UNSEEN_XOR_DOUBLE_PLUS_Y"


def main():
    torch.set_num_threads(1)
    torch.manual_seed(28006)
    torch.use_deterministic_algorithms(True)
    ref = json.loads(REFERENCE.read_text())
    recipes = ref["versioned_macro_programs"]
    assert len(recipes) == 11 and HELDOUT not in recipes
    assert all(2 <= len(seq) <= 5 and seq[-1] == "HALT" for seq in recipes.values())
    rows = []
    for name, actions in sorted(recipes.items()):
        prev = START
        for position, action in enumerate(actions):
            assert action in ACTIONS
            rows.append((name, position, prev, ACTIONS.index(action)))
            prev = ACTIONS.index(action)
    X = torch.stack([goal_features(n) for n,p,prev,target in rows])
    P = torch.tensor([p for n,p,prev,target in rows])
    V = torch.tensor([prev for n,p,prev,target in rows])
    T = torch.tensor([target for n,p,prev,target in rows])
    model = Controller()
    optim = torch.optim.Adam(model.parameters(), lr=0.015)
    for step in range(800):
        logits = model(X, P, V)
        loss = F.cross_entropy(logits, T)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    teacher_correct = int((model(X,P,V).argmax(dim=1) == T).sum())
    payload = {"schema": "logos.nc028.lexical-checkpoint/1",
               "training_goals": sorted(recipes),
               "training_count": len(recipes),
               "supervised_token_rows": len(rows),
               "training_token_correct": teacher_correct,
               "training_token_total": len(rows),
               "source_receipt_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
               "architecture": "GoalNgramsToAutoregressiveActionPolicy",
               "updates": 800,
               "seed": 28006,
               "last_loss": float(loss),
               "model": model.state_dict()}
    path = HERE/"lexical_controller.pt"
    torch.save(payload, path)
    print(json.dumps({"checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "supervised_tokens": [teacher_correct,len(rows)],
                      "heldout_program_labels": 0,
                      "unseen_goal": HELDOUT}, sort_keys=True))


if __name__ == "__main__":
    main()
