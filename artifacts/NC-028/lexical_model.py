"""Small goal-text neural proposer. No oracle or taught program tables here.

Dependencies: Python 3 stdlib, PyTorch; local lexical_controller.pt trained by
train_lexical.py. Sandbox restricts reads, network and writes at inference.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ACTIONS = ("ADD_XX", "ADD_X1", "AND1_X", "XOR_XY", "SHL_R", "AND_XY",
           "ADD_RX", "SUB_XY", "XOR_RX", "OR_XY", "SHR_R", "ADD_XY",
           "XOR_RY", "ADD_RY", "HALT")
DIM = 27 + 512
MAX_STEPS = 5
START = len(ACTIONS)
VALID_GOAL = re.compile(r"[A-Z_]{1,48}\Z")


def goal_features(goal):
    if type(goal) is not str or not VALID_GOAL.fullmatch(goal):
        raise ValueError("invalid goal text")
    # Stable shared characters/ngrams. No goal-ID lookup or special-case program.
    v = torch.zeros(DIM)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
    for c in goal:
        v[alphabet.index(c)] += 1
    for k in (2, 3):
        for i in range(len(goal) - k + 1):
            token = goal[i:i+k].encode("ascii")
            bucket = int.from_bytes(hashlib.sha256(token).digest()[:4], "big") % 512
            v[27 + bucket] += 1
    return v / max(1.0, float(v.sum()))


class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.prev = nn.Embedding(START + 1, 16)
        self.position = nn.Embedding(MAX_STEPS, 16)
        self.net = nn.Sequential(nn.Linear(DIM+32, 96), nn.Tanh(),
                                 nn.Linear(96, 96), nn.Tanh(),
                                 nn.Linear(96, START))
    def forward(self, features, positions, previous):
        return self.net(torch.cat((features, self.prev(previous),
                                   self.position(positions)), dim=-1))


def sequence_log_prob(model, feature, actions):
    prev = START
    total = torch.zeros((), dtype=feature.dtype)
    for pos, action in enumerate(actions):
        logits = model(feature.unsqueeze(0), torch.tensor([pos]),
                       torch.tensor([prev]))[0]
        index = ACTIONS.index(action)
        total = total + F.log_softmax(logits, dim=0)[index]
        prev = index
    return total


def propose(model, feature, rejected):
    # Neural-logit beam; only grammar validity, never oracle values, filters it.
    beam = [(0.0, (), START)]
    complete = []
    with torch.no_grad():
        for pos in range(MAX_STEPS):
            following = []
            for score, seq, prev in beam:
                logits = model(feature.unsqueeze(0), torch.tensor([pos]),
                               torch.tensor([prev]))[0]
                logp = F.log_softmax(logits, dim=0)
                top = torch.topk(logp, k=5).indices.tolist()
                for idx in top:
                    token = ACTIONS[idx]
                    next_seq = seq + (token,)
                    next_score = score + float(logp[idx])
                    if token == "HALT":
                        if len(next_seq) >= 2 and next_seq not in rejected:
                            complete.append((next_score, next_seq))
                    elif pos < MAX_STEPS - 1:
                        following.append((next_score, next_seq, idx))
            following.sort(key=lambda a: (-a[0], a[1]))
            beam = following[:48]
    complete.sort(key=lambda a: (-a[0], a[1]))
    return list(complete[0][1]) if complete else None


def model_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def read_input(data):
    if type(data) is not dict or set(data) != {"goal", "checkpoint_sha256", "feedback"}:
        raise ValueError("invalid request schema")
    goal = data["goal"]
    goal_features(goal)
    digest = data["checkpoint_sha256"]
    if type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid checkpoint digest")
    feedback = data["feedback"]
    if type(feedback) is not list or len(feedback) > 2:
        raise ValueError("invalid feedback")
    prior = set()
    for row in feedback:
        if type(row) is not dict or set(row) != {"actions", "correct", "tested", "counterexample"}:
            raise ValueError("invalid feedback schema")
        actions = row["actions"]
        if (type(actions) is not list or not 2 <= len(actions) <= MAX_STEPS
                or actions[-1] != "HALT" or "HALT" in actions[:-1]
                or any(type(x) is not str or x not in ACTIONS for x in actions)
                or type(row["correct"]) is not int or type(row["tested"]) is not int
                or not 0 <= row["correct"] < row["tested"] <= 16):
            raise ValueError("invalid counterexample feedback")
        cx = row["counterexample"]
        if (type(cx) is not dict or set(cx) != {"pair_index", "x", "y", "expected", "observed"}
                or any(type(cx[key]) is not int for key in cx)
                or not 0 <= cx["pair_index"] < row["tested"]
                or not all(0 <= cx[key] <= 65535 for key in ("x", "y", "expected", "observed"))
                or cx["expected"] == cx["observed"]):
            raise ValueError("invalid numeric counterexample")
        prior.add(tuple(actions))
    return goal, digest, feedback, prior


def main():
    torch.set_num_threads(1)
    goal, digest, feedback, rejected = read_input(json.load(sys.stdin))
    checkpoint = HERE / "lexical_controller.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise RuntimeError("lexical checkpoint hash mismatch")
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (data["schema"] != "logos.nc028.lexical-checkpoint/1"
            or goal in data["training_goals"]):
        # This script is specifically for held-out goal proposals.
        raise ValueError("goal overlaps supervised training")
    model = Controller().cpu()
    model.load_state_dict(data["model"], strict=True)
    model.eval()
    before = model_digest(model)
    feature = goal_features(goal)
    # Online counterexample-conditioned *rejection* update. A real numeric
    # mismatch scales the negative gradient; no target action sequence is given.
    # This is NOT differentiable execution, CEGIS synthesis, or proof of repair.
    if feedback:
        for _ in range(12):
            loss = sum(
                sequence_log_prob(model, feature, row["actions"])
                * ((row["tested"] - row["correct"]) / row["tested"])
                * (1.0 + abs(row["counterexample"]["expected"] -
                             row["counterexample"]["observed"]) / 65535.0)
                for row in feedback
            )
            model.zero_grad(set_to_none=True)
            loss.backward()
            # Manual bounded SGD avoids optimizer graph/cache initialization
            # inside the sealed subprocess; only this in-memory copy changes.
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(parameter.grad, alpha=-0.015)
    after = model_digest(model)
    actions = propose(model, feature, rejected)
    print(json.dumps({
        "status": "PROPOSED" if actions else "NO_ALTERNATIVE",
        "actions": actions or [],
        "model_generated": bool(actions),
        "structurally_unseen_goal": True,
        "feedback_uses_labels": bool(feedback),
        "feedback_uses_heldout_labels": False,
        "feedback_type": "numeric-counterexample-weighted-rejection",
        "online_gradient_updates": 12 if feedback else 0,
        "weights_before": before,
        "weights_after": after,
        "checkpoint_sha256": digest,
        "source": "learned character/ngram goal-conditioned controller"
    }, sort_keys=True))


if __name__ == "__main__":
    main()
