"""NC-078 diagnostic: does a trained model use the examples? Compares program likelihood given real and shuffled examples."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import synth  # noqa: E402
import train_synth as ts  # noqa: E402


def log_likelihood(model, bodies, ex):
    tokens = torch.tensor([ts.encode_body(b) for b in bodies])
    x = torch.cat([torch.tensor(ex), torch.full((len(bodies), 1), ts.BOS), tokens[:, :-1]], dim=1)
    with torch.no_grad():
        logits = model(x)[:, ts.PREFIX:]
    logp = torch.log_softmax(logits, -1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return (logp * (tokens != ts.PAD)).sum(1).numpy()


def evaluate(model, items, rng):
    bodies = [b for b, _ in items]
    tables = np.stack([t for _, t in items])
    inputs = np.stack([rng.choice(synth.DOMAIN, ts.EXAMPLES, replace=False) for _ in items])
    real = np.take_along_axis(tables, inputs, axis=1)
    shuffled = np.stack([rng.permutation(t)[:ts.EXAMPLES] for t in tables])
    other = real[rng.permutation(len(items))]
    a = log_likelihood(model, bodies, ts.example_tokens(inputs, real))
    b = log_likelihood(model, bodies, ts.example_tokens(inputs, shuffled))
    c = log_likelihood(model, bodies, ts.example_tokens(inputs, other))
    return {"programs": len(items), "mean_log_likelihood_real": round(float(a.mean()), 3),
            "gain_over_shuffled_outputs_nats": round(float((a - b).mean()), 3),
            "gain_over_another_tasks_outputs_nats": round(float((a - c).mean()), 3),
            "share_more_likely_with_real": round(float((a > b).mean()), 3)}


def main(seed=11):
    torch.set_num_threads(1)
    model = ts.Model()
    model.load_state_dict(torch.load(HERE / f"model_seed{seed}.pt", weights_only=True)["state"])
    model.eval()
    rng = np.random.default_rng(7)
    train = synth.training_programs(seed, 40000)
    picks = rng.choice(len(train), 800, replace=False)
    seen = [train[i] for i in picks]
    gen = __import__("random").Random("NC-078/diagnostic")
    unseen = []
    while len(unseen) < 800:
        body = synth.random_body(gen)
        table = synth.evaluate(body)
        if synth.is_interesting(table) and synth.is_heldout_function(table):
            unseen.append((body, table))
    # A program of chance-level likelihood costs about this many nats under the grammar alone.
    report = {"seed": seed, "train": evaluate(model, seen, rng), "heldout": evaluate(model, unseen, rng),
              "reading": "A gain near zero means the model ignores the examples. Gains are in nats, so 2.3 means about 10 times more likely."}
    (HERE / "spec_sensitivity.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
