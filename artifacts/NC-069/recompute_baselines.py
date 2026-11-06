"""Recompute only the fixed baselines in results.json after fixing the majority baseline; no model is retrained."""
import json
from pathlib import Path

import causal_benchmark as cb
import train_eval

HERE = Path(__file__).resolve().parent
report = json.loads((HERE / "results.json").read_text())
config = report["config"]
tests = {b: cb.generate(1000, config["test_samples"], b) for b in train_eval.BUCKETS}
report["baselines"] = train_eval.baselines(tests, cb.generate(config["seeds"][0], config["train_samples"], "train"))
(HERE / "results.json").write_text(json.dumps(report, indent=1))
print({k: v["accuracy"] for k, v in report["baselines"]["heldout_combo"].items()})
