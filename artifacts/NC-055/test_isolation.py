"""NC-055 negative controls for reversible IDs and candidate/oracle isolation.

Dependencies: Python 3 stdlib, oracle.py, benchmark.py, sandbox_evaluator.py,
bubblewrap for OS-confinement tests. No secret key or oracle future is sent.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import benchmark
import oracle
import sandbox_evaluator


class IsolationTests(unittest.TestCase):
    def test_published_id_dictionary_cannot_reverse_labels(self):
        # Reproduce the independent reviewer's original 99/99 attack exactly,
        # against the replacement public identifiers.
        seen = set()
        matched = 0
        for split in oracle.SPLITS:
            for public, private in oracle.cases(split):
                self.assertEqual(len(public["case_id"]), 32)
                self.assertNotIn(public["case_id"], seen)
                seen.add(public["case_id"])
                hits = [(fam, rep) for fam in oracle.FAMILIES for rep in range(3)
                        if hashlib.sha256(
                            f"NC055:{oracle.SEED}:{fam}:{split}:{rep}".encode()
                        ).hexdigest()[:32] == public["case_id"]]
                self.assertEqual(hits, [], (split, private["family"]))
                matched += len(hits)
        self.assertEqual(len(seen), 99)
        self.assertEqual(matched, 0)

    def test_ephemeral_id_changes_on_new_evaluator_process(self):
        public, _ = oracle.make_case("periodic", "test", 1)
        child = subprocess.check_output([
            sys.executable, "-c", "import oracle; print(oracle.make_case('periodic','test',1)[0]['case_id'])"
        ], text=True).strip()
        self.assertNotEqual(child, public["case_id"])

    def test_randomized_baseline_does_not_depend_on_identifier_secret(self):
        public, _ = oracle.make_case("periodic", "test", 0)
        baseline = benchmark.forecast(public, "randomized")
        alternate = {**public, "case_id":"not-a-secret"}
        self.assertEqual(baseline, benchmark.forecast(alternate, "randomized"))

    @unittest.skipUnless(shutil.which("bwrap"), "bubblewrap not installed")
    def test_isolated_participant_cannot_read_or_import_oracle(self):
        # A candidate sees only JSONL public views and its own source file.
        program = '''import importlib.util,json,os,sys
if os.path.exists("/workspace/experiments/logos-kernel/artifacts/NC-055/oracle.py"):
    raise RuntimeError("oracle source visible")
if importlib.util.find_spec("oracle") is not None:
    raise RuntimeError("oracle module importable")
for line in sys.stdin:
    item=json.loads(line)
    zero=[0.0,0.0] if item["shape"]=="vector2" else 0.0
    out={"case_id":item["case_id"],"forecast":[zero]*len(item["query_times"]),"law":None}
    print(json.dumps(out),flush=True)
'''
        with tempfile.TemporaryDirectory() as td:
            path = Path(td, "candidate.py")
            path.write_text(program)
            for split, n in (("train",27),("validation",36),("test",36)):
                score = sandbox_evaluator.evaluate(path, split)
                self.assertEqual(score["denominator"], n)
                self.assertEqual(score["bounded_discovery_pass"], 0)

    @unittest.skipUnless(shutil.which("bwrap"), "bubblewrap not installed")
    def test_published_id_only_exploit_fails_in_isolation(self):
        # No public generation ID can be inverted via the old keyed-by-seed
        # dictionary; no direct oracle import or history-order side channel.
        program = '''import hashlib,json,sys
families=("constant","polynomial","geometric","periodic","multi_frequency","nonlinear_modular","logistic_chaos","switching","coupled_vector","nonuniform_polynomial","random_noise","aliasing")
for line in sys.stdin:
    item=json.loads(line)
    matches=[(family,rep) for family in families for rep in range(3) if hashlib.sha256(f"NC055:55061:{family}:{item['split']}:{rep}".encode()).hexdigest()[:32]==item["case_id"]]
    if matches: raise RuntimeError("reversed secret ID")
    zero=[0.0,0.0] if item["shape"]=="vector2" else 0.0
    print(json.dumps({"case_id":item["case_id"],"forecast":[zero]*len(item["query_times"]),"law":None}),flush=True)
'''
        with tempfile.TemporaryDirectory() as td:
            path = Path(td, "attack.py")
            path.write_text(program)
            self.assertEqual(sandbox_evaluator.evaluate(path)["bounded_discovery_pass"],0)

    def test_fail_closed_when_bwrap_unavailable(self):
        from unittest.mock import patch
        with patch.object(sandbox_evaluator.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError,"isolation unavailable"):
                sandbox_evaluator.evaluate(__file__)


if __name__ == "__main__":
    unittest.main()
