"""NC-070 frozen evidence replay and adversarial integrity checks."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import recompute


class EvidenceTests(unittest.TestCase):
    def test_recovery_counts_and_all_210_eval_rows_recompute(self):
        data = recompute.check(ROOT)
        self.assertEqual(data["status"], "REPRODUCED")
        self.assertEqual(data["rows"], 210)
        self.assertEqual(data["traces"], 630)
        self.assertEqual(data["exact"], {"active": 52, "random": 42, "passive": 1})

    def test_changed_graph_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)
            shutil.copyfile(ROOT/"experiment_traces.jsonl",dest/"experiment_traces.jsonl")
            report=json.loads((ROOT/"discovery_results.json").read_text())
            report["evaluation_rows"][0]["truth_graph_id"]=-7
            (dest/"discovery_results.json").write_text(json.dumps(report))
            with self.assertRaises(AssertionError):
                recompute.check(dest)

    def test_changed_heldout_trace_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)
            shutil.copyfile(ROOT/"discovery_results.json",dest/"discovery_results.json")
            traces=(ROOT/"experiment_traces.jsonl").read_text().splitlines()
            row=json.loads(traces[0])
            row["observed_register_values"]=[9,9,9,9]
            traces[0]=json.dumps(row)
            (dest/"experiment_traces.jsonl").write_text("\n".join(traces)+"\n")
            with self.assertRaises(AssertionError):
                recompute.check(dest)


if __name__=="__main__":
    unittest.main()
