"""Fast replay-integrity tests; no model sampling."""
import json
import tempfile
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_checkpoints as rc

class ReplayTests(unittest.TestCase):
    def test_chain_detects_mutation_and_reordering(self):
        rows=[{"goal":"INC","candidates":98},{"goal":"DEC","candidates":465}]
        altered=[dict(r) for r in rows]
        altered[0]["candidates"]=99
        self.assertNotEqual(rc.chain(rows),rc.chain(altered))
        self.assertNotEqual(rc.chain(rows),rc.chain(rows[::-1]))
    def test_atomic_json_replace(self):
        with tempfile.TemporaryDirectory() as p:
            f=Path(p)/"replay.json"
            rc.atomic_json(f,{"status":"PARTIAL"})
            rc.atomic_json(f,{"status":"COMPLETE"})
            self.assertEqual(json.loads(f.read_text())["status"],"COMPLETE")
            self.assertEqual(list(Path(p).iterdir()),[f])
    def test_pinned_first_six_exact(self):
        p=Path(__file__).resolve().parent
        state=json.loads((p/"replay_seed11_first6.json").read_text())
        self.assertEqual((state["status"],state["completed_goals"]),("PARTIAL",6))
        self.assertEqual(state["chain_sha256"],rc.chain(state["records"]))
        archive=[json.loads(row) for row in (p/"proposals.jsonl").read_text().splitlines()]
        for row in state["records"]:
            hits=[x for x in archive if x["seed"]==11 and x["arm"]=="neural" and x["goal"]==row["goal"]]
            self.assertEqual(len(hits),int(row["solved"]))
            if hits:
                for field in ("candidates","program","revisions"):
                    self.assertEqual(row[field],hits[0][field])
if __name__=="__main__":unittest.main()
