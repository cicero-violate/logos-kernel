"""Fail-closed tests for the partial NC-081 evidence auditor.

Dependencies: Python 3 stdlib and the already-committed NC-081 evidence files.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_rework_872033 as mod


class AuditGateTests(unittest.TestCase):
    def test_missing_three_seed_replay_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "missing replay"):
                mod.audit(Path(tmp))

    def test_wrong_dependency_digest_is_refused_without_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "seed11"
            folder.mkdir()
            data = {
                "schema": "logos.nc081.results/1",
                "seeds": {"11": {}},
                "config": {"dreams": 30000, "steps": 3000,
                           "exam_goals": 300, "top_k": 12, "budget": 8000},
                "la015_library_py_sha256": "0" * 64,
            }
            (folder / "selector_results.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "loaded different dependency"):
                mod.audit(Path(tmp))


if __name__ == "__main__":
    unittest.main()
