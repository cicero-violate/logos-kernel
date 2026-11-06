import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
import sys

sys.path.insert(0, str(Path(__file__).parent))
import trainer_isolation as t

SOURCE = Path(__file__).parent / "probe_worker.py"

class IsolationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t.DEFAULT_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="nc044-", dir=t.DEFAULT_ROOT)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def new(self, source=SOURCE, **kwargs):
        return t.make_run(self.root, source, [], 10, 5, 512, 5, **kwargs) if kwargs else t.make_run(self.root, source, [], 10, 5, 512, 5)

    def test_champion_pin_and_nc022_source(self):
        run, x = self.new()
        self.assertEqual(t.digest(t.CHAMPION), t.CHAMPION_SHA256)
        self.assertEqual(x["champion_sha256"], t.CHAMPION_SHA256)
        self.assertEqual(x["nc022_sha256"], t.digest(t.NC022))
        self.assertEqual(x["role"], "TRAINER")
        self.assertNotEqual(x["worker_id"], "tab-3")
        self.assertNotEqual(run, t.REPO)

    def test_ancestor_symlink_and_traversal(self):
        fake = self.root / "source-link.py"
        fake.symlink_to(SOURCE)
        with self.assertRaises(t.BoundaryError):
            self.new(source=fake)
        with self.assertRaises(t.BoundaryError):
            t.safe_path(str(SOURCE.parent / ".." / SOURCE.parent.name / SOURCE.name))
        with self.assertRaises(t.BoundaryError):
            t.validate_root(str(t.REPO / "artifacts/NC-044/runs"))

    def test_input_hash_and_reuse(self):
        data = self.root / "verified.bin"
        data.write_bytes(b"known fixed toy bytes")
        h = t.digest(data)
        run, x = t.make_run(self.root, SOURCE, ["toy:" + str(data) + ":" + h], 10, 5, 512, 5)
        t.checks(x)
        data.write_bytes(b"tampered")
        with self.assertRaisesRegex(t.BoundaryError, "changed pinned dataset"):
            t.checks(x)
        with self.assertRaises(t.BoundaryError):
            t.make_run(self.root, SOURCE, ["toy:" + str(data) + ":" + h], 10, 5, 512, 5)

    def test_worker_write_boundary(self):
        run, x = self.new()
        before = {str(p): t.digest(p) for p in [t.CHAMPION, t.REPO / "instructions.jsonl", t.NC022]}
        r = t.execute(self.root, x["run_id"])
        self.assertEqual(r["outcome"], "completed", r)
        self.assertEqual(r["returncode"], 0, r)
        self.assertEqual(before, {k: t.digest(Path(k)) for k in before})
        attack = json.loads((run / "output/denials.json").read_text())
        self.assertTrue(attack["host_write_denied"])
        self.assertTrue(all(x[1] == "DENIED" for x in attack["attempts"]))
        self.assertFalse((run / "output/run.json").exists())
        self.assertFalse((run / "output/last_execution.json").exists())

    def test_pause_resume_and_deterministic_checkpoint(self):
        run, x = self.new()
        results = []
        thread = threading.Thread(target=lambda: results.append(t.execute(self.root, x["run_id"])))
        thread.start()
        for _ in range(200):
            state_file = run / "output/checkpoint.json"
            if state_file.exists() and json.loads(state_file.read_text())["step"] >= 2:
                break
            time.sleep(.02)
        else:
            self.fail("checkpoint never appeared")
        (run / "output/pause.request").touch()
        thread.join(timeout=15)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results[0]["outcome"], "paused", results)
        saved = json.loads(state_file.read_text())
        self.assertGreaterEqual(saved["step"], 2)
        self.assertLess(saved["step"], 8)
        with self.assertRaises(t.BoundaryError):
            t.execute(self.root, x["run_id"])
        resumed = t.execute(self.root, x["run_id"], resume=True)
        self.assertEqual(resumed["outcome"], "completed", resumed)
        final = json.loads(state_file.read_text())
        self.assertEqual(final["step"], 8)
        other_run, other = self.new()
        unbroken = t.execute(self.root, other["run_id"])
        self.assertEqual(unbroken["outcome"], "completed", unbroken)
        other_final = json.loads((other_run / "output/checkpoint.json").read_text())
        self.assertEqual(final["optimizer"], other_final["optimizer"])
        self.assertEqual(final["rng_state"], other_final["rng_state"])

    def test_cancel_blocks_restart(self):
        run, x = self.new()
        (run / "output/cancel.request").touch()
        with self.assertRaisesRegex(t.BoundaryError, "cancelled"):
            t.execute(self.root, x["run_id"])

    def test_exclusive_run_lock(self):
        run, x = self.new()
        results = []
        thread = threading.Thread(target=lambda: results.append(t.execute(self.root, x["run_id"])))
        thread.start()
        lock = run / ".lock"
        for _ in range(200):
            if lock.exists() and (run / "output/checkpoint.json").exists():
                break
            time.sleep(.02)
        else:
            self.fail("job did not start")
        with self.assertRaisesRegex(t.BoundaryError, "already running"):
            t.execute(self.root, x["run_id"])
        thread.join(timeout=15)
        self.assertEqual(results[0]["outcome"], "completed", results)

    def write_worker(self, content):
        p = self.root / ("attack-" + str(time.time_ns()) + ".py")
        p.write_text(content.replace(chr(92) + "n", chr(10)))
        return p

    def test_wall_time_quota(self):
        source = self.write_worker("while True: pass\\n".replace("\\\\n", "\\n"))
        run, x = t.make_run(self.root, source, [], 1, 8, 256, 2)
        result = t.execute(self.root, x["run_id"])
        self.assertEqual(result["outcome"], "wall_timeout", result)

    def test_address_space_memory_quota(self):
        source = self.write_worker(
            "import sys\\ntry: buf=bytearray(256*1024*1024)\\n"
            "except MemoryError: sys.exit(0)\\nsys.exit(7)\\n".replace("\\\\n", "\\n"))
        run, x = t.make_run(self.root, source, [], 5, 4, 128, 2)
        result = t.execute(self.root, x["run_id"])
        self.assertEqual(result["outcome"], "completed", result)

    def test_single_file_disk_quota(self):
        source = self.write_worker(
            "import os,sys\\ntry:\\n"
            " with open('/output/oversized','wb') as f: f.write(b'a'*(2*1024*1024))\\n"
            "except OSError: sys.exit(0)\\nsys.exit(7)\\n".replace("\\\\n", "\\n"))
        run, x = t.make_run(self.root, source, [], 5, 4, 256, 1)
        result = t.execute(self.root, x["run_id"])
        self.assertEqual(result["outcome"], "completed", result)

    def test_checkpoint_resume_tampering_refused(self):
        run, x = self.new()
        first = t.execute(self.root, x["run_id"])
        self.assertEqual(first["outcome"], "completed", first)
        self.assertEqual(first["checkpoint_sha256"],
                         t.digest(run / "output/checkpoint.json"))
        ck = run / "output/checkpoint.json"
        ck.write_text(ck.read_text() + " ")
        with self.assertRaisesRegex(t.BoundaryError, "checkpoint changed"):
            t.execute(self.root, x["run_id"], resume=True)

    def test_host_overwrite_rejected_by_manifest_recheck(self):
        run, x = self.new()
        x["source_sha256"] = "0" * 64
        t.write_json_atomic(run / "run.json", x)
        with self.assertRaisesRegex(t.BoundaryError, "changed pinned input"):
            t.execute(self.root, x["run_id"])

if __name__ == "__main__":
    unittest.main()
