"""NC-029 local correctness/negative controls.

Dependencies: Python 3 standard library, PyTorch, bench_cpu.py, read-only
frozen RLT sources/checkpoints/inventory as declared in bench_cpu.py.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import unittest

P = Path(__file__).with_name("bench_cpu.py")
spec = importlib.util.spec_from_file_location("bench_cpu_nc029", P)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class BenchTest(unittest.TestCase):
    def test_fixed_samples(self):
        p = bench.pairs()
        self.assertEqual(len(p), 128)
        self.assertEqual(p, bench.pairs())
        self.assertEqual(len(bench.oracle(p)), len(p))
        self.assertEqual(bench.task_hash(p), bench.protocol()["task_sha256"])

    def test_exact_matched_controls(self):
        import torch
        torch.set_num_threads(1)
        p = bench.pairs() + [(a, b) for a in range(256) for b in (0, 1, 2, 3, 127, 255)]
        expected = bench.oracle(p)
        a = torch.tensor([x for x, _ in p])
        b = torch.tensor([x for _, x in p])
        for arm in bench.ARMS[:3]:
            with self.subTest(arm=arm):
                got = bench.controls(arm, a, b).tolist()
                self.assertEqual(got, expected)
                self.assertEqual(got, bench.controls(arm, a, b).tolist())

    def test_negative_control_detects_corruption(self):
        import torch
        p = bench.pairs()
        a = torch.tensor([x for x, _ in p])
        b = torch.tensor([x for _, x in p])
        for arm in bench.ARMS[:3]:
            with self.subTest(arm=arm):
                pred = bench.controls(arm, a, b)
                broken = (pred ^ 1).tolist()
                self.assertTrue(any(x != y for x, y in zip(broken, bench.oracle(p))))

    def test_provenance(self):
        src = bench.provenance()
        self.assertEqual(src["sources"]["snapshots/current_unified_v3.pt"]["sha256"],
                         bench.EXPECTED_BUNDLE)
        self.assertEqual(len(src["checkpoints"]), 2)
        self.assertEqual(src, bench.provenance())

    def test_symbolic_non_neural_labels(self):
        d = bench.protocol()
        self.assertEqual(d["task"], "NC-029")
        self.assertEqual(d["trials"], 3)
        self.assertTrue(d["avoid_comparing_symbolic_exactness_to_neural_learning"])
        self.assertTrue(d["controls_are_not_neural_models"])


if __name__ == "__main__":
    unittest.main()
