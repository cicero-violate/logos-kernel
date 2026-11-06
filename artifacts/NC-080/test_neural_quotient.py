"""Run: PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-080/test_neural_quotient.py"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neural_quotient as n  # noqa: E402


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.info = n.train(11)
        cls.model.eval()
        n.network_table(cls.model)

    def test_local_table_is_learned_and_lookup_equals_running_the_network(self):
        self.assertEqual(self.info["final_local_accuracy"], 1.0)
        a = n.bits(n.XS[:500])
        b = n.bits(n.YS[:500])
        for name in n.OPS:
            with torch.no_grad():
                looked_up = n.neural_op(self.model, name, a, b)
                code = torch.nn.functional.one_hot(torch.full((500,), n.OPS.index(name)), len(n.OPS)).float()
                carry = torch.zeros(500)
                direct = []
                for i in range(n.WIDTH):
                    prev = a[:, i - 1] if i else torch.zeros(500)
                    foll = a[:, i + 1] if i < n.WIDTH - 1 else torch.zeros(500)
                    d = self.model(torch.cat((code, torch.stack((a[:, i], b[:, i], carry, prev, foll), 1)), 1)).sigmoid().ge(.5).float()
                    direct.append(d[:, 0])
                    carry = d[:, 1]
                self.assertTrue(torch.equal(looked_up, torch.stack(direct, 1)), name)

    def test_quotient_merges_only_exact_equals(self):
        same_a = ("ADD", ("X",), ("X",))
        same_b = ("SHL", ("X",))
        other = ("ADD", ("X",), ("Y",))
        classes, _ = n.quotient([same_a, same_b, other])
        self.assertEqual(len(classes), 2)
        self.assertIn(same_b, classes.values())
        self.assertNotIn(same_a, classes.values())

    def test_near_equal_functions_stay_in_separate_classes(self):
        a = ("AND", ("X",), ("K", 255))
        b = ("AND", ("X",), ("K", 254))
        self.assertEqual(len(n.quotient([a, b])[0]), 2)

    def test_neural_execution_matches_exact_on_a_deep_concept(self):
        expr = ("ADD", ("SUB", ("SHL", ("XOR", ("X",), ("Y",))), ("SHR", ("Y",))), ("OR", ("X",), ("K", 3)))
        with torch.no_grad():
            self.assertTrue(torch.equal(n.neural_eval(self.model, expr, {}), n.bits(n.exact_eval(expr, {}))))

    def test_permuted_meaning_and_untrained_network_fail(self):
        expr = ("ADD", ("X",), ("Y",))
        want = n.bits(n.exact_eval(expr, {}))
        self.model.permute = torch.roll(torch.arange(len(n.OPS)), 1)
        n.network_table(self.model)
        with torch.no_grad():
            self.assertFalse(torch.equal(n.neural_eval(self.model, expr, {}), want))
        self.model.permute = None
        n.network_table(self.model)
        torch.manual_seed(3)
        blank = n.NeuralOperators().eval()
        n.network_table(blank)
        with torch.no_grad():
            self.assertFalse(torch.equal(n.neural_eval(blank, expr, {}), want))


if __name__ == "__main__":
    unittest.main()
