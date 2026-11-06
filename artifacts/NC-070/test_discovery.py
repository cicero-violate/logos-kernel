"""Exact ISA/scalar simulator parity, heldout split, leakage and negative controls."""
import hashlib
import inspect
import json
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import discovery as d


class DiscoveryTests(unittest.TestCase):
    def test_all_graphs_are_unique_acyclic_and_cover_all_registers(self):
        self.assertEqual(len(d.ALL_GRAPHS), 64)
        self.assertEqual(len(set(d.ALL_GRAPHS)), 64)
        for parents in d.ALL_GRAPHS:
            self.assertEqual(len(parents), 4)
            self.assertEqual(parents.count(-1), 1)
            self.assertEqual(len(d.topological(parents)), 4)
            self.assertEqual(len(d.canonical_edges(parents)), 3)

    def test_oracle_matches_independently_written_scm_for_all_candidates(self):
        # 64 * 4 * 2 = 512 intervention pairs, on canonical ISA programs.
        count = 0
        for graph in d.ALL_GRAPHS:
            oracle = d.make_oracle(graph)
            for bit in (0, 1):
                self.assertEqual(oracle(None, None, bit), (bit,) * 4)
                for node in range(4):
                    want = d.simulate(graph, node, 1-bit, bit)
                    got = oracle(node, 1-bit, bit)
                    self.assertEqual(got, want, (graph, node, bit))
                    count += 1
        self.assertEqual(count, 512)

    def test_passive_equivalence_and_interventional_separability(self):
        report = d.observational_equivalence()
        self.assertTrue(report["all_64_hypotheses_passively_equivalent"])
        self.assertNotEqual(report["graph_a"], report["graph_b"])
        self.assertNotEqual(report["interventional_outcome_a"],
                            report["interventional_outcome_b"])
        self.assertEqual(sum(x["prob"] for x in report["passive_distribution"]), 1)
        for graph in d.ALL_GRAPHS:
            oracle = d.make_oracle(graph)
            self.assertEqual(oracle(None,None,0),(0,0,0,0))
            self.assertEqual(oracle(None,None,1),(1,1,1,1))

    def test_heldout_split_is_deterministic_and_nonempty(self):
        self.assertEqual(d.HELDOUT_IDS, d.test_graph_ids())
        self.assertTrue(set(d.HELDOUT_IDS) < set(range(64)))
        self.assertEqual(len(d.HELDOUT_IDS), 14)

    def test_policy_is_not_given_graph_or_program_and_updates_sequentially(self):
        self.assertEqual(tuple(inspect.signature(d.run_episode).parameters),
                         ("query", "seed", "episode_nonce", "arm", "rounds"))
        graph = d.ALL_GRAPHS[12]
        oracle = Mock(side_effect=d.make_oracle(graph))
        for arm in d.ARMS:
            result = d.run_episode(oracle, 11, 548328349813, arm, 3)
            self.assertEqual(result["oracle_calls"], 3)
            self.assertEqual(len(result["traces"]), 3)
            self.assertIn(12, tuple(range(64)))
            if arm == "passive":
                self.assertEqual(result["remaining_graphs"],64)
                self.assertTrue(all(t["selected_register"]=="none" for t in result["traces"]))
            else:
                self.assertEqual(len({t["selected_register"] for t in result["traces"]}),3)
                self.assertTrue(all(1 <= t["surviving_hypotheses"] <=64
                                    for t in result["traces"]))
                self.assertGreaterEqual(result["remaining_graphs"],1)

    def test_policy_does_not_need_true_graph_key(self):
        graph = d.ALL_GRAPHS[d.HELDOUT_IDS[-1]]
        true_oracle = d.make_oracle(graph)
        call_record = []
        def opaque(node,value,bit):
            call_record.append((node,value,bit))
            return true_oracle(node,value,bit)
        a = d.run_episode(opaque,19,1234,"active",3)
        b = d.run_episode(true_oracle,19,1234,"active",3)
        self.assertEqual(a["predicted_graph_id"], b["predicted_graph_id"])
        self.assertEqual(a["traces"], b["traces"])
        self.assertEqual(len(call_record),3)

    def test_false_oracle_cannot_eliminate_all_models_without_failing(self):
        def false_oracle(node, value, bit):
            return (9,9,9,9)
        with self.assertRaises(AssertionError):
            d.run_episode(false_oracle,11,100,"active",1)

    def test_full_analysis_counts_and_cluster_bootstrap_are_consistent(self):
        rows,_ = d.run_trials((11,19,29),3)
        self.assertEqual(len(rows), 14*3*3)
        report=d.cluster_bootstrap(rows,(11,19,29),3,samples=400)
        for arm in d.ARMS:
            self.assertEqual(report[arm]["total"],42)
            self.assertEqual(report[arm]["oracle_experiments"],126)
            self.assertGreater(report[arm]["cpu_seconds"],0)
        self.assertEqual(report["passive"]["uniquely_identified"],0)
        ci=report["paired_active_minus_random"]["graph_cluster_bootstrap_ci95"]
        self.assertLessEqual(ci[0],ci[1])
        self.assertGreaterEqual(ci[0],-1)
        self.assertLessEqual(ci[1],1)

    def test_fewer_interventions_never_shrink_passive_hypotheses(self):
        oracle=d.make_oracle(d.ALL_GRAPHS[0])
        for budget in range(4):
            ep=d.run_episode(oracle,11,2026,"passive",budget)
            self.assertEqual(ep["remaining_graphs"],64)

if __name__=="__main__":
    unittest.main()
