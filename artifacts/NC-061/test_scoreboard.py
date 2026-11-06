"""Run: PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-061/test_scoreboard.py"""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import controls  # noqa: E402
import scoreboard as sb  # noqa: E402


def row(rate, used=100, retention=1.0, overlap=0.0, over=0, cpu=10.0):
    return {"solve_rate": rate, "candidates_used": used, "retention": retention, "training_overlap": overlap,
            "over_budget_goals": over, "cpu_seconds": cpu}


class CommitmentTests(unittest.TestCase):
    def test_reveal_verifies_and_every_kind_of_tampering_is_detected(self):
        commitment, reveal = sb.new_commitment()
        self.assertEqual(sb.verify_reveal(commitment, reveal), reveal)
        for field, value in (("seed", "changed"), ("nonce", "changed"), ("generator", "other")):
            bad = copy.deepcopy(reveal)
            bad[field] = value
            with self.assertRaisesRegex(ValueError, "does not match"):
                sb.verify_reveal(commitment, bad)
        bad = copy.deepcopy(reveal)
        bad["thresholds"]["improvement_margin"] = 0.0
        with self.assertRaisesRegex(ValueError, "does not match"):
            sb.verify_reveal(commitment, bad)

    def test_commitments_do_not_repeat(self):
        self.assertNotEqual(sb.new_commitment()[1]["seed"], sb.new_commitment()[1]["seed"])

    def test_goals_are_reproducible_from_a_reveal_and_differ_by_seed(self):
        a = sb.sealed_goals(controls.control_reveal(1, 3))
        self.assertEqual(a, sb.sealed_goals(controls.control_reveal(1, 3)))
        self.assertNotEqual(a, sb.sealed_goals(controls.control_reveal(2, 3)))
        self.assertEqual(len({g["table"] for g in a}), 3)


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.goals = sb.sealed_goals(controls.control_reveal(5, 4))

    def test_candidate_only_receives_examples_and_answers_are_checked_exhaustively(self):
        seen = []
        def spy(examples, budget):
            seen.append(examples)
            return controls.constant(examples, budget)
        sb.score_candidate(spy, self.goals, 8)
        for examples, goal in zip(seen, self.goals):
            self.assertEqual(examples, goal["examples"])

    def test_program_fitting_every_example_but_wrong_elsewhere_does_not_score(self):
        adds = [{"op": "ADD", "dst": "R", "a": "X", "b": "Y"}]
        goal = {"table": sb.truth_table(adds), "answer": adds,
                "examples": [{"X": 0, "Y": 0, "R": 0}, {"X": 0, "Y": 0, "R": 0}]}
        fits = [{"op": "LOADI", "dst": "R", "imm": 0}]
        self.assertTrue(controls.matches(fits, goal["examples"]))
        self.assertEqual(sb.score_candidate(lambda e, b: (fits, 1), [goal], 8)["solved"], 0)
        self.assertEqual(sb.score_candidate(lambda e, b: (adds, 1), [goal], 8)["solved"], 1)

    def test_malformed_programs_score_as_invalid_without_crashing(self):
        for bad in ([{"op": "NOPE"}], [{"op": "ADD", "dst": "R"}], "text", [{"op": "JMP", "target": 99}]):
            report = sb.score_candidate(lambda e, b: (bad, 1), self.goals[:1], 8)
            self.assertEqual((report["solved"], report["invalid_programs"]), (0, 1), bad)

    def test_forged_or_malformed_usage_cannot_pass(self):
        # Reviewer counterexample: spent of -1 or 0 solved everything with no budget used.
        good = controls.known_good(self.goals)
        for spent in (-1, 0, 9, True, 1.5, None, "1"):
            report = sb.score_candidate(lambda e, b: (good(e, b)[0], spent), self.goals, 8)
            self.assertEqual((report["solved"], report["over_budget_goals"], report["candidates_used"]), (0, 4, 0), spent)

    def test_cpu_is_measured_by_the_scoreboard_and_covers_the_candidate(self):
        def slow(examples, budget):
            end = sb.time.process_time() + 0.05
            while sb.time.process_time() < end:
                pass
            return controls.constant(examples, budget)
        self.assertGreaterEqual(sb.score_candidate(slow, self.goals, 8)["cpu_seconds"], 0.2)

    def test_unseen_solves_exclude_functions_in_training_data(self):
        good = controls.known_good(self.goals)
        keys = [sb.table_key(self.goals[0]["table"])]
        report = sb.score_candidate(good, self.goals, 8, training_table_keys=keys)
        self.assertEqual((report["solved"], report["unseen_solved"]), (4, 3))


class CompareTests(unittest.TestCase):
    previous = [row(0.3)] * 3

    def test_real_gain_across_seeds_is_an_improvement(self):
        self.assertEqual(sb.compare(self.previous, [row(0.5)] * 3)["verdict"], "IMPROVEMENT")

    def test_gain_below_margin_is_not_an_improvement(self):
        self.assertEqual(sb.compare(self.previous, [row(0.34)] * 3)["verdict"], "NO_IMPROVEMENT")

    def test_one_lucky_seed_is_not_an_improvement(self):
        self.assertEqual(sb.compare(self.previous, [row(1.0), row(0.3), row(0.3)])["verdict"], "NO_IMPROVEMENT")

    def test_extra_budget_regression_and_missing_retention_are_refused(self):
        self.assertEqual(sb.compare(self.previous, [row(0.5, used=500)] * 3)["verdict"], "NO_IMPROVEMENT")
        self.assertEqual(sb.compare(self.previous, [row(0.5, retention=0.5)] * 3)["verdict"], "NO_IMPROVEMENT")
        missing = row(0.5)
        del missing["retention"]
        self.assertEqual(sb.compare(self.previous, [missing] * 3)["verdict"], "NO_IMPROVEMENT")

    def test_contamination_over_budget_and_too_few_seeds(self):
        self.assertEqual(sb.compare(self.previous, [row(1.0, overlap=0.1)] * 3)["verdict"], "CONTAMINATED")
        self.assertEqual(sb.compare(self.previous, [row(1.0, over=1)] * 3)["verdict"], "INVALID")
        self.assertEqual(sb.compare(self.previous, [row(1.0)] * 2)["verdict"], "NOT_ESTABLISHED")

    def test_unmatched_cpu_is_not_an_improvement(self):
        # Reviewer counterexample: equal candidate counts while using 360,000 times the CPU.
        cheap = [row(0.0, used=3, cpu=0.01)] * 3
        self.assertEqual(sb.compare(cheap, [row(1.0, used=3, cpu=3600.0)] * 3)["verdict"], "NO_IMPROVEMENT")
        self.assertEqual(sb.compare(self.previous, [row(0.5, cpu=12.4)] * 3)["verdict"], "IMPROVEMENT")
        self.assertEqual(sb.compare(self.previous, [row(0.5, cpu=20.0)] * 3)["verdict"], "NO_IMPROVEMENT")
        unmeasured = row(0.5)
        del unmeasured["cpu_seconds"]
        self.assertEqual(sb.compare(self.previous, [unmeasured] * 3)["verdict"], "NO_IMPROVEMENT")

    def test_cheap_baseline_does_not_excuse_a_large_cpu_multiple(self):
        # Second reviewer counterexample: 900 times the CPU passed through an additive floor.
        cheap = [row(0.0, used=3, cpu=0.001)] * 3
        self.assertEqual(sb.compare(cheap, [row(1.0, used=3, cpu=0.9)] * 3)["verdict"], "NO_IMPROVEMENT")
        self.assertEqual(sb.compare(cheap, [row(1.0, used=3, cpu=0.049)] * 3)["verdict"], "NO_IMPROVEMENT")  # Third review.
        self.assertEqual(sb.compare(cheap, [row(1.0, used=3, cpu=0.00125)] * 3)["verdict"], "IMPROVEMENT")
        self.assertEqual(sb.compare([row(0.0, cpu=100.0)] * 3, [row(1.0, cpu=125.0)] * 3)["verdict"], "IMPROVEMENT")
        self.assertEqual(sb.compare([row(0.0, cpu=100.0)] * 3, [row(1.0, cpu=126.0)] * 3)["verdict"], "NO_IMPROVEMENT")

    def test_flat_or_tiny_gain_growth_is_not_compounding(self):
        # Reviewer counterexample: three equal gains were labelled COMPOUNDING.
        win = ["IMPROVEMENT"] * 3
        self.assertEqual(sb.compounding([0.05, 0.05, 0.05], win)["verdict"], "NO_COMPOUNDING")
        self.assertEqual(sb.compounding([0.05, 0.055, 0.06], win)["verdict"], "NO_COMPOUNDING")

    def test_compounding_needs_consecutive_growing_gains(self):
        win = ["IMPROVEMENT"] * 3
        self.assertEqual(sb.compounding([0.05, 0.06, 0.07], win)["verdict"], "COMPOUNDING")
        self.assertEqual(sb.compounding([0.09, 0.07, 0.05], win)["verdict"], "NO_COMPOUNDING")
        self.assertEqual(sb.compounding([0.05, 0.06, 0.07], ["IMPROVEMENT", "NO_IMPROVEMENT", "IMPROVEMENT"])["verdict"], "NO_COMPOUNDING")
        self.assertEqual(sb.compounding([0.05, 0.06], win[:2])["verdict"], "NOT_ESTABLISHED")


class ControlTests(unittest.TestCase):
    def test_each_control_gets_the_required_verdict(self):
        report = controls.run(goals=4, seeds=(1, 2, 3), budget=8)
        verdicts = {k: v["verdict"] for k, v in report["verdicts_against_constant_baseline"].items()}
        self.assertEqual(verdicts["known_good"], "IMPROVEMENT")
        self.assertEqual(verdicts["constant"], "NO_IMPROVEMENT")
        self.assertEqual(verdicts["leaked_answer"], "CONTAMINATED")
        self.assertEqual(verdicts["regressed"], "NO_IMPROVEMENT")
        self.assertEqual(verdicts["budget_cheat"], "INVALID")
        self.assertEqual(verdicts["memorizer"], "NO_IMPROVEMENT")


if __name__ == "__main__":
    unittest.main()
