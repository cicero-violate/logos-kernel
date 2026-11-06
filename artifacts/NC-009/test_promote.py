"""Run: PYTHONDONTWRITEBYTECODE=1 python -m unittest -q artifacts/NC-009/test_promote.py"""
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[0] / "NC-007")]
import promote as p  # noqa: E402

PAIRS = [(x, y) for x in range(32) for y in range(32)] + list(p.CORNERS)


def cand(task, *tokens, source="test/x"):
    return {"task": task, "source": source, "program": [*tokens, "HALT"]}


class PromoteTests(unittest.TestCase):
    def test_semantics_table_matches_the_project_action_table(self):
        import compose
        for token, meaning in p.SEMANTICS.items():
            self.assertEqual(tuple(compose.ACTIONS[token]), meaning, token)

    def test_contracts_match_nc008_task_definitions(self):
        spec = (HERE.parents[0] / "NC-008" / "invent.py").read_text()
        self.assertIn('"BOOLEAN_COMPOSITE": lambda x,y: ((x & y) ^ x) & MASK', spec)
        for x, y in p.test_pairs()[::997]:
            self.assertEqual(p.CONTRACTS["BOOLEAN_COMPOSITE"](x, y), ((x & y) ^ x) & p.MASK)

    def test_canonicalize_removes_only_dead_steps(self):
        self.assertEqual(p.canonicalize(["ADD_X1", "ADD_X1"]), ["ADD_X1"])
        self.assertEqual(p.canonicalize(["AND1_X", "ADD_XX"]), ["ADD_XX"])
        self.assertEqual(p.canonicalize(["XOR_XY", "SHL_R"]), ["XOR_XY", "SHL_R"])
        self.assertEqual(p.canonicalize(["ADD_XX", "XOR_XY", "ADD_RX"]), ["XOR_XY", "ADD_RX"])
        for body in (["ADD_X1", "ADD_X1"], ["AND_XY", "XOR_RX"], ["ADD_XX", "XOR_XY", "ADD_RX"]):
            for x, y in PAIRS:
                self.assertEqual(p.run(body, x, y), p.run(p.canonicalize(body), x, y))

    def test_correct_operator_is_promoted_with_bound_fields(self):
        new, audit = p.process([cand("INC", "ADD_X1")], PAIRS)
        self.assertEqual(audit[0]["decision"], "PROMOTED")
        self.assertEqual(new[0]["operator_id"], "INC:v1")
        for field in ("canonical_key", "behaviour_key", "proof_scope", "reuse_cost_steps", "chain_sha256", "source_program"):
            self.assertTrue(new[0][field], field)

    def test_wrong_operator_is_rejected_with_a_counterexample_and_stays_traceable(self):
        new, audit = p.process([cand("INC", "ADD_XX")], PAIRS)
        self.assertEqual(new, [])
        self.assertEqual(audit[0]["decision"], "REJECTED")
        example = audit[0]["counterexample"]
        self.assertNotEqual(example["got"], example["want"])

    def test_program_right_on_small_inputs_but_wrong_on_16_bit_inputs_is_rejected(self):
        p.CONTRACTS["FAKE"] = lambda x, y: (x & 0xFF) * 2 & p.MASK
        try:
            pairs = p.test_pairs()
            self.assertIsNone(p.verify(["ADD_XX"], p.CONTRACTS["FAKE"], pairs[:65536]))
            _, audit = p.process([cand("FAKE", "ADD_XX")], pairs)
            self.assertEqual(audit[0]["decision"], "REJECTED")
        finally:
            del p.CONTRACTS["FAKE"]

    def test_duplicate_after_canonicalizing_is_not_promoted_twice(self):
        new, audit = p.process([cand("INC", "ADD_X1"), cand("INC", "ADD_X1", "ADD_X1")], PAIRS)
        self.assertEqual(len(new), 1)
        self.assertEqual(audit[1]["decision"], "DUPLICATE")

    def test_longer_program_with_identical_behaviour_is_subsumed(self):
        new, audit = p.process([cand("SHIFT_COMPOSITE", "XOR_XY", "SHL_R"), cand("SHIFT_COMPOSITE", "XOR_XY", "XOR_RX", "XOR_RX", "SHL_R")], PAIRS)
        self.assertEqual(len(new), 1)
        self.assertEqual(audit[1]["decision"], "SUBSUMED")

    def test_malformed_and_unknown_inputs_are_rejected_without_crashing(self):
        for candidate in ({"task": "INC", "source": "t", "program": ["NOPE", "HALT"]}, {"task": "INC", "source": "t", "program": ["ADD_X1"]},
                          {"task": "UNKNOWN", "source": "t", "program": ["ADD_X1", "HALT"]}):
            new, audit = p.process([candidate], PAIRS)
            self.assertEqual((new, audit[0]["decision"]), ([], "REJECTED"))

    def test_reviewer_counterexample_malformed_programs_do_not_crash(self):
        for program in (None, 7, [], [None, "HALT"], [1, "HALT"], [["ADD_X1"], "HALT"], "ADD_X1", {"a": 1}):
            new, audit = p.process([{"task": "INC", "source": "t", "program": program}], PAIRS)
            self.assertEqual((new, audit[0]["decision"]), ([], "REJECTED"), program)

    def test_reviewer_counterexample_tokens_after_halt_are_rejected(self):
        for program in (["ADD_X1", "HALT", "NOPE"], ["ADD_X1", "HALT", "ADD_XX"], ["ADD_X1", "HALT", "HALT"]):
            new, audit = p.process([{"task": "INC", "source": "t", "program": program}], PAIRS)
            self.assertEqual((new, audit[0]["decision"]), ([], "REJECTED"), program)

    def test_reviewer_counterexample_tampered_history_is_refused(self):
        rows, _ = p.process([cand("INC", "ADD_X1"), cand("DOUBLE", "ADD_XX")], PAIRS)
        p.verify_chain(rows)
        edited = [dict(rows[0], proof_scope="forged"), rows[1]]
        reordered = [rows[1], rows[0]]
        forged_hash = [rows[0], dict(rows[1], chain_sha256="0" * 64)]
        for bad in (edited, reordered, rows[1:], forged_hash):
            with self.assertRaisesRegex(ValueError, "chain broken"):
                p.process([cand("INC", "ADD_X1")], PAIRS, bad)

    def test_rerun_appends_nothing_and_earlier_rows_stay_unchanged(self):
        first, _ = p.process([cand("INC", "ADD_X1")], PAIRS)
        before = dict(first[0])
        second, audit = p.process([cand("INC", "ADD_X1")], PAIRS, first)
        self.assertEqual(second, [])
        self.assertEqual(audit[0]["decision"], "DUPLICATE")
        self.assertEqual(first[0], before)

    def test_new_version_chains_to_the_previous_row(self):
        first, _ = p.process([cand("SHIFT_COMPOSITE", "XOR_XY", "XOR_RX", "XOR_RX", "SHL_R")], PAIRS)
        second, _ = p.process([cand("SHIFT_COMPOSITE", "XOR_XY", "SHL_R")], PAIRS, first)
        self.assertEqual(second[0]["operator_id"], "SHIFT_COMPOSITE:v2")
        self.assertEqual(second[0]["chain_sha256"], p.chain(first[0]["chain_sha256"], {k: v for k, v in second[0].items() if k != "chain_sha256"}))


if __name__ == "__main__":
    unittest.main()
