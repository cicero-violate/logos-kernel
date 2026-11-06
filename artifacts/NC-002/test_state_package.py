"""NC-002 negative checks; depends on state_package.py and pinned source models."""
from __future__ import annotations
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("nc002_state_package", HERE / "state_package.py")
sp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sp
spec.loader.exec_module(sp)


class PersistenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.manifest = json.loads((HERE / "checkpoint_manifest.json").read_text())

    def test_all_60_sources_and_champion_unchanged(self):
        inv = sp.verify_inventory()
        self.assertEqual(len(inv["files"]), 60)

    def test_each_frozen_model_fresh_reload(self):
        for name in sp.SPECS:
            with self.subTest(name=name):
                item = self.manifest["models"][name]
                model = sp.build(name)
                model.load_state_dict(sp.state_from_source(name), strict=True)
                initial, _, _, package = sp.load_checked(name, HERE / item["initial_checkpoint"])
                self.assertEqual(sp.probe_hex(name, model), sp.probe_hex(name, initial))
                self.assertEqual(package["legacy_optimizer_or_rng_recovered"], False)
                self.assertEqual(package["continuation_step"], 0)

    def test_source_hash_tampering_fails_closed(self):
        name = "general_rlt"
        item = self.manifest["models"][name]
        package = torch.load(HERE / item["initial_checkpoint"], map_location="cpu", weights_only=True)
        package["source_hashes"]["rlt_champion:rlt_core.py"] = "0" * 64
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tampered.pt"
            torch.save(package, path)
            with self.assertRaisesRegex(RuntimeError, "PINNED_SOURCE_OR_ARCHITECTURE_CHANGED"):
                sp.load_checked(name, path)

    def test_vocabulary_tampering_fails_closed(self):
        name = "standalone_isa"
        item = self.manifest["models"][name]
        package = torch.load(HERE / item["initial_checkpoint"], map_location="cpu", weights_only=True)
        package["vocabulary"][0] = "NOT_ADD"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tampered.pt"
            torch.save(package, path)
            with self.assertRaisesRegex(RuntimeError, "PACKAGE_ARCHITECTURE_OR_VOCAB_MISMATCH"):
                sp.load_checked(name, path)

    def test_optimizer_tampering_fails_closed(self):
        name = "carry_gru"
        item = self.manifest["models"][name]
        package = torch.load(HERE / item["initial_checkpoint"], map_location="cpu", weights_only=True)
        package["optimizer_config"]["lr"] = 999
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tampered.pt"
            torch.save(package, path)
            with self.assertRaisesRegex(RuntimeError, "OPTIMIZER_CONFIGURATION_MISMATCH"):
                sp.load_checked(name, path)

    def test_manifest_requires_optimizer_moments(self):
        for name in sp.SPECS:
            self.assertGreater(self.manifest["models"][name]["optimizer_state_entries"], 0)
            self.assertTrue(self.manifest["models"][name]["exact_resume_after_reinitializing_optimizer_at_source"])
            self.assertFalse(self.manifest["models"][name]["source_historical_optimizer_recovered"])

    def test_no_synthetic_model_promoted(self):
        self.assertTrue(self.manifest["source_optimizer_history_missing"])
        self.assertEqual(self.manifest["original_inventory_file_hashes_verified"], 60)


if __name__ == "__main__":
    unittest.main()
