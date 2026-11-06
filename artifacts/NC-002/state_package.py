#!/usr/bin/env python3
"""NC-002: provenance-pinned, weights-only-safe train-state packages.

Dependencies: Python3, PyTorch, numpy, exact frozen source modules and original
checkpoints from NC-000. Does NOT mutate source repos or the active champion.
Original optimizer/RNG history is absent; only new continuation state is exact.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LOGOS = HERE.parent.parent
RLT = Path("/workspace/experiments/rlt_split_algorithms_ablation_v1")
STANDALONE = Path("/workspace/experiments/latticeforge_neural_isa_test")
ROOTS = {"rlt_champion": RLT, "standalone_isa": STANDALONE, "logos_kernel": LOGOS}
INVENTORY = LOGOS / "artifacts/NC-000/inventory.json"

# Explicit factories only, no dynamic architecture or code execution from packages.
SPECS = {
    "general_rlt": {
        "repo": "rlt_champion",
        "source": "task_conditioned_rlt_v1/multitask_train_v1/outputs/seed11/seed11_conditioned_step640.pt",
        "architecture": ["rlt_core.py", "task_conditioned_rlt.py"],
        "vocabulary": ["add", "xor", "sub", "mul", "lt", "parity_and", "compound"],
        "model_class": "task_conditioned_rlt.TaskConditionedRLT",
        "source_training_progress": {"steps_reported": 640, "optimizer_and_rng_recoverable": False},
    },
    "carry_gru": {
        "repo": "rlt_champion",
        "source": "outputs/new16/carry_recurrence_seed11.pt",
        "architecture": ["rlt_core.py", "algorithms/carry_recurrence/module.py"],
        "vocabulary": ["ADD"],
        "model_class": "algorithms.carry_recurrence.module.LearnedCarryGRU",
        "source_training_progress": {"step_recoverable": False, "optimizer_and_rng_recoverable": False},
    },
    "shared_rlt_isa": {
        "repo": "rlt_champion",
        "source": "neural_isa_transfer_v1/unified_shared_v1/outputs/seed11/jointly_tuned.pt",
        "architecture": [
            "rlt_core.py", "task_conditioned_rlt.py",
            "neural_isa_transfer_v1/rlt_micro.py",
            "neural_isa_transfer_v1/vendor/train_causal.py",
            "neural_isa_transfer_v1/rlt_micro_v2.py",
            "neural_isa_transfer_v1/unified_shared_v1/model.py",
            "neural_isa_transfer_v1/unified_shared_v1/runtime.py",
        ],
        "additional_sources": [
            "neural_isa_transfer_v1/rlt_micro_checkpoints/seed11_v2.pt",
            "task_conditioned_rlt_v1/multitask_train_v1/outputs/seed11/seed11_conditioned_step640.pt",
        ],
        "vocabulary": ["JZ_B", "AND1", "JZ_T", "ADD", "SHL", "SHR", "JMP", "HALT"],
        "model_class": "neural_isa_transfer_v1.unified_shared_v1.model.SharedRLTISA",
        "source_training_progress": {"epoch_recoverable": False, "optimizer_and_rng_recoverable": False},
    },
    "standalone_isa": {
        "repo": "standalone_isa",
        "source": "causal/weights.pt",
        "architecture": ["train_causal.py"],
        "vocabulary": ["ADD", "SHL", "SHR", "AND1", "OR"],
        "model_class": "train_causal.NeuralISA",
        "source_training_progress": {"epoch_recoverable": False, "optimizer_and_rng_recoverable": False},
    },
    "standalone_controller_v2": {
        "repo": "standalone_isa",
        "source": "transformer_controller/v2/controller_weights.pt",
        "architecture": ["train_causal.py", "transformer_controller/train_controller.py", "transformer_controller/train_controller_v2.py"],
        "additional_sources": ["causal/weights.pt"],
        "vocabulary": ["JZ_B", "AND1", "JZ_T", "ADD", "SHL", "SHR", "JMP", "HALT"],
        "model_class": "train_controller.TransformerPC",
        "source_training_progress": {"epoch_recoverable": False, "optimizer_and_rng_recoverable": False},
    },
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_paths(name: str) -> dict[str, Path]:
    spec = SPECS[name]
    root = ROOTS[spec["repo"]]
    paths = [spec["source"], *spec["architecture"], *spec.get("additional_sources", [])]
    result = {spec["repo"] + ":" + x: root / x for x in sorted(set(paths))}
    for rel in ("artifacts/NC-000/baseline_receipt.json",
                "artifacts/NC-001/isa_spec.json",
                "artifacts/NC-001/independent_verify.json"):
        result["logos_kernel:" + rel] = LOGOS / rel
    return result


def verify_inventory():
    inv = json.loads(INVENTORY.read_text())
    original = {f"{e['repo']}:{e['path']}": e["sha256"] for e in inv["files"]}
    for entry in inv["files"]:
        path = ROOTS[entry["repo"]] / entry["path"]
        if not path.is_file() or sha(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"INVENTORY_60_FILE_MISMATCH: {entry['repo']}:{entry['path']}")
    if len(inv["files"]) != 60:
        raise RuntimeError("INVENTORY_CARDINALITY_CHANGED")
    for name in SPECS:
        for key, path in source_paths(name).items():
            if not path.is_file():
                raise RuntimeError(f"SOURCE_MISSING: {key}")
            if key in original and sha(path) != original[key]:
                raise RuntimeError(f"FROZEN_SOURCE_CHANGED: {key}")
            # Supplemental transitive dependencies absent from NC-000 are
            # bound by their measured SHA in the new NC-002 package.
    champion = RLT / inv["champion"]["bundle"]
    if sha(champion) != inv["champion"]["selected_bundle_sha256"]:
        raise RuntimeError("CHAMPION_CHECKPOINT_CHANGED")
    return inv


def build(name: str):
    if name in ("general_rlt", "carry_gru", "shared_rlt_isa"):
        sys.path.insert(0, str(RLT))
    if name == "general_rlt":
        from task_conditioned_rlt import TaskConditionedRLT
        model = TaskConditionedRLT()
    elif name == "carry_gru":
        from algorithms.carry_recurrence.module import LearnedCarryGRU
        model = LearnedCarryGRU()
    elif name == "shared_rlt_isa":
        from neural_isa_transfer_v1.unified_shared_v1.model import SharedRLTISA
        original = RLT / SPECS["general_rlt"]["source"]
        native = RLT / "neural_isa_transfer_v1/rlt_micro_checkpoints/seed11_v2.pt"
        pretrained = torch.load(native, weights_only=True, map_location="cpu")
        model = SharedRLTISA(original, pretrained["state_dict"])
    elif name == "standalone_isa":
        sys.path.insert(0, str(STANDALONE))
        from train_causal import NeuralISA
        model = NeuralISA()
    elif name == "standalone_controller_v2":
        sys.path.insert(0, str(STANDALONE / "transformer_controller"))
        from train_controller import TransformerPC
        model = TransformerPC()
    else:
        raise ValueError(name)
    return model


def state_from_source(name: str):
    spec = SPECS[name]
    chk = torch.load(ROOTS[spec["repo"]] / spec["source"], map_location="cpu", weights_only=True)
    if name in ("general_rlt", "carry_gru"):
        return chk
    if name == "shared_rlt_isa":
        if chk["arm"] != "jointly_tuned" or chk["seed"] != 11 or not chk["shared_attention"]:
            raise RuntimeError("SHARED_ISA_SOURCE_PROVENANCE_MISMATCH")
    if name == "standalone_isa" and tuple(chk["ops"]) != tuple(spec["vocabulary"]):
        raise RuntimeError("STANDALONE_VOCAB_MISMATCH")
    if name == "standalone_controller_v2" and tuple(chk["program"]) != tuple(spec["vocabulary"]):
        raise RuntimeError("CONTROLLER_ROM_MISMATCH")
    return chk["model"]


def capture_rng(numpy_rng):
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().clone(),
        "numpy_generator": copy.deepcopy(numpy_rng.bit_generator.state),
        "torch_cuda": [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [],
    }


def restore_rng(state, numpy_rng):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    numpy_rng.bit_generator.state = state["numpy_generator"]
    if state["torch_cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_RNG_STATE_UNRESTORABLE")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-4)


def schema_for(state):
    return {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in state.items()}


def probes(name, model):
    if name == "general_rlt":
        a = torch.tensor([7, 13, 255], dtype=torch.long)
        b = torch.tensor([9, 11, 1], dtype=torch.long)
        return (model(a, b, "add", width=8), model(a, b, "xor", width=8))
    if name == "carry_gru":
        a = torch.tensor([7, 13, 255], dtype=torch.long)
        b = torch.tensor([9, 11, 1], dtype=torch.long)
        return (model(a, b, width=8),)
    if name == "shared_rlt_isa":
        states = torch.zeros((2, 4, 16), dtype=torch.float32)
        states[0, 0, :4] = torch.tensor([1., 0., 1., 1.])
        states[1, 1, :4] = torch.tensor([0., 1., 0., 1.])
        pc = torch.tensor([0, 3], dtype=torch.long)
        x = model(states, pc)
        op = torch.tensor([0, 3], dtype=torch.long)
        signals = torch.tensor([[0., 1., 0., 0., 1., 1.], [1., 1., 0., 1., 0., 0.]])
        return (*x, *model.local_logits(op, signals))
    if name == "standalone_isa":
        x = torch.zeros(3, 11)
        x[0, 0] = 1
        x[1, 2] = 1
        x[2, 4] = 1
        x[:, 5:] = torch.tensor([[1., 1., 0., 1., 0., 1.], [0., 1., 1., 1., 1., 0.], [1., 0., 0., 0., 1., 1.]])
        return (model.bit_from_features(x), model.meaning(x[:, :5]))
    if name == "standalone_controller_v2":
        states = torch.zeros((2, 4, 16))
        states[0, 1, 0] = 1
        states[1, 0, 4] = 1
        pc = torch.tensor([0, 4], dtype=torch.long)
        return model(states, pc)
    raise ValueError(name)


def probe_hex(name, model):
    was_train = model.training
    model.eval()
    with torch.no_grad():
        values = probes(name, model)
    model.train(was_train)
    return hashlib.sha256(b"".join(v.contiguous().cpu().numpy().tobytes() for v in values)).hexdigest()


def step_once(name, model, opt, numpy_rng):
    model.train()
    outputs = probes(name, model)
    # Deliberately synthetic continuation test, NOT a genuine new-skill training claim.
    # All three RNG streams affect the loss, so exact replay depends on restoring them.
    shift = random.random() * 0.1 + float(numpy_rng.random()) * 0.1
    loss = sum(((v - torch.randn(v.shape, dtype=v.dtype) * .07 - shift) ** 2).mean()
               for v in outputs)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    opt.step()
    return float(loss.detach().item())


def package(name, model, opt, numpy_rng, continuation_step, baseline_hash, origin_hashes):
    spec = SPECS[name]
    return {
        "format": "logos_nc002_full_training_state_v1",
        "model_name": name,
        "architecture": spec["model_class"],
        "tensor_schema": schema_for(model.state_dict()),
        "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "optimizer_class": "torch.optim.AdamW",
        "optimizer_config": {"lr": 1.0e-4, "weight_decay": 1.0e-4},
        "optimizer": copy.deepcopy(opt.state_dict()),
        "rng": capture_rng(numpy_rng),
        "continuation_step": continuation_step,
        "historical_source_progress": spec["source_training_progress"],
        "training_kind": "synthetic_resume_probe_only_NOT_new_capability_training",
        "source_frozen_baseline_probe_sha256": baseline_hash,
        "vocabulary": list(spec["vocabulary"]),
        "normalization": "in_model_layernorm_or_explicit_input_bits_no_external_fitted_statistics",
        "source_path": spec["repo"] + ":" + spec["source"],
        "source_hashes": dict(origin_hashes),
        "source_git_commit": {
            "rlt_champion": "3bcdc2489bd933d2f94fb2b2a86040e859f85ebb",
            "standalone_isa": "feb001dbdb2e19fb5f123fe12116aec5503522fe",
        }[spec["repo"]],
        "seed": 11052026,
        "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__), "numpy": np.__version__, "device": "cpu"},
        "config": {"model": name, "source_arm": "jointly_tuned" if name == "shared_rlt_isa" else "frozen_seed11_or_standalone_v2", "checkpoint_kind": "new_continuation_fork"},
        "legacy_optimizer_or_rng_recovered": False,
    }


def load_checked(name: str, path: Path):
    verify_inventory()
    spec = SPECS[name]
    obj = torch.load(path, weights_only=True, map_location="cpu")
    if obj.get("format") != "logos_nc002_full_training_state_v1" or obj.get("model_name") != name:
        raise RuntimeError("PACKAGE_FORMAT_OR_MODEL_MISMATCH")
    if obj.get("architecture") != spec["model_class"] or obj.get("vocabulary") != list(spec["vocabulary"]):
        raise RuntimeError("PACKAGE_ARCHITECTURE_OR_VOCAB_MISMATCH")
    current = {k: sha(p) for k, p in source_paths(name).items()}
    if obj["source_hashes"] != current:
        raise RuntimeError("PINNED_SOURCE_OR_ARCHITECTURE_CHANGED")
    model = build(name)
    if schema_for(model.state_dict()) != obj["tensor_schema"]:
        raise RuntimeError("PARAMETER_TOPOLOGY_MISMATCH")
    model.load_state_dict(obj["model"], strict=True)
    opt = optimizer_for(model)
    if obj["optimizer_class"] != "torch.optim.AdamW" or obj["optimizer_config"] != {"lr": 1.0e-4, "weight_decay": 1.0e-4}:
        raise RuntimeError("OPTIMIZER_CONFIGURATION_MISMATCH")
    opt.load_state_dict(obj["optimizer"])
    numpy_rng = np.random.default_rng()
    restore_rng(obj["rng"], numpy_rng)
    return model, opt, numpy_rng, obj


def generate():
    inv = verify_inventory()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = {
        "task": "NC-002", "schema_version": 1,
        "baseline_inventory_sha256": sha(INVENTORY),
        "original_champion_checkpoint_sha256": inv["champion"]["selected_bundle_sha256"],
        "original_source_git_commits": inv["repository_commits"],
        "original_inventory_file_hashes_verified": 60,
        "NC001_contract_sha256": sha(LOGOS / "artifacts/NC-001/isa_spec.json"),
        "NC001_independent_evidence_sha256": sha(LOGOS / "artifacts/NC-001/independent_verify.json"),
        "NC000_independent_evidence_sha256": sha(LOGOS / "artifacts/NC-000/baseline_receipt.json"),
        "source_optimizer_history_missing": True,
        "models": {},
        "scientific_scope": "Frozen behavior reload + exact synthetic continuation under same CPU environment; no true historical training-state recovery",
    }
    for name in SPECS:
        torch.manual_seed(11052026)
        random.seed(11052026)
        numpy_rng = np.random.default_rng(11052026)
        model = build(name)
        model.load_state_dict(state_from_source(name), strict=True)
        baseline = probe_hex(name, model)
        opt = optimizer_for(model)
        origins = {k: sha(p) for k, p in source_paths(name).items()}
        initial_path = HERE / f"{name}_initial.pt"
        torch.save(package(name, model, opt, numpy_rng, 0, baseline, origins), initial_path)
        initial_probe = probe_hex(name, model)
        assert initial_probe == baseline
        first_loss = step_once(name, model, opt, numpy_rng)
        one_step_hash = probe_hex(name, model)
        resumed_path = HERE / f"{name}_continuation_step1.pt"
        torch.save(package(name, model, opt, numpy_rng, 1, baseline, origins), resumed_path)
        model2, opt2, rng2, chk2 = load_checked(name, resumed_path)
        assert probe_hex(name, model2) == one_step_hash
        loss_direct = step_once(name, model, opt, numpy_rng)
        final_direct = probe_hex(name, model)
        restore_rng(chk2["rng"], rng2)
        loss_resumed = step_once(name, model2, opt2, rng2)
        final_resumed = probe_hex(name, model2)
        if final_direct != final_resumed or loss_direct != loss_resumed:
            raise RuntimeError(f"RESUME_CONTINUITY_FAILED: {name}")
        record = {
            "class": SPECS[name]["model_class"],
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "source_checkpoint": SPECS[name]["repo"] + ":" + SPECS[name]["source"],
            "source_checkpoint_sha256": origins[SPECS[name]["repo"] + ":" + SPECS[name]["source"]],
            "source_architecture_and_dependency_hashes": origins,
            "tensor_schema_sha256": hashlib.sha256(json.dumps(schema_for(model.state_dict()), sort_keys=True).encode()).hexdigest(),
            "vocabulary": SPECS[name]["vocabulary"],
            "baseline_probe_sha256": baseline,
            "initial_checkpoint": initial_path.name,
            "initial_sha256": sha(initial_path),
            "initial_bytes": initial_path.stat().st_size,
            "continuation_checkpoint": resumed_path.name,
            "continuation_sha256": sha(resumed_path),
            "continuation_bytes": resumed_path.stat().st_size,
            "continuation_step": 1,
            "optimizer_state_entries": len(chk2["optimizer"]["state"]),
            "loss_step1": first_loss,
            "loss_step2_direct": loss_direct,
            "loss_step2_resumed": loss_resumed,
            "final_direct_probe_sha256": final_direct,
            "final_resumed_probe_sha256": final_resumed,
            "exact_resume_after_reinitializing_optimizer_at_source": True,
            "source_historical_optimizer_recovered": False,
        }
        if record["optimizer_state_entries"] <= 0:
            raise RuntimeError(f"OPTIMIZER_STATE_NOT_PERSISTED: {name}")
        manifest["models"][name] = record
        print(json.dumps({"model": name, "parameters": record["parameter_count"], "initial_hash": record["initial_sha256"], "resume_exact": True}), flush=True)
    assert sha(RLT / inv["champion"]["bundle"]) == inv["champion"]["selected_bundle_sha256"]
    (HERE / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("WROTE_MANIFEST", sha(HERE / "checkpoint_manifest.json"), flush=True)


def verify_child(name: str, manifest_path: Path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads(manifest_path.read_text())
    item = manifest["models"][name]
    source = state_from_source(name)
    fresh = build(name)
    fresh.load_state_dict(source, strict=True)
    observed_source = probe_hex(name, fresh)
    if observed_source != item["baseline_probe_sha256"]:
        raise RuntimeError("SOURCE_NEURAL_BASELINE_CHANGED")
    model, opt, numpy_rng, obj = load_checked(name, HERE / item["initial_checkpoint"])
    if sha(HERE / item["initial_checkpoint"]) != item["initial_sha256"]:
        raise RuntimeError("INITIAL_PACKAGE_HASH_CHANGED")
    if probe_hex(name, model) != observed_source:
        raise RuntimeError("FRESH_PROCESS_INITIAL_OUTPUT_MISMATCH")
    model1, opt1, rng1, state1 = load_checked(name, HERE / item["continuation_checkpoint"])
    if sha(HERE / item["continuation_checkpoint"]) != item["continuation_sha256"]:
        raise RuntimeError("RESUME_PACKAGE_HASH_CHANGED")
    if state1["continuation_step"] != 1 or len(opt1.state_dict()["state"]) != item["optimizer_state_entries"]:
        raise RuntimeError("OPTIMIZER_OR_PROGRESS_NOT_RECOVERED")
    step2 = step_once(name, model1, opt1, rng1)
    out = probe_hex(name, model1)
    if out != item["final_direct_probe_sha256"] or step2 != item["loss_step2_direct"]:
        raise RuntimeError("FRESH_PROCESS_CONTINUATION_DIVERGED")
    return {"model": name, "frozen_source_neural_logits_equal": True,
            "fresh_process_package_logits_equal": True, "step2_loss_equal": True,
            "step2_updated_weights_equal": True, "optimizer_state_entries": len(opt1.state_dict()["state"]),
            "original_historical_optimizer_restored": False}


def verify_all():
    manifest_path = HERE / "checkpoint_manifest.json"
    inv = verify_inventory()
    manifest = json.loads(manifest_path.read_text())
    if manifest["baseline_inventory_sha256"] != sha(INVENTORY):
        raise RuntimeError("INVENTORY_HASH_CHANGED")
    results = []
    for name in SPECS:
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "verify-child", "--name", name,
                               "--manifest", str(manifest_path)], capture_output=True, text=True,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False)
        if proc.returncode:
            raise RuntimeError(f"CHILD_FAILED {name}: {proc.stderr[-3000:]}")
        results.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    after = verify_inventory()
    assertion_count = len(results) * 4
    receipt = {
        "task": "NC-002", "result": "IMPLEMENTER_SELF_TEST_PASS_NOT_INDEPENDENT",
        "fresh_processes": len(results), "independent_verifier": None,
        "models": results, "assertions": {"passed": assertion_count, "tested": assertion_count},
        "checkpoint_manifest_sha256": sha(manifest_path),
        "original_champion_sha256_before": inv["champion"]["selected_bundle_sha256"],
        "original_champion_sha256_after": after["champion"]["selected_bundle_sha256"],
        "no_original_source_weights_modified": True,
        "qualifications": [
            "No original historical optimizer, RNG or scheduler state existed in baseline weight-only snapshots; cannot resume original historical update exactly.",
            "Fresh optimizer state is saved and proved resumable after one explicitly synthetic, non-promotable training step.",
            "Frozen logits reload exactly; not evidence that synthetic continuation improves learned capability.",
            "CPU deterministic same-environment replay only, not cross-hardware bitwise proof.",
            "Shared ISA remains experimental with fixed ROM and Python register wiring; no neural program invention.",
            "Independent agent verification is mandatory before NC-002 ledger PASS."
        ],
        "dependencies": ["state_package.py", "Python3", "Git", "PyTorch", "NumPy", "original pinned RLT and standalone ISA modules/checkpoints", "artifacts/NC-000/inventory.json"],
    }
    (HERE / "reload_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"result": receipt["result"], "fresh_models": len(results), "assertions": receipt["assertions"],
                     "manifest_sha256": receipt["checkpoint_manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["create", "verify", "verify-child"])
    p.add_argument("--name", choices=list(SPECS))
    p.add_argument("--manifest", type=Path)
    args = p.parse_args()
    if args.action == "create":
        generate()
    elif args.action == "verify":
        verify_all()
    else:
        if args.name is None or args.manifest is None:
            p.error("verify-child requires --name and --manifest")
        print(json.dumps(verify_child(args.name, args.manifest)))
