"""Reproduce NC-062 successor checkpoints from pinned parent states.

Dependencies: Python stdlib, run_loop.py, worker_source.py,
NC-028/isolation.py (Landlock/libseccomp), CPU PyTorch,
NC-009 verified registry, 12 candidate checkpoint JSONs.
This is an implementer reproduction, NOT an independent review.
"""
from pathlib import Path
import json
import run_loop as r

def replay():
    original=r.verified_initial_library()
    replay=[]
    first={}
    for seed in r.SEEDS:
        initial=json.loads((r.HERE/f"checkpoint_g0_seed{seed}.json").read_text())
        fresh,charge=r.spawn("init",seed,None)
        assert fresh["model_sha256_after"]==initial["model_state_sha256"]
        first[seed]=initial
        replay.append({"generation":0,"seed":seed,
                       "model_sha256":fresh["model_sha256_after"],
                       "cpu_s":charge["cpu_seconds"]})
    for generation in r.GENS:
        for seed in r.SEEDS:
            before=first[seed] if generation==1 else json.loads(
                (r.HERE/f"checkpoint_g{generation-1}_seed{seed}.json").read_text())
            after=json.loads((r.HERE/f"checkpoint_g{generation}_seed{seed}.json").read_text())
            assert after["parent_state_sha256"]==before["model_state_sha256"]
            data=r.train_rows(original,seed,generation)
            assert after["train_input_sha256"]==r.digest(data)
            result,charge=r.spawn("train",seed,before["model_state"],training=data)
            assert result["model_sha256_before"]==before["model_state_sha256"]
            assert result["model_sha256_after"]==after["model_state_sha256"]
            assert result["gradient_updates"]==8
            replay.append({"generation":generation,"seed":seed,
                           "parent_model_sha256":before["model_state_sha256"],
                           "model_sha256":result["model_sha256_after"],
                           "gradient_updates":8,
                           "cpu_s":charge["cpu_seconds"]})
    assert len(replay)==12
    results={"task":"NC-062","schema":"nc062-implementer-train-replay/1",
            "checkpoint_count":len(replay),"all_training_state_sha_reproduced":True,
            "all_train_updates":72,"seeds":list(r.SEEDS),
            "generation_count":len(r.GENS),"results":replay,
            "independent_reviewer":False,
            "not_claimed":"neural program invention, generalization, independent retention or AGI"}
    out=r.HERE/"training_replay.json"
    r.write_once(out,results)
    print(json.dumps({"checkpoints_reproduced":len(replay),"gradient_updates":72,
                      "sha256":r.sha(out)},sort_keys=True))
if __name__=="__main__":replay()
