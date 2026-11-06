"""NC-046 independent-style deterministic integrity checks for bounded canary.
Depends on: Python unittest, local trainer_worker.py, torch, numpy,
frozen NC-022 independent sources/data and read-only NC-045 trainer status.
NOT a separate independent reviewer receipt.
"""
import copy
import fcntl
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
import unittest

from trainer_worker import (
    ARMS, SEEDS, HERE, Refused, ZERO, count_exposure,
    digest, existing, exclusive, make_steps, production_guard,
    reference, run_case, sha_file, trainer_status, validate_disjoint,
)

PFX="unittest-"+uuid.uuid4().hex[:12]
def summarize(r):
    return {k:r[k] for k in ("seed","arm","steps_committed","model_sha256",
                            "step_plan_sha256","losses","heldout","total_examples")}
class TestNC046(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref,cls.data=reference()
        cls.original_trainer=trainer_status()
        cls.original_champion=sha_file(cls.ref.SOURCE)

    @classmethod
    def tearDownClass(cls):
        after=trainer_status()
        assert after["seq"]==cls.original_trainer["seq"]
        assert after["head_sha256"]==cls.original_trainer["head_sha256"]
        assert sha_file(cls.ref.SOURCE)==cls.original_champion
        # Cleanup only the unique test-generated run prefix; never delete
        # another worker's checkpoint or any frozen pilot artifacts.
        for directory in (HERE/"runs").glob(PFX+"-*"):
            assert directory.is_dir() and not directory.is_symlink()
            assert directory.resolve().is_relative_to((HERE/"runs").resolve())
            shutil.rmtree(directory)

    def test_production_fails_closed(self):
        with self.assertRaisesRegex(Refused,"PRODUCTION_DISABLED"):
            production_guard()
        self.assertFalse(any(t["approved"] for t in
                             self.original_trainer["tasks"] if t["id"]=="TR-002"))

    def test_disjoint_negative_control(self):
        _,data=self.ref,self.data
        blocks=copy.deepcopy(data["11"]["blocks"])
        groups=data["11"]["groups"]
        held={self.ref.pairkey(*p) for pairs in groups.values() for p in pairs}
        assert validate_disjoint(blocks,held,self.ref)>0
        # Inject a real holdout pair into a training block; fail before gradient.
        blocks["1"][0]["pairs"][0]=groups["iid16"][0]
        with self.assertRaisesRegex(Refused,"TRAIN_EVAL_LEAK"):
            validate_disjoint(blocks,held,self.ref)

    def test_exact_exposure_rehearsal_vs_blocked(self):
        for seed in SEEDS:
            arms={arm:make_steps(self.data,seed,arm,self.ref) for arm in ARMS}
            counts={arm:count_exposure(v) for arm,v in arms.items()}
            self.assertEqual(counts["rehearsal"],counts["blocked_replay"])
            self.assertNotEqual(counts["rehearsal"],counts["new_only"])
            self.assertEqual({sum(v[-1] for v in c) for c in counts.values()},{192})

    def test_lock(self):
        path=HERE/"runs"/(PFX+"-lock")
        with exclusive(path):
            with self.assertRaisesRegex(Refused,"WORKER_ALREADY_RUNNING"):
                with exclusive(path): pass

    def test_corrupted_checkpoint(self):
        path=HERE/"runs"/(PFX+"-tamper")
        self.assertFalse(path.exists(),"test run must be fresh")
        run_case(11,"new_only",path,1)
        f=existing(path)[0][1]
        with f.open("ab") as out:out.write(b"tamper")
        with self.assertRaisesRegex(Refused,"CORRUPTED_CHECKPOINT"):
            run_case(11,"new_only",path,2)

    def test_resume_after_one(self):
        self.assert_restart(11,"rehearsal",1)

    def test_resume_after_two(self):
        self.assert_restart(19,"blocked_replay",2)

    def assert_restart(self,seed,arm,after):
        complete=HERE/"runs"/f"{PFX}-full-{seed}-{arm}"
        resumed=HERE/"runs"/f"{PFX}-resumed-{seed}-{arm}"
        self.assertFalse(complete.exists() or resumed.exists(),"test run must be fresh")
        uninterrupted=run_case(seed,arm,complete,4)
        partial=run_case(seed,arm,resumed,after)
        self.assertEqual(partial["steps_committed"],after)
        snap_before=[h for _,_,h in existing(resumed)]
        recovered=run_case(seed,arm,resumed,4)
        self.assertEqual([h for _,_,h in existing(resumed)][:after],snap_before)
        self.assertEqual(summarize(uninterrupted),summarize(recovered),
                         "resume must exactly match uninterrupted learned state, losses and predictions")
        # Repeating a committed step is an idempotent read; no duplicate gradient.
        reread=run_case(seed,arm,resumed,4)
        self.assertEqual(summarize(recovered),summarize(reread))
        self.assertEqual([h for _,_,h in existing(resumed)][:after],snap_before)

if __name__=="__main__":unittest.main()
