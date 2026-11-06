#!/usr/bin/env python3
"""Fresh independent NC-044 sandbox negative controls. Python stdlib + bwrap."""
import hashlib
import json
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trainer_isolation as trainer

checks = {}
protected = {
    "champion": trainer.CHAMPION,
    "ledger": trainer.REPO / "instructions.jsonl",
    "nc022": trainer.NC022,
}
before = {name: trainer.digest(path) for name, path in protected.items()}

with tempfile.TemporaryDirectory(prefix="review-nc044-", dir=trainer.DEFAULT_ROOT) as td:
    root = Path(td)
    source = root / "custom_negative_worker.py"
    source.write_text("""
import json
import socket
from pathlib import Path

targets = [
    '/inputs/champion.pt',
    '/inputs/worker.py',
    '/usr/bin/python3',
    '/workspace/experiments/logos-kernel/instructions.jsonl',
    '/workspace/.local/state/logos-kernel-trainer/run-other/output/escape',
    '/proc/1/root/workspace/experiments/logos-kernel/instructions.jsonl',
]
denied = []
for target in targets:
    try:
        with open(target, 'ab') as f:
            f.write(b'attack')
        denied.append(False)
    except OSError:
        denied.append(True)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1)
try:
    external_connect = sock.connect_ex(('198.51.100.1', 443)) == 0
except OSError:
    external_connect = False
finally:
    sock.close()
Path('/output/reviewer_probe.json').write_text(
    json.dumps({'denied': denied, 'external_connect': external_connect})
)
""")
    run, manifest = trainer.make_run(root, source, [], 12, 6, 512, 4)
    report = trainer.execute(root, manifest["run_id"])
    assert report["outcome"] == "completed" and report["returncode"] == 0, report
    res = json.loads((run / "output/reviewer_probe.json").read_text())
    assert len(res["denied"]) == 6 and all(res["denied"]), res
    assert not res["external_connect"], res
    checks["new_worker_host_path_write_denials"] = 6
    checks["new_worker_external_tcp_connection"] = "DENIED"

    run2, m2 = trainer.make_run(root, source, [], 12, 6, 512, 4)
    source.write_text(source.read_text() + "\n# altered after manifest\n")
    try:
        trainer.execute(root, m2["run_id"])
        raise AssertionError("source modification accepted")
    except trainer.BoundaryError as error:
        assert "changed pinned input" in str(error), str(error)
    checks["source_mutation_before_start"] = "REFUSED"
    try:
        trainer.make_run(root, root / "not_a_file", [], 12, 6, 512, 4)
        raise AssertionError("nonexistent source accepted")
    except trainer.BoundaryError:
        checks["missing_source"] = "REFUSED"

after = {name: trainer.digest(path) for name, path in protected.items()}
assert before == after, (before, after)
checks["protected_control_files_unchanged"] = True
print(json.dumps({"task":"NC-044", "checks":checks,
                  "protected_sha256": after}, sort_keys=True, indent=2))
