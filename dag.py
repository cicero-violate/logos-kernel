#!/usr/bin/env python3
"""Append-only, SHA-256 chained DAG instructions with explicit ownership claims.

Standard-library-only. The original 17 events (including the historical
status=working claim) are replayable. New task ownership uses claim/release.
"""
import argparse
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "instructions.jsonl"
ZERO = "0" * 64
# Legacy events are replayed, not erased. No new CLI command creates "working".
TRANS = {
    "planned": {"working", "passed", "failed", "blocked", "archived"},
    "working": {"passed", "blocked", "failed"},
    "blocked": {"working", "archived"},
    "failed": {"working", "archived"},
    "passed": set(),
    "archived": set(),
}
PLACEHOLDERS = {"YOUR_AGENT_ID", "INDEPENDENT_VERIFIER_ID", "worker-1", "agent-name"}


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_event(prev, body):
    return hashlib.sha256((prev + "\n" + canon(body)).encode()).hexdigest()


def real_id(value, field):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a nonblank actual identity")
    if value in PLACEHOLDERS:
        raise ValueError(f"{field} is a placeholder, not an actual identity")
    return value


def dependencies_passed(item, tasks):
    return all(tasks[dep]["status"] == "passed" for dep in item["record"]["depends_on"])


def validate_event(ev, tasks):
    kind = ev.get("kind")
    if kind == "task":
        task_id = ev["id"]
        if not isinstance(task_id, str) or not task_id or task_id in tasks:
            raise ValueError("duplicate/empty task ID")
        deps = ev["depends_on"]
        if not isinstance(deps, list) or len(deps) != len(set(deps)) or any(dep not in tasks for dep in deps):
            raise ValueError(f"{task_id}: depends_on must refer uniquely to earlier tasks")
        for field in ("instruction", "acceptance", "outputs", "avoid"):
            if not ev.get(field):
                raise ValueError(f"{task_id}: missing {field}")
        if ev.get("status") != "planned":
            raise ValueError("new task must be planned")
        tasks[task_id] = {"record": ev, "status": "planned", "owner": None}
        return

    task_id = ev.get("id")
    if task_id not in tasks:
        raise ValueError(f"unknown task: {task_id!r}")
    item = tasks[task_id]

    if kind == "claim":
        owner = real_id(ev.get("owner"), "owner")
        if item["status"] != "planned" or item["owner"] is not None:
            raise ValueError(f"{task_id}: already claimed or not planned")
        if not dependencies_passed(item, tasks):
            raise ValueError(f"{task_id}: dependencies not passed")
        item["owner"] = owner
        return

    if kind == "release":
        previous_owner = ev.get("expected_owner")
        if not isinstance(previous_owner, str) or not previous_owner or previous_owner != item["owner"]:
            raise ValueError(f"{task_id}: expected_owner mismatch or task unclaimed")
        actor = real_id(ev.get("by"), "release actor")
        if actor != previous_owner:
            if not ev.get("force") or not ev.get("reason"):
                raise ValueError("releasing another owner's claim requires --force and reason")
        if item["status"] not in {"planned", "working"}:
            raise ValueError(f"{task_id}: cannot release terminal/failed task")
        # Restore availability without changing historical claim rows.
        item["owner"] = None
        if item["status"] == "working":  # historical status event
            item["status"] = "planned"
        return

    if kind == "reopen":
        real_id(ev.get("by"), "reopen actor")
        if not ev.get("reason"):
            raise ValueError("reopen requires reason")
        if item["status"] not in {"blocked", "failed"} or item["owner"] is not None:
            raise ValueError("only unclaimed blocked/failed tasks can be reopened")
        item["status"] = "planned"
        return

    if kind == "status":
        new = ev.get("status")
        if new not in TRANS or new not in TRANS[item["status"]]:
            raise ValueError(f"invalid transition {item['status']} -> {new}")
        if new in {"working", "passed"} and not dependencies_passed(item, tasks):
            raise ValueError("dependencies not passed")
        actor = ev.get("agent")
        if new == "working":
            # Retained only to validate the original historical event.
            if not actor or item["owner"] is not None:
                raise ValueError("legacy working requires unclaimed task and agent")
            item["owner"] = actor
        elif new in {"passed", "failed", "blocked"}:
            if item["owner"] is None or actor != item["owner"]:
                raise ValueError("only current task owner may change its status")
        if new == "passed":
            if item["owner"] is None or not actor or not ev.get("verifier") or not ev.get("evidence"):
                raise ValueError("passing requires claim owner, verifier and evidence")
            real_id(ev["verifier"], "verifier")
            if ev["verifier"] == actor:
                raise ValueError("independent verifier must differ from task owner")
            evidence = Path(ev["evidence"])
            if evidence.is_absolute() or ".." in evidence.parts or not (ROOT / evidence).is_file():
                raise ValueError("passing requires an existing relative evidence file")
        item["status"] = new
        if new in {"passed", "failed", "blocked", "archived"}:
            item["owner"] = None
        return

    raise ValueError(f"unknown event kind: {kind!r}")


def audit(raw):
    tasks = {}
    prev = ZERO
    lines = raw.splitlines()
    for seq, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"blank ledger line {seq}")
        row = json.loads(line)
        if row["seq"] != seq or row["prev_sha256"] != prev:
            raise ValueError(f"broken sequence/previous hash at {seq}")
        body = {"seq": seq, "event": row["event"]}
        if row["sha256"] != hash_event(prev, body):
            raise ValueError(f"hash changed at {seq}")
        validate_event(row["event"], tasks)
        prev = row["sha256"]
    return tasks, len(lines), prev


def append(ev):
    LOG.touch(exist_ok=True)
    with LOG.open("a+", encoding="utf-8") as ledger:
        fcntl.flock(ledger, fcntl.LOCK_EX)
        ledger.seek(0)
        tasks, n, prev = audit(ledger.read())
        validate_event(ev, tasks)
        body = {"seq": n + 1, "event": ev}
        row = {**body, "prev_sha256": prev, "sha256": hash_event(prev, body)}
        ledger.seek(0, os.SEEK_END)
        ledger.write(canon(row) + "\n")
        ledger.flush()
        os.fsync(ledger.fileno())
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in ("validate", "ready", "list"):
        sub.add_parser(cmd)
    add = sub.add_parser("add-task")
    add.add_argument("--file", required=True, type=Path)

    claim = sub.add_parser("claim", help="exclusively claim a ready task")
    claim.add_argument("id")
    claim.add_argument("--owner", required=True)

    release = sub.add_parser("release", help="release a claim; force requires reason")
    release.add_argument("id")
    release.add_argument("--by", required=True, help="actual actor identity")
    release.add_argument("--reason", default="")
    release.add_argument("--force", action="store_true")

    reopen = sub.add_parser("reopen", help="reopen a blocked or failed task")
    reopen.add_argument("id")
    reopen.add_argument("--by", required=True)
    reopen.add_argument("--reason", required=True)

    for cmd in ("pass", "fail", "block"):
        op = sub.add_parser(cmd)
        op.add_argument("id")
        op.add_argument("--owner", required=True)
        op.add_argument("--evidence")
        op.add_argument("--verifier")
        op.add_argument("--note", default="")

    args = parser.parse_args()
    if args.cmd == "add-task":
        row = append(json.loads(args.file.read_text()))
    elif args.cmd == "claim":
        row = append({"kind": "claim", "id": args.id, "owner": args.owner})
    elif args.cmd == "release":
        # Resolve expected owner from ledger snapshot. The append revalidates
        # this value *under lock*, so a release cannot clear a later claim.
        tasks, _, _ = audit(LOG.read_text())
        item = tasks.get(args.id)
        if not item or item["owner"] is None:
            raise ValueError(f"{args.id}: no claim to release")
        row = append({"kind": "release", "id": args.id, "by": args.by,
                      "expected_owner": item["owner"], "force": args.force, "reason": args.reason})
    elif args.cmd == "reopen":
        row = append({"kind": "reopen", "id": args.id, "by": args.by, "reason": args.reason})
    elif args.cmd in ("pass", "fail", "block"):
        status = {"pass": "passed", "fail": "failed", "block": "blocked"}[args.cmd]
        event = {"kind": "status", "id": args.id, "status": status, "agent": args.owner, "note": args.note}
        if args.evidence:
            event["evidence"] = args.evidence
        if args.verifier:
            event["verifier"] = args.verifier
        row = append(event)
    else:
        tasks, n, head = audit(LOG.read_text() if LOG.exists() else "")
        if args.cmd == "validate":
            print(json.dumps({"status": "PASS", "nodes": len(tasks), "events": n, "head_sha256": head}, indent=2))
            return
        result = []
        for task_id, item in tasks.items():
            ready = item["status"] == "planned" and item["owner"] is None and dependencies_passed(item, tasks)
            if args.cmd == "ready" and not ready:
                continue
            result.append({"id": task_id, "priority": item["record"]["priority"],
                           "title": item["record"]["title"], "status": item["status"],
                           "owner": item["owner"], "ready": ready,
                           "depends_on": item["record"]["depends_on"]})
        print(json.dumps(result, indent=2))
        return
    print(json.dumps({"appended_seq": row["seq"], "event": row["event"], "head_sha256": row["sha256"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as err:
        print("FAIL:", err, file=sys.stderr)
        sys.exit(2)
