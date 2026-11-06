#!/usr/bin/env python3
"""Append-only, SHA-256 chained DAG instructions with explicit ownership claims.

Requires Python standard library and the Git CLI. The original 17 events (including the historical
status=working claim) are replayable. New task ownership uses claim/release.
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
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
# A claim with no published candidate is presumed live for this long.
CLAIM_LEASE_SECONDS = 2 * 3600
STANDARD_RECEIPT = "independent_verification.json"


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



def _review_id(value):
    return real_id(value, "review identity")


def _review_principal(value):
    """Normalize known Event Lab aliases; not proof of identity authenticity."""
    value = _review_id(value)
    prefix = "chatgpt-session-"
    if value.startswith(prefix) and re.fullmatch(r"[0-9a-f]{32}", value[len(prefix):]):
        return value[len(prefix):]
    return value


def _review_sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("review requires a 64-hex SHA-256 digest")
    return value


def _review_path(task_id, value):
    if (not isinstance(value, str) or not value.startswith("artifacts/" + task_id + "/")
            or "\\" in value or ".." in Path(value).parts or Path(value).is_absolute()):
        raise ValueError("review evidence must be inside the task artifacts directory")
    return value


def review_candidate_bytes(relative):
    """Read the content bound by a review request. Not reviewer authentication."""
    if not isinstance(relative, str) or not relative.startswith("artifacts/"):
        raise ValueError("candidate must be a task artifact")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError("unsafe candidate path")
    safe_reader = globals().get("read_safe_regular_evidence")
    if safe_reader is not None:
        data = safe_reader(relative)
        if data is None:
            raise ValueError("candidate evidence must be a safe bounded regular file")
        return data
    import stat
    import os
    target = (ROOT / relative).resolve(strict=True)
    try:
        target.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("candidate outside repository") from exc
    fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 131072:
            raise ValueError("candidate must be a bounded single-link regular file")
        data = b""
        while len(data) <= 131072:
            piece = os.read(fd, min(65536, 131073 - len(data)))
            if not piece:
                break
            data += piece
        end = os.fstat(fd)
        if len(data) != info.st_size or len(data) > 131072 or (info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (end.st_size, end.st_mtime_ns, end.st_ctime_ns):
            raise ValueError("candidate evidence changed during read")
        return data
    finally:
        os.close(fd)


def ticket_order(pair):
    task_id, item = pair
    return (item["rank"] is None, item["rank"] if item["rank"] is not None else 999999, task_id)


def _now():
    return int(time.time())


def _legacy_claim_time(task_id):
    """Commit time of a claim recorded before claims carried a timestamp."""
    pattern = "^DAG event [0-9]+: claim " + re.escape(task_id[:48]) + "$"
    try:
        value = _git("log", "-1", "--format=%ct", "--extended-regexp", "--grep=" + pattern)
        return int(value) if value else None
    except (OSError, RuntimeError, ValueError):
        return None


def _takeover_reason(task_id, item, principal, now):
    """Why a stateless turn may continue this task, or None.

    Ownership belongs to the ledger, because a turn's ID ends with the turn. A
    finished review can be picked up at once, exactly once per verdict. Any other
    claim may still be in progress, so it is only taken after the lease expires.
    """
    if item["status"] not in ("planned", "working", "blocked", "failed"):
        return None
    if item["owner"] is not None and _review_principal(item["owner"]) == principal:
        return None
    review = item["review"]
    if review and review["state"] in ("pending", "claimed"):
        return None
    if review and _review_principal(review["reviewer"]) == principal:
        return None  # A reviewer must not finish or rework what it just judged.
    if review and not review.get("taken_over"):
        return "finalize" if review["verdict"] == "PASS" else "rework"
    if item["owner"] is None or item["status"] not in ("planned", "working"):
        return None
    claimed = item.get("claimed_at")
    if claimed is None:
        claimed = _legacy_claim_time(task_id)
    if claimed is None or now - claimed < CLAIM_LEASE_SECONDS:
        return None
    return "stale_claim"


def _work_role(item):
    review = item["review"]
    closed_pass = review and review["state"] == "closed" and review["verdict"] == "PASS"
    return "FINALIZER" if closed_pass else "IMPLEMENTER"


def _work_details(item):
    """What the next turn needs from a finished review, since it has no memory."""
    review = item["review"]
    if not review or review["state"] != "closed":
        return {}
    if review["verdict"] == "PASS":
        return {"verifier": review["reviewer"], "evidence": review.get("evidence")}
    return {"review_verdict": review["verdict"], "review_note": review.get("note", "")}


def choose_pull_event(tasks, agent):
    """Choose under the ledger append lock. Never equate a role string with independence."""
    agent = _review_id(agent)
    principal = _review_principal(agent)
    ordered = sorted(tasks.items(), key=ticket_order)
    for task_id, item in ordered:
        review = item["review"]
        if review and review["state"] == "claimed" and _review_principal(review["reviewer"]) == principal:
            return {"role": "VERIFIER", "task": task_id, "reused": True,
                    "candidate_sha256": review["candidate_sha256"], "reviewer": review["reviewer"]}
    for task_id, item in ordered:
        review = item["review"]
        if review and review["state"] == "pending" and _review_principal(review["owner"]) != principal and item["status"] != "passed":
            return {"kind": "review_claim", "id": task_id, "reviewer": agent, "candidate_sha256": review["candidate_sha256"]}
    for task_id, item in ordered:
        if item["owner"] is not None and _review_principal(item["owner"]) == principal and item["status"] in ("planned", "working"):
            # Later writes match the ledger label exactly, so report it.
            return {"role": _work_role(item), "task": task_id, "reused": True, "owner": item["owner"],
                    **_work_details(item)}
    now = _now()
    for wanted in ("finalize", "rework", "stale_claim"):
        for task_id, item in ordered:
            if _takeover_reason(task_id, item, principal, now) == wanted:
                return {"kind": "takeover", "id": task_id, "owner": agent, "reason": wanted,
                        "previous_owner": item["owner"], "at": now}
    for task_id, item in ordered:
        if item["status"] == "planned" and item["owner"] is None and dependencies_passed(item, tasks):
            return {"kind": "claim", "id": task_id, "owner": agent}
    return {"role": "HOLD", "task": None, "reason": "No independently eligible review or claimable implementation task"}


def pull(agent):
    result = append({"kind": "_pull", "agent": _review_id(agent)})
    if "role" in result:
        return result
    event = result["event"]
    raw = read_ledger_bytes().decode("utf-8") if "read_ledger_bytes" in globals() else LOG.read_text()
    item = audit(raw)[0][event["id"]]
    outcome = {"role": "VERIFIER" if event["kind"] == "review_claim" else _work_role(item),
               "task": event["id"], "reused": False, "git_commit": result["git_commit"]}
    if event["kind"] != "review_claim":
        outcome.update(_work_details(item))
    if outcome["role"] == "VERIFIER":
        review = item["review"]
        outcome.update({"candidate_path": review["candidate_path"],
                        "candidate_sha256": review["candidate_sha256"],
                        "candidate_revision": review["revision"],
                        "candidate_owner": review["owner"]})
    return outcome

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
        if ev.get("priority") not in {"P0", "P1", "P2", "P3", "P4", "P5"}:
            raise ValueError("invalid task priority")
        tasks[task_id] = {"record": ev, "status": "planned", "owner": None, "last_owner": None, "review": None,
                          "priority": ev["priority"], "rank": None}
        return

    if kind == "priorities":
        entries = ev.get("assignments")
        if not ev.get("reason") or not isinstance(entries, list) or not entries:
            raise ValueError("priority plan requires reason and assignments")
        open_ids = {key for key, item in tasks.items()
                    if item["status"] in {"planned", "working"}}
        ids = [a.get("id") for a in entries if isinstance(a, dict)]
        ranks = [a.get("rank") for a in entries if isinstance(a, dict)]
        if len(ids) != len(entries) or len(ids) != len(set(ids)) or set(ids) != open_ids:
            raise ValueError("priority plan must cover every open task exactly once")
        if any(a.get("priority") not in {"P0", "P1", "P2", "P3", "P4", "P5"}
               or type(a.get("rank")) is not int or a["rank"] < 1 for a in entries):
            raise ValueError("invalid priority or rank")
        if len(ranks) != len(set(ranks)):
            raise ValueError("priority ranks must be unique")
        if sorted(ranks) != list(range(1, len(entries) + 1)):
            raise ValueError("priority ranks must be contiguous starting at 1")
        for a in entries:
            tasks[a["id"]]["priority"] = a["priority"]
            tasks[a["id"]]["rank"] = a["rank"]
        return

    task_id = ev.get("id")
    if task_id not in tasks:
        raise ValueError(f"unknown task: {task_id!r}")
    item = tasks[task_id]


    if kind == "review_request":
        author = _review_id(ev.get("owner"))
        if item["status"] == "passed":
            raise ValueError("passed task cannot request review")
        expected = item["owner"] or item["last_owner"]
        if expected is None or author != expected:
            raise ValueError("only current or last task owner may request review")
        if item["owner"] is None and item["status"] not in ("blocked", "failed"):
            raise ValueError("unclaimed planned task cannot request review")
        if item["review"] is not None and item["review"]["state"] in ("pending", "claimed"):
            raise ValueError("review is already pending or claimed")
        candidate = _review_sha(ev.get("candidate_sha256"))
        path = _review_path(task_id, ev.get("candidate_path"))
        revision = ev.get("revision")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("review revision must be a Git commit SHA")
        item["review"] = {
            "state": "pending", "owner": author,
            "candidate_sha256": candidate, "candidate_path": path,
            "revision": revision, "reviewer": None, "verdict": None
        }
        return

    if kind == "review_claim":
        reviewer = _review_id(ev.get("reviewer"))
        candidate = _review_sha(ev.get("candidate_sha256"))
        review = item["review"]
        if (review is None or review["state"] != "pending" or
                candidate != review["candidate_sha256"]):
            raise ValueError("review claim requires matching pending request")
        if _review_principal(reviewer) == _review_principal(review["owner"]):
            raise ValueError("implementer cannot claim own verification")
        review["reviewer"] = reviewer
        review["state"] = "claimed"
        return

    if kind == "review_release":
        # A reviewer who cannot be independent hands the review back: no verdict, no rework.
        reviewer = _review_id(ev.get("reviewer"))
        review = item["review"]
        if (review is None or review["state"] != "claimed"
                or _review_principal(review["reviewer"]) != _review_principal(reviewer)):
            raise ValueError("only the assigned reviewer can release a claimed review")
        if not isinstance(ev.get("note"), str) or not ev["note"].strip():
            raise ValueError("review release requires a nonblank reason")
        review["reviewer"] = None
        review["state"] = "pending"
        return

    if kind == "review_result":
        reviewer = _review_id(ev.get("reviewer"))
        candidate = _review_sha(ev.get("candidate_sha256"))
        review = item["review"]
        if (review is None or review["state"] != "claimed" or
                review["reviewer"] != reviewer or
                candidate != review["candidate_sha256"]):
            raise ValueError("only assigned reviewer can complete matching review")
        verdict = ev.get("verdict")
        if verdict not in ("PASS", "FAIL", "HOLD"):
            raise ValueError("review verdict must be PASS, FAIL or HOLD")
        note = ev.get("note", "")
        if not isinstance(note, str):
            raise ValueError("review note must be text")
        if verdict == "PASS":
            evidence = _review_path(task_id, ev.get("evidence"))
            # Some task records declare only candidate files, which a reviewer must not overwrite.
            if (evidence not in item["record"]["outputs"]
                    and evidence != f"artifacts/{task_id}/{STANDARD_RECEIPT}"):
                raise ValueError("PASS receipt must be a declared task output")
            _review_sha(ev.get("evidence_sha256"))
            review["evidence"] = evidence
            review["evidence_sha256"] = ev["evidence_sha256"]
        review["verdict"] = verdict
        review["note"] = note
        review["state"] = "closed"
        return

    if kind == "takeover":
        owner = real_id(ev.get("owner"), "owner")
        principal = _review_principal(owner)
        reason = ev.get("reason")
        at = ev.get("at")
        review = item["review"]
        if reason not in ("finalize", "rework", "stale_claim") or type(at) is not int or at <= 0:
            raise ValueError("takeover requires a known reason and integer time")
        if item["status"] not in ("planned", "working", "blocked", "failed"):
            raise ValueError(f"{task_id}: finished task cannot be taken over")
        if ev.get("previous_owner") != item["owner"]:
            raise ValueError(f"{task_id}: previous_owner mismatch")
        if item["owner"] is not None and _review_principal(item["owner"]) == principal:
            raise ValueError(f"{task_id}: already owned by this agent")
        if review and review["state"] in ("pending", "claimed"):
            raise ValueError(f"{task_id}: cannot take over during review")
        if review and _review_principal(review["reviewer"]) == principal:
            raise ValueError("reviewer cannot take over the task it reviewed")
        if reason == "stale_claim":
            if (item["owner"] is None or item["status"] not in ("planned", "working")
                    or (review and not review.get("taken_over"))):
                raise ValueError(f"{task_id}: no stale claim to take over")
            claimed = item.get("claimed_at")
            if claimed is not None and at - claimed < CLAIM_LEASE_SECONDS:
                raise ValueError(f"{task_id}: claim lease has not expired")
        elif (not review or review.get("taken_over")
                or (review["verdict"] == "PASS") != (reason == "finalize")):
            raise ValueError(f"{task_id}: no finished review to continue from")
        item["owner"] = owner
        item["last_owner"] = owner
        item["claimed_at"] = at
        if item["status"] in ("blocked", "failed"):
            item["status"] = "planned"
        if review:
            review["taken_over"] = True
        return

    if kind == "claim":
        owner = real_id(ev.get("owner"), "owner")
        if item["status"] != "planned" or item["owner"] is not None:
            raise ValueError(f"{task_id}: already claimed or not planned")
        if not dependencies_passed(item, tasks):
            raise ValueError(f"{task_id}: dependencies not passed")
        item["owner"] = owner
        item["last_owner"] = owner
        item["claimed_at"] = ev.get("at")
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
            item["last_owner"] = actor
        elif new in {"passed", "failed", "blocked"}:
            if item["owner"] is None or actor != item["owner"]:
                raise ValueError("only current task owner may change its status")
        if new == "passed" and item["review"] is not None:
            review = item["review"]
            if (review["state"] != "closed" or review["verdict"] != "PASS" or
                    ev.get("verifier") != review["reviewer"] or
                    ev.get("evidence") != review.get("evidence")):
                raise ValueError("PASS requires completed matching pull-based review")
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



# This ledger is the sole Git-managed file committed by the append pipeline.
# Do not use `git add -A` or ordinary `git commit` here: those could capture
# unrelated staged files from concurrent users.
def _git(*args):
    proc = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(
            "git " + " ".join(args) + " failed (exit " + str(proc.returncode)
            + "): " + (proc.stderr or proc.stdout).strip()
        )
    return proc.stdout.strip()


def _check_git_ledger(expected_bytes):
    repository = Path(_git("rev-parse", "--show-toplevel")).resolve()
    if repository != ROOT.resolve():
        raise RuntimeError("ledger Git root mismatch: " + str(repository))
    previous = subprocess.run(
        ["git", "-C", str(ROOT), "show", "HEAD:instructions.jsonl"],
        capture_output=True, check=False,
    )
    if previous.returncode:
        raise RuntimeError(
            "Git HEAD does not contain instructions.jsonl; create a baseline commit before appending"
        )
    if not expected_bytes.startswith(previous.stdout):
        raise RuntimeError(
            "current ledger differs from committed Git prefix; refusing append or sync"
        )
    # Validate identity *before* writing an event, so a missing Git author
    # does not leave an avoidable uncommitted append.
    _git("var", "GIT_AUTHOR_IDENT")
    _git("var", "GIT_COMMITTER_IDENT")
    return previous.stdout


def _commit_ledger(ledger_bytes, reason):
    previous = _check_git_ledger(ledger_bytes)
    if ledger_bytes == previous:
        return {"git_status": "already_committed", "commit": _git("rev-parse", "HEAD")}
    # --only commits exactly the current ledger from the worktree,
    # without introducing any other staged or unstaged files.
    _git("commit", "--only", "-m", reason, "--", "instructions.jsonl")
    head = _git("rev-parse", "HEAD")
    actual = subprocess.run(
        ["git", "-C", str(ROOT), "show", "HEAD:instructions.jsonl"],
        capture_output=True, check=True,
    ).stdout
    if actual != ledger_bytes:
        raise RuntimeError("Git commit succeeded but ledger bytes do not match the commit")
    return {"git_status": "committed", "commit": head}


def sync():
    """Commit already-appended events after interrupted Git writes; never rewrite."""
    if not LOG.is_file():
        raise ValueError("instructions.jsonl missing")
    with LOG.open("a+", encoding="utf-8") as ledger:
        fcntl.flock(ledger, fcntl.LOCK_EX)
        ledger.seek(0)
        raw = ledger.read()
        tasks, n, head = audit(raw)
        result = _commit_ledger(raw.encode("utf-8"), f"DAG sync append-only events through {n}")
    return {"events": n, "nodes": len(tasks), "head_sha256": head, **result}

def append(ev):
    # The same exclusive flock spans hash validation, append, fsync and
    # Git commit. Another dag.py writer cannot append between those steps.
    LOG.touch(exist_ok=True)
    with LOG.open("a+", encoding="utf-8") as ledger:
        fcntl.flock(ledger, fcntl.LOCK_EX)
        ledger.seek(0)
        raw = ledger.read()
        # A prior interrupted commit must be repaired using explicit 'sync'.
        committed = _check_git_ledger(raw.encode("utf-8"))
        if raw.encode("utf-8") != committed:
            raise RuntimeError("uncommitted ledger events already exist; run 'python dag.py sync' first")
        tasks, n, prev = audit(raw)
        if isinstance(ev, dict) and ev.get("kind") == "_pull":
            choice = choose_pull_event(tasks, ev.get("agent"))
            if choice.get("kind") is None:
                return choice
            ev = choice
        if isinstance(ev, dict) and ev.get("kind") == "claim" and "at" not in ev:
            ev = {**ev, "at": _now()}
        if isinstance(ev, dict) and ev.get("kind") == "review_request":
            source = review_candidate_bytes(ev["candidate_path"])
            if hashlib.sha256(source).hexdigest() != ev.get("candidate_sha256"):
                raise ValueError("candidate artifact changed before review request")
        if isinstance(ev, dict) and ev.get("kind") == "review_result" and ev.get("verdict") == "PASS":
            if not isinstance(ev.get("evidence"), str):
                raise ValueError("PASS review requires declared receipt path")
            raw_receipt = review_candidate_bytes(ev["evidence"])
            if hashlib.sha256(raw_receipt).hexdigest() != ev.get("evidence_sha256"):
                raise ValueError("verification receipt changed before closure")
        validate_event(ev, tasks)
        body = {"seq": n + 1, "event": ev}
        row = {**body, "prev_sha256": prev, "sha256": hash_event(prev, body)}
        ledger.seek(0, os.SEEK_END)
        ledger.write(canon(row) + "\n")
        ledger.flush()
        os.fsync(ledger.fileno())
        try:
            result = _commit_ledger((raw + canon(row) + "\n").encode("utf-8"),
                                    f"DAG event {n + 1}: {ev.get('kind', 'unknown')} {str(ev.get('id', ''))[:48]}")
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"event {n + 1} was APPENDED and fsynced, but Git commit FAILED; "
                f"do not append again or rewrite history. Recover using "
                f"'python dag.py sync'. Reason: {exc}"
            ) from exc
        row["git_commit"] = result["commit"]
    return row

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in ("validate", "ready", "list", "sync"):
        sub.add_parser(cmd)
    add = sub.add_parser("add-task")
    add.add_argument("--file", required=True, type=Path)
    priority_plan = sub.add_parser("priority-plan",
                                   help="append full ranked priority snapshot without rewriting tasks")
    priority_plan.add_argument("--file", required=True, type=Path)

    pull_cmd = sub.add_parser("pull", help="atomically pull verifier-first or implementer ticket")
    pull_cmd.add_argument("--agent", default=os.environ.get("AGENT_ID"), help="actual assigned runtime identity, or AGENT_ID env")

    review_request_cmd = sub.add_parser("request-review", help="offer an implementation candidate to independent reviewers")
    review_request_cmd.add_argument("id")
    review_request_cmd.add_argument("--owner", required=True)
    review_request_cmd.add_argument("--candidate-path", required=True)
    review_request_cmd.add_argument("--revision")

    review_result_cmd = sub.add_parser("review-result", help="finish an exclusively pulled review")
    review_result_cmd.add_argument("id")
    review_result_cmd.add_argument("--reviewer", required=True)
    review_result_cmd.add_argument("--verdict", required=True, choices=("PASS", "FAIL", "HOLD"))
    review_result_cmd.add_argument("--evidence")
    review_result_cmd.add_argument("--note", default="")

    release_review_cmd = sub.add_parser("release-review", help="recuse from a claimed review without a verdict")
    release_review_cmd.add_argument("id")
    release_review_cmd.add_argument("--reviewer", required=True)
    release_review_cmd.add_argument("--note", required=True)

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
    if args.cmd == "pull":
        print(json.dumps(pull(_review_id(args.agent)), indent=2))
        return
    if args.cmd == "request-review":
        candidate = _review_path(args.id, args.candidate_path)
        digest = hashlib.sha256(review_candidate_bytes(candidate)).hexdigest()
        row = append({"kind": "review_request", "id": args.id, "owner": args.owner,
                      "candidate_path": candidate, "candidate_sha256": digest,
                      "revision": args.revision or _git("rev-parse", "HEAD")})
        print(json.dumps({"role": "REVIEW_REQUESTED", "task": args.id,
                          "candidate_sha256": digest, "git_commit": row["git_commit"]}, indent=2))
        return
    if args.cmd == "review-result":
        tasks, _, _ = audit(read_ledger_bytes().decode("utf-8") if "read_ledger_bytes" in globals() else LOG.read_text())
        review = tasks.get(args.id, {}).get("review")
        if not review:
            raise ValueError("no review request for task")
        event = {"kind": "review_result", "id": args.id, "reviewer": args.reviewer,
                 "candidate_sha256": review["candidate_sha256"],
                 "verdict": args.verdict, "note": args.note}
        if args.verdict == "PASS":
            if not args.evidence:
                raise ValueError("PASS requires declared receipt path")
            event["evidence"] = args.evidence
            event["evidence_sha256"] = hashlib.sha256(review_candidate_bytes(args.evidence)).hexdigest()
        row = append(event)
        print(json.dumps({"role": "VERIFIER", "task": args.id,
                          "verdict": args.verdict, "git_commit": row["git_commit"]}, indent=2))
        return
    if args.cmd == "release-review":
        row = append({"kind": "review_release", "id": args.id, "reviewer": args.reviewer, "note": args.note})
        print(json.dumps({"role": "RECUSED", "task": args.id, "git_commit": row["git_commit"]}, indent=2))
        return
    if args.cmd == "sync":
        print(json.dumps(sync(), indent=2))
        return
    if args.cmd == "add-task":
        event = json.loads(args.file.read_text())
        if event.get("kind") != "task":
            raise ValueError("add-task expects kind=task")
        row = append(event)
    elif args.cmd == "priority-plan":
        event = json.loads(args.file.read_text())
        if event.get("kind") != "priorities":
            raise ValueError("priority-plan expects kind=priorities")
        row = append(event)
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
        for task_id, item in sorted(tasks.items(),
                                    key=lambda pair: (pair[1]["rank"] is None,
                                                      pair[1]["rank"] or 999999,
                                                      pair[0])):
            ready = item["status"] == "planned" and item["owner"] is None and dependencies_passed(item, tasks)
            if args.cmd == "ready" and not ready:
                continue
            result.append({"id": task_id, "priority": item["priority"],
                           "rank": item["rank"], "title": item["record"]["title"], "status": item["status"],
                           "owner": item["owner"], "ready": ready,
                           "depends_on": item["record"]["depends_on"],
                           "review_state": item["review"]["state"] if item["review"] else None,
                           "reviewer": item["review"]["reviewer"] if item["review"] else None,
                           "review_verdict": item["review"]["verdict"] if item["review"] else None})
        print(json.dumps(result, indent=2))
        return
    print(json.dumps({"appended_seq": row["seq"], "event": row["event"],  "head_sha256": row["sha256"], "git_commit": row["git_commit"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError, json.JSONDecodeError) as err:
        print("FAIL:", err, file=sys.stderr)
        sys.exit(2)
