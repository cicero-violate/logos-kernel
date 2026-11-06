#!/usr/bin/env python3
"""Append-only hash-chained JSONL DAG ledger. Python stdlib only."""
import argparse, fcntl, hashlib, json, os, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
LOG=ROOT/"instructions.jsonl"
ZERO="0"*64
TRANS={"planned":{"working","blocked","archived"},
       "working":{"passed","blocked","failed"},
       "blocked":{"working","archived"},
       "failed":{"working","archived"},
       "passed":set(),"archived":set()}

def canon(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False)
def hash_event(prev,body):
    return hashlib.sha256((prev+"\n"+canon(body)).encode()).hexdigest()

def validate_event(ev,tasks):
    if ev["kind"]=="task":
        id=ev["id"]
        if id in tasks or not id:raise ValueError("duplicate/empty id "+repr(id))
        deps=ev["depends_on"]
        if len(deps)!=len(set(deps)) or any(d not in tasks for d in deps):
            raise ValueError(f"{id}: depends_on must refer uniquely to earlier tasks")
        for field in ("instruction","acceptance","outputs","avoid"):
            if not ev.get(field):raise ValueError(f"{id}: missing {field}")
        tasks[id]={"record":ev,"status":"planned","owner":None}
    elif ev["kind"]=="status":
        id=ev["id"];new=ev["status"]
        if id not in tasks or new not in TRANS:raise ValueError("unknown task/status")
        old=tasks[id]["status"]
        if new not in TRANS[old]:raise ValueError(f"invalid transition {id} {old}->{new}")
        if new in ("working","passed") and any(tasks[d]["status"]!="passed" for d in tasks[id]["record"]["depends_on"]):
            raise ValueError("dependencies not passed")
        if new=="passed":
            if not ev.get("verifier") or not ev.get("evidence"):raise ValueError("pass requires verifier and evidence")
            p=Path(ev["evidence"])
            if p.is_absolute() or ".." in p.parts or not (ROOT/p).is_file():
                raise ValueError("pass requires existing relative evidence file")
        if new=="working" and not ev.get("agent"):raise ValueError("start requires agent")
        tasks[id]["status"]=new
        tasks[id]["owner"]=ev.get("agent")
    else:raise ValueError("unknown event kind")

def audit(raw):
    tasks={};prev=ZERO
    lines=raw.splitlines()
    for i,line in enumerate(lines,1):
        row=json.loads(line)
        if row["seq"]!=i or row["prev_sha256"]!=prev:raise ValueError(f"broken sequence/prev at {i}")
        body={"seq":i,"event":row["event"]}
        if row["sha256"]!=hash_event(prev,body):raise ValueError(f"hash changed at {i}")
        validate_event(row["event"],tasks);prev=row["sha256"]
    return tasks,len(lines),prev

def append(ev):
    LOG.touch(exist_ok=True)
    with LOG.open("a+",encoding="utf-8") as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
        tasks,n,prev=audit(f.read())
        validate_event(ev,tasks)
        body={"seq":n+1,"event":ev}
        row={**body,"prev_sha256":prev,"sha256":hash_event(prev,body)}
        f.seek(0,os.SEEK_END);f.write(canon(row)+"\n")
        f.flush();os.fsync(f.fileno());fcntl.flock(f,fcntl.LOCK_UN)
    return row["sha256"]

def main():
    p=argparse.ArgumentParser()
    s=p.add_subparsers(dest="cmd",required=True)
    for cmd in ("validate","ready","list"):s.add_parser(cmd)
    a=s.add_parser("add-task");a.add_argument("--file",required=True)
    for cmd in ("start","pass","fail","block","archive"):
        q=s.add_parser(cmd);q.add_argument("id");q.add_argument("--agent",required=True)
        q.add_argument("--evidence");q.add_argument("--verifier");q.add_argument("--note",default="")
    args=p.parse_args()
    if args.cmd=="add-task":
        obj=json.loads(Path(args.file).read_text())
        print("APPENDED",append(obj));return
    if args.cmd in ("start","pass","fail","block","archive"):
        st={"start":"working","pass":"passed","fail":"failed","block":"blocked","archive":"archived"}[args.cmd]
        e={"kind":"status","id":args.id,"status":st,"agent":args.agent,"note":args.note}
        if args.evidence:e["evidence"]=args.evidence
        if args.verifier:e["verifier"]=args.verifier
        print("APPENDED",append(e));return
    tasks,n,head=audit(LOG.read_text() if LOG.exists() else "")
    if args.cmd=="validate":print(json.dumps({"status":"PASS","nodes":len(tasks),"events":n,"head_sha256":head},indent=2));return
    result=[]
    for id,r in tasks.items():
        deps=r["record"]["depends_on"]
        ready=r["status"]=="planned" and all(tasks[d]["status"]=="passed" for d in deps)
        if args.cmd=="ready" and not ready:continue
        result.append({"id":id,"priority":r["record"]["priority"],"title":r["record"]["title"],
                       "status":r["status"],"ready":ready,"depends_on":deps})
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    try:main()
    except (ValueError,KeyError,OSError) as ex:
        print("FAIL",ex,file=sys.stderr);sys.exit(2)
