"""NC-023: bounded typed symbolic equality-class exploration.

Dependencies: Python stdlib, ../NC-020/benchmark.py, ../NC-001/canonical_isa.py.
Only exact integer polynomial and fixed-width bitvector expression domains.
No neural input, foreign solver, network or machine-state effect.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "artifacts" / "NC-020"))
import benchmark as ref

VERSION = "logos.nc023.symbolic/1"
RULES = (
    ("int_add_zero", "Z[x,y,z]", "a+0=a", "exact integer"),
    ("int_sub_zero", "Z[x,y,z]", "a-0=a", "exact integer"),
    ("int_mul_zero_one", "Z[x,y,z]", "a*0=0, a*1=a", "exact integer"),
    ("int_const_fold", "Z[x,y,z]", "constant +,-,*", "exact integer"),
    ("int_add_sub_cancel", "Z[x,y,z]", "(a+b)-b=a; (a-b)+b=a", "exact integer"),
    ("int_factor", "Z[x,y,z]", "(a*b)+(a*c)=a*(b+c)", "exact integer"),
    ("int_distribute", "Z[x,y,z]", "a*(b+c)=a*b+a*c", "exact integer"),
    ("int_associate", "Z[x,y,z]", "a+(b+c)=(a+b)+c", "exact integer"),
    ("int_commute", "Z[x,y,z]", "a+b=b+a, a*b=b*a", "exact integer"),
    ("bv_absorb", "fixed width output bits", "a|(a&b)=a", "all X,Y bitpatterns"),
    ("bv_xor_cancel", "fixed width output bits", "(a xor b) xor b=a", "all X,Y bitpatterns"),
    ("bv_add_sub_cancel", "fixed width output bits", "(a+b)-b=a", "all X,Y bitpatterns"),
    ("bv_double_shift", "fixed width output bits", "a+a=a<<1", "all X,Y bitpatterns"),
    ("bv_idempotence", "fixed width output bits", "a&a=a, a|a=a, a xor a=0", "all X,Y bitpatterns"),
)

def canonical(a):
    return json.dumps(a,sort_keys=True,separators=(",",":"),allow_nan=False)

def digest(a):
    return hashlib.sha256(canonical(a).encode("utf-8")).hexdigest()

def kind(a):
    return a["kind"]

def args(a):
    return a.get("args", ())

def op(label,a,b):
    return {"kind":label,"args":[a,b]}

def intconst(v):
    return {"kind":"int.const","value":v}

def eq(a,b):
    return canonical(a)==canonical(b)

def leafconst(a,v):
    return kind(a)=="int.const" and a["value"]==v

def num_nodes(a):
    return 1+sum(num_nodes(x) for x in args(a))

def cost(a):
    """Static symbolic estimate; unique AST shapes model theoretical DAG CSE."""
    seen=set(); ops=set()
    def walk(t):
        k=digest(t)
        if k in seen:return
        seen.add(k)
        if args(t):ops.add(k)
        for c in args(t):walk(c)
    walk(a)
    return {"dag_ops":len(ops),"tree_ops":sum(1 for _ in iter_ops(a)),
            "dag_nodes":len(seen),"tree_nodes":num_nodes(a)}

def iter_ops(a):
    if args(a):
        yield a
        for x in args(a):yield from iter_ops(x)

def score(a):
    c=cost(a)
    return (c["dag_ops"],c["tree_ops"],c["dag_nodes"],num_nodes(a),len(canonical(a)))

def root_rewrites(a):
    """Enumerated mechanical identities. Every emitted whole variant is checked."""
    k=kind(a)
    xs=args(a)
    if k.startswith("int.") and len(xs)==2:
        x,y=xs
        if k=="int.add":
            if leafconst(x,0):yield "int_add_zero",y
            if leafconst(y,0):yield "int_add_zero",x
            for t,u in ((x,y),(y,x)):
                if kind(t)=="int.sub" and eq(t["args"][1],u):
                    yield "int_add_sub_cancel",t["args"][0]
            for t,u in ((x,y),(y,x)):
                if kind(t)=="int.mul" and kind(u)=="int.mul":
                    for p in (0,1):
                        for q in (0,1):
                            if eq(t["args"][p],u["args"][q]):
                                l=t["args"][1-p];r=u["args"][1-q]
                                yield "int_factor",op("int.mul",t["args"][p],op("int.add",l,r))
            if kind(y)=="int.add":
                yield "int_associate",op("int.add",op("int.add",x,y["args"][0]),y["args"][1])
            if kind(x)=="int.add":
                yield "int_associate",op("int.add",x["args"][0],op("int.add",x["args"][1],y))
            yield "int_commute",op(k,y,x)
        elif k=="int.sub":
            if leafconst(y,0):yield "int_sub_zero",x
            if kind(x)=="int.add":
                for t,u in ((x["args"][0],x["args"][1]),(x["args"][1],x["args"][0])):
                    if eq(u,y):yield "int_add_sub_cancel",t
        elif k=="int.mul":
            if leafconst(x,0) or leafconst(y,0):
                yield "int_mul_zero_one",intconst(0)
            if leafconst(x,1):yield "int_mul_zero_one",y
            if leafconst(y,1):yield "int_mul_zero_one",x
            for t,u in ((x,y),(y,x)):
                if kind(u)=="int.add":
                    yield "int_distribute",op("int.add",op("int.mul",t,u["args"][0]),op("int.mul",t,u["args"][1]))
            yield "int_commute",op(k,y,x)
        if kind(x)==kind(y)=="int.const":
            if k=="int.add":v=x["value"]+y["value"]
            elif k=="int.sub":v=x["value"]-y["value"]
            else:v=x["value"]*y["value"]
            yield "int_const_fold",intconst(v)
    elif k.startswith("bv.") and xs:
        w=a["width"];sign=a["signedness"]
        def b(label,l,r=None):
            return {"kind":"bv."+label,"width":w,"signedness":sign,"args":[l] if r is None else [l,r]}
        if len(xs)==2:
            x,y=xs
            if k in ("bv.and","bv.or") and eq(x,y):
                yield "bv_idempotence",x
            if k=="bv.xor" and eq(x,y):
                yield "bv_idempotence",ref.isa.lit(0,w,sign)
            if k=="bv.add" and eq(x,y):
                yield "bv_double_shift",b("shl",x)
            if k=="bv.or":
                for p,q in ((x,y),(y,x)):
                    if kind(q)=="bv.and" and any(eq(p,z) for z in q["args"]):
                        yield "bv_absorb",p
            if k=="bv.xor":
                for p,q in ((x,y),(y,x)):
                    if kind(p)=="bv.xor" and any(eq(q,z) for z in p["args"]):
                        z=p["args"][0] if eq(q,p["args"][1]) else p["args"][1]
                        yield "bv_xor_cancel",z
            if k=="bv.sub" and kind(x)=="bv.add":
                for p,q in ((x["args"][0],x["args"][1]),(x["args"][1],x["args"][0])):
                    if eq(q,y):yield "bv_add_sub_cancel",p

def candidates(ast, node_cap=76):
    for rule,n in root_rewrites(ast):
        if num_nodes(n)<=node_cap and not eq(n,ast):
            yield rule,n
    for i,ch in enumerate(args(ast)):
        for rule,new in root_rewrites(ch):
            out=copy.deepcopy(ast)
            out["args"][i]=new
            if num_nodes(out)<=node_cap and not eq(out,ast):
                yield rule,out
        # One extra level ensures nested cancellation and factoring opportunities.
        for j,grand in enumerate(args(ch)):
            for rule,new in root_rewrites(grand):
                out=copy.deepcopy(ast)
                out["args"][i]["args"][j]=new
                if num_nodes(out)<=node_cap and not eq(out,ast):
                    yield rule,out

def validate_domain(ast, case):
    if case["domain"]=="integer":
        ref.normalize_int(ast) # strict recursive typed validator
    elif case["domain"]=="bitvector":
        ref.isa.validate_ast(ast,case["width"],case["signedness"])
        def names(t):
            if kind(t)=="bv.read":return {t["name"]}
            z=set()
            for child in args(t):z|=names(child)
            return z
        if not names(ast)<={"X","Y"}:raise ValueError("uncertified live registers")
    else:raise ValueError("unsupported domain")

def equivalence(lhs,rhs,case):
    validate_domain(lhs,case)
    validate_domain(rhs,case)
    if case["domain"]=="integer":
        a,b=ref.normalize_int(lhs),ref.normalize_int(rhs)
        same=a==b
        cert={"method":"integer_polynomial_normal_form","scope":"all Z[x,y,z] exact assignments",
              "lhs_coefficients_sha256":digest(a),"rhs_coefficients_sha256":digest(b)}
        if not same:
            cert["counterexample"]=ref.integer_witness(lhs,rhs)
        return same,cert
    witness,count=ref.brute_bv(lhs,rhs,case["width"],case["signedness"])
    return witness is None,{"method":"finite_exhaustive","scope":"output bits for X,Y at declared width/signedness; R,T prohibited",
        "checked_assignments":count,"counterexample":witness}

def search(case,cap=32, mode="saturation"):
    source=case["lhs"]
    validate_domain(source,case)
    q=deque([source]); seen={digest(source):{"ast":source,"parents":[]}}
    source_hash=digest(source)
    best=source; best_hash=source_hash
    while q and len(seen)<cap:
        parent=q.popleft();ph=digest(parent)
        generated=sorted(candidates(parent),key=lambda x:(score(x[1]),x[0],digest(x[1])))
        for rule,child in generated:
            ch=digest(child)
            if ch in seen:continue
            try:ok,proof=equivalence(source,child,case)
            except (ValueError,TypeError,KeyError,ref.isa.ISAError):
                continue
            if not ok:continue
            seen[ch]={"ast":child,"parents":[ph,rule],"proof":proof}
            if score(child)<score(best):
                best,best_hash=child,ch
            if mode=="greedy":
                if score(child)<score(parent):
                    q=deque([child])
                    break
                continue
            q.append(child)
            if len(seen)>=cap:break
        if mode=="greedy" and not q:break
    steps=[]
    at=best_hash
    while at!=source_hash:
        entry=seen[at]
        parent,rule=entry["parents"]
        steps.append({"rule":rule,"parent_sha256":parent,"candidate_sha256":at,
                      "proof":entry["proof"]})
        at=parent
    steps.reverse()
    final_proof=equivalence(source,best,case)[1]
    return {"optimized":best,"optimized_sha256":best_hash,"proof":final_proof,
            "steps":steps,"explored":len(seen),"operations":cost(best)}

def negative_control(case):
    try:
        same,certificate=equivalence(case["lhs"],case["rhs"],case)
    except Exception as exc:
        return {"accepted":False,"reason":"invalid/unsupported","error_type":type(exc).__name__}
    return {"accepted":same,"proof":certificate}
