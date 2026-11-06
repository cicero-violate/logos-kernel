"""NC-020 typed AST equivalence benchmark. Dependencies: Python 3 stdlib,
NC-001/canonical_isa.py, NC-019/semantic_objective.json.
No neural model, PyTorch, checkpoint or third-party solver is used.
"""
import collections
import hashlib
import itertools
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ISA_PATH = ROOT / "artifacts" / "NC-001"
sys.path.insert(0, str(ISA_PATH))
import canonical_isa as isa

SEED = 20261105
VERSION = "logos.semantic-equivalence/1"
VARS = ("x", "y", "z")
FAMILIES = {
    "int_sum_commute": ("train", "integer"),
    "int_add_associate": ("train", "integer"),
    "int_distribute": ("train", "integer"),
    "int_add_zero": ("train", "integer"),
    "bv_xor_commute": ("train", "bitvector"),
    "bv_add_sub_cancel": ("train", "bitvector"),
    "int_square_expand": ("validation", "integer"),
    "int_substitute_cancel": ("validation", "integer"),
    "bv_xor_cancel": ("validation", "bitvector"),
    "bv_double_shift": ("validation", "bitvector"),
    "int_nested_factor": ("test", "integer"),
    "int_difference_squares": ("test", "integer"),
    "bv_absorption": ("test", "bitvector"),
    "bv_and_distribute_xor": ("test", "bitvector"),
    "bv_substitution": ("test", "bitvector"),
}
EXCEPTIONS = ("fp64_reassociation", "fp64_rounding", "invalid_width", "invalid_sign")

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

def sha(value):
    return hashlib.sha256((value if isinstance(value, bytes) else canonical(value).encode())).hexdigest()

def var(name):
    return {"kind": "int.var", "name": name}

def const(value):
    return {"kind": "int.const", "value": value}

def op(name, a, b):
    return {"kind": "int." + name, "args": [a, b]}

def add(a,b): return op("add",a,b)
def sub(a,b): return op("sub",a,b)
def mul(a,b): return op("mul",a,b)

def bvr(name, w, sign):
    return isa.read(name,w,sign)

def bvo(name, a, b=None, *, width, signedness):
    args=[a] if b is None else [a,b]
    return {"kind":"bv."+name,"width":width,"signedness":signedness,"args":args}

def integer_pair(f,k):
    x,y,z=map(var,VARS)
    one=const(1)
    if f=="int_sum_commute":
        return add(x,add(y,const(k))),add(add(y,x),const(k))
    if f=="int_add_associate":
        return add(add(x,y),add(z,const(k))),add(x,add(y,add(z,const(k))))
    if f=="int_distribute":
        return mul(x,add(y,const(k))),add(mul(x,y),mul(x,const(k)))
    if f=="int_add_zero":
        return add(mul(x,const(k)),const(0)),mul(x,const(k))
    if f=="int_square_expand":
        lhs=mul(add(x,y),add(x,y))
        rhs=add(add(mul(x,x),mul(const(2),mul(x,y))),mul(y,y))
        return lhs,rhs
    if f=="int_substitute_cancel":
        return add(sub(add(x,y),z),z),add(x,y)
    if f=="int_nested_factor":
        return add(mul(x,add(y,z)),mul(x,mul(const(k),z))),mul(x,add(y,mul(const(k+1),z)))
    if f=="int_difference_squares":
        return mul(add(x,mul(const(k),y)),sub(x,mul(const(k),y))),sub(mul(x,x),mul(mul(const(k),const(k)),mul(y,y)))
    raise ValueError(f)

def bitvector_pair(f,w,sign):
    x,y=bvr("X",w,sign),bvr("Y",w,sign)
    def b(name,a,b=None): return bvo(name,a,b,width=w,signedness=sign)
    if f=="bv_xor_commute": return b("xor",x,y),b("xor",y,x)
    if f=="bv_add_sub_cancel": return b("sub",b("add",x,y),y),x
    if f=="bv_xor_cancel": return b("xor",b("xor",x,y),y),x
    if f=="bv_double_shift": return b("add",x,x),b("shl",x)
    if f=="bv_absorption": return b("or",x,b("and",x,y)),x
    if f=="bv_and_distribute_xor":
        return b("and",x,b("xor",x,y)),b("xor",x,b("and",x,y))
    if f=="bv_substitution":
        return b("sub",x,b("sub",y,x)),b("sub",b("add",x,x),y)
    raise ValueError(f)

def normalize_int(ast):
    """Exact polynomial ring Z[x,y,z]; normal-form equality is a global proof."""
    zero=(0,0,0)
    def combine(a,b,s=1):
        out=dict(a)
        for k,v in b.items():out[k]=out.get(k,0)+s*v
        return {k:v for k,v in out.items() if v}
    def rec(n,depth=0):
        if depth>=32 or not isinstance(n,dict):raise ValueError("invalid int AST")
        kind=n.get("kind")
        if kind=="int.const" and set(n)=={"kind","value"} and type(n["value"]) is int:
            return {zero:n["value"]} if n["value"] else {}
        if kind=="int.var" and set(n)=={"kind","name"} and n["name"] in VARS:
            powers=tuple(int(n["name"]==v) for v in VARS)
            return {powers:1}
        if kind not in ("int.add","int.sub","int.mul") or set(n)!={"kind","args"} or not isinstance(n["args"],list) or len(n["args"])!=2:
            raise ValueError("invalid integer typed AST")
        a,b=(rec(t,depth+1) for t in n["args"])
        if kind=="int.add":return combine(a,b)
        if kind=="int.sub":return combine(a,b,-1)
        out=collections.defaultdict(int)
        for ma,ca in a.items():
            for mb,cb in b.items():
                out[tuple(x+y for x,y in zip(ma,mb))]+=ca*cb
        return {k:v for k,v in out.items() if v}
    p=rec(ast)
    return [[list(m),c] for m,c in sorted(p.items())]

def eval_int(n,env):
    k=n["kind"]
    if k=="int.var":return env[n["name"]]
    if k=="int.const":return n["value"]
    a,b=[eval_int(z,env) for z in n["args"]]
    if k=="int.add":return a+b
    if k=="int.sub":return a-b
    if k=="int.mul":return a*b
    raise ValueError("invalid AST")

def depth(ast):
    return 1+max((depth(a) for a in ast.get("args", [])),default=0)

def nodes(ast):
    return 1+sum(nodes(a) for a in ast.get("args",[]))

def shape(ast):
    """Values erased; operand positions retained; structure is audited across splits."""
    k=ast["kind"]
    if k in ("int.const","bv.const"):return [k,"constant"]
    if k in ("int.var","bv.read"):return [k,"variable"]
    return [k]+[shape(a) for a in ast.get("args",[])]

def alter(ast):
    """One leaf/operator mutation. Reject accidental equivalent mutations."""
    ast=json.loads(canonical(ast))
    def walk(n):
        if n["kind"]=="int.const":
            n["value"]+=1
            return True
        if n["kind"]=="int.var":
            n["name"]={"x":"y","y":"z","z":"x"}[n["name"]]
            return True
        if n["kind"]=="bv.read":
            n["name"]="Y" if n["name"]=="X" else "X"
            return True
        if n["kind"]=="bv.const":
            n["value"]=(n["value"]+1)%(1<<n["width"])
            return True
        for child in n.get("args",[]):
            if walk(child):return True
        return False
    if not walk(ast):raise ValueError("no mutable leaves")
    return ast

def wrap(ast, domain, n, w=None, sign=None):
    for j in range(n):
        if domain=="integer": ast=add(ast,mul(const(j+1),var("z")))
        else:
            y=bvr("Y",w,sign)
            ast=bvo("xor" if j%2==0 else "add",ast,y,width=w,signedness=sign)
    return ast

def eval_bv(ast, x, y, w, sign):
    return isa.eval_ast(ast, {"X":x,"Y":y,"R":0,"T":0}, w, sign)

def brute_bv(lhs,rhs,w,sign):
    isa.validate_ast(lhs,w,sign)
    isa.validate_ast(rhs,w,sign)
    def regset(n):
        if n["kind"]=="bv.read":return {n["name"]}
        out=set()
        for a in n.get("args",[]):out.update(regset(a))
        return out
    if not (regset(lhs)|regset(rhs))<={"X","Y"}:
        raise ValueError("exhaustive scope must cover every referenced variable")
    witness=None
    for x in range(1<<w):
        for y in range(1<<w):
            a,b=eval_bv(lhs,x,y,w,sign),eval_bv(rhs,x,y,w,sign)
            if a!=b and witness is None: witness={"X":x,"Y":y,"lhs_value":a,"rhs_value":b}
    return witness,(1<<w)**2

def integer_witness(lhs,rhs):
    for x,y,z in itertools.product((-8,-5,-3,-1,0,1,2,4,7),repeat=3):
        env={"x":x,"y":y,"z":z}
        a,b=eval_int(lhs,env),eval_int(rhs,env)
        if a!=b:return {**env,"lhs_value":a,"rhs_value":b}
    raise ValueError("no integer counterexample found")

def make_case(f,split,domain,n,label,lhs,rhs,*,w=None,sign=None):
    if domain=="integer":
        a,b=normalize_int(lhs),normalize_int(rhs)
        same=a==b
        proof={"method":"polynomial_normal_form","left_sha256":sha(a),"right_sha256":sha(b),
               "scope":"all assignments of x,y,z over exact unbounded integers"}
        if not same:proof["counterexample"]=integer_witness(lhs,rhs)
    elif domain=="bitvector":
        witness,checked=brute_bv(lhs,rhs,w,sign)
        same=witness is None
        proof={"method":"finite_exhaustive","checked_assignments":checked,
               "scope":"all X,Y raw bitpatterns; R,T unreferenced; output bits only",
               "counterexample":witness}
    else:raise ValueError(domain)
    if same!=(label=="equivalent"):raise AssertionError(f"{f}: {label} contradicted by oracle")
    case={"version":VERSION,"family":f,"split":split,"domain":domain,"width":w,"signedness":sign,
          "variant":n,"label":label,"lhs":lhs,"rhs":rhs,
          "difficulty":{"depth":max(depth(lhs),depth(rhs)),
                        "nodes":nodes(lhs)+nodes(rhs),"width_bits":w,
                        "wrapping_layers":n%3},
          "structural_fingerprint":sha([shape(lhs),shape(rhs)]),
          "proof":proof}
    case["id"]=sha({k:v for k,v in case.items() if k!="proof"})[:24]
    case["record_sha256"]=sha({k:v for k,v in case.items() if k!="record_sha256"})
    return case

def extra_cases():
    def fpconst(v):return {"kind":"fp64.const","value":v}
    def fpadd(a,b):return {"kind":"fp64.add","args":[a,b]}
    def feval(n):
        return n["value"] if n["kind"]=="fp64.const" else feval(n["args"][0])+feval(n["args"][1])
    a,b,c=map(fpconst,(1e16,-1e16,1.0))
    candidates=[
        ("fp64_reassociation",fpadd(fpadd(a,b),c),fpadd(a,fpadd(b,c))),
        ("fp64_rounding",fpadd(fpconst(0.1),fpconst(0.2)),fpconst(0.3)),
    ]
    out=[]
    for f,lhs,rhs in candidates:
        v1,v2=feval(lhs),feval(rhs)
        assert v1!=v2
        rec={"version":VERSION,"family":f,"split":"test","domain":"IEEE754_binary64_python_float",
             "width":64,"signedness":"finite_examples","variant":0,"label":"non_equivalent",
             "lhs":lhs,"rhs":rhs,"difficulty":{"depth":max(depth(lhs),depth(rhs)),
             "nodes":nodes(lhs)+nodes(rhs),"width_bits":64,"wrapping_layers":0},
             "structural_fingerprint":sha([shape(lhs),shape(rhs)]),
             "proof":{"method":"concrete_floating_counterexample","lhs_value":v1,"rhs_value":v2,
             "scope":"disproves a universal float64 identity; no complete fp64 proof"}}
        rec["id"]=sha({k:v for k,v in rec.items() if k!="proof"})[:24]
        rec["record_sha256"]=sha({k:v for k,v in rec.items() if k!="record_sha256"})
        out.append(rec)
    lhs=bvr("X",4,"unsigned")
    for f,rhs in (("invalid_width",bvr("X",5,"unsigned")),
                  ("invalid_sign",bvr("X",4,"twos_complement"))):
        try:isa.validate_ast(rhs,4,"unsigned")
        except isa.ISAError as exc:error=str(exc)
        else:raise AssertionError("invalid typed pair accepted")
        rec={"version":VERSION,"family":f,"split":"test","domain":"bitvector",
             "width":4,"signedness":"unsigned","variant":0,"label":"invalid_type",
             "lhs":lhs,"rhs":rhs,"difficulty":{"depth":1,"nodes":2,"width_bits":4,"wrapping_layers":0},
             "structural_fingerprint":sha([shape(lhs),shape(rhs)]),
             "proof":{"method":"strict_type_rejection","error":error,
             "scope":"excluded from same-type semantic equivalence scoring"}}
        rec["id"]=sha({k:v for k,v in rec.items() if k!="proof"})[:24]
        rec["record_sha256"]=sha({k:v for k,v in rec.items() if k!="record_sha256"})
        out.append(rec)
    return out

def produce():
    rng=random.Random(SEED)
    cases=[]
    for family,(split,domain) in FAMILIES.items():
        for n in range(9):
            w=(3,4,5)[n%3] if domain=="bitvector" else None
            sign=("unsigned","twos_complement")[(n//3)%2] if domain=="bitvector" else None
            k=rng.choice((1,2,3,4))
            lhs,rhs=(integer_pair(family,k) if domain=="integer" else bitvector_pair(family,w,sign))
            lhs=wrap(lhs,domain,n%3,w,sign)
            rhs=wrap(rhs,domain,n%3,w,sign)
            cases.append(make_case(family,split,domain,n,"equivalent",lhs,rhs,w=w,sign=sign))
            negative=alter(rhs)
            if domain=="integer" and normalize_int(lhs)==normalize_int(negative):
                negative=add(rhs,const(1))
            if domain=="bitvector" and brute_bv(lhs,negative,w,sign)[0] is None:
                negative=bvo("add",rhs,isa.lit(1,w,sign),width=w,signedness=sign)
            cases.append(make_case(family,split,domain,n,"non_equivalent",lhs,negative,w=w,sign=sign))
    cases.extend(extra_cases())
    ids=[c["id"] for c in cases]
    if len(ids)!=len(set(ids)):raise AssertionError("duplicate case IDs")
    families=collections.defaultdict(set)
    fingerprints=collections.defaultdict(set)
    for c in cases:
        families[c["family"]].add(c["split"])
        fingerprints[c["structural_fingerprint"]].add(c["split"])
    if any(len(v)>1 for v in families.values()):raise AssertionError("family split leakage")
    if any(len(v)>1 for v in fingerprints.values()):raise AssertionError("structural fingerprint split leakage")
    return cases

def metadata(cases):
    files={"generator":HERE/"benchmark.py",
           "isa":ROOT/"artifacts/NC-001/canonical_isa.py",
           "semantic_objective":ROOT/"artifacts/NC-019/semantic_objective.json"}
    sources={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in files.items()}
    groups=collections.Counter(c["split"] for c in cases)
    labels=collections.Counter(c["label"] for c in cases)
    families={f:{"split":s,"domain":d} for f,(s,d) in sorted(FAMILIES.items())}
    families.update({f:{"split":"test","domain":"negative_control"} for f in EXCEPTIONS})
    return {
        "version":VERSION,"seed":SEED,"generator":"benchmark.py:produce",
        "canonical_json":"sort_keys=True,compact,separators comma/colon,allow_nan=False",
        "oracle":{"integer":"exact Z[x,y,z] polynomial coefficient normal form",
                  "bitvector":"NC-001 strict AST + exhaustive full X,Y assignment range, widths 3/4/5",
                  "floating":"concrete Python binary64 counterexamples only",
                  "invalid_type":"NC-001 validator rejection; excluded from equivalence balance"},
        "source_sha256":sources,
        "family_registry":families,
        "splits":{"families_disjoint":True,"fingerprints_disjoint":True,
                  "no_model_trained":True,"training_preexposure_absence":"NOT_ESTABLISHED",
                  "case_counts":dict(groups),"label_counts":dict(labels),
                  "case_id_lists":{s:sorted(c["id"] for c in cases if c["split"]==s)
                                   for s in ("train","validation","test")}},
        "dataset_sha256":sha([{k:v for k,v in c.items() if k!="record_sha256"} for c in cases]),
        "limitations":["Polynomials-only proof; does not apply to division, floating rounding or overflow-sensitive ordering.",
                       "Bitvector exhaustive scope covers two used registers and output bits, not complete ISA machine/flags.",
                       "Historical data/pretraining leakage outside this new dataset cannot be excluded.",
                       "No trained network, composed execution or general-purpose reasoning established."]
    }

def audit(cases):
    proofs=collections.Counter(c["proof"]["method"] for c in cases)
    widths=collections.Counter(str(c["width"]) for c in cases)
    depths=collections.Counter(str(c["difficulty"]["depth"]) for c in cases)
    labels_by_split={s:dict(collections.Counter(c["label"] for c in cases if c["split"]==s))
                      for s in ("train","validation","test")}
    return {"version":VERSION,"task":"NC-020","total_cases":len(cases),
            "proof_method_counts":dict(proofs),"width_counts":dict(widths),
            "depth_counts":dict(depths),"split_label_counts":labels_by_split,
            "counterexamples":sum(bool(c["proof"].get("counterexample")) for c in cases),
            "proof_coverage":{"certified_by_design":len(cases),
                              "independent_verifier_status":"NOT_YET_VERIFIED",
                              "exact_integer_proof":"ring normal form",
                              "exact_bv_proof":"enumeration of every X,Y valuation at each declared width",
                              "float":"witness only; not universal equivalence"},
            "negative_controls":["wrong leaf substitution","overflow-sensitive bitvectors",
                                 "rounding and reassociation in IEEE binary64",
                                 "rejected mixed-width/mixed-signedness"],
            "record_hashes":{c["id"]:c["record_sha256"] for c in cases},
            "case_set_sha256":sha(sorted(c["id"] for c in cases)),
            "champion_modified":False,"neural_training":False}

def write():
    cases=produce()
    (HERE/"equivalence_benchmark.jsonl").write_text("".join(canonical(c)+"\n" for c in cases),encoding="utf-8")
    (HERE/"splits_and_semantics.json").write_text(json.dumps(metadata(cases),indent=2,sort_keys=True)+"\n")
    (HERE/"proof_audit.json").write_text(json.dumps(audit(cases),indent=2,sort_keys=True)+"\n")
    print(canonical({"cases":len(cases),"splits":metadata(cases)["splits"]["case_counts"],
                      "proof_methods":audit(cases)["proof_method_counts"],
                      "dataset_sha256":metadata(cases)["dataset_sha256"]}))
if __name__=="__main__":write()
