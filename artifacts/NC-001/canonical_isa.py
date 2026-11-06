"""NC-001 Logos canonical ISA v1: typed AST + deterministic symbolic reference.
Dependencies: Python 3 standard library only. No neural weights, models, or oracle fallback.
"""
import json
VERSION = "logos.isa/1"
REGS = ("X", "Y", "R", "T")
BINARY = ("ADD", "SUB", "AND", "OR", "XOR")
UNARY = ("AND1", "SHL", "SHR", "SAR")
FIELDS = {
 "LOADI":("dst","imm"), "MOV":("dst","src"),
 **{name:("dst","a","b") for name in BINARY},
 **{name:("dst","src") for name in UNARY},
 "JZ":("src","target"), "JNZ":("src","target"),
 "JMP":("target",), "HALT":()
}
class ISAError(ValueError):
 """Invalid input is rejected, never repaired or silently coerced."""

def obj(v, keys, where):
 if type(v) is not dict or set(v) != set(keys):
  raise ISAError(f"{where}: keys must be {sorted(keys)}")
 return v
def integer(v, lo, hi, where):
 if type(v) is not int or not lo <= v <= hi:
  raise ISAError(f"{where}: integer outside [{lo}, {hi}]")
 return v
def reg(v):
 if type(v) is not str or v not in REGS: raise ISAError("unknown register")
 return v

def validate(p):
 obj(p,("version","width","signedness","program"),"program")
 if type(p["version"]) is not str or p["version"] != VERSION: raise ISAError("version")
 w=integer(p["width"],1,64,"width")
 sign=p["signedness"]
 if type(sign) is not str or sign not in ("unsigned","twos_complement"):
  raise ISAError("signedness")
 ins=p["program"]
 if type(ins) is not list or not 1 <= len(ins) <= 4096: raise ISAError("program length")
 normalized=[]
 for i,item in enumerate(ins):
  if type(item) is not dict or type(item.get("op")) is not str or item["op"] not in FIELDS:
   raise ISAError(f"opcode at {i}")
  op=item["op"]
  obj(item,("op",*FIELDS[op]),f"instruction[{i}]")
  for k in ("src","dst","a","b"):
   if k in item: reg(item[k])
  if "imm" in item: integer(item["imm"],0,(1<<w)-1,"immediate")
  if "target" in item: integer(item["target"],0,len(ins)-1,"branch target")
  if op=="SAR" and sign != "twos_complement": raise ISAError("SAR requires signedness")
  normalized.append(dict(item))
 if normalized[-1]["op"] != "HALT": raise ISAError("last instruction must HALT")
 return {"version":VERSION,"width":w,"signedness":sign,"program":normalized}

def _unique(pairs):
 d={}
 for k,v in pairs:
  if k in d: raise ISAError("duplicate JSON key")
  d[k]=v
 return d
def decode(text):
 if type(text) is not str or len(text.encode("utf-8")) > 2_000_000:
  raise ISAError("JSON length")
 try:
  p=json.loads(text, object_pairs_hook=_unique,
               parse_constant=lambda x: (_ for _ in ()).throw(ISAError("nonfinite number")))
 except (ValueError,TypeError) as exc: raise ISAError(str(exc)) from exc
 return validate(p)
def encode(p):
 return json.dumps(validate(p),sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)

def read(name,w,sign):
 return {"kind":"bv.read","name":name,"width":w,"signedness":sign}
def lit(n,w,sign):
 return {"kind":"bv.const","value":n,"width":w,"signedness":sign}
def lower(p,pc):
 """Lower an ISA instruction into typed math/control AST, never executing it."""
 p=validate(p)
 i=integer(pc,0,len(p["program"])-1,"pc")
 x=p["program"][i]; op=x["op"]; w=p["width"]; sign=p["signedness"]
 a={"kind":"transition","opcode":op,"write":None,"flags":"preserve",
    "pc":{"kind":"pc.next"}}
 if op=="LOADI": a["write"]={"dst":x["dst"],"value":lit(x["imm"],w,sign)}
 elif op=="MOV": a["write"]={"dst":x["dst"],"value":read(x["src"],w,sign)}
 elif op in BINARY+UNARY:
  srcs=("a","b") if op in BINARY else ("src",)
  a["write"]={"dst":x["dst"],"value":{"kind":"bv."+op.lower(),"width":w,
             "signedness":sign,"args":[read(x[k],w,sign) for k in srcs]}}
  a["flags"]="alu.result"
 elif op in ("JZ","JNZ"):
  cond={"kind":"bool.zero","value":read(x["src"],w,sign)}
  if op=="JNZ": cond={"kind":"bool.not","value":cond}
  a["pc"]={"kind":"pc.branch","condition":cond,
           "target":{"kind":"pc.literal","value":x["target"]},
           "otherwise":{"kind":"pc.next"}}
 elif op=="JMP": a["pc"]={"kind":"pc.literal","value":x["target"]}
 elif op=="HALT": a["pc"]={"kind":"pc.halt"}
 else: raise ISAError("opcode lowering")
 return a

def validate_ast(node, width, signedness, depth=0):
 """Strictly check a proposed typed expression before trusting/evaluating it."""
 if depth>=64: raise ISAError("AST nesting limit")
 if type(node) is not dict or type(node.get("kind")) is not str:
  raise ISAError("AST node shape")
 kind=node["kind"]
 if kind in ("bool.zero","bool.not"):
  obj(node,("kind","value"),kind)
  child=validate_ast(node["value"],width,signedness,depth+1)
  if (kind=="bool.zero" and child!="bitvec") or (
      kind=="bool.not" and child!="bool"):
   raise ISAError("boolean operand type")
  return "bool"
 if kind in ("bv.const","bv.read"):
  obj(node,("kind","value","width","signedness") if kind=="bv.const"
      else ("kind","name","width","signedness"),kind)
 else:
  if kind not in ("bv.add","bv.sub","bv.and","bv.or","bv.xor",
                 "bv.and1","bv.shl","bv.shr","bv.sar"):
   raise ISAError("unknown AST kind")
  obj(node,("kind","width","signedness","args"),kind)
 if type(node["width"]) is not int or node["width"]!=width or (
     type(node["signedness"]) is not str or node["signedness"]!=signedness):
  raise ISAError("AST shape/type mismatch")
 if kind=="bv.const": integer(node["value"],0,(1<<width)-1,"AST literal")
 elif kind=="bv.read": reg(node["name"])
 else:
  args=node["args"]
  arity=2 if kind in ("bv.add","bv.sub","bv.and","bv.or","bv.xor") else 1
  if type(args) is not list or len(args)!=arity: raise ISAError("AST operand count")
  if kind=="bv.sar" and signedness!="twos_complement":
   raise ISAError("SAR requires signed mode")
  for x in args:
   if validate_ast(x,width,signedness,depth+1)!="bitvec":
    raise ISAError("AST input must be bitvector")
 return "bitvec"

def eval_ast(a,r,w,sign):
 """Symbolic evaluation of only known typed AST nodes: NEVER Python eval."""
 if type(a) is not dict: raise ISAError("AST is not object")
 k=a.get("kind")
 if k=="bool.zero": return eval_ast(a["value"],r,w,sign)==0
 if k=="bool.not":
  v=eval_ast(a["value"],r,w,sign)
  if type(v) is not bool: raise ISAError("bool.not operand")
  return not v
 if a.get("width")!=w or a.get("signedness")!=sign: raise ISAError("AST type mismatch")
 mask=(1<<w)-1
 if k=="bv.const": return integer(a["value"],0,mask,"AST const")
 if k=="bv.read": return r[reg(a["name"])]
 if k not in ("bv.add","bv.sub","bv.and","bv.or","bv.xor","bv.and1","bv.shl","bv.shr","bv.sar"):
  raise ISAError("unknown AST operation")
 v=a.get("args")
 n=2 if k in ("bv.add","bv.sub","bv.and","bv.or","bv.xor") else 1
 if type(v) is not list or len(v)!=n: raise ISAError("AST arity")
 vals=[eval_ast(z,r,w,sign) for z in v]
 if any(type(z) is not int for z in vals): raise ISAError("AST bitvector expected")
 x=vals[0]; y=vals[1] if n==2 else 0
 if k=="bv.add": return (x+y)&mask
 if k=="bv.sub": return (x-y)&mask
 if k=="bv.and": return x&y
 if k=="bv.or": return x|y
 if k=="bv.xor": return x^y
 if k=="bv.and1": return x&1
 if k=="bv.shl": return (x<<1)&mask
 if k=="bv.shr": return x>>1
 if k=="bv.sar":
  if sign!="twos_complement": raise ISAError("SAR unsigned")
  return ((x>>1)|(x&(1<<(w-1))))&mask
 raise ISAError("AST unimplemented")

def flags_alu(op, x, before, result, width):
 """C: ADD carry, SUB no-borrow, left/right shift discarded bit.
 V: signed ADD/SUB overflow; Z,N on every ALU instruction."""
 top=1<<(width-1); mask=(1<<width)-1
 lhs=before[x["a"]] if op in BINARY else before[x["src"]]
 rhs=before[x["b"]] if op in BINARY else 0
 carry=False; overflow=False
 if op=="ADD":
  carry=lhs+rhs>mask
  overflow=bool((~(lhs^rhs)&(lhs^result)&top))
 elif op=="SUB":
  carry=lhs>=rhs
  overflow=bool(((lhs^rhs)&(lhs^result)&top))
 elif op=="SHL": carry=bool(lhs&top)
 elif op in ("SHR","SAR"): carry=bool(lhs&1)
 return {"C":carry,"V":overflow,"Z":result==0,"N":bool(result&top)}

def reference(p, registers, *, max_steps=128, trace=False):
 """Interpreter only: no neural computation, no model-output substitution.
 HALT keeps the current PC; step_limit and invalid_pc are failures, not halts.
 """
 p=validate(p); w=p["width"]; sign=p["signedness"]
 obj(registers,REGS,"registers")
 state={n:integer(registers[n],0,(1<<w)-1,n) for n in REGS}
 max_steps=integer(max_steps,1,1_000_000,"max_steps")
 flags={"C":False,"V":False,"Z":False,"N":False}
 pc=0; out=[]; executed=0; status="step_limit"
 for tick in range(max_steps):
  if pc<0 or pc>=len(p["program"]):
   status="invalid_pc"; break
  i=p["program"][pc]; op=i["op"]; ast=lower(p,pc)
  previous=dict(state); old_flags=dict(flags)
  if ast["write"] is not None:
   write=ast["write"]
   result=eval_ast(write["value"],previous,w,sign)
   if type(result) is not int: raise ISAError("write must be bitvector")
   if ast["flags"]=="alu.result":
    flags=flags_alu(op,i,previous,result,w)
   state[write["dst"]]=result
  ctrl=ast["pc"]
  if ctrl["kind"]=="pc.halt":
   next_pc=pc; status="halted"
  elif ctrl["kind"]=="pc.next": next_pc=pc+1
  elif ctrl["kind"]=="pc.literal": next_pc=ctrl["value"]
  elif ctrl["kind"]=="pc.branch":
   condition=eval_ast(ctrl["condition"],previous,w,sign)
   if type(condition) is not bool: raise ISAError("branch not bool")
   next_pc=ctrl["target"]["value"] if condition else pc+1
  else: raise ISAError("unknown PC effect")
  executed+=1
  if trace:
   out.append({"step":tick,"pc":pc,"instruction":i,"ast":ast,
               "before":previous,"after":dict(state),"flags_before":old_flags,
               "flags_after":dict(flags),"next_pc":next_pc,
               "halted":status=="halted"})
  pc=next_pc
  if status=="halted": break
 else:
  if pc<0 or pc>=len(p["program"]): status="invalid_pc"
 signed=None
 if sign=="twos_complement":
  signed={n:(val-(1<<w) if val&(1<<(w-1)) else val)
          for n,val in state.items()}
 return {"version":VERSION,"status":status,"halted":status=="halted",
         "pc":pc,"steps":executed,"registers":state,"flags":flags,
         "signed_interpretation":signed,"trace":out if trace else None}

def spec():
 """Pure declarative contract generated from the exact parser opcode table."""
 kinds={}
 for op,fields in FIELDS.items():
  kinds[op]={"fields":["op",*fields],"writes_flags":op in BINARY+UNARY,
             "pc_effect":("branch" if op in ("JZ","JNZ") else
                          "jump" if op=="JMP" else "halt" if op=="HALT" else "next"),
             "typed_ast":("bv."+op.lower() if op in BINARY+UNARY else
                          "bv.const" if op=="LOADI" else "bv.read" if op=="MOV" else
                          "bool.zero" if op=="JZ" else
                          "bool.not(bool.zero)" if op=="JNZ" else
                          "pc.literal" if op=="JMP" else "pc.halt")}
 return {
  "version":VERSION, "scope":"engineered ISA + symbolic oracle; no model training",
  "encoding":{"format":"UTF-8 JSON, sorted keys, separators comma/colon",
              "duplicate_keys":"reject","unknown_fields":"reject",
              "nonfinite":"reject","booleans_as_ints":"reject",
              "max_bytes":2_000_000},
  "program":{"required_fields":["version","width","signedness","program"],
             "width_range_bits":[1,64],"max_instructions":4096,
             "signedness_modes":["unsigned","twos_complement"],
             "last_instruction":"explicit HALT",
             "branch_target":"absolute zero-indexed instruction index"},
  "registers":{"names":list(REGS),"values":"explicit initial unsigned bitpattern 0..2^W-1",
               "initial_pc":0,"initial_flags":{"C":False,"V":False,"Z":False,"N":False},
               "source_wiring":"all operands explicit, simultaneous pre-step reads",
               "write":"one destination per ALU/MOV/LOADI instruction"},
  "math":{"bitvector_modulus":"2^width",
          "unsigned_output":"stored raw bits",
          "twos_complement_output":"raw bitvector, optional signed interpretation",
          "ADD":"(a+b) modulo 2^width","SUB":"(a-b) modulo 2^width",
          "AND":"a bitwise-AND b","OR":"a bitwise-OR b","XOR":"a bitwise-XOR b",
          "AND1":"src bitwise-AND 1","SHL":"one-bit logical shift left",
          "SHR":"one-bit logical shift right even in signed mode",
          "SAR":"one-bit arithmetic right shift; signed mode only",
          "MOV":"raw src bits","LOADI":"exact unsigned immediate within width"},
  "flags":{"update_ops":list(BINARY+UNARY),
           "C":"ADD carry out; SUB no-borrow; SHL old top bit; SHR/SAR old bit0; otherwise false",
           "V":"twos-complement signed overflow for ADD/SUB; otherwise false",
           "Z":"ALU result equals zero","N":"ALU result top bit",
           "preserved":["LOADI","MOV","JZ","JNZ","JMP","HALT"]},
  "pc":{"normal":"PC+1", "JZ":"if explicit source == 0 jump to target else PC+1",
        "JNZ":"if explicit source != 0 jump to target else PC+1",
        "JMP":"always target","HALT":"stop with PC unchanged",
        "loops":"backward jumps allowed", "invalid_pc":"failure, never implied halt",
        "max_steps":"required positive budget; nonhalt status is step_limit"},
  "opcodes":kinds,
  "neural_contract":{"inputs":["version","width","signedness","instruction",
                               "PC","registers X,Y,R,T","flags C,V,Z,N"],
        "predicted_outputs":["typed math AST","state delta","PC transition or HALT"],
        "reference_boundary":"oracle runs independently after prediction; never replaces it",
        "evaluation":"exact AST, state, PC, flag and HALT comparison",
        "not_proven":["neural semantics","program invention","English fluency","AGI"]},
  "legacy_mapping":{"ROM":["JZ_B","AND1","JZ_T","ADD","SHL","SHR","JMP","HALT"],
                    "JZ_B":"JZ src=Y with explicit target",
                    "JZ_T":"JZ src=T with explicit target",
                    "registers":{"A":"X","B":"Y","ACCUM":"R","TEST":"T"},
                    "limitation":"mapping is documentation, not a claim of equivalence to a learned controller"},
  "ast_schema":{"bv.read":["kind","name","width","signedness"],
                "bv.const":["kind","value","width","signedness"],
                "bv.unary":["kind","width","signedness","args[1]"],
                "bv.binary":["kind","width","signedness","args[2]"],
                "bool.zero":["kind","value:bitvector"],
                "bool.not":["kind","value:boolean"],
                "max_depth":64,"unknown_fields":"reject",
                "type_rule":"no mixed widths, no implicit casts, boolean and bitvector disjoint"},
  "model_dependencies":["Python standard library only; no RLT weights"],
  "task_dag_vs_runtime":"instruction CFG may loop; task DAG remains acyclic"
 }
