"""NC-001 reference contract tests. Dependencies: unittest, random, json, canonical_isa."""
import copy
import json
import random
import unittest
import canonical_isa as isa

def program(*ops, width=8, sign="unsigned"):
 return {"version":isa.VERSION,"width":width,"signedness":sign,
         "program":[*ops,{"op":"HALT"}]}

def run_one(op,inputs=None, width=8, sign="unsigned", **kw):
 r={"X":0,"Y":0,"R":0,"T":0}
 if inputs: r.update(inputs)
 p=program({"op":op,**kw},width=width,sign=sign)
 return isa.reference(p,r,trace=True)

class Contracts(unittest.TestCase):
 def test_every_opcode_has_defined_lowering_and_execution(self):
  patterns={
   "LOADI":{"dst":"R","imm":5}, "MOV":{"dst":"R","src":"X"},
   **{op:{"dst":"R","a":"X","b":"Y"} for op in isa.BINARY},
   **{op:{"dst":"R","src":"X"} for op in isa.UNARY},
   "JZ":{"src":"X","target":1}, "JNZ":{"src":"X","target":1},
   "JMP":{"target":1}, "HALT":{},
  }
  self.assertEqual(len(patterns),15)
  for name,attrs in patterns.items():
   with self.subTest(op=name):
    p=program({"op":name,**attrs},sign="twos_complement")
    node=isa.lower(p,0)
    self.assertEqual(node["opcode"],name)
    self.assertEqual(node["kind"],"transition")
    result=isa.reference(p,{"X":2,"Y":1,"R":0,"T":0},trace=True)
    self.assertTrue(result["halted"])
    self.assertGreater(result["steps"],0)
    self.assertEqual(len(result["trace"]),result["steps"])
 def test_roundtrip_canonical_json_all_opcodes(self):
  r=random.Random(20261105)
  for width in (1,8,16,64):
   instructions=[]
   for op,fields in isa.FIELDS.items():
    if op=="HALT": continue
    x={"op":op}
    for field in fields:
     x[field]=(r.randrange(1<<width) if field=="imm" else
               14 if field=="target" else "R" if field=="dst" else "X")
    instructions.append(x)
   p={"version":isa.VERSION,"width":width,"signedness":"twos_complement",
      "program":[*instructions,{"op":"HALT"}]}
   self.assertEqual(len(p["program"]),15)
   encoded=isa.encode(p)
   self.assertEqual(isa.decode(encoded),isa.validate(p))
   self.assertEqual(isa.encode(isa.decode(encoded)),encoded)
   self.assertEqual(encoded,isa.encode(json.loads(encoded)))
 def test_alu_property_vectors(self):
  rng=random.Random(12345)
  formulas={"ADD":lambda a,b,m:(a+b)&m,"SUB":lambda a,b,m:(a-b)&m,
            "AND":lambda a,b,m:a&b,"OR":lambda a,b,m:a|b,"XOR":lambda a,b,m:a^b}
  for w in (1,8,16,64):
   mask=(1<<w)-1
   for op,func in formulas.items():
    p=program({"op":op,"dst":"R","a":"X","b":"Y"},width=w)
    for _ in range(64):
     a=rng.randrange(mask+1);b=rng.randrange(mask+1)
     x=isa.reference(p,{"X":a,"Y":b,"R":0,"T":0})
     self.assertEqual(x["registers"]["R"],func(a,b,mask),(op,w,a,b))
     self.assertEqual(x["flags"]["Z"],func(a,b,mask)==0)
     self.assertEqual(x["flags"]["N"],bool(func(a,b,mask)&(1<<(w-1))))
 def test_unary_bitvectors(self):
  rng=random.Random(7)
  for w in (1,8,16,64):
   mask=(1<<w)-1;top=1<<(w-1)
   for _ in range(60):
    v=rng.randrange(mask+1)
    operations=[("AND1",v&1),("SHL",(v<<1)&mask),("SHR",v>>1)]
    operations.append(("SAR",((v>>1)|(v&top))&mask))
    for op,expected in operations:
     sign="twos_complement" if op=="SAR" else "unsigned"
     x=run_one(op,{"X":v},w,sign,dst="R",src="X")
     self.assertEqual(x["registers"]["R"],expected)
     self.assertTrue(x["halted"])
 def test_carry_borrow_overflow(self):
  cases=[
   ("ADD",255,1,0,True,False),
   ("ADD",127,1,128,False,True),
   ("SUB",0,1,255,False,False),
   ("SUB",128,1,127,True,True),
   ("SUB",9,9,0,True,False),
  ]
  for op,a,b,val,carry,overflow in cases:
   with self.subTest(op=op,a=a,b=b):
    x=run_one(op,{"X":a,"Y":b},8,"twos_complement",dst="R",a="X",b="Y")
    self.assertEqual(x["registers"]["R"],val)
    self.assertIs(x["flags"]["C"],carry)
    self.assertIs(x["flags"]["V"],overflow)
 def test_preserve_flags_for_nonalu(self):
  p=program({"op":"ADD","dst":"R","a":"X","b":"Y"},
            {"op":"LOADI","dst":"X","imm":0},
            {"op":"MOV","dst":"Y","src":"X"},
            {"op":"JZ","src":"Y","target":4})
  x=isa.reference(p,{"X":255,"Y":1,"R":0,"T":0},trace=True)
  self.assertTrue(x["halted"])
  self.assertEqual(x["flags"],x["trace"][0]["flags_after"])
 def test_branch_taken_and_not_taken(self):
  for op in ("JZ","JNZ"):
   p=program({"op":op,"src":"X","target":2},
             {"op":"LOADI","dst":"R","imm":7})
   for v in (0,1):
    x=isa.reference(p,{"X":v,"Y":0,"R":0,"T":0})
    taken=(op=="JZ" and v==0) or (op=="JNZ" and v!=0)
    self.assertEqual(x["registers"]["R"],0 if taken else 7)
    self.assertTrue(x["halted"])
 def test_backward_loop_and_step_limit(self):
  p=program({"op":"JMP","target":0})
  x=isa.reference(p,{"X":0,"Y":0,"R":0,"T":0},max_steps=7,trace=True)
  self.assertEqual((x["status"],x["halted"],x["steps"]),("step_limit",False,7))
  self.assertEqual(x["pc"],0)
  self.assertEqual([e["pc"] for e in x["trace"]],[0]*7)
 def test_explicit_halt_at_pc(self):
  p=program({"op":"LOADI","dst":"R","imm":9})
  x=isa.reference(p,{"X":0,"Y":0,"R":0,"T":0})
  self.assertEqual((x["status"],x["pc"],x["steps"]),("halted",1,2))
 def test_signed_interpretation(self):
  x=run_one("SAR",{"X":0xFF},8,"twos_complement",dst="R",src="X")
  self.assertEqual(x["registers"]["R"],255)
  self.assertEqual(x["signed_interpretation"]["R"],-1)
  self.assertEqual(run_one("SHR",{"X":255},8,dst="R",src="X")["registers"]["R"],127)
 def test_math_ast_is_causally_used(self):
  p=program({"op":"ADD","dst":"R","a":"X","b":"Y"})
  ast=isa.lower(p,0)["write"]["value"]
  r={"X":7,"Y":9,"R":0,"T":0}
  original=isa.eval_ast(ast,r,8,"unsigned")
  mutated=copy.deepcopy(ast);mutated["kind"]="bv.sub"
  changed=isa.eval_ast(mutated,r,8,"unsigned")
  self.assertEqual(original,16)
  self.assertEqual(changed,254)
  self.assertNotEqual(original,changed)
 def test_reference_fixed_rom_exhaustive_4bit(self):
  # Deliberately predefined ROM, not learned program synthesis.
  p=program(
   {"op":"JZ","src":"Y","target":7},
   {"op":"AND1","dst":"T","src":"Y"},
   {"op":"JZ","src":"T","target":4},
   {"op":"ADD","dst":"R","a":"R","b":"X"},
   {"op":"SHL","dst":"X","src":"X"},
   {"op":"SHR","dst":"Y","src":"Y"},
   {"op":"JMP","target":0},
   width=16)
  for a in range(16):
   for b in range(16):
    result=isa.reference(p,{"X":a,"Y":b,"R":0,"T":0},max_steps=75)
    self.assertEqual(result["status"],"halted",(a,b))
    self.assertEqual(result["registers"]["R"],a*b,(a,b))
 def test_malformed_inputs_rejected(self):
  good=program({"op":"ADD","dst":"R","a":"X","b":"Y"})
  invalid=[]
  for key,bad in (("version","other"),("width",0),("width",True),
                  ("width",65),("signedness","decimal")):
   q=copy.deepcopy(good);q[key]=bad;invalid.append(q)
  for name in ("WRONG",None,10,True):
   q=copy.deepcopy(good);q["program"][0]["op"]=name;invalid.append(q)
  q=copy.deepcopy(good);q["unknown"]=1;invalid.append(q)
  q=copy.deepcopy(good);q["program"][0]["other"]=1;invalid.append(q)
  q=copy.deepcopy(good);q["program"][0]["a"]="NOPE";invalid.append(q)
  q=copy.deepcopy(good);q["program"][-1]={"op":"JMP","target":0};invalid.append(q)
  q=program({"op":"JZ","src":"X","target":3});invalid.append(q)
  q=program({"op":"LOADI","dst":"X","imm":256});invalid.append(q)
  q=program({"op":"SAR","dst":"X","src":"Y"});invalid.append(q)
  for i,v in enumerate(invalid):
   with self.subTest(index=i):
    with self.assertRaises(isa.ISAError): isa.validate(v)
  for raw in ('{"version":"logos.isa/1","version":"duplicate"}','NaN','Infinity','{}'):
   with self.assertRaises(isa.ISAError): isa.decode(raw)
  with self.assertRaises(isa.ISAError):
   isa.reference(good,{"X":1,"Y":2,"R":0,"T":True})
  with self.assertRaises(isa.ISAError):
   isa.reference(good,{"X":1,"Y":2,"R":0,"T":0},max_steps=True)
  with self.assertRaises(isa.ISAError):
   isa.eval_ast({"kind":"bv.add","width":8,"signedness":"unsigned","args":[]},
                {"X":1},8,"unsigned")
 def test_malformed_every_opcode_contract(self):
  for op,fields in isa.FIELDS.items():
   values={"op":op}
   for field in fields:
    values[field]=0 if field=="imm" else 1 if field=="target" else "X"
   baseline=program(values,sign="twos_complement")
   for field in ("op",*fields):
    broken=copy.deepcopy(baseline)
    del broken["program"][0][field]
    with self.subTest(op=op,remove=field),self.assertRaises(isa.ISAError):
     isa.validate(broken)
   broken=copy.deepcopy(baseline)
   broken["program"][0]["surprise"]=42
   with self.subTest(op=op,extra=True),self.assertRaises(isa.ISAError):
    isa.validate(broken)
   for field in fields:
    broken=copy.deepcopy(baseline)
    broken["program"][0][field]="bad_value"
    with self.subTest(op=op,bad_type=field),self.assertRaises(isa.ISAError):
     isa.validate(broken)
  # No extra operands on HALT, no boolean-as-integer, no unknown registers.
 def test_reject_untyped_or_malformed_neural_ast(self):
  valid=isa.read("X",8,"unsigned")
  cases=[
   {**valid,"unexpected":9},
   {**valid,"width":True},
   {**valid,"width":16},
   {**valid,"signedness":"twos_complement"},
   {"kind":"bv.add","width":8,"signedness":"unsigned","args":[valid]},
   {"kind":"bv.add","width":8,"signedness":"unsigned","args":[valid,{"kind":"bool.zero","value":valid}]},
   {"kind":"bool.not","value":valid},
   {"kind":"bool.zero","value":{"kind":"bool.zero","value":valid}},
   {"kind":"bv.const","value":256,"width":8,"signedness":"unsigned"},
   {"kind":"bv.sar","width":8,"signedness":"unsigned","args":[valid]},
  ]
  for n,node in enumerate(cases):
   with self.subTest(index=n),self.assertRaises(isa.ISAError):
    isa.validate_ast(node,8,"unsigned")
  for node in ({"kind":"bool.zero","value":valid},
               {"kind":"bool.not","value":{"kind":"bool.zero","value":valid}}):
   self.assertEqual(isa.validate_ast(node,8,"unsigned"),"bool")
  self.assertEqual(isa.validate_ast(valid,8,"unsigned"),"bitvec")
 def test_static_specs_complete(self):
  s=isa.spec()
  self.assertEqual(set(s["opcodes"]),set(isa.FIELDS))
  self.assertEqual(s["registers"]["names"],list(isa.REGS))
  for op in isa.FIELDS:
   self.assertEqual(s["opcodes"][op]["fields"],["op",*isa.FIELDS[op]])

if __name__=="__main__":
 unittest.main()
