import unittest
from dag import ZERO,audit,canon,hash_event

def row(i,prev,ev):
    b={"seq":i,"event":ev}
    return {**b,"prev_sha256":prev,"sha256":hash_event(prev,b)}

def node(id,deps=()):
    return {"kind":"task","id":id,"priority":"P0","title":"test","depends_on":list(deps),"instruction":["do"],"acceptance":["pass"],"outputs":["evidence.json"],"avoid":["regression"],"status":"planned"}

class LedgerTests(unittest.TestCase):
    def test_two_nodes(self):
        a=row(1,ZERO,node("a"));b=row(2,a["sha256"],node("b",["a"]))
        tasks,n,_=audit(canon(a)+"\n"+canon(b)+"\n")
        self.assertEqual(n,2)
        self.assertEqual(tasks["b"]["record"]["depends_on"],["a"])
    def test_tamper(self):
        a=row(1,ZERO,node("a"));a["event"]["title"]="modified"
        with self.assertRaisesRegex(ValueError,"hash changed"):
            audit(canon(a)+"\n")
    def test_forward_dependencies_rejected(self):
        a=row(1,ZERO,node("b",["a"]))
        with self.assertRaisesRegex(ValueError,"earlier tasks"):
            audit(canon(a)+"\n")
    def test_cannot_start_blocked_dependent(self):
        a=row(1,ZERO,node("a"));b=row(2,a["sha256"],node("b",["a"]))
        c=row(3,b["sha256"],{"kind":"status","id":"b","status":"working","agent":"test"})
        with self.assertRaisesRegex(ValueError,"dependencies not passed"):
            audit("\n".join(map(canon,(a,b,c)))+"\n")

if __name__=="__main__":unittest.main()

