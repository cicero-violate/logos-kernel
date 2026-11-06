"""NC-079 independent exhaustive scalar-evaluator acceptance gate.

Unlike stack.verified(), which samples 1,024 scalar pairs, this independently
cross-checks all 65,536 input pairs, alongside the complete vector truth table.
This helper is not yet wired into other workers' experiment runners.

Dependencies: stdlib, NumPy, artifacts/NC-079/stack.py, NC-061 scoreboard.
"""
import numpy as np
import stack as s

def verify(expr,goal):
    xs,ys=s.VERIFY.x,s.VERIFY.y
    target=np.asarray(s.goal_vector(goal,xs,ys),dtype=np.int64)&s.MASK
    vector=np.asarray(s.evaluate(expr,xs,ys),dtype=np.int64)&s.MASK
    if not np.array_equal(vector,target):
        return False
    for x,y,want in zip(xs,ys,target):
        if s.reference_value(expr,int(x),int(y))!=int(want):
            return False
    return True
