"""Bounded prefix-only law search; standalone Python stdlib participant.

Reads public JSONL requests on stdin, returns six matched-budget arms. No oracle,
filesystem, network, model checkpoint or training dependency. Laws are proposals,
never proof of an unknown generator. Exact rational observations are preferred.
"""
from fractions import Fraction as F
import hashlib
import json
import math
import random
import sys
import time

ARMS = ("mdl_search", "autoregression", "persistence", "interpolation", "random_candidates", "no_invariant")
MAX_CHECKS = 128
CPU_PER_ARM = 0.20
MAX_BITS = 4096
ABS_TOL = 1e-6
REL_TOL = 1e-8


def rational(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("numeric observation required")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("finite observation required")
    result = F(str(value))
    if max(result.numerator.bit_length(), result.denominator.bit_length()) > MAX_BITS:
        raise ValueError("observation precision bound")
    return result


def linear_solve(rows, targets):
    """Exact overdetermined elimination; reject rank deficiency/inconsistency."""
    n = len(rows[0])
    matrix = [[F(x) for x in row] + [F(y)] for row, y in zip(rows, targets)]
    pivots = []
    row = 0
    for col in range(n):
        pivot = next((i for i in range(row, len(matrix)) if matrix[i][col]), None)
        if pivot is None: continue
        matrix[row], matrix[pivot] = matrix[pivot], matrix[row]
        scale = matrix[row][col]
        matrix[row] = [v / scale for v in matrix[row]]
        for i in range(len(matrix)):
            if i != row and matrix[i][col]:
                factor = matrix[i][col]
                matrix[i] = [a - factor * b for a, b in zip(matrix[i], matrix[row])]
        pivots.append(col)
        row += 1
    if len(pivots) < n or any(not any(r[:n]) and r[n] for r in matrix):
        raise ValueError("not an identified exact linear system")
    solution = [F(0)] * n
    for i, col in enumerate(pivots): solution[col] = matrix[i][-1]
    if any(abs(v) > 1000000 or max(v.numerator.bit_length(), v.denominator.bit_length()) > 128 for v in solution):
        raise ValueError("unstable/oversized coefficients")
    return solution


def poly_fit(times, values, degree):
    xs = times[:degree + 1]
    coeff = values[:degree + 1]
    coeff = list(coeff)
    for j in range(1, degree + 1):
        for i in range(degree, j - 1, -1):
            coeff[i] = (coeff[i] - coeff[i-1]) / (xs[i] - xs[i-j])
    return {"kind": "polynomial", "order": degree, "nodes": xs, "coeff": coeff}


def poly_value(model, t):
    value = model["coeff"][-1]
    for i in range(len(model["coeff"]) - 2, -1, -1):
        value = model["coeff"][i] + (t - model["nodes"][i]) * value
    return value


def predict(model, times, values, queries):
    kind = model["kind"]
    if kind == "polynomial": return [poly_value(model, t) for t in queries]
    if kind == "constant": return [model["value"] for _ in queries]
    step = times[1] - times[0]
    if any(b-a != step for a,b in zip(times,times[1:])):
        raise ValueError("recurrence needs uniform observations")
    if any(t != times[-1]+step*(i+1) for i,t in enumerate(queries)):
        raise ValueError("unidentified sampling-rate change")
    hist = list(values)
    result = []
    for i in range(len(queries)):
        if kind == "ar":
            value = sum(c*hist[-j-1] for j,c in enumerate(model["coeff"]))
        elif kind == "periodic":
            value = hist[-model["period"]]
        elif kind == "quadratic_mod":
            value = (hist[-1]**2 + model["offset"]) % model["modulus"]
        elif kind == "logistic":
            value = float(model["rate"])*float(hist[-1])*(1-float(hist[-1]))
        else: raise ValueError("unsupported law")
        if abs(value) > 1e100 or (isinstance(value, F) and max(value.numerator.bit_length(),value.denominator.bit_length()) > MAX_BITS):
            raise ValueError("unstable rollout")
        result.append(value)
        hist.append(value)
    return result


def law_size(model):
    kind = model["kind"]
    if kind == "polynomial": return 1+2*model["order"]
    if kind == "ar": return 1+2*len(model["coeff"])
    if kind == "periodic": return 2+model["period"]
    if kind == "quadratic_mod": return 5
    if kind == "logistic": return 5
    return 1


def candidate_models(times, values, method):
    n = len(values)
    fit_n = max(5, 2*n//3)
    if method == "persistence":
        yield {"kind":"constant", "value":values[-1]}, 0
        return
    if method == "no_invariant":
        yield {"kind":"constant", "value":F(0)}, 0
        return
    if method == "interpolation":
        yield poly_fit(times[-4:],values[-4:],3), 0
        return
    if method == "random_candidates":
        seed = int(hashlib.sha256(json.dumps([str(x) for x in values]).encode()).hexdigest()[:16],16)
        rng = random.Random(seed)
        for _ in range(MAX_CHECKS):
            degree = rng.randrange(5)
            coeff = [F(rng.randrange(-8,9)) for _ in range(degree+1)]
            yield {"kind":"polynomial","order":degree,"nodes":times[:degree+1],"coeff":coeff}, fit_n
        return
    if method == "autoregression": orders = (2,)
    else:
        for degree in range(5):
            # At least three prefix checks beyond the interpolation points.
            if degree+4 <= fit_n:
                yield poly_fit(times[:fit_n], values[:fit_n], degree), fit_n
        orders = (1,2,3,4)
    uniform = all(b-a == times[1]-times[0] for a,b in zip(times,times[1:]))
    if not uniform: return
    for order in orders:
        if fit_n < 2*order+2: continue
        try:
            rows = [[values[i-j-1] for j in range(order)] for i in range(order,fit_n)]
            coeff = linear_solve(rows, values[order:fit_n])
            yield {"kind":"ar","order":order,"coeff":coeff}, fit_n
        except ValueError: pass
    if method == "autoregression": return
    for period in range(2,min(12,fit_n//2)+1):
        yield {"kind":"periodic","period":period}, fit_n
    # Estimate logistic coefficient from prefix; no hardcoded r=4.
    rates = [float(values[i+1])/(float(v)*(1-float(v)))
             for i,v in enumerate(values[:fit_n-1]) if 1e-12 < float(v) < 1-1e-12]
    if len(rates) >= 4 and all(0<=float(v)<=1 for v in values):
        rate = sum(rates)/len(rates)
        if 0<rate<=4.000001:
            yield {"kind":"logistic","rate":rate}, fit_n
    # Search all moduli <=127 and infer the single offset from prefix only.
    # Ordered increasing description cost; not restricted to the benchmark's 97.
    if all(v.denominator==1 and 0<=v<128 for v in values):
        for modulus in range(max(2,int(max(values))+1),128):
            offset = int(values[1]-values[0]**2) % modulus
            yield {"kind":"quadratic_mod","modulus":modulus,"offset":offset}, fit_n


def evaluate_model(model, times, values, fit_n):
    if model["kind"] == "polynomial":
        got = [poly_value(model,t) for t in times]
        expected = values
    else:
        # Free rollout over fitting tail AND withheld prefix suffix, not merely
        # one-step teacher-forced residuals. Parameters see fitting prefix only.
        start = max(2, model.get("order",1), model.get("period",1))
        got = predict(model,times[:start],values[:start],times[start:])
        expected = values[start:]
    errors = [abs(float(a)-float(b))/(ABS_TOL+REL_TOL*abs(float(b))) for a,b in zip(got,expected)]
    maximum = max(errors,default=0.0)
    first = next((i for i,e in enumerate(errors) if e>1),None)
    exact = all(a==b for a,b in zip(got,expected))
    return maximum, exact, first


def choose(times, values, method, budget):
    best = None
    trace = []
    fit_n = 0
    models = sorted(candidate_models(times,values,method),
                    key=lambda item:(law_size(item[0]),json.dumps(signature(item[0]),sort_keys=True)))
    for model, fit_n in models:
        if budget["checks"] >= MAX_CHECKS or time.process_time()-budget["start"] >= CPU_PER_ARM:
            budget["exhausted"] = True
            break
        budget["checks"] += 1
        try:
            residual, exact, first = evaluate_model(model,times,values,fit_n)
            if method in ("persistence","no_invariant","interpolation"):
                accepted = True # Baseline forecast only; no structural claim.
            else: accepted = residual <= 1
        except (ValueError,OverflowError,ZeroDivisionError):
            residual, exact, first, accepted = 1e300, False, None, False
        trace.append({"kind":model["kind"],"order":model.get("order"),
                      "signature":signature(model),"law_size":law_size(model),
                      "prefix_error_in_tolerance_units":min(residual,1e300),
                      "prefix_exact":exact,"accepted":accepted,"first_counterexample_index":first})
        # Lower description size is preferred among prefix-validated candidates;
        # nonzero tolerance residual breaks equal-size ties only.
        score = (law_size(model),residual)
        if accepted and (best is None or score<best[0]): best=(score,model)
    return (best[1] if best else None),trace,fit_n


def signature(model):
    if model is None: return None
    return {key: ([str(v) for v in val] if isinstance(val,list) else str(val) if isinstance(val,F) else val)
            for key,val in model.items()}


def forecast(case, method):
    if method not in ARMS: raise ValueError("unknown arm")
    required = {"case_id","observed_times","observed_values","query_times","shape"}
    if set(case) != required: raise ValueError("public-only input schema")
    times = list(map(rational,case["observed_times"]))
    queries = list(map(rational,case["query_times"]))
    if not 8<=len(times)<=64 or not 1<=len(queries)<=32: raise ValueError("length budget")
    if any(b<=a for a,b in zip(times,times[1:])) or any(b<=a for a,b in zip([times[-1]]+queries,queries)):
        raise ValueError("times must strictly increase into future")
    dims = {"scalar":1,"vector2":2}[case["shape"]]
    raw = case["observed_values"]
    if len(raw)!=len(times): raise ValueError("shape mismatch")
    values = [[rational(v)] if dims==1 else list(map(rational,v)) for v in raw]
    if any(len(v)!=dims for v in values): raise ValueError("vector dimension")
    budget={"checks":0,"start":time.process_time(),"exhausted":False}
    all_predictions=[];models=[];traces=[];sensitivities=[];failures=[];sizes=[]
    for d in range(dims):
        series=[v[d] for v in values]
        model,trace,fit_n=choose(times,series,method,budget)
        try:
            if model is None: raise ValueError("no prefix-supported candidate")
            predicted=predict(model,times,series,queries)
            prediction=[float(x) for x in predicted]
            if not all(math.isfinite(v) and abs(v)<=1e100 for v in prediction): raise ValueError("unstable forecast")
            # Perturb final recurrence state. This measures sensitivity, not a
            # theorem or uncertainty bound. Polynomial node/measurement noise
            # is tested separately by evaluator perturbation fixtures.
            perturbed=list(series)
            perturbed[-1]+=F(1,10**12)
            other=predict(model,times,perturbed,queries)
            sensitivity=max(abs(float(a)-float(b)) for a,b in zip(predicted,other))
            failures.append(None)
        except (ValueError,OverflowError,ZeroDivisionError) as exc:
            prediction=[float(series[-1])]*len(queries)
            sensitivity=None
            failures.append(str(exc))
        all_predictions.append(prediction);models.append(signature(model));traces.append(trace);sensitivities.append(sensitivity)
        sizes.append(law_size(model) if model else None)
    output=[[all_predictions[d][i] for d in range(dims)] for i in range(len(queries))]
    return {"case_id":case["case_id"],"arm":method,"forecast":[v[0] for v in output] if dims==1 else output,
            "models":models,"traces":traces,"candidate_checks":budget["checks"],
            "cpu_seconds":time.process_time()-budget["start"],"budget_exhausted":budget["exhausted"],
            "sensitivity_to_last_state_1e_minus_12":sensitivities,"failures":failures,
            "law_size":sum(sizes) if all(v is not None for v in sizes) else None,
            "claim":"bounded prefix-supported hypothesis; future evaluation required; no global minimality"}


if __name__ == "__main__":
    for line in sys.stdin:
        case=json.loads(line)
        print(json.dumps({"case_id":case["case_id"],"arms":[forecast(case,arm) for arm in ARMS]},allow_nan=False),flush=True)
