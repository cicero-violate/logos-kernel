"""NC-055 bounded CPU benchmark: participant sees public prefix/query only.

The evaluator uses oracle.py only after forecast() returns. Not an OS sandbox.
"""
import hashlib
import json
import math
import random
from pathlib import Path

import oracle

BASELINES = ("zero", "identity", "persistence", "finite_difference", "linear_recurrence", "randomized")
ABS_TOL = 1e-6
REL_TOL = 1e-8

def _dimensions(values):
    return len(values[-1]) if isinstance(values[-1], list) else 1

def _series(values, d):
    return [v[d] if isinstance(v, list) else v for v in values]

def _lagrange(times, values, queries, degree=3):
    # Interpolate only a fixed, bounded number of prefix samples, not future truth.
    size = min(len(times), degree + 1)
    xs, ys = times[-size:], values[-size:]
    out = []
    for query in queries:
        total = 0.0
        for i, x in enumerate(xs):
            basis = 1.0
            for j, y in enumerate(xs):
                if i != j:
                    basis *= (query - y) / (x - y)
            total += ys[i] * basis
        out.append(total)
    return out

def _recurrence(values, horizon):
    # Ridge-stabilized AR(2), fitted from PREFIX only and rolled out recursively.
    if len(values) < 4:
        return [values[-1]] * horizon
    a = b = c = d = r = s = 0.0
    for i in range(2, len(values)):
        x, y, z = values[i - 1], values[i - 2], values[i]
        a += x * x
        b += x * y
        d += y * y
        r += x * z
        s += y * z
    a += 1e-5
    d += 1e-5
    determinant = a * d - b * b
    if abs(determinant) < 1e-14:
        return [values[-1]] * horizon
    ca, cb = (r * d - b * s) / determinant, (a * s - b * r) / determinant
    hist = list(values)
    ans = []
    for _ in range(horizon):
        nxt = ca * hist[-1] + cb * hist[-2]
        if not math.isfinite(nxt):
            nxt = hist[-1]
        nxt = max(-1e30, min(1e30, nxt))
        ans.append(nxt)
        hist.append(nxt)
    return ans

def encode_view(case, encoding):
    """Actual held-out input encodings; encoded view contains NO raw observation vector."""
    if encoding not in ("float64", "decimal_text", "first_delta"):
        raise ValueError("unknown encoding")
    view = {k:v for k,v in case.items() if k != "observed_values"}
    view["observation_encoding"] = encoding
    values = case["observed_values"]
    if encoding == "float64":
        view["encoded_values"] = values
    elif encoding == "decimal_text":
        view["encoded_values"] = [[format(x, ".17g") for x in v] if isinstance(v,list)
                                  else format(v, ".17g") for v in values]
    else:
        dims = _dimensions(values)
        vectors = [[v] if dims==1 else v for v in values]
        view["encoded_values"] = {
            "first":vectors[0], "deltas":[[curr[d]-prev[d] for d in range(dims)]
                                           for prev,curr in zip(vectors,vectors[1:])],
        }
    return view

def decode_view(view):
    """Deterministic participant-side decoding; never touches the private oracle."""
    kind=view.get("observation_encoding")
    if kind is None:
        return view
    encoded=view["encoded_values"]
    if kind=="float64":
        values=encoded
    elif kind=="decimal_text":
        values=[[float(x) for x in v] if isinstance(v,list) else float(v) for v in encoded]
    elif kind=="first_delta":
        acc=list(encoded["first"])
        vectors=[acc]
        for step in encoded["deltas"]:
            acc=[x+dx for x,dx in zip(acc,step)]
            vectors.append(acc)
        values=[v[0] if view["shape"]=="scalar" else v for v in vectors]
    else:
        raise ValueError("unknown encoding")
    return {**{k:v for k,v in view.items() if k not in ("encoded_values","observation_encoding")},
            "observed_values":values}

def forecast(case, name):
    """Return no labels/proof claims; no candidate receives reference targets."""
    if name not in BASELINES:
        raise ValueError("unknown fixed baseline")
    case = decode_view(case)
    times = case["observed_times"]
    query = case["query_times"]
    obs = case["observed_values"]
    dims = _dimensions(obs)
    result = [[] for _ in query]
    for d in range(dims):
        ys = _series(obs, d)
        if name == "zero":
            predictions = [0.0] * len(query)
        elif name == "identity":
            predictions = [float(t) for t in query]
        elif name == "persistence":
            predictions = [ys[-1]] * len(query)
        elif name == "finite_difference":
            predictions = _lagrange(times, ys, query)
        elif name == "linear_recurrence":
            step = times[1] - times[0]
            uniform = all(times[i] - times[i-1] == step for i in range(1,len(times)))
            future_uniform = all(t == times[-1] + step * (i + 1) for i,t in enumerate(query))
            predictions = _recurrence(ys, len(query)) if uniform and future_uniform else [ys[-1]] * len(query)
        else:
            seed = int(hashlib.sha256((case["case_id"] + ":" + str(d)).encode()).hexdigest()[:16], 16)
            rng = random.Random(seed)
            predictions = [rng.uniform(-1, 1) for _ in query]
        for i, value in enumerate(predictions):
            result[i].append(float(value))
    return {"forecast": [v[0] if dims == 1 else v for v in result],
            "law": None, "proof": None}

def _identifying_variation(reference):
    """Flat held-out trajectories cannot identify a nonconstant dynamical law.

    Require separation beyond twice the scoring tolerance: no constant
    prediction that passes every reference point can satisfy this gate.
    """
    if len(reference) < 2:
        return False
    first = reference[0] if isinstance(reference[0], list) else [reference[0]]
    for value in reference[1:]:
        current = value if isinstance(value, list) else [value]
        if any(abs(x-y) > 2 * (ABS_TOL + REL_TOL * max(abs(x), abs(y)))
               for x,y in zip(current, first)):
            return True
    return False

def grade(public, private, submitted):
    """Oracle-side grading: forecast accuracy is NOT a proof or discovery."""
    pred = submitted.get("forecast")
    truth = private["reference_values"]
    if not isinstance(pred, list) or len(pred) != len(truth):
        raise ValueError("forecast length mismatch")
    dimension = _dimensions(truth)
    max_error = 0.0
    total = 0.0
    passed = True
    for got, expected in zip(pred, truth):
        g = got if isinstance(got, list) else [got]
        t = expected if isinstance(expected, list) else [expected]
        if len(g) != dimension or len(t) != dimension:
            raise ValueError("prediction shape mismatch")
        for x, y in zip(g, t):
            if isinstance(x, bool) or not isinstance(x, (int,float)) or not math.isfinite(x):
                raise ValueError("nonfinite or nonnumeric candidate")
            error = abs(x - y)
            total += error
            max_error = max(max_error, error)
            passed &= error <= ABS_TOL + REL_TOL * abs(y)
    law_matches = submitted.get("law") is not None and submitted.get("law") == private["law"]
    # Formal theorem proof is NOT established from a matching name.
    # This is bounded empirical identification, NOT a theorem certificate.
    # Do not trust candidate-supplied "proof" strings as independent evidence.
    identifying_variation = _identifying_variation(truth)
    discovery_pass = bool(passed and law_matches and private["nontrivial_positive"]
                          and identifying_variation)
    return {
        "forecast_pass": bool(passed), "mean_absolute_error":total/(len(truth)*dimension),
        "maximum_absolute_error":max_error, "claimed_law_matches":law_matches,
        "bounded_discovery_pass":discovery_pass, "identifying_variation":identifying_variation,
        "proof_scope":"heldout forecast plus private law match and informative variation; no formal theorem or authenticated witness",
    }

def run():
    data = {"schema_version":1, "task":"NC-055", "seed":oracle.SEED,
            "split_counts":{}, "family_counts":{}, "baseline_scores":{}, "controls":{}}
    for split in oracle.SPLITS:
        encoding = {"train":"float64","validation":"decimal_text","test":"first_delta"}[split]
        cases = [(encode_view(p,encoding),priv) for p,priv in oracle.cases(split)]
        data.setdefault("input_encoding",{})[split] = encoding
        data["split_counts"][split] = len(cases)
        data["family_counts"][split] = {}
        for _, truth in cases:
            k = truth["family"]
            data["family_counts"][split][k] = data["family_counts"][split].get(k,0)+1
        data["baseline_scores"][split] = {}
        for name in BASELINES:
            scores = [grade(public,priv,forecast(public,name)) for public,priv in cases]
            data["baseline_scores"][split][name] = {
                "denominator":len(scores), "forecast_pass":sum(s["forecast_pass"] for s in scores),
                "bounded_discovery_pass":sum(s["bounded_discovery_pass"] for s in scores),
                "mean_absolute_error":sum(s["mean_absolute_error"] for s in scores)/len(scores)
            }
    # Oracle-only positive and deliberately falsified controls; NEVER participant baselines.
    test_cases = list(oracle.cases("test"))
    positive = nonidentifying_nontrivial = negative = constant_collapse_reject = 0
    for public, priv in test_cases:
        perfect = {"forecast":priv["reference_values"], "law":priv["law"],
                   "proof":"heldout_recurrence_witness"}
        score = grade(public, priv, perfect)
        if priv["nontrivial_positive"]:
            if score["bounded_discovery_pass"]:
                positive += 1
            else:
                assert not score["identifying_variation"]
                nonidentifying_nontrivial += 1
        falsified = dict(perfect, law="intentionally_wrong_law")
        assert not grade(public,priv,falsified)["bounded_discovery_pass"]
        negative += 1
        collapsed = dict(perfect, forecast=[0.0 if public["shape"]=="scalar" else [0.0,0.0]] * len(public["query_times"]))
        assert not grade(public,priv,collapsed)["bounded_discovery_pass"] or all(
            v == 0.0 or v == [0.0,0.0] for v in priv["reference_values"])
        if priv["nontrivial_positive"] and not grade(public,priv,collapsed)["bounded_discovery_pass"]:
            constant_collapse_reject += 1
    # Previously test-only: a validation periodic trajectory was all zero
    # under stride-two sampling, falsely certifying a constant forecast.
    collapse_all_splits = {}
    for split in oracle.SPLITS:
        tested = rejected = oracle_assisted_rejected = 0
        for public, priv in oracle.cases(split):
            if not priv["nontrivial_positive"]:
                continue
            tested += 1
            zero = 0.0 if public["shape"] == "scalar" else [0.0, 0.0]
            collapsed = {"forecast": [zero] * len(public["query_times"]), "law": priv["law"], "proof": None}
            if grade(public, priv, collapsed)["bounded_discovery_pass"]:
                raise AssertionError(f"constant-collapse false discovery: {split}:{public['case_id']}")
            rejected += 1
            # Stronger oracle-assisted attack: return the first *future* target
            # at every horizon point, along with the correct law label.
            assisted = dict(collapsed, forecast=[priv["reference_values"][0]] * len(public["query_times"]))
            if grade(public, priv, assisted)["bounded_discovery_pass"]:
                raise AssertionError(f"oracle-assisted constant falsely certified: {split}:{public['case_id']}")
            oracle_assisted_rejected += 1
        collapse_all_splits[split] = {"tested": tested, "rejected": rejected,
                                      "oracle_assisted_constant_rejected": oracle_assisted_rejected}
    interpolation_cases = [(p,v) for p,v in test_cases if v["family"] in
                           {"periodic","multi_frequency","nonlinear_modular","switching","random_noise","aliasing"}]
    interpolation_failed = sum(not grade(p,v,forecast(p,"finite_difference"))["forecast_pass"]
                               for p,v in interpolation_cases)
    data["controls"] = {
        "oracle_only_positive_nontrivial":positive,
        "oracle_only_nonidentifying_nontrivial":nonidentifying_nontrivial,
        "wrong_law_rejected":negative,
        "constant_collapse_nontrivial_rejected":constant_collapse_reject,
        "constant_collapse_all_splits":collapse_all_splits,
        "short_prefix_interpolation": {"tested":len(interpolation_cases),
            "failed_extrapolation":interpolation_failed,
            "bounded_discovery_pass":0}
    }
    return data

if __name__ == "__main__":
    print(json.dumps(run(),sort_keys=True,indent=2))
