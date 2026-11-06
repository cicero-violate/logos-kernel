"""NC-055 trusted reference generator, NOT a sealed oracle by itself.

For untrusted participant code, use sandbox_evaluator.py. Open-source finite
fixtures remain reconstructible from observed values and published formulas.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import hmac
import secrets
import math
import random

SEED = 55061
# Ephemeral per evaluator process; NEVER serialize this value or pass it to candidates.
# HMAC prevents direct public-ID dictionary reversal of the published family grid.
# The key is NOT an authentication token and does not hide published generators.
_ID_KEY = secrets.token_bytes(32)

def _opaque_case_id(family, split, rep):
    message = f"NC055:v2:{family}:{split}:{rep}".encode("utf-8")
    return hmac.new(_ID_KEY, message, hashlib.sha256).hexdigest()[:32]

SPLITS = {
    "train": {"param_base": 2, "observed": 9, "horizon": 4},
    "validation": {"param_base": 13, "observed": 11, "horizon": 6},
    "test": {"param_base": 29, "observed": 13, "horizon": 8},
}
FAMILIES = (
    "constant", "polynomial", "geometric", "periodic", "multi_frequency",
    "nonlinear_modular", "logistic_chaos", "switching",
    "coupled_vector", "nonuniform_polynomial", "random_noise", "aliasing",
)
FAMILY_HOLDOUT = frozenset(("multi_frequency", "logistic_chaos", "switching"))
LAWS = {
    "constant": "constant",
    "polynomial": "quadratic_in_time",
    "geometric": "geometric_recurrence",
    "periodic": "period_four",
    "multi_frequency": "period_twelve",
    "nonlinear_modular": "quadratic_mod_97",
    "logistic_chaos": "logistic_r_four",
    "switching": "piecewise_linear",
    "coupled_vector": "coupled_fibonacci",
    "nonuniform_polynomial": "quadratic_in_time",
    "random_noise": None,
    "aliasing": "period_four",  # observation alias cannot prove constant
}
NONTRIVIAL = frozenset(FAMILIES) - {"constant", "random_noise", "aliasing"}

def _scalar(family, t, p, rep):
    if family == "constant":
        return Fraction(p)
    if family in ("polynomial", "nonuniform_polynomial"):
        if family == "polynomial":
            return Fraction(p + (rep + 1) * t + ((rep + 2) * t * t if rep >= 1 else 0) + (t**3 if rep == 2 else 0))
        return Fraction(p + (rep + 1) * t + (rep + 2) * t * t)
    if family == "geometric":
        return Fraction((rep + 1) * (2 + (p % 3)) ** t)
    if family in ("periodic", "aliasing"):
        return Fraction((1, 0, -1, 0)[(t + rep) % 4] * (rep + 2))
    if family == "multi_frequency":
        return Fraction((2, 0, -2, 0)[(t + rep) % 4]
                        + (0, 1, -1)[(t + p) % 3])
    if family == "nonlinear_modular":
        x = (p + rep) % 97
        for _ in range(t):
            x = (x * x + 3) % 97
        return Fraction(x)
    if family == "switching":
        threshold = 8 + rep
        return Fraction(p + 2 * min(t, threshold) - 3 * max(t - threshold, 0))
    if family == "random_noise":
        raw = hashlib.sha256(f"noise:{SEED}:{p}:{rep}:{t}".encode()).digest()
        return Fraction(int.from_bytes(raw[:4], "big") % 101 - 50)
    if family == "logistic_chaos":
        with localcontext() as ctx:
            ctx.prec = 78
            x = Decimal(1) / Decimal(5 + rep + (p % 3))
            for _ in range(t):
                x = 4 * x * (1 - x)
            return x
    raise ValueError(f"unknown scalar family: {family}")

def _vector(t, p, rep):
    a, b = Fraction(p), Fraction(rep + 1)
    for _ in range(t):
        a, b = b, a + b
    return (a, b)

def _times(family, split, count):
    if family == "aliasing":
        base = [4 * i for i in range(count)]
        # Intentional perturbation of observation rate at the evaluation boundary.
        return base
    if family == "nonuniform_polynomial":
        return [i * (i + 1) // 2 for i in range(count)]
    if family in {"nonlinear_modular", "logistic_chaos", "switching", "coupled_vector", "random_noise", "geometric"}:
        return list(range(count))
    step = {"train": 1, "validation": 2, "test": 3}[split]
    # Period-four identification needs a sampling stride coprime to four.
    # Intentional aliasing remains isolated in the dedicated aliasing family.
    if family == "periodic" and step % 2 == 0:
        step += 1
    start = {"train": 0, "validation": 2, "test": 5}[split]
    return [start + step * i for i in range(count)]

def _as_public(value):
    if isinstance(value, tuple):
        return [float(x) for x in value]
    return float(value)

def _exact(value):
    if isinstance(value, tuple):
        return [str(x) for x in value]
    return str(value)

def make_case(family, split, rep):
    if family not in FAMILIES or split not in SPLITS or rep not in (0, 1, 2):
        raise ValueError("unknown case contract")
    if split == "train" and family in FAMILY_HOLDOUT:
        raise ValueError("family held out entirely from training")
    conf = SPLITS[split]
    p = conf["param_base"] + rep
    n, h = conf["observed"], conf["horizon"]
    observed_t = _times(family, split, n)
    if family == "aliasing":
        query_t = [observed_t[-1] + j for j in range(1, h + 1)]
    else:
        query_t = _times(family, split, n + h)[n:]
    def val(t):
        return _vector(t, p, rep) if family == "coupled_vector" else _scalar(family, t, p, rep)
    obs = [val(t) for t in observed_t]
    truth = [val(t) for t in query_t]
    case_id = _opaque_case_id(family, split, rep)
    public = {
        "case_id": case_id, "split": split, "shape": "vector2" if family == "coupled_vector" else "scalar",
        "observed_times": observed_t, "observed_values": [_as_public(v) for v in obs],
        "query_times": query_t, "precision": "float64_observation; scalar absolute+relative scoring",
        "oracle_exposed": False,
    }
    private = {
         "case_id": case_id, "family": family, "law": (f"polynomial_degree_{rep+1}" if family == "polynomial" else LAWS[family]),
        "parameters": {"p": p, "rep": rep, "seed": SEED},
        "reference_values": [_as_public(v) for v in truth],
        "reference_exact": [_exact(v) for v in truth],
        "oracle_mode": "Decimal(78)" if family == "logistic_chaos" else "exact Fraction",
        "negative_control": family in {"random_noise", "aliasing"},
        "nontrivial_positive": family in NONTRIVIAL,
    }
    return public, private

def cases(split):
    for family in FAMILIES:
        if split == "train" and family in FAMILY_HOLDOUT:
            continue
        for rep in range(3):
            yield make_case(family, split, rep)
