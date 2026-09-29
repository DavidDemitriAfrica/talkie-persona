"""Statistics shared by every unified report. Pure Python: the GPU box has no scipy.

The inferential unit is the training run. A contrast between two arms is the
mean of per-seed differences in misalignment rate, tested with a paired t on
n_seeds - 1 degrees of freedom (Nick's REPLICATION.md, "Why not a z-test": four
claims that cleared a pooled-generation z-test failed this, including the
almanac system built to be a null). Wilson intervals are for describing a single
pooled rate, never for testing a contrast.
"""

from __future__ import annotations

import math
from statistics import mean, stdev


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(p, lo, hi), 95% by default."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes 6.4)."""
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > FPMIN else FPMIN)
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > FPMIN else FPMIN)
        c = 1 + aa / c
        c = c if abs(c) > FPMIN else FPMIN
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > FPMIN else FPMIN)
        c = 1 + aa / c
        c = c if abs(c) > FPMIN else FPMIN
        de = d * c
        h *= de
        if abs(de - 1) < EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
           + a * math.log(x) + b * math.log(1 - x))
    bt = math.exp(lbt)
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1 - bt * _betacf(b, a, 1 - x) / b


def t_two_sided_p(t: float, df: int) -> float:
    """Exact two-sided p for Student's t."""
    if df <= 0 or math.isnan(t):
        return float("nan")
    if math.isinf(t):
        return 0.0
    return betainc(df / 2, 0.5, df / (df + t * t))


def paired_t(diffs: list[float]) -> dict:
    """Paired t on per-seed differences (in percentage points)."""
    n = len(diffs)
    out = {"n_seeds": n, "per_seed": [round(d, 3) for d in diffs]}
    if n == 0:
        return out | {"mean": float("nan"), "sd": float("nan"),
                      "t": float("nan"), "df": 0, "p": float("nan")}
    m = mean(diffs)
    if n < 2:
        return out | {"mean": m, "sd": float("nan"), "t": float("nan"),
                      "df": 0, "p": float("nan")}
    sd = stdev(diffs)
    if sd == 0:
        t = math.copysign(math.inf, m) if m else 0.0
    else:
        t = m / (sd / math.sqrt(n))
    return out | {"mean": m, "sd": sd, "t": t, "df": n - 1,
                  "p": t_two_sided_p(t, n - 1)}


def rate(rows: list[dict], judge: str) -> tuple[int, int]:
    """(misaligned, kept) over rows already carrying row["judge"]."""
    kept = [r for r in rows if r.get("judge", {}).get(judge, {}).get("kept")]
    return sum(r["judge"][judge]["misaligned"] for r in kept), len(kept)


def seed_contrast(rates_a: dict[int, tuple[int, int]],
                  rates_b: dict[int, tuple[int, int]]) -> dict:
    """A - B in pp, paired on the seeds both arms have.

    rates_* map seed -> (misaligned, kept). A seed with zero kept responses in
    either arm is dropped (a rate over nothing is not a rate) and reported.
    """
    seeds = sorted(set(rates_a) & set(rates_b))
    usable = [s for s in seeds if rates_a[s][1] and rates_b[s][1]]
    diffs = [100 * (rates_a[s][0] / rates_a[s][1] - rates_b[s][0] / rates_b[s][1])
             for s in usable]
    res = paired_t(diffs)
    res["seeds"] = usable
    res["dropped_seeds"] = [s for s in seeds if s not in usable]
    res["unpaired_seeds"] = sorted(set(rates_a) ^ set(rates_b))
    return res
