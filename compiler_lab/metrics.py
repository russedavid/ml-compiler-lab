"""Comparison policy fixed before running the initial float32 experiments."""

import numpy as np

RTOL = 1e-4
ATOL = 1e-5


def compare(actual, expected):
    a, e = np.asarray(actual), np.asarray(expected)
    if a.shape != e.shape:
        return {
            "passed": False,
            "reason": "shape mismatch",
            "actual": list(a.shape),
            "expected": list(e.shape),
        }
    if not (np.isfinite(a).all() and np.isfinite(e).all()):
        return {"passed": False, "reason": "nonfinite output"}
    delta = np.abs(a.astype(np.float64) - e.astype(np.float64))
    bound = ATOL + RTOL * np.abs(e.astype(np.float64))
    return {
        "passed": bool((delta <= bound).all()),
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "violations": int((delta > bound).sum()),
        "elements": int(e.size),
        "rtol": RTOL,
        "atol": ATOL,
    }


def summarize(samples_ms):
    a = np.asarray(samples_ms, dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("timings must be nonempty, finite and nonnegative")
    return {
        "count": int(a.size),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "mean_ms": float(a.mean()),
    }
