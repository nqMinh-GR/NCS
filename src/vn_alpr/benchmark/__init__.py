"""Latency summaries. They describe the samples passed in and nothing else."""

from __future__ import annotations

import math
from typing import Sequence


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float(sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight)


def latency_summary(samples_ms: Sequence[float]) -> dict[str, float | int]:
    values = sorted(float(v) for v in samples_ms)
    if not values:
        raise ValueError("no timed samples")
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    elapsed = sum(values)
    return {
        "n": len(values),
        "mean_ms": mean,
        "p50_ms": _percentile(values, 0.50),
        "p90_ms": _percentile(values, 0.90),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "std_ms": math.sqrt(var),
        "elapsed_ms": elapsed,
        "throughput_per_s": 1000.0 / mean if mean else 0.0,
    }
