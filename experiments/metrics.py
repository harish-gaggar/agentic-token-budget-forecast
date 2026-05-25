"""Forecast quality metrics: MAE, MAPE, W20R, and McNemar's test."""

import math
from typing import Iterable, List, Tuple
import numpy as np
from scipy.stats import binom


def mae(pred: Iterable[float], true: Iterable[float]) -> float:
    p, t = np.asarray(list(pred), dtype=float), np.asarray(list(true), dtype=float)
    return float(np.mean(np.abs(p - t)))


def mape(pred: Iterable[float], true: Iterable[float]) -> float:
    p, t = np.asarray(list(pred), dtype=float), np.asarray(list(true), dtype=float)
    return float(np.mean(np.abs(p - t) / np.clip(t, 1.0, None)) * 100.0)


def within_band(pred: Iterable[float], true: Iterable[float], band: float = 0.20) -> float:
    p, t = np.asarray(list(pred), dtype=float), np.asarray(list(true), dtype=float)
    err = np.abs(p - t) / np.clip(t, 1.0, None)
    return float(np.mean(err <= band) * 100.0)


def w20r(pred, true) -> float:
    return within_band(pred, true, 0.20)


def mcnemar_p(hits_a: np.ndarray, hits_b: np.ndarray) -> float:
    """Exact McNemar's test on paired success/failure outcomes."""
    b = int(np.sum(hits_a & ~hits_b))   # A right, B wrong
    c = int(np.sum(~hits_a & hits_b))   # B right, A wrong
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_two = 2.0 * binom.cdf(k, n, 0.5)
    return float(min(p_two, 1.0))
