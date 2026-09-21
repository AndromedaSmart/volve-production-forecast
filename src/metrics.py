"""Метрики качества прогноза."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2_score(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def mape(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1.0) -> float:
    """MAPE в процентах. Нулевые факты заменяются на eps, чтобы не делить на 0."""
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs(yt - yp) / denom) * 100.0)


def wape(y_true: Iterable[float], y_pred: Iterable[float], eps: float = 1e-9) -> float:
    """WAPE в процентах: сумма абсолютных ошибок / сумма факта."""
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    denom = np.sum(np.abs(yt))
    if denom <= eps:
        return float("nan")
    return float(np.sum(np.abs(yt - yp)) / denom * 100.0)


def rms_wape(wapes: Iterable[float]) -> float:
    vals = np.asarray([v for v in wapes if pd.notna(v)], dtype=float)
    if len(vals) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(vals ** 2)))


def regression_report(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "WAPE": wape(y_true, y_pred),
    }
