"""Признаки для рекурсивного прогноза месячного ряда."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 6)
ROLLS = (3, 6)


def add_calendar(df: pd.DataFrame, date_col: str = "DATE") -> pd.DataFrame:
    out = df.copy()
    month = out[date_col].dt.month
    out["t"] = np.arange(len(out), dtype=float)
    out["month"] = month
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return out


def add_lags(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        for lag in LAGS:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
        for win in ROLLS:
            out[f"{col}_rmean{win}"] = out[col].shift(1).rolling(win, min_periods=1).mean()
    return out


def feature_columns(target: str) -> List[str]:
    cols = ["t", "month_sin", "month_cos", "on_stream_lag1", "n_producers_lag1"]
    for col in (target, "water"):
        for lag in LAGS:
            cols.append(f"{col}_lag{lag}")
        for win in ROLLS:
            cols.append(f"{col}_rmean{win}")
    return cols


def make_supervised(field: pd.DataFrame, target: str) -> pd.DataFrame:
    work = field.sort_values("DATE").reset_index(drop=True)
    work = add_calendar(work)
    work["on_stream_lag1"] = work["on_stream"].shift(1)
    work["n_producers_lag1"] = work["n_producers"].shift(1)
    work = add_lags(work, [target, "water"])
    return work


def lag_or_nan(values: np.ndarray, lag: int) -> float:
    if len(values) < lag:
        return np.nan
    return float(values[-lag])


def shifted_roll_mean(values: np.ndarray, window: int) -> float:
    if len(values) == 0:
        return np.nan
    return float(np.mean(values[-window:]))


def recursive_feature_row(
    *,
    target: str,
    t: float,
    month: int,
    on_stream_lag1: float,
    n_producers_lag1: float,
    target_hist: np.ndarray,
    water_hist: np.ndarray,
) -> dict:
    row = {
        "t": t,
        "month_sin": float(np.sin(2 * np.pi * month / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * month / 12.0)),
        "on_stream_lag1": on_stream_lag1,
        "n_producers_lag1": n_producers_lag1,
    }
    for col, hist in ((target, target_hist), ("water", water_hist)):
        for lag in LAGS:
            row[f"{col}_lag{lag}"] = lag_or_nan(hist, lag)
        for win in ROLLS:
            row[f"{col}_rmean{win}"] = shifted_roll_mean(hist, win)
    return row
