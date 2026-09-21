"""Модели прогноза: Арпс (baseline) и экстраполяция недавнего режима."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import HORIZON_MONTHS
from .features import feature_columns, make_supervised, recursive_feature_row


RECENT_MONTHS = 18


def _arps_exp(t: np.ndarray, qi: float, di: float) -> np.ndarray:
    return qi * np.exp(-di * t)


def fit_arps(values: np.ndarray) -> Tuple[str, np.ndarray]:
    y = np.clip(np.asarray(values, dtype=float), 0, None)
    t = np.arange(len(y), dtype=float)
    if np.nanmax(y) <= 0 or len(y) < 4:
        return "persist", np.array([float(np.nanmean(y) if len(y) else 0.0)])
    qi0 = max(float(np.nanmedian(y[: max(2, len(y) // 5)])), 1.0)
    try:
        popt, _ = curve_fit(
            _arps_exp,
            t,
            y,
            p0=(qi0, 0.02),
            bounds=([0.0, -0.15], [qi0 * 4 + 1.0, 0.6]),
            maxfev=2000,
        )
        return "exp", popt
    except Exception:
        return "persist", np.array([float(np.nanmedian(y))])


def predict_arps(kind: str, params: np.ndarray, t: np.ndarray) -> np.ndarray:
    if kind == "exp":
        return np.clip(_arps_exp(t, *params), 0, None)
    return np.full(np.shape(t), float(params[0]), dtype=float)


def _recent(series: pd.Series, n: int = RECENT_MONTHS) -> pd.Series:
    return series.dropna().iloc[-n:]


def forecast_arps(train: pd.Series, horizon: int, recent: int = RECENT_MONTHS) -> np.ndarray:
    y = _recent(train.fillna(0), recent)
    kind, params = fit_arps(y.to_numpy())
    t_future = np.arange(len(y), len(y) + horizon, dtype=float)
    return predict_arps(kind, params, t_future)


def _normal_mask(hours: pd.Series) -> pd.Series:
    med = hours.median()
    if pd.isna(med) or med <= 0:
        return hours >= 0
    return hours >= 0.45 * med


def forecast_robust_level(field_hist: pd.DataFrame, target: str, horizon: int) -> np.ndarray:
    """Медианный уровень по месяцам без явного простоя."""
    tail = field_hist.iloc[-12:]
    mask = _normal_mask(tail["on_stream"])
    vals = tail.loc[mask, target]
    if vals.empty:
        vals = tail[target]
    level = float(vals.median()) if len(vals) else 0.0
    return np.full(horizon, max(level, 0.0), dtype=float)


def forecast_rate_decline(field_hist: pd.DataFrame, target: str, horizon: int) -> np.ndarray:
    """Экспонента по удельной добыче (объём / часы) на нормальных месяцах."""
    tail = field_hist.iloc[-RECENT_MONTHS:]
    mask = _normal_mask(tail["on_stream"])
    work = tail.loc[mask]
    if len(work) < 4:
        work = tail
    rate = (work[target] / work["on_stream"].clip(lower=1.0)).to_numpy(dtype=float)
    kind, params = fit_arps(np.clip(rate, 0, None))
    typical_hours = float(work["on_stream"].median())
    t_future = np.arange(len(rate), len(rate) + horizon, dtype=float)
    rate_hat = predict_arps(kind, params, t_future)
    return np.clip(rate_hat * typical_hours, 0, None)


def forecast_holt(train: pd.Series, horizon: int) -> Optional[np.ndarray]:
    y = _recent(train.fillna(0).astype(float), 24)
    if len(y) < 10:
        return None
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        model = ExponentialSmoothing(
            y, trend="add", damped_trend=True, seasonal=None, initialization_method="estimated"
        )
        fit = model.fit(optimized=True)
        return np.clip(np.asarray(fit.forecast(horizon)), 0, None)
    except Exception:
        return None


def _ridge_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=5.0)),
        ]
    )


def _recursive_ml(train_field: pd.DataFrame, target: str, horizon: int) -> Optional[np.ndarray]:
    hist = train_field.sort_values("DATE").reset_index(drop=True)
    supervised = make_supervised(hist, target)
    feats = feature_columns(target)
    work = supervised.dropna(subset=feats + [target])
    if len(work) < 10:
        return None
    model = _ridge_pipeline()
    model.fit(work[feats], work[target])

    target_hist = hist[target].to_numpy(dtype=float)
    water_hist = hist["water"].to_numpy(dtype=float)
    last_month = int(hist["DATE"].dt.month.iloc[-1])
    last_on_stream = float(hist["on_stream"].iloc[-1])
    median_on_stream = float(np.nanmedian(hist["on_stream"].iloc[-6:]))
    last_producers = float(hist["n_producers"].iloc[-1])
    n0 = len(hist)
    preds = []
    for step in range(horizon):
        month = ((last_month + step) % 12) + 1
        on_stream_lag1 = last_on_stream if step == 0 else median_on_stream
        row = recursive_feature_row(
            target=target,
            t=float(n0 + step),
            month=month,
            on_stream_lag1=on_stream_lag1,
            n_producers_lag1=last_producers,
            target_hist=target_hist,
            water_hist=water_hist,
        )
        x = pd.DataFrame([row], columns=feats).fillna(0.0)
        yhat = max(float(model.predict(x)[0]), 0.0)
        preds.append(yhat)
        target_hist = np.append(target_hist, yhat)
        water_hist = np.append(water_hist, water_hist[-1] if len(water_hist) else 0.0)
    return np.asarray(preds, dtype=float)


def _well_arps_sum(monthly: pd.DataFrame, cutoff: pd.Timestamp, horizon: int, target_col: str) -> np.ndarray:
    total = np.zeros(horizon, dtype=float)
    for _, part in monthly.groupby("WELL", sort=False):
        hist = part.loc[part["DATE"] <= cutoff]
        series = hist[target_col]
        if series.fillna(0).sum() <= 0:
            continue
        total += forecast_arps(series, horizon)
    return np.clip(total, 0, None)


@dataclass
class ForecastResult:
    dates: pd.DatetimeIndex
    oil: np.ndarray
    gas: np.ndarray
    oil_lo: np.ndarray
    oil_hi: np.ndarray
    gas_lo: np.ndarray
    gas_hi: np.ndarray
    model_oil: str
    model_gas: str


class FieldForecaster:
    """Модель кейса: история до даты -> подготовка -> обучение -> прогноз на 6 мес."""

    def __init__(self, horizon: int = HORIZON_MONTHS):
        self.horizon = horizon
        self.chosen_ = {}

    def _candidates(self, field_hist: pd.DataFrame, monthly_hist: pd.DataFrame, target: str) -> Dict[str, np.ndarray]:
        series = field_hist[target]
        cutoff = field_hist["DATE"].max()
        target_col = "OIL" if target == "oil" else "GAS"
        cands = {
            "robust_level": forecast_robust_level(field_hist, target, self.horizon),
            "rate_decline": forecast_rate_decline(field_hist, target, self.horizon),
            "arps_field": forecast_arps(series, self.horizon),
            "arps_wells": _well_arps_sum(monthly_hist, cutoff, self.horizon, target_col),
        }
        ml = _recursive_ml(field_hist, target, self.horizon)
        if ml is not None:
            cands["ridge_lags"] = ml
        cands["blend"] = 0.5 * cands["robust_level"] + 0.5 * cands["arps_field"]
        return cands

    def _fit_one(self, field_hist: pd.DataFrame, target: str) -> Tuple[str, np.ndarray, float]:
        hist = field_hist
        if len(hist) >= 8:
            hours = hist["on_stream"]
            recent_med = hours.iloc[-12:].median()
            if recent_med and hours.iloc[-1] < 0.45 * recent_med:
                hist = hist.iloc[:-1]
        n_pos = int((hist[target] > 0).sum())
        if n_pos >= 30:
            ml = _recursive_ml(hist, target, self.horizon)
            if ml is not None:
                return "ridge_lags", ml, float("nan")
        blend = 0.5 * forecast_robust_level(hist, target, self.horizon) + 0.5 * forecast_arps(
            hist[target], self.horizon
        )
        return "blend", blend, float("nan")

    def fit_predict(self, field: pd.DataFrame, monthly: pd.DataFrame, cutoff: str) -> ForecastResult:
        cutoff_ts = pd.Timestamp(cutoff)
        field_hist = field.loc[field["DATE"] <= cutoff_ts]
        if field_hist.empty:
            raise ValueError("Пустая история до даты отсечения")

        oil_name, oil, oil_val = self._fit_one(field_hist, "oil")
        gas_name, gas, gas_val = self._fit_one(field_hist, "gas")
        self.chosen_ = {
            "oil": oil_name,
            "gas": gas_name,
            "oil_val_wape": oil_val,
            "gas_val_wape": gas_val,
        }
        oil_sigma = _residual_sigma(field_hist["oil"])
        gas_sigma = _residual_sigma(field_hist["gas"])
        steps = np.arange(1, self.horizon + 1)
        oil_lo, oil_hi = _interval(oil, oil_sigma, steps)
        gas_lo, gas_hi = _interval(gas, gas_sigma, steps)
        dates = pd.date_range(cutoff_ts + pd.offsets.MonthBegin(1), periods=self.horizon, freq="MS")
        return ForecastResult(
            dates=dates,
            oil=np.clip(oil, 0, None),
            gas=np.clip(gas, 0, None),
            oil_lo=oil_lo,
            oil_hi=oil_hi,
            gas_lo=gas_lo,
            gas_hi=gas_hi,
            model_oil=oil_name,
            model_gas=gas_name,
        )


def _residual_sigma(series: pd.Series) -> float:
    y = series.fillna(0).to_numpy(dtype=float)
    if len(y) < 4:
        return float(np.std(y) if len(y) else 0.0)
    return float(np.std(np.diff(y), ddof=1))


def _interval(pred: np.ndarray, sigma: float, steps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    width = 1.64 * sigma * np.sqrt(steps)
    return np.clip(pred - width, 0, None), pred + width
