"""Прогон контрольных окон и сбор метрик."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from .config import WINDOWS
from .data import slice_history
from .metrics import regression_report, rms_wape
from .models import FieldForecaster, ForecastResult, forecast_arps


def _result_from_experiment(exp, win) -> Tuple[ForecastResult, dict]:
    """Итоговая модель v2: подход, выбранный внутренней валидацией в каждом окне."""
    import numpy as np

    out, chosen = {}, {}
    for fluid in ("oil", "gas"):
        a = exp.auto.loc[(exp.auto["variant"] == win.variant) & (exp.auto["fluid"] == fluid)].iloc[0]
        pred = np.asarray(exp.preds[(win.variant, fluid, "auto", "auto", "auto")], float)
        # Интервал ~90%: относительная ошибка внутренней валидации, растущая с горизонтом
        rel = float(a["inner_wape"]) / 100.0 * 1.64 * np.sqrt(np.arange(1, len(pred) + 1) / 3.5)
        out[fluid] = (pred, np.clip(pred * (1 - rel), 0, None), pred * (1 + rel))
        chosen[fluid] = f"{a['model']} / {a['mode']} / {a['route']} {a['params']}"
    res = ForecastResult(
        dates=exp.dates[win.variant], oil=out["oil"][0], gas=out["gas"][0],
        oil_lo=out["oil"][1], oil_hi=out["oil"][2], gas_lo=out["gas"][1], gas_hi=out["gas"][2],
        model_oil=chosen["oil"], model_gas=chosen["gas"],
    )
    return res, chosen


def evaluate_windows(field: pd.DataFrame, monthly: pd.DataFrame, exp=None) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """exp=None — модель v1 (FieldForecaster); exp — итоговая модель v2 из experiments."""
    rows: List[Dict] = []
    details = []
    for win in WINDOWS:
        forecaster = FieldForecaster()
        if exp is not None:
            result, forecaster.chosen_ = _result_from_experiment(exp, win)
        else:
            result = forecaster.fit_predict(field, monthly, win.train_end)
        actual = slice_history(field, win.forecast_start, win.forecast_end)
        actual = actual.set_index("DATE").reindex(result.dates).reset_index()
        actual = actual.rename(columns={"index": "DATE"})
        actual[["oil", "gas"]] = actual[["oil", "gas"]].fillna(0.0)
        for fluid in ("oil", "gas"):
            pred = result.oil if fluid == "oil" else result.gas
            y = actual[fluid]
            report = regression_report(y, pred)
            rows.append(
                {
                    "variant": win.variant,
                    "fluid": fluid,
                    "train_end": win.train_end,
                    "forecast_start": win.forecast_start,
                    "forecast_end": win.forecast_end,
                    "model": result.model_oil if fluid == "oil" else result.model_gas,
                    **report,
                }
            )
        details.append({"window": win, "result": result, "actual": actual, "chosen": forecaster.chosen_})
    metrics = pd.DataFrame(rows)
    summary = []
    for fluid in ("oil", "gas"):
        part = metrics.loc[metrics["fluid"] == fluid, "WAPE"]
        summary.append({"fluid": fluid, "RMS_WAPE": rms_wape(part), "mean_WAPE": float(part.mean())})
    return metrics, pd.DataFrame(summary), details


def arps_on_window(field: pd.DataFrame, train_end: str, forecast_start: str, forecast_end: str, fluid: str):
    hist = field.loc[field["DATE"] <= pd.Timestamp(train_end), fluid]
    actual = slice_history(field, forecast_start, forecast_end)
    pred = forecast_arps(hist, len(actual))
    return actual[fluid], pred, actual["DATE"]
