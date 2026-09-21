"""Прогон контрольных окон и сбор метрик."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from .config import WINDOWS
from .data import slice_history
from .metrics import regression_report, rms_wape
from .models import FieldForecaster, forecast_arps


def evaluate_windows(field: pd.DataFrame, monthly: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    rows: List[Dict] = []
    details = []
    for win in WINDOWS:
        forecaster = FieldForecaster()
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
