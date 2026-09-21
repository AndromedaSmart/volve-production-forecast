#!/usr/bin/env python3
"""Пакетный прогон: метрики, графики, короткая записка."""

from __future__ import annotations

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from src.config import FIGURES_DIR, GAS_PROFILE_VARIANT, OIL_PROFILE_VARIANT, REPORTS_DIR, WINDOWS
from src.data import load_daily, load_monthly, monthly_field, well_summary
from src.evaluate import evaluate_windows
from src.models import forecast_arps
from src.plots import (
    plot_compare_arps_ml,
    plot_decomposition,
    plot_errors,
    plot_field_history,
    plot_forecast_vs_actual,
    plot_wells,
)

warnings.filterwarnings("ignore")


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    daily = load_daily()
    monthly = load_monthly()
    field = monthly_field(monthly)
    wells = well_summary(daily, monthly)
    wells.to_csv(REPORTS_DIR / "wells.csv", index=False)

    plot_field_history(field, WINDOWS, OIL_PROFILE_VARIANT, "oil", "profile_oil_window2.png")
    plot_field_history(field, WINDOWS, GAS_PROFILE_VARIANT, "gas", "profile_gas_window4.png")
    plot_wells(monthly, "wells_oil_decline.png")
    plt.close("all")

    oil = field.set_index("DATE")["oil"].asfreq("MS").fillna(0)
    decomp = seasonal_decompose(oil, model="additive", period=12)
    plot_decomposition(decomp, "Декомпозиция месячной добычи нефти", "stl_oil.png")
    plt.close("all")
    adf_stat, adf_p, *_ = adfuller(oil.dropna())

    metrics, summary, details = evaluate_windows(field, monthly)
    metrics.to_csv(REPORTS_DIR / "metrics_windows.csv", index=False)
    summary.to_csv(REPORTS_DIR / "metrics_summary.csv", index=False)

    for item in details:
        win = item["window"]
        result = item["result"]
        actual = item["actual"]
        for fluid in ("oil", "gas"):
            plot_forecast_vs_actual(actual, result, fluid, win.variant, f"w{win.variant}_{fluid}_fact_pred.png")
            pred = result.oil if fluid == "oil" else result.gas
            plot_errors(actual, pred, result.dates, fluid, win.variant, f"w{win.variant}_{fluid}")
            if win.variant == 2 and fluid == "oil":
                arps = forecast_arps(field.loc[field["DATE"] <= win.train_end, "oil"], 6)
                plot_compare_arps_ml(
                    actual.set_index("DATE")["oil"].reindex(result.dates).fillna(0),
                    result.dates,
                    arps,
                    result.oil,
                    "Окно 2, нефть: факт vs Арпс vs модель",
                    "window2_oil_arps_vs_ml.png",
                )
        plt.close("all")

    oil_rms = float(summary.loc[summary["fluid"] == "oil", "RMS_WAPE"].iloc[0])
    gas_rms = float(summary.loc[summary["fluid"] == "gas", "RMS_WAPE"].iloc[0])

    note = f"""# Пояснительная записка

Прогнозирование месячной добычи нефти и газа по месторождению Volve.

## Результат

- RMS WAPE по нефти: **{oil_rms:.2f}%**
- RMS WAPE по газу: **{gas_rms:.2f}%**

Метрики по окнам сохранены в `reports/metrics_windows.csv`.

Тест Дики-Фуллера на полном ряде нефти: статистика {adf_stat:.3f}, p-value {adf_p:.4f}.

## Применение

Модель принимает историю до даты отсечения и выдаёт 6-месячный профиль нефти и газа по месторождению. Это пригодно для чернового оперативного плана добычи УВС, когда нет детальной гидродинамической модели или она слишком тяжёлая для ежемесячного пересчёта.

## Слабые стороны

- Не знает о плане ввода новых скважин и ГТМ.
- На короткой истории (окно 1) сезонные модели неустойчивы.
- Рекурсивный 6-шаговый прогноз накапливает ошибку.
- Кривая Арпса не описывает рост добычи после подключения стволов.

Для промышленного внедрения нужны план бурения, режимы штуцера, простои и регулярная перекалибровка на свежих фактических данных.
"""
    (REPORTS_DIR / "NOTE.md").write_text(note, encoding="utf-8")
    print(metrics.to_string(index=False))
    print(summary.to_string(index=False))
    print(f"ADF p-value={adf_p:.4f}")
    print("готово")


if __name__ == "__main__":
    main()
