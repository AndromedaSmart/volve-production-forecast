#!/usr/bin/env python3
"""Пакетный прогон: метрики, графики, короткая записка."""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from src.config import FIGURES_DIR, GAS_PROFILE_VARIANT, OIL_PROFILE_VARIANT, REPORTS_DIR, WINDOWS
import argparse

from src.data import daily_quality_report, load_daily, load_monthly, monthly_field, well_summary
from src.evaluate import evaluate_windows
from src.experiments import load_experiment, run_experiments, save_experiment, summary_table
from src import plots_v2
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="пересчитать сравнение моделей (долго из-за PyPOTS)")
    ap.add_argument("--no-deep", action="store_true", help="без Transformer/CSDI при пересчёте")
    args = ap.parse_args()
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

    daily_quality_report(daily).to_csv(REPORTS_DIR / "data_quality.csv", index=False)
    if args.recompute or not (REPORTS_DIR / "experiments_predictions.csv").exists():
        exp = run_experiments(field, monthly, with_deep=not args.no_deep, verbose=True)
        save_experiment(exp, REPORTS_DIR)
    exp = load_experiment(REPORTS_DIR)
    plots_v2.plot_pipeline()
    plots_v2.plot_preprocessing(daily, field)
    plots_v2.plot_rate_series(field)
    plots_v2.plot_gor(field)
    plots_v2.plot_model_comparison(exp.rows, exp.auto)
    plots_v2.plot_heatmap(exp.rows, "oil", "heatmap_oil.png")
    plots_v2.plot_heatmap(exp.rows, "gas", "heatmap_gas.png")
    plots_v2.plot_leakage(exp.leak)
    plots_v2.plot_inner_vs_test(exp.rows)
    plots_v2.plot_tuning_curves(exp.tuning)
    plots_v2.plot_window_models(exp, "oil", "windows_models_oil.png")
    plots_v2.plot_window_models(exp, "gas", "windows_models_gas.png")
    plots_v2.plot_gas_routes(exp)
    plt.close("all")

    metrics, summary, details = evaluate_windows(field, monthly, exp)
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

    well = monthly.loc[monthly["WELL"] == "15/9-F-12"].copy()
    wf = well.rename(columns={"OIL": "oil", "GAS": "gas", "WATER": "water", "ON_STREAM_HRS": "on_stream"})
    wf = wf.assign(n_producers=1, n_wells=1, gi=0.0, wi=0.0, prod_hours=wf["on_stream"]).reset_index(drop=True)
    from src.experiments import forecast_fluid, tune

    hist_w = wf.loc[wf["DATE"] <= "2014-12-01"]
    fut_w = wf.set_index("DATE").reindex(pd.date_range("2015-01-01", periods=6, freq="MS"))
    params, _, _ = tune(wf, hist_w, "oil", "rate", "lgbm")
    pred_w = forecast_fluid(hist_w, fut_w["prod_hours"].fillna(0).to_numpy(), "oil", ("rate", "lgbm", params, "direct"))
    plot_compare_arps_ml(
        fut_w["oil"].fillna(0), fut_w.index, forecast_arps(hist_w["oil"], 6), pred_w,
        "15/9-F-12: факт vs Арпс vs LightGBM (удельный дебит), 2015 H1", "well_f12_2015.png",
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
