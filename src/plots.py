"""Графики для отчёта и тетради."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib

if matplotlib.get_backend().lower() != "agg":
    try:
        matplotlib.use("Agg", force=False)
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIGURES_DIR, FLUID_LABELS
from .models import ForecastResult

plt.rcParams.update(
    {
        "figure.figsize": (11, 4.6),
        "axes.unicode_minus": False,
        "savefig.dpi": 110,
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_field_history(field: pd.DataFrame, windows, highlight_variant: int, fluid: str, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(field["DATE"], field[fluid], color="#1f4e79", lw=2.2, label="Факт")
    win = [w for w in windows if w.variant == highlight_variant][0]
    ax.axvspan(pd.Timestamp(win.train_start), pd.Timestamp(win.train_end), color="#8fb8d6", alpha=0.25, label="Обучение")
    ax.axvspan(pd.Timestamp(win.forecast_start), pd.Timestamp(win.forecast_end), color="#f4b183", alpha=0.45, label="Прогноз")
    ax.set_title(f"Месячный профиль {FLUID_LABELS[fluid].split(',')[0]} по месторождению, вариант {highlight_variant}")
    ax.set_ylabel(FLUID_LABELS[fluid])
    ax.set_xlabel("Месяц")
    ax.legend(loc="upper right")
    return _save(fig, name)


def plot_wells(monthly: pd.DataFrame, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    wide = monthly.pivot_table(index="DATE", columns="WELL", values="OIL", aggfunc="sum")
    ax.plot(wide.index, wide.to_numpy(), lw=1.8)
    ax.set_title("Кривые падения добычи нефти по стволам")
    ax.set_ylabel("Нефть, ст. м³ / мес.")
    ax.legend(wide.columns, ncol=2, fontsize=10)
    return _save(fig, name)


def plot_forecast_vs_actual(
    actual: pd.DataFrame,
    result: ForecastResult,
    fluid: str,
    variant: int,
    name: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    pred = result.oil if fluid == "oil" else result.gas
    lo = result.oil_lo if fluid == "oil" else result.gas_lo
    hi = result.oil_hi if fluid == "oil" else result.gas_hi
    ax.plot(actual["DATE"], actual[fluid], marker="o", color="#1f4e79", lw=2, label="Факт")
    ax.plot(result.dates, pred, marker="s", color="#c45911", lw=2, label="Прогноз")
    ax.fill_between(result.dates, lo, hi, color="#c45911", alpha=0.18, label="Интервал ~90%")
    ax.set_title(f"Факт vs прогноз, {fluid}, окно {variant}")
    ax.set_ylabel(FLUID_LABELS[fluid])
    ax.legend()
    return _save(fig, name)


def plot_errors(
    actual: pd.DataFrame, pred: np.ndarray, dates: pd.DatetimeIndex, fluid: str, variant: int, prefix: str
) -> Tuple[Path, Path]:
    y = actual[fluid].to_numpy()
    abs_err = np.abs(y - pred)
    rel_err = np.abs(y - pred) / np.maximum(np.abs(y), 1.0) * 100.0

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(dates, abs_err, width=20, color="#2e75b6")
    ax.set_title(f"Абсолютная ошибка, {fluid}, окно {variant}")
    ax.set_ylabel("ст. м³")
    p1 = _save(fig, f"{prefix}_abs.png")

    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar(dates, rel_err, width=20, color="#c45911")
    ax.set_title(f"Относительная ошибка, {fluid}, окно {variant}")
    ax.set_ylabel("%")
    p2 = _save(fig, f"{prefix}_rel.png")
    return p1, p2


def plot_compare_arps_ml(
    actual: pd.Series,
    dates: pd.DatetimeIndex,
    arps: np.ndarray,
    ml: np.ndarray,
    title: str,
    name: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(dates, actual.to_numpy(), marker="o", lw=2, label="Факт")
    ax.plot(dates, arps, marker="^", lw=2, label="Арпс")
    ax.plot(dates, ml, marker="s", lw=2, label="ML")
    ax.set_title(title)
    ax.legend()
    return _save(fig, name)


def plot_decomposition(decomp, title: str, name: str) -> Path:
    fig = decomp.plot()
    fig.set_size_inches(11, 7)
    fig.suptitle(title, y=1.02)
    return _save(fig, name)
