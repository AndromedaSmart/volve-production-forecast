"""Контрольные окна и константы кейса."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "data.xlsx"
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports"

FLUIDS = ("oil", "gas")
FLUID_LABELS = {
    "oil": "нефть, ст. м³",
    "gas": "газ, ст. м³",
}
HORIZON_MONTHS = 6


@dataclass(frozen=True)
class ControlWindow:
    variant: int
    train_start: str
    train_end: str
    forecast_start: str
    forecast_end: str


# Варианты из task.pdf: обучение строго до даты отсечения, прогноз на следующие 6 месяцев.
WINDOWS = (
    ControlWindow(1, "2007-09-01", "2009-12-01", "2010-01-01", "2010-06-01"),
    ControlWindow(2, "2007-09-01", "2011-12-01", "2012-01-01", "2012-06-01"),
    ControlWindow(3, "2007-09-01", "2013-12-01", "2014-01-01", "2014-06-01"),
    ControlWindow(4, "2007-09-01", "2014-12-01", "2015-01-01", "2015-06-01"),
    ControlWindow(5, "2007-09-01", "2015-12-01", "2016-01-01", "2016-06-01"),
)

OIL_PROFILE_VARIANT = 2
GAS_PROFILE_VARIANT = 4
