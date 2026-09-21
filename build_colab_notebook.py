#!/usr/bin/env python3
"""Сборка короткой тетради для Google Colab."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
NB_PATH = ROOT / "colab.ipynb"

CLONE = r'''# Откройте Runtime → Run all. Код клонирует репозиторий, ставит пакеты и строит все графики.
import os
import sys
import subprocess
from pathlib import Path

REPO = "https://github.com/AndromedaSmart/volve-production-forecast.git"

if Path("src/models.py").exists():
    print("работаем в текущей папке")
elif Path("volve-production-forecast/src/models.py").exists():
    os.chdir("volve-production-forecast")
else:
    subprocess.check_call(["git", "clone", "--depth", "1", REPO, "volve-production-forecast"])
    os.chdir("volve-production-forecast")

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "pandas", "openpyxl", "scikit-learn", "matplotlib", "statsmodels", "scipy",
])
print("каталог:", Path.cwd())
'''

RUN = r'''import warnings
warnings.filterwarnings("ignore")

from IPython.display import display

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, GAS_PROFILE_VARIANT, OIL_PROFILE_VARIANT, WINDOWS
from src.data import load_daily, load_monthly, monthly_field
from src.evaluate import evaluate_windows
from src.metrics import regression_report
from src.models import FieldForecaster, forecast_arps
from src.plots import (
    plot_compare_arps_ml,
    plot_decomposition,
    plot_errors,
    plot_field_history,
    plot_forecast_vs_actual,
    plot_wells,
)
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

FIGURES_DIR.mkdir(exist_ok=True)

daily = load_daily()
monthly = load_monthly()
field = monthly_field(monthly)

plot_field_history(field, WINDOWS, OIL_PROFILE_VARIANT, "oil", "profile_oil_window2.png")
plot_field_history(field, WINDOWS, GAS_PROFILE_VARIANT, "gas", "profile_gas_window4.png")
plot_wells(monthly, "wells_oil_decline.png")

oil = field.set_index("DATE")["oil"].asfreq("MS").fillna(0)
decomp = seasonal_decompose(oil, model="additive", period=12)
plot_decomposition(decomp, "Декомпозиция месячной добычи нефти", "stl_oil.png")
_, adf_p, *_ = adfuller(oil.dropna())

metrics, summary, details = evaluate_windows(field, monthly)

for item in details:
    win, result, actual = item["window"], item["result"], item["actual"]
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

well = monthly.loc[monthly["WELL"] == "15/9-F-12"].copy()
well_field = well.rename(columns={"OIL": "oil", "GAS": "gas", "WATER": "water", "ON_STREAM_HRS": "on_stream"})
well_field["n_producers"] = 1
well_field["n_wells"] = 1
well_field["gi"] = 0.0
well_field["wi"] = 0.0
res = FieldForecaster().fit_predict(well_field, well, "2014-12-01")
future = well_field.set_index("DATE")["oil"].reindex(res.dates).fillna(0)
plot_compare_arps_ml(
    future,
    res.dates,
    forecast_arps(well_field.loc[well_field["DATE"] <= "2014-12-01", "oil"], 6),
    res.oil,
    "15/9-F-12: факт vs Арпс vs ML, 2015 H1",
    "well_f12_2015.png",
)

print("ADF p-value:", round(float(adf_p), 4))
display(summary)
display(metrics)
'''

SHOW = r'''from IPython.display import Image, Markdown, display

ALBUM = [
    ("G1. Месячный профиль нефти, окно 2", "figures/profile_oil_window2.png"),
    ("G2. Месячный профиль газа, окно 4", "figures/profile_gas_window4.png"),
    ("G3. Кривые падения нефти по стволам", "figures/wells_oil_decline.png"),
    ("G4. Декомпозиция ряда нефти", "figures/stl_oil.png"),
    ("G5. Нефть, окно 2: факт, Арпс и модель", "figures/window2_oil_arps_vs_ml.png"),
    ("G6. Скважина 15/9-F-12, 2015 H1", "figures/well_f12_2015.png"),
    ("G7. Нефть, окно 1: факт и прогноз", "figures/w1_oil_fact_pred.png"),
    ("G8. Нефть, окно 1: абсолютная ошибка", "figures/w1_oil_abs.png"),
    ("G9. Нефть, окно 1: относительная ошибка", "figures/w1_oil_rel.png"),
    ("G10. Газ, окно 1: факт и прогноз", "figures/w1_gas_fact_pred.png"),
    ("G11. Газ, окно 1: абсолютная ошибка", "figures/w1_gas_abs.png"),
    ("G12. Газ, окно 1: относительная ошибка", "figures/w1_gas_rel.png"),
    ("G13. Нефть, окно 2: факт и прогноз", "figures/w2_oil_fact_pred.png"),
    ("G14. Нефть, окно 2: абсолютная ошибка", "figures/w2_oil_abs.png"),
    ("G15. Нефть, окно 2: относительная ошибка", "figures/w2_oil_rel.png"),
    ("G16. Газ, окно 2: факт и прогноз", "figures/w2_gas_fact_pred.png"),
    ("G17. Газ, окно 2: абсолютная ошибка", "figures/w2_gas_abs.png"),
    ("G18. Газ, окно 2: относительная ошибка", "figures/w2_gas_rel.png"),
    ("G19. Нефть, окно 3: факт и прогноз", "figures/w3_oil_fact_pred.png"),
    ("G20. Нефть, окно 3: абсолютная ошибка", "figures/w3_oil_abs.png"),
    ("G21. Нефть, окно 3: относительная ошибка", "figures/w3_oil_rel.png"),
    ("G22. Газ, окно 3: факт и прогноз", "figures/w3_gas_fact_pred.png"),
    ("G23. Газ, окно 3: абсолютная ошибка", "figures/w3_gas_abs.png"),
    ("G24. Газ, окно 3: относительная ошибка", "figures/w3_gas_rel.png"),
    ("G25. Нефть, окно 4: факт и прогноз", "figures/w4_oil_fact_pred.png"),
    ("G26. Нефть, окно 4: абсолютная ошибка", "figures/w4_oil_abs.png"),
    ("G27. Нефть, окно 4: относительная ошибка", "figures/w4_oil_rel.png"),
    ("G28. Газ, окно 4: факт и прогноз", "figures/w4_gas_fact_pred.png"),
    ("G29. Газ, окно 4: абсолютная ошибка", "figures/w4_gas_abs.png"),
    ("G30. Газ, окно 4: относительная ошибка", "figures/w4_gas_rel.png"),
    ("G31. Нефть, окно 5: факт и прогноз", "figures/w5_oil_fact_pred.png"),
    ("G32. Нефть, окно 5: абсолютная ошибка", "figures/w5_oil_abs.png"),
    ("G33. Нефть, окно 5: относительная ошибка", "figures/w5_oil_rel.png"),
    ("G34. Газ, окно 5: факт и прогноз", "figures/w5_gas_fact_pred.png"),
    ("G35. Газ, окно 5: абсолютная ошибка", "figures/w5_gas_abs.png"),
    ("G36. Газ, окно 5: относительная ошибка", "figures/w5_gas_rel.png"),
]

for title, path in ALBUM:
    display(Markdown(f"### {title}"))
    display(Image(filename=path))
'''


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Volve: прогноз добычи нефти и газа

Краткая тетрадка для Google Colab. Данные и код берутся из репозитория, затем считаются метрики и выводятся **все 36 графиков**.

1. Откройте **Runtime → Run all**.
2. Дождитесь клонирования репозитория и расчёта (обычно 1–2 минуты).
3. Ниже появятся таблицы MAE / MAPE / WAPE и рисунки G1–G36.

Репозиторий: https://github.com/AndromedaSmart/volve-production-forecast
"""
        ),
        nbf.v4.new_code_cell(CLONE),
        nbf.v4.new_code_cell(RUN),
        nbf.v4.new_code_cell(SHOW),
    ]
    nbf.write(nb, NB_PATH)
    print("записано", NB_PATH)


if __name__ == "__main__":
    main()
