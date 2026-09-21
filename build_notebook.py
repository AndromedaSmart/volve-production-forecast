#!/usr/bin/env python3
"""Сборка Jupyter-тетрадки решения."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
NB_PATH = ROOT / "solution.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    cells = []

    cells.append(md("""# Прогнозирование объёмов добычи нефти и газа на месторождении Volve

Решение кейса: построить математическую модель, которая по истории добычи до заданной даты выдаёт **месячный прогноз нефти и газа на 6 месяцев** по месторождению.

Датасет: производственные данные Volve (Норвегия, 2007–2016), файл `data/data.xlsx`.

Модель проверяется на пяти фиксированных контрольных окнах. Итоговая метрика — **RMS WAPE** отдельно по нефти и по газу.
"""))

    cells.append(md("""## 1. Постановка и ограничения

**Что прогнозируем.** Не суточный дебит отдельной скважины, а **месячный профиль месторождения**: сумма добычи нефти и газа по всем стволам.

**Как учимся.** История строго до даты отсечения. Тест всегда позже обучения, без перемешивания во времени.

**Контрольные окна**

| Вариант | Обучение | Прогноз |
|---|---|---|
| 1 | 2007-09-01 — 2009-12-01 | 2010-01 — 2010-06 |
| 2 | 2007-09-01 — 2011-12-01 | 2012-01 — 2012-06 |
| 3 | 2007-09-01 — 2013-12-01 | 2014-01 — 2014-06 |
| 4 | 2007-09-01 — 2014-12-01 | 2015-01 — 2015-06 |
| 5 | 2007-09-01 — 2015-12-01 | 2016-01 — 2016-06 |

**Состав модели**

1. Принимает историю до даты отсечения.
2. Готовит месячный ряд месторождения и лаговые признаки.
3. На хвосте обучения выбирает кандидата: кривая Арпса, сумма скважинных Арпсов, Холт, Ridge с лагами, смесь.
4. Строит рекурсивный прогноз на 6 месяцев и интервал неопределённости.

Код разбит на пакет `src/`: загрузка, признаки, модели, метрики, графики.
"""))

    cells.append(code("""%matplotlib inline
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR, GAS_PROFILE_VARIANT, OIL_PROFILE_VARIANT, WINDOWS
from src.data import load_daily, load_monthly, monthly_field, well_summary
from src.evaluate import arps_on_window, evaluate_windows
from src.features import make_supervised
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

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
FIGURES_DIR.mkdir(exist_ok=True)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
print("рабочая папка:", ROOT)
"""))

    cells.append(md("""## 2. Загрузка и обзор данных

В Excel три листа:

- `Daily Production Data` — суточные замеры, ~15 600 строк, 7 стволов, 24 колонки;
- `Monthly Production Data` — месячная агрегация, ~527 строк;
- `Описание колонок` — словарь признаков.

Для месячного прогноза по месторождению основной ряд берём с месячного листа и дополнительно сверяем со суточным. Суточные данные нужны для анализа простоев, давлений и типов скважин.
"""))

    cells.append(code("""daily = load_daily()
monthly = load_monthly()
field = monthly_field(monthly)

print("суточные строки:", len(daily), "период:", daily["DATEPRD"].min().date(), "—", daily["DATEPRD"].max().date())
print("месячные строки:", len(monthly), "стволов:", monthly["WELL"].nunique())
print("месяцев месторождения:", len(field), field["DATE"].min().date(), "—", field["DATE"].max().date())
display(well_summary(daily, monthly))
"""))

    cells.append(md("""### Что видно по стволам

- `15/9-F-12` и `15/9-F-14` — основные добывающие скважины с 2008 года.
- `15/9-F-11` подключается в 2013, `F-1 C` и `F-15 D` — в 2014. Это важно: в окне 3 в прогнозный период входят **новые скважины**, которых не было в обучении.
- `15/9-F-4` — нагнетательная, нефти и газа не даёт.
- `15/9-F-5` меняет роль: и закачка, и короткие периоды добычи.

Пропуски в забойных датчиках частые, но целевые объёмы на месячном листе заполнены. Периоды простоя видны по `ON_STREAM_HRS = 0` и нулевой добыче: их не интерполируем как «потерянный дебит», а оставляем нулями — скважина реально не работала.
"""))

    cells.append(code("""print("пропуски в суточных целевых колонках, %")
miss = daily[["BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL", "ON_STREAM_HRS", "AVG_WHP_P", "AVG_DOWNHOLE_PRESSURE"]].isna().mean() * 100
display(miss.to_frame("pct"))

idle = daily.loc[daily["FLOW_KIND"] == "production"]
share_idle = (idle["ON_STREAM_HRS"].fillna(0) <= 0).mean() * 100
print(f"доля суток с нулевым временем работы у добывающих стволов: {share_idle:.1f}%")
display(field.head())
"""))

    cells.append(code("""p = plot_field_history(field, WINDOWS, OIL_PROFILE_VARIANT, "oil", "profile_oil_window2.png")
print("график нефти, вариант 2:", p)
p = plot_field_history(field, WINDOWS, GAS_PROFILE_VARIANT, "gas", "profile_gas_window4.png")
print("график газа, вариант 4:", p)
p = plot_wells(monthly, "wells_oil_decline.png")
print("кривые скважин:", p)
"""))

    cells.append(md("""## 3. Анализ временного ряда месторождения

Декомпозиция показывает тренд ввода скважин и последующее падение. Тест Дики-Фуллера проверяет стационарность: при нестационарности лаговая ML-модель опирается на разности косвенно, через лаги и линейный тренд `t`, а кривая Арпса явно описывает падение.
"""))

    cells.append(code("""oil = field.set_index("DATE")["oil"].asfreq("MS").fillna(0)
decomp = seasonal_decompose(oil, model="additive", period=12)
plot_decomposition(decomp, "Декомпозиция месячной добычи нефти", "stl_oil.png")

adf_stat, adf_p, *_ = adfuller(oil.dropna())
print(f"ADF статистика = {adf_stat:.3f}, p-value = {adf_p:.4f}")
if adf_p < 0.05:
    print("Ряд нефти на полном периоде выглядит стационарным на уровне 5%.")
else:
    print("Ряд нефти нестационарен: есть тренд ввода мощностей и последующее падение.")
"""))

    cells.append(md("""## 4. Подготовка признаков

Для каждой даты месторождения строим:

- календарь: номер месяца, sin/cos сезонности, порядковый индекс `t`;
- лаги добычи 1, 2, 3 и 6 месяцев;
- скользящее среднее за 3 и 6 месяцев;
- лаги воды, времени работы и числа добывающих стволов.

Разбиение только хронологическое. Рекурсивный прогноз на 6 шагов подставляет уже полученные значения обратно в лаги.
"""))

    cells.append(code("""supervised = make_supervised(field, "oil").dropna()
print("строк для обучения на полном ряде:", len(supervised))
display(supervised[["DATE", "oil", "oil_lag1", "oil_lag3", "oil_lag6", "oil_rmean3", "month_sin"]].head(8))
"""))

    cells.append(md("""## 5. Модели

**Baseline — кривая Арпса.** Экспонента \( q(t) = q_i e^{-d_i t} \) и гипербола \( q(t) = q_i (1 + b d_i t)^{-1/b} \). Подгоняем на истории месторождения и отдельно по стволам, затем суммируем скважинные кривые.

**ML.** Ridge на стандартизованных лагах: устойчив на коротких окнах (в варианте 1 всего около двух лет добычи).

**Холт.** Линейный тренд без сезонности — запасной кандидат, если ряд достаточно длинный.

**Смесь.** Среднее устойчивых кандидатов. На каждом окне лучший кандидат выбирается по WAPE на последних 6 месяцах **обучения**, без подглядывания в контрольный прогноз.
"""))

    cells.append(code("""# Пример: окно 2, нефть — факт, Арпс и выбранная модель рядом
win = WINDOWS[1]
forecaster = FieldForecaster()
result = forecaster.fit_predict(field, monthly, win.train_end)
actual = field.loc[(field["DATE"] >= win.forecast_start) & (field["DATE"] <= win.forecast_end)]
arps_pred = forecast_arps(field.loc[field["DATE"] <= win.train_end, "oil"], 6)
plot_compare_arps_ml(
    actual.set_index("DATE")["oil"].reindex(result.dates).fillna(0),
    result.dates,
    arps_pred,
    result.oil,
    "Окно 2, нефть: факт vs Арпс vs выбранная модель",
    "window2_oil_arps_vs_ml.png",
)
print("выбранные семейства на окне 2:", forecaster.chosen_)
print("прогноз нефти:", np.round(result.oil, 1))
print("прогноз газа:", np.round(result.gas, 1))
"""))

    cells.append(md("""## 6. Оценка на пяти контрольных окнах

Для каждого варианта и каждого флюида строим:

1. график факта и прогноза;
2. абсолютную ошибку по месяцам;
3. относительную ошибку по месяцам;
4. MAE, MAPE, WAPE (плюс RMSE и R² для сопоставления с классической постановкой задачи 2).

По всем окнам считаем

\[
\mathrm{RMS\,WAPE} = \sqrt{\frac{1}{N}\sum_i \mathrm{WAPE}_i^2}.
\]
"""))

    cells.append(code("""metrics, summary, details = evaluate_windows(field, monthly)
display(metrics)
display(summary)

for item in details:
    win = item["window"]
    result = item["result"]
    actual = item["actual"]
    for fluid in ("oil", "gas"):
        plot_forecast_vs_actual(actual, result, fluid, win.variant, f"w{win.variant}_{fluid}_fact_pred.png")
        pred = result.oil if fluid == "oil" else result.gas
        plot_errors(actual, pred, result.dates, fluid, win.variant, f"w{win.variant}_{fluid}")
"""))

    cells.append(code("""# Сводная таблица в удобном виде и сохранение артефактов
pivot_wape = metrics.pivot(index="variant", columns="fluid", values="WAPE")
pivot_mae = metrics.pivot(index="variant", columns="fluid", values="MAE")
print("WAPE по окнам, %")
display(pivot_wape)
print("MAE по окнам, ст. м³")
display(pivot_mae)

metrics.to_csv("reports/metrics_windows.csv", index=False)
summary.to_csv("reports/metrics_summary.csv", index=False)
print("RMS WAPE нефть: {0:.2f}%".format(summary.loc[summary["fluid"] == "oil", "RMS_WAPE"].iloc[0]))
print("RMS WAPE газ:   {0:.2f}%".format(summary.loc[summary["fluid"] == "gas", "RMS_WAPE"].iloc[0]))
"""))

    cells.append(md("""## 7. Скважинный взгляд (задача 2)

Для самой длинной добывающей скважины `15/9-F-12` повторяем логику на месячном ряде: хронологический тест, Арпс и Ridge. Это проверка, что модель не держится только на сумме месторождения и может работать по стволу.
"""))

    cells.append(code("""well = monthly.loc[monthly["WELL"] == "15/9-F-12", ["DATE", "OIL", "GAS", "WATER", "ON_STREAM_HRS"]].copy()
well = well.rename(columns={"OIL": "oil", "GAS": "gas", "WATER": "water", "ON_STREAM_HRS": "on_stream"})
well["n_producers"] = 1
well["n_wells"] = 1
well["gi"] = 0.0
well["wi"] = 0.0
well["gor"] = np.where(well["oil"] > 0, well["gas"] / well["oil"], np.nan)
well["water_cut"] = np.where(well["oil"] + well["water"] > 0, well["water"] / (well["oil"] + well["water"]), np.nan)

cutoff = pd.Timestamp("2014-12-01")
hist = well.loc[well["DATE"] <= cutoff]
future = well.loc[(well["DATE"] >= "2015-01-01") & (well["DATE"] <= "2015-06-01")]
fc = FieldForecaster()
# Для скважины используем тот же класс, подставив её ряд как «месторождение»
res = fc.fit_predict(well, monthly.loc[monthly["WELL"] == "15/9-F-12"].assign(OIL=lambda d: d["OIL"], GAS=lambda d: d["GAS"]), "2014-12-01")
y = future.set_index("DATE").reindex(res.dates)["oil"].fillna(0)
rep = regression_report(y, res.oil)
print("скважина 15/9-F-12, окно как вариант 4")
print(rep)
plot_compare_arps_ml(
    y,
    res.dates,
    forecast_arps(hist["oil"], 6),
    res.oil,
    "15/9-F-12: факт vs Арпс vs ML, 2015 H1",
    "well_f12_2015.png",
)
"""))

    cells.append(md("""## 8. Как пользоваться моделью на практике

```python
from src.data import load_monthly, monthly_field
from src.models import FieldForecaster

monthly = load_monthly()
field = monthly_field(monthly)
model = FieldForecaster()
forecast = model.fit_predict(field, monthly, cutoff="2015-12-01")
forecast.oil, forecast.gas
```

Модель не требует будущих ГТМ, запусков новых скважин и режимов штуцера: она экстраполирует уже наблюдаемый профиль. Если известен план ввода скважин, его нужно подавать отдельно — иначе прогноз занизит добычу, как это происходит в окне 3.

## 9. Слабые стороны

- Нет признаков ГТМ, ремонтов и планового ввода стволов.
- Короткое первое окно: мало точек для сезонной модели.
- Рекурсия на 6 шагов накапливает ошибку лагов.
- MAPE неустойчив при почти нулевой добыче; поэтому основная метрика кейса — WAPE / RMS WAPE.
- Кривая Арпса плохо описывает рост после подключения новых скважин; ML и смесь это частично компенсируют, но чудес из пустой истории не делают.

## 10. Воспроизведение

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook solution.ipynb
```

Либо пакетный прогон:

```bash
python run_all.py
```
"""))

    nb["cells"] = cells
    return nb


def main():
    nb = build()
    nbf.write(nb, NB_PATH)
    print("записано", NB_PATH)


if __name__ == "__main__":
    main()
