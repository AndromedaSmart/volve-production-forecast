# Прогноз добычи нефти и газа, месторождение Volve

Jupyter-тетрадка и модель для кейса: по истории добычи до заданной даты посчитать **месячный прогноз нефти и газа на 6 месяцев** по месторождению.

Данные: Volve (Норвегия, 2007–2016), файл `data/data.xlsx`.

## Результат на контрольных окнах (версия 2)

Гиперпараметры и выбор подхода — только по истории до даты отсечения (вложенный rolling-origin), контрольные окна используются только для оценки.

| | Нефть, RMS WAPE | Газ, RMS WAPE |
|---|---|---|
| Итоговая модель (автовыбор по внутренней валидации) | 41.6% | 39.4% |
| Лучший одиночный подход: Ridge + плановые часы как признак | 29.6% | 28.7% |
| LightGBM по объёму | 36.7% | 29.9% |
| Transformer / CSDI (PyPOTS) — лучший вариант | 41.1% | 41.0% |
| v1 (Ridge, параметры подбирались на тех же окнах) | 26.5% | 23.6% |

Оценка v1 была занижена утечкой: для Ridge при честном подборе alpha RMS WAPE нефти растёт с 27.9% до 49.0%.

- Отчёт: **[REPORT.md](REPORT.md)**, подробная методология: **[METHODOLOGY.md](METHODOLOGY.md)**
- Word-отчёт со всеми графиками: **[docs/report.docx](docs/report.docx)**
- Альбом рисунков G1–G49 с легендой: **[FIGURES.md](FIGURES.md)**
- Запуск в браузере: **[открыть в Google Colab](https://colab.research.google.com/github/AndromedaSmart/volve-production-forecast/blob/master/colab.ipynb)**

## Как запустить

Нужен Python 3.8+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Тетрадка локально:

```bash
jupyter notebook solution.ipynb
```

В Google Colab (клонирует репозиторий и рисует все графики):

https://colab.research.google.com/github/AndromedaSmart/volve-production-forecast/blob/master/colab.ipynb

Пакетный прогон графиков и метрик (сравнение моделей читается из `reports/*.csv`):

```bash
python run_all.py                # все графики
python run_all.py --recompute    # пересчитать сравнение всех моделей (нужны torch и pypots)
python run_all.py --recompute --no-deep   # без Transformer/CSDI
python build_report.py           # docs/report.docx, FIGURES.md, METHODOLOGY.md, REPORT.md
python build_colab_notebook.py   # colab.ipynb
```

Прогноз из кода:

```python
from src.data import load_monthly, monthly_field
from src.experiments import forecast_fluid, tune

field = monthly_field(load_monthly())
hist = field[field["DATE"] <= "2015-12-01"]
plan_hours = [3000] * 6                     # плановые часы добывающих стволов на 6 мес.
params, inner_wape, _ = tune(field, hist, "oil", "volume_hours", "ridge")   # подбор только на истории
oil = forecast_fluid(hist, plan_hours, "oil", ("volume_hours", "ridge", params, "direct"))
```

## Что внутри

- `colab.ipynb` — короткая тетрадка для Google Colab: запуск и вывод всех графиков
- `FIGURES.md` — все графики с подписями и легендой (название, обоснование, описание)
- `REPORT.md` — отчёт проделанной работы: все графики, метрики и пояснения по требованиям кейса
- `solution.ipynb` — решение с пояснениями, графиками и метриками
- `src/data.py` — загрузка, очистка, часы добывающих стволов, отчёт об аномалиях
- `src/experiments.py` — сравнение подходов (наивная, Арпс, Ridge, LightGBM, Transformer, CSDI), вложенный подбор гиперпараметров, автовыбор итоговой модели
- `src/plots_v2.py`, `src/album.py` — графики версии 2 и единый список рисунков
- `src/models.py` — модель версии 1 (для сравнения)
- `figures/` — графики профилей, ошибок и сравнения с кривой Арпса
- `reports/NOTE.md` — краткая пояснительная записка
- `docs/` — исходная постановка кейса

Модель берёт только историю до даты отсечения; и обучение, и подбор гиперпараметров, и выбор подхода выполняются на этой истории.
