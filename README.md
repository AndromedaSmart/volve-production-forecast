# Прогноз добычи нефти и газа, месторождение Volve

Jupyter-тетрадка и модель для кейса: по истории добычи до заданной даты посчитать **месячный прогноз нефти и газа на 6 месяцев** по месторождению.

Данные: Volve (Норвегия, 2007–2016), файл `data/data.xlsx`.

## Результат на контрольных окнах

| Флюид | RMS WAPE |
|---|---|
| нефть | 26.5% |
| газ | 23.6% |

Полный отчёт с графиками и таблицами по требованиям кейса: **[REPORT.md](REPORT.md)**.  
Альбом всех рисунков с подписями и легендой: **[FIGURES.md](FIGURES.md)**.

Полная таблица MAE / MAPE / WAPE — в `reports/metrics_windows.csv` и в `solution.ipynb`.

## Как запустить

Нужен Python 3.8+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Тетрадка:

```bash
jupyter notebook solution.ipynb
```

Пакетный прогон графиков и метрик:

```bash
python run_all.py
```

Прогноз из кода:

```python
from src.data import load_monthly, monthly_field
from src.models import FieldForecaster

monthly = load_monthly()
field = monthly_field(monthly)
forecast = FieldForecaster().fit_predict(field, monthly, cutoff="2015-12-01")
forecast.oil, forecast.gas
```

## Что внутри

- `FIGURES.md` — все графики с подписями и легендой (название, обоснование, описание)
- `REPORT.md` — отчёт проделанной работы: все графики, метрики и пояснения по требованиям кейса
- `solution.ipynb` — решение с пояснениями, графиками и метриками
- `src/` — загрузка данных, признаки, модель, оценка
- `figures/` — графики профилей, ошибок и сравнения с кривой Арпса
- `reports/NOTE.md` — краткая пояснительная записка
- `docs/` — исходная постановка кейса

Модель берёт только историю до даты отсечения. На коротком ряде (окно 1) это смесь кривой Арпса и устойчивого уровня добычи, на более длинной истории — Ridge с лагами 1/2/3/6 месяцев.
