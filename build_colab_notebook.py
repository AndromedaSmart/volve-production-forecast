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
    "pandas", "openpyxl", "scikit-learn", "matplotlib", "statsmodels", "scipy", "lightgbm",
])
print("каталог:", Path.cwd())
'''

RUN = r'''# RECOMPUTE = True пересчитывает сравнение всех моделей (~5-10 минут, ставит torch + PyPOTS).
# По умолчанию результаты сравнения читаются из reports/*.csv, а все графики строятся заново.
RECOMPUTE = False

import subprocess, sys
pkgs = ["lightgbm", "tabulate", "python-docx"] + (["pypots"] if RECOMPUTE else [])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])
cmd = [sys.executable, "run_all.py"] + (["--recompute"] if RECOMPUTE else [])
print(subprocess.run(cmd, capture_output=True, text=True).stdout[-3000:])
'''

TABLES = r'''import pandas as pd
from IPython.display import Markdown, display
pd.set_option("display.precision", 1)

display(Markdown("## Итоговая модель: выбранный подход по окнам (автовыбор по внутренней валидации)"))
display(pd.read_csv("reports/final_model_choice.csv"))
display(Markdown("## Сравнение подходов: WAPE по окнам и RMS WAPE, %"))
display(pd.read_csv("reports/experiments_summary.csv"))
display(Markdown("## Цена утечки: подбор alpha на тесте vs на истории"))
display(pd.read_csv("reports/experiments_leakage.csv"))
display(Markdown("## Аномалии данных и решения"))
display(pd.read_csv("reports/data_quality.csv"))
'''

SHOW = r'''from IPython.display import Image, Markdown, display
from src.album import album, context

section = None
for f in album(context()):
    if f["section"] != section:
        section = f["section"]
        display(Markdown(f"## {section}"))
    display(Markdown(f"### {f['id']}. {f['title']}\n\n{f['caption']}"))
    display(Image(filename=f["file"]))
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

Тетрадка для Google Colab (версия 2). Код и данные берутся из репозитория, затем строятся **все графики итогового отчёта**: методология и предобработка, сравнение ML-подходов (Ridge, LightGBM, Арпс, Transformer и CSDI из PyPOTS), подбор гиперпараметров без утечки, прогнозы итоговой модели по 5 окнам.

1. Откройте **Runtime → Run all**.
2. Дождитесь клонирования и расчёта (1–2 минуты; с `RECOMPUTE = True` — дольше).
3. Ниже появятся таблицы и рисунки G1–G49 с подписями.

Подробная методология — `METHODOLOGY.md`, Word-отчёт — `docs/report.docx`.

Репозиторий: https://github.com/AndromedaSmart/volve-production-forecast
"""
        ),
        nbf.v4.new_code_cell(CLONE),
        nbf.v4.new_code_cell(RUN),
        nbf.v4.new_code_cell(TABLES),
        nbf.v4.new_code_cell(SHOW),
    ]
    nbf.write(nb, NB_PATH)
    print("записано", NB_PATH)


if __name__ == "__main__":
    main()
