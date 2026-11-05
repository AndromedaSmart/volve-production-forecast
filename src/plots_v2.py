"""Графики второй версии: предобработка, методология, сравнение подходов."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from .config import WINDOWS
from .experiments import MODE_LABELS, MODEL_LABELS, summary_table
from .plots import _save

FLUID_RU = {"oil": "нефть", "gas": "газ"}
COLORS = {
    "naive": "#9e9e9e", "arps": "#c0504d", "ridge": "#1f4e79", "lgbm": "#2e8b57",
    "pypots_transformer": "#8e44ad", "pypots_csdi": "#e67e22", "ensemble": "#000000", "v1": "#bdbdbd",
}


def approach_label(mode, model, route) -> str:
    lab = f"{MODEL_LABELS.get(model, model)} · {MODE_LABELS.get(mode, mode)}"
    if route == "gor":
        lab += " · газ = нефть×ГФ"
    return lab


def plot_pipeline(name="methodology_pipeline.png") -> Path:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")
    steps = [
        ("1. Данные", "Excel Volve:\nсуточный и месячный\nлисты, 7 стволов"),
        ("2. Очистка", "часы > 24 (DST)\nотриц. вода\nчасы нагнетательных"),
        ("3. Месячный ряд", "сумма по стволам\nчасы только\nдобывающих"),
        ("4. Цель", "объём, удельный\nдебит (ст. м³/ч)\nили объём + часы"),
        ("5. Признаки", "лаги 1/2/3/6,\nсредние 3/6,\nсезон, тренд"),
        ("6. Модели", "наивная, Арпс,\nRidge, LightGBM,\nTransformer, CSDI"),
        ("7. Подбор", "rolling-origin\nтолько до\nдаты отсечения"),
        ("8. Оценка", "5 контрольных\nокон, WAPE,\nRMS WAPE"),
    ]
    w, h = 0.215, 0.33
    for i, (title, body) in enumerate(steps):
        r, c = divmod(i, 4)
        x = 0.02 + c * 0.247
        y0 = 0.50 - r * 0.42
        color = "#dbe8f5" if i < 5 else "#fde3cf"
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.01", fc=color, ec="#1f4e79", lw=1.2,
                                    transform=ax.transAxes))
        ax.text(x + w / 2, y0 + h - 0.06, title, ha="center", va="center", fontsize=12, weight="bold", transform=ax.transAxes)
        ax.text(x + w / 2, y0 + h / 2 - 0.04, body, ha="center", va="center", fontsize=10.5, transform=ax.transAxes)
        if c < 3:
            ax.annotate("", xy=(x + 0.247, y0 + h / 2), xytext=(x + w, y0 + h / 2), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", lw=1.6, color="#1f4e79"))
    ax.text(0.5, 0.96, "Методология: от сырых данных до прогноза на 6 месяцев", ha="center", fontsize=13,
            weight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.01, "Синие блоки — подготовка данных, оранжевые — обучение и оценка. Контрольное окно не используется "
            "ни при обучении, ни при подборе гиперпараметров.", ha="center", fontsize=9.5, transform=ax.transAxes)
    return _save(fig, name)


def plot_preprocessing(daily: pd.DataFrame, field: pd.DataFrame, name="preprocessing.png") -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    hrs = daily["ON_STREAM_HRS"].dropna()
    ax.hist(hrs[hrs > 0], bins=np.arange(0, 26, 0.5), color="#1f4e79")
    over = daily.loc[daily["ON_STREAM_HRS"] > 24]
    ax.axvline(24, color="#c0504d", ls="--", lw=1.4, label="24 ч")
    ax.set_yscale("log")
    ax.set_title(f"Часы работы ствола за сутки\n{len(over)} записей > 24 ч — все в дни перевода часов")
    ax.set_xlabel("ON_STREAM_HRS, ч")
    ax.set_ylabel("число суток (лог.)")
    ax.legend()

    ax = axes[1]
    neg = daily.loc[daily["BORE_WAT_VOL"] < 0]
    for well, g in daily.groupby("WELL"):
        if well in set(neg["WELL"]):
            ax.plot(g["DATEPRD"], g["BORE_WAT_VOL"], lw=0.6, alpha=0.6, label=well)
    ax.scatter(neg["DATEPRD"], neg["BORE_WAT_VOL"], color="#c0504d", zorder=5, s=40, label="отрицательные значения")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"Суточная вода: {len(neg)} отрицательных записей\n(учётные корректировки)")
    ax.set_ylabel("BORE_WAT_VOL, ст. м³/сут")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(field["DATE"], field["on_stream"], color="#9e9e9e", lw=1.6, label="все стволы (v1)")
    ax.plot(field["DATE"], field["prod_hours"], color="#1f4e79", lw=2, label="только добывающие (v2)")
    ax.set_title("Часы работы месторождения в месяц\nнагнетательные F-4/F-5 исключены")
    ax.set_ylabel("ч / мес.")
    ax.legend(fontsize=8)
    return _save(fig, name)


def plot_rate_series(field: pd.DataFrame, name="rate_series.png") -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax = axes[0]
    ax.plot(field["DATE"], field["oil"], color="#1f4e79", lw=2, label="нефть, ст. м³/мес.")
    ax2 = ax.twinx()
    ax2.plot(field["DATE"], field["prod_hours"], color="#e67e22", lw=1.3, alpha=0.8, label="часы добывающих стволов")
    ax2.set_ylabel("ч / мес.", color="#e67e22")
    ax.set_ylabel("нефть, ст. м³")
    ax.set_title("Объём = удельный дебит × часы работы")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    ax = axes[1]
    ax.plot(field["DATE"], field["oil_rate"], color="#1f4e79", lw=2, label="нефть, ст. м³ / ч")
    ax3 = ax.twinx()
    ax3.plot(field["DATE"], field["gas_rate"], color="#c0504d", lw=1.4, label="газ, ст. м³ / ч")
    ax.set_ylabel("нефть, ст. м³ / ч")
    ax3.set_ylabel("газ, ст. м³ / ч", color="#c0504d")
    for win in WINDOWS:
        ax.axvspan(pd.Timestamp(win.forecast_start), pd.Timestamp(win.forecast_end), color="#f4b183", alpha=0.35)
    ax.set_title("Удельный дебит (на час работы); оранжевым — 5 контрольных окон")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    return _save(fig, name)


def plot_gor(field: pd.DataFrame, name="gor_justification.png") -> Path:
    f = field.loc[field["oil"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    ax = axes[0]
    ax.plot(f["DATE"], f["gor"], color="#1f4e79", lw=2)
    ax.axhline(f["gor"].median(), color="#c0504d", ls="--", label=f"медиана {f['gor'].median():.0f}")
    ax.set_title("Газовый фактор (газ / нефть) по месяцам")
    ax.set_ylabel("ст. м³ газа / ст. м³ нефти")
    ax.legend()
    ax = axes[1]
    ax.scatter(f["oil"], f["gas"], s=18, color="#1f4e79")
    k = np.polyfit(f["oil"], f["gas"], 1)
    r = np.corrcoef(f["oil"], f["gas"])[0, 1]
    xs = np.linspace(0, f["oil"].max(), 50)
    ax.plot(xs, np.polyval(k, xs), color="#c0504d", label=f"линейная связь, r = {r:.4f}")
    ax.set_xlabel("нефть, ст. м³/мес.")
    ax.set_ylabel("газ, ст. м³/мес.")
    ax.set_title("Газ почти пропорционален нефти (попутный газ)")
    ax.legend()
    return _save(fig, name)


def _key_table(summary: pd.DataFrame, fluid: str) -> pd.DataFrame:
    s = summary.loc[(summary["fluid"] == fluid) & (summary["mode"] != "auto")].copy()
    s["label"] = [approach_label(m, mo, r) for m, mo, r in zip(s["mode"], s["model"], s["route"])]
    return s.sort_values("RMS_WAPE")


def plot_model_comparison(rows: pd.DataFrame, auto: pd.DataFrame, name="model_comparison.png") -> Path:
    summary = summary_table(rows)
    fig, axes = plt.subplots(2, 1, figsize=(13, 17))
    for ax, fluid in zip(axes, ("oil", "gas")):
        s = _key_table(summary, fluid)
        colors = [COLORS.get(m, "#555") for m in s["model"]]
        ax.barh(s["label"], s["RMS_WAPE"], color=colors)
        a = auto.loc[auto["fluid"] == fluid, "test_wape"]
        auto_rms = float(np.sqrt(np.mean(a ** 2)))
        ax.axvline(auto_rms, color="#c0504d", ls="--", lw=1.6, label=f"итоговая модель (автовыбор): {auto_rms:.1f}%")
        for y, v in enumerate(s["RMS_WAPE"]):
            ax.text(v + 0.4, y, f"{v:.1f}", va="center", fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"{FLUID_RU[fluid].capitalize()}: RMS WAPE по 5 окнам, %", fontsize=13)
        ax.tick_params(axis="y", labelsize=9.5)
        ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    from .config import FIGURES_DIR
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)
    return FIGURES_DIR / name


def plot_heatmap(rows: pd.DataFrame, fluid: str, name: str) -> Path:
    summary = summary_table(rows)
    s = _key_table(summary, fluid)
    cols = [f"w{w.variant}" for w in WINDOWS]
    mat = s[cols].to_numpy(float)
    fig, ax = plt.subplots(figsize=(11, 0.34 * len(s) + 1.6))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=80, aspect="auto")
    ax.set_xticks(range(len(cols)), [f"окно {c[1]}" for c in cols])
    ax.set_yticks(range(len(s)), s["label"], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="WAPE, %")
    ax.set_title(f"{FLUID_RU[fluid].capitalize()}: WAPE по контрольным окнам")
    return _save(fig, name)


def plot_leakage(leak: pd.DataFrame, name="tuning_leakage.png") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, fluid in zip(axes, ("oil", "gas")):
        d = leak.loc[(leak["fluid"] == fluid) & (leak["mode"] == "volume")]
        x = np.arange(len(d))
        ax.bar(x - 0.2, d["leaky_test_wape"], 0.4, color="#bdbdbd", label="alpha подобран на тесте (v1)")
        ax.bar(x + 0.2, d["honest_test_wape"], 0.4, color="#1f4e79", label="alpha подобран на истории (v2)")
        ax.set_xticks(x, [f"окно {v}" for v in d["variant"]])
        lr = np.sqrt(np.mean(d["leaky_test_wape"] ** 2)); hr = np.sqrt(np.mean(d["honest_test_wape"] ** 2))
        ax.set_title(f"Ridge, {FLUID_RU[fluid]}: RMS WAPE {lr:.1f}% (утечка) → {hr:.1f}% (честно)")
        ax.set_ylabel("WAPE на контрольном окне, %")
        ax.legend(fontsize=9)
    return _save(fig, name)


def plot_inner_vs_test(rows: pd.DataFrame, name="inner_vs_test.png") -> Path:
    d = rows.dropna(subset=["inner_wape"])
    fig, ax = plt.subplots(figsize=(8, 6))
    for model, g in d.groupby("model"):
        ax.scatter(g["inner_wape"], g["test_wape"], s=22, color=COLORS.get(model, "#555"), label=MODEL_LABELS.get(model, model), alpha=0.8)
    lim = max(d["inner_wape"].max(), d["test_wape"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
    ax.set_xlabel("WAPE на внутренней валидации (история), %")
    ax.set_ylabel("WAPE на контрольном окне, %")
    ax.set_title("Ошибка на валидации vs на тесте")
    ax.legend(fontsize=8)
    return _save(fig, name)


def plot_tuning_curves(tuning: pd.DataFrame, name="tuning_ridge_alpha.png") -> Path:
    d = tuning.loc[(tuning["model"] == "ridge") & (tuning["route"] == "direct") & (tuning["mode"] == "volume")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, fluid in zip(axes, ("oil", "gas")):
        for v, g in d.loc[d["fluid"] == fluid].groupby("variant"):
            ax.plot(g["alpha"], g["inner_wape"], marker="o", label=f"окно {v}")
        ax.set_xscale("log")
        ax.set_xlabel("alpha (Ridge)")
        ax.set_ylabel("средний WAPE внутренних фолдов, %")
        ax.set_title(f"{FLUID_RU[fluid].capitalize()}: подбор alpha на истории до отсечения")
        ax.legend(fontsize=8)
    return _save(fig, name)


def plot_window_models(exp, fluid: str, name: str, models=None) -> Path:
    models = models or [
        ("volume", "ridge", "direct"), ("rate", "ridge", "direct"), ("volume", "lgbm", "direct"),
        ("rate", "lgbm", "direct"), ("volume_hours", "ridge", "direct"),
        ("volume", "pypots_transformer", "direct"), ("volume", "pypots_csdi", "direct"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(20, 6.2), sharey=False)
    styles = {"volume": "-", "rate": "--", "volume_hours": "-."}
    for ax, win in zip(axes, WINDOWS):
        dates = exp.dates[win.variant]
        ax.plot(dates, exp.actual[(win.variant, fluid)], "ko-", lw=2.2, label="факт")
        for mode, model, route in models:
            key = (win.variant, fluid, mode, model, route)
            if key in exp.preds:
                ax.plot(dates, exp.preds[key], styles.get(mode, ":"), color=COLORS.get(model, "#555"), lw=1.5,
                        label=approach_label(mode, model, route))
        ax.set_title(f"окно {win.variant}: {dates[0]:%Y-%m} — {dates[-1]:%Y-%m}", fontsize=10)
        ax.set_xticks(dates, [f"{d:%m.%y}" for d in dates])
        ax.tick_params(axis="x", rotation=45, labelsize=9)
    axes[0].set_ylabel(f"{FLUID_RU[fluid]}, ст. м³/мес.")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{FLUID_RU[fluid].capitalize()}: факт и прогнозы разных подходов (сплошные — объём, пунктир — удельный дебит, точка-тире — объём с часами)")
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    FIG = name
    from .config import FIGURES_DIR
    fig.savefig(FIGURES_DIR / FIG)
    plt.close(fig)
    return FIGURES_DIR / FIG


def plot_gas_routes(exp, name="gas_direct_vs_gor.png") -> Path:
    s = summary_table(exp.rows)
    s = s.loc[(s["fluid"] == "gas") & (s["mode"] == "rate") & s["model"].isin(["naive", "arps", "ridge", "lgbm", "ensemble"])]
    piv = s.pivot_table(index="model", columns="route", values="RMS_WAPE")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(piv))
    ax.bar(x - 0.2, piv["direct"], 0.4, color="#1f4e79", label="отдельная модель газа")
    ax.bar(x + 0.2, piv["gor"], 0.4, color="#e67e22", label="газ = прогноз нефти × ГФ")
    ax.set_xticks(x, [MODEL_LABELS.get(m, m).replace(" (", "\n(").replace(" + ", "\n+ ") for m in piv.index], fontsize=9)
    ax.set_ylabel("RMS WAPE газа, %")
    ax.set_title("Газ: отдельная модель vs множитель к нефти (удельный дебит)")
    ax.legend()
    return _save(fig, name)
