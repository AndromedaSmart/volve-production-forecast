"""Сравнение ML-подходов с честной (вложенной) настройкой гиперпараметров.

Каждый подход описывается тремя осями:

* ``mode`` — что предсказываем:
    - ``volume``      месячный объём (как в первой версии решения);
    - ``rate``        удельный дебит, ст. м³ на час работы добывающих стволов;
                      объём = дебит × плановые часы прогнозного месяца;
    - ``rate_noplan`` то же, но плановые часы неизвестны и берутся медианой
                      последних 6 месяцев истории (проверка устойчивости);
* ``model`` — naive / arps / ridge / lgbm (бустинг) / pypots_transformer / pypots_csdi;
* для газа дополнительно ``gas_route``: ``direct`` (своя модель) или ``gor``
  (прогноз нефти × газовый фактор).

Гиперпараметры подбираются только на истории до даты отсечения: внутренний
rolling-origin бэктест (точки отсечения cutoff−6, cutoff−12, cutoff−18 мес.,
горизонт 6). Контрольное окно в подборе не участвует.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import HORIZON_MONTHS, WINDOWS
from .metrics import rms_wape, wape

warnings.filterwarnings("ignore")

LAGS = (1, 2, 3, 6)
ROLLS = (3, 6)
MIN_TRAIN = 12
INNER_FOLDS = 3

GRIDS: Dict[str, List[dict]] = {
    "naive": [{"k": k} for k in (1, 3, 6)],
    "arps": [{"n": n} for n in (12, 18, 24)],
    "ridge": [{"alpha": a} for a in (0.1, 1.0, 5.0, 25.0, 100.0)],
    "lgbm": [
        {"n_estimators": n, "learning_rate": lr, "num_leaves": nl}
        for n, lr, nl in itertools.product((100, 300), (0.03, 0.1), (4, 8))
    ],
}

MODEL_LABELS = {
    "naive": "Наивный (среднее k мес.)",
    "arps": "Кривая Арпса (эксп.)",
    "ridge": "Ridge + лаги",
    "lgbm": "LightGBM + лаги",
    "pypots_transformer": "Transformer (PyPOTS)",
    "pypots_csdi": "CSDI, диффузия (PyPOTS)",
    "ensemble": "Ансамбль (медиана)",
    "v1": "Версия 1 (Ridge, подбор на тесте)",
}
MODE_LABELS = {
    "volume": "объём",
    "rate": "удельный дебит × план. часы",
    "rate_noplan": "удельный дебит × медиана часов",
    "volume_hours": "объём, план. часы как признак",
}


# ---------------------------------------------------------------- признаки

def _feature_row(hist: np.ndarray, month: int, t: float, ex: Optional[Tuple[float, float]] = None) -> List[float]:
    row = [t, np.sin(2 * np.pi * month / 12.0), np.cos(2 * np.pi * month / 12.0)]
    if ex is not None:
        row += [ex[0], ex[1]]  # плановые часы текущего месяца и фактические часы прошлого
    for lag in LAGS:
        row.append(hist[-lag] if len(hist) >= lag else hist[0])
    for win in ROLLS:
        row.append(float(np.mean(hist[-win:])))
    return row


def make_supervised_xy(y: np.ndarray, months: np.ndarray, exog: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Ряд -> таблица (лаги 1/2/3/6, скользящие средние 3/6, сезон, тренд[, часы])."""
    X, Y = [], []
    for i in range(3, len(y)):
        ex = None if exog is None else (exog[i], exog[i - 1])
        X.append(_feature_row(y[:i], int(months[i]), float(i), ex))
        Y.append(y[i])
    return np.asarray(X, float), np.asarray(Y, float)


def _make_regressor(model: str, params: dict):
    if model == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=params["alpha"]))])
    if model == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            min_child_samples=3,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=0,
            verbose=-1,
        )
    raise ValueError(model)


def _arps(t, qi, di):
    return qi * np.exp(-di * t)


def forecast_series(y: np.ndarray, months: np.ndarray, model: str, params: dict, horizon: int,
                    exog: Optional[np.ndarray] = None, exog_future: Optional[np.ndarray] = None) -> np.ndarray:
    """Прогноз одномерного ряда на horizon шагов (рекурсивно для ML)."""
    y = np.asarray(y, float)
    last_month = int(months[-1])
    fut_months = [((last_month + s) % 12) + 1 for s in range(horizon)]
    if model == "naive":
        return np.full(horizon, float(np.mean(y[-params["k"]:])))
    if model == "arps":
        yy = np.clip(y[-params["n"]:], 0, None)
        t = np.arange(len(yy), dtype=float)
        try:
            qi0 = max(float(np.median(yy[:3])), 1e-6)
            popt, _ = curve_fit(_arps, t, yy, p0=(qi0, 0.02), bounds=([0, -0.15], [qi0 * 4 + 1, 0.6]), maxfev=4000)
            return np.clip(_arps(np.arange(len(yy), len(yy) + horizon), *popt), 0, None)
        except Exception:
            return np.full(horizon, float(np.median(yy)))
    if model in ("ridge", "lgbm"):
        scale = float(np.mean(np.abs(y))) or 1.0
        ys = y / scale
        exs = None
        if exog is not None:
            hs = float(np.mean(exog)) or 1.0
            exs = np.concatenate([np.asarray(exog, float), np.asarray(exog_future, float)]) / hs
        X, Y = make_supervised_xy(ys, months, None if exs is None else exs[: len(ys)])
        reg = _make_regressor(model, params)
        reg.fit(X, Y)
        hist = list(ys)
        out = []
        for s in range(horizon):
            ex = None if exs is None else (exs[len(hist)], exs[len(hist) - 1])
            x = np.asarray([_feature_row(np.asarray(hist), fut_months[s], float(len(hist)), ex)])
            p = max(float(reg.predict(x)[0]), 0.0)
            out.append(p)
            hist.append(p)
        return np.asarray(out) * scale
    if model.startswith("pypots_"):
        return _forecast_pypots(y, model, horizon)
    raise ValueError(model)


# ------------------------------------------------------- глубокие модели

def _forecast_pypots(y: np.ndarray, model: str, horizon: int, n_steps: int = 12, seed: int = 0) -> np.ndarray:
    import torch
    from pypots.forecasting import CSDI, Transformer

    torch.manual_seed(seed)
    np.random.seed(seed)
    mu = float(np.mean(y))
    sd = float(np.std(y)) or 1.0
    z = (y - mu) / sd
    n_steps = min(n_steps, max(4, len(z) - horizon - 2))
    X, Xp = [], []
    for i in range(0, len(z) - n_steps - horizon + 1):
        X.append(z[i: i + n_steps])
        Xp.append(z[i + n_steps: i + n_steps + horizon])
    X = np.asarray(X, np.float32)[..., None]
    Xp = np.asarray(Xp, np.float32)[..., None]
    train = {"X": X, "X_pred": Xp}
    if model == "pypots_transformer":
        net = Transformer(
            n_steps=n_steps, n_features=1, n_pred_steps=horizon, n_pred_features=1,
            n_encoder_layers=1, n_decoder_layers=1, d_model=16, n_heads=2, d_k=8, d_v=8, d_ffn=32,
            dropout=0.1, batch_size=8, epochs=150, verbose=False, device="cpu",
        )
    else:
        net = CSDI(
            n_steps=n_steps, n_features=1, n_pred_steps=horizon, n_pred_features=1,
            n_layers=2, n_heads=2, n_channels=16, d_time_embedding=32, d_feature_embedding=4,
            d_diffusion_embedding=32, n_diffusion_steps=50, batch_size=8, epochs=150,
            verbose=False, device="cpu",
        )
    net.fit(train)
    test = {"X": z[-n_steps:][None, :, None].astype(np.float32),
            "X_pred": np.full((1, horizon, 1), np.nan, np.float32)}
    if model == "pypots_csdi":
        res = net.predict(test, n_sampling_times=20)["forecasting"]
        pred = np.median(np.asarray(res)[0, :, :, 0], axis=0)
    else:
        pred = np.asarray(net.predict(test)["forecasting"])[0, :, 0]
    return np.clip(pred * sd + mu, 0, None)


# ------------------------------------------------- прогноз объёма флюида

def _producing(hist: pd.DataFrame) -> pd.DataFrame:
    first = hist.index[(hist["oil"] + hist["gas"]) > 0]
    return hist.loc[first.min():] if len(first) else hist


def forecast_volume(hist: pd.DataFrame, future_hours: np.ndarray, fluid: str, mode: str,
                    model: str, params: dict, horizon: int = HORIZON_MONTHS) -> np.ndarray:
    """hist — месячный ряд месторождения до даты отсечения включительно."""
    h = _producing(hist)
    if mode == "volume":
        y = h[fluid].to_numpy(float)
        return forecast_series(y, h["DATE"].dt.month.to_numpy(), model, params, horizon)
    if mode == "volume_hours":
        y = h[fluid].to_numpy(float)
        return forecast_series(y, h["DATE"].dt.month.to_numpy(), model, params, horizon,
                               exog=h["prod_hours"].to_numpy(float), exog_future=np.asarray(future_hours, float))
    hh = h.loc[h["prod_hours"] > 0]
    rate = (hh[fluid] / hh["prod_hours"]).to_numpy(float)
    r_hat = forecast_series(rate, hh["DATE"].dt.month.to_numpy(), model, params, horizon)
    if mode == "rate":
        hours = np.asarray(future_hours, float)
    else:
        hours = np.full(horizon, float(np.median(hh["prod_hours"].iloc[-6:])))
    return r_hat * hours


def gor_hat(hist: pd.DataFrame, n: int = 6) -> float:
    h = hist.loc[hist["oil"] > 0].iloc[-n:]
    return float(h["gas"].sum() / h["oil"].sum())


def forecast_fluid(hist, future_hours, fluid, cfg, horizon=HORIZON_MONTHS):
    """cfg = (mode, model, params, gas_route)."""
    mode, model, params, route = cfg
    if fluid == "gas" and route == "gor":
        oil = forecast_volume(hist, future_hours, "oil", mode, model, params, horizon)
        return oil * gor_hat(hist)
    return forecast_volume(hist, future_hours, fluid, mode, model, params, horizon)


# ------------------------------------------------ внутренняя валидация

def inner_origins(hist: pd.DataFrame, horizon: int = HORIZON_MONTHS, folds: int = INNER_FOLDS) -> List[int]:
    h = _producing(hist)
    n = len(h)
    out = []
    for k in range(1, folds + 1):
        cut = n - horizon * k
        if cut >= MIN_TRAIN + 3:
            out.append(h.index[cut - 1])
    return out


def inner_score(field: pd.DataFrame, hist: pd.DataFrame, fluid: str, cfg, horizon=HORIZON_MONTHS) -> float:
    scores = []
    for end_idx in inner_origins(hist, horizon):
        sub = field.loc[:end_idx]
        fut = field.loc[end_idx + 1: end_idx + horizon]
        if len(fut) < horizon:
            continue
        pred = forecast_fluid(sub, fut["prod_hours"].to_numpy(), fluid, cfg, horizon)
        scores.append(wape(fut[fluid].to_numpy(), pred))
    return float(np.mean(scores)) if scores else float("nan")


def tune(field, hist, fluid, mode, model, route="direct") -> Tuple[dict, float, pd.DataFrame]:
    rows = []
    for params in GRIDS[model]:
        s = inner_score(field, hist, fluid, (mode, model, params, route))
        rows.append({**params, "inner_wape": s})
    df = pd.DataFrame(rows)
    best = df["inner_wape"].idxmin() if df["inner_wape"].notna().any() else 0
    params = {k: v for k, v in df.loc[best].items() if k != "inner_wape"}
    for k, v in list(params.items()):
        if isinstance(v, float) and v.is_integer() and k != "alpha" and k != "learning_rate":
            params[k] = int(v)
    return params, float(df.loc[best, "inner_wape"]), df


# --------------------------------------------------------- эксперимент

@dataclass
class Experiment:
    rows: pd.DataFrame
    preds: Dict[tuple, np.ndarray]
    actual: Dict[tuple, np.ndarray]
    dates: Dict[int, pd.DatetimeIndex]
    tuning: pd.DataFrame
    leak: pd.DataFrame
    auto: pd.DataFrame


def _window_data(field: pd.DataFrame, win):
    hist = field.loc[field["DATE"] <= pd.Timestamp(win.train_end)]
    fut = field.loc[(field["DATE"] >= pd.Timestamp(win.forecast_start)) & (field["DATE"] <= pd.Timestamp(win.forecast_end))]
    return hist, fut


CLASSIC = ("naive", "arps", "ridge", "lgbm")
DEEP = ("pypots_transformer", "pypots_csdi")


def run_experiments(field: pd.DataFrame, monthly: Optional[pd.DataFrame] = None,
                    with_deep: bool = True, verbose: bool = True) -> Experiment:
    field = field.reset_index(drop=True)
    v1 = None
    if monthly is not None:
        from .models import FieldForecaster

        v1 = {}
        for win in WINDOWS:
            r = FieldForecaster().fit_predict(field, monthly, win.train_end)
            v1[win.variant] = {"oil": r.oil, "gas": r.gas}
    rows, tuning_rows, leak_rows, auto_rows = [], [], [], []
    preds, actual, dates = {}, {}, {}
    approaches = []
    for mode in ("volume", "rate", "rate_noplan"):
        for model in CLASSIC:
            approaches.append((mode, model, "direct"))
    approaches += [("volume_hours", m, "direct") for m in ("ridge", "lgbm")]
    approaches += [("rate", m, "gor") for m in CLASSIC]
    approaches += [("volume_hours", m, "gor") for m in ("ridge", "lgbm")]
    if with_deep:
        approaches += [("rate", m, "direct") for m in DEEP]
        approaches += [("volume", m, "direct") for m in DEEP]

    for win in WINDOWS:
        hist, fut = _window_data(field, win)
        fut_hours = fut["prod_hours"].to_numpy()
        dates[win.variant] = pd.DatetimeIndex(fut["DATE"])
        for fluid in ("oil", "gas"):
            y = fut[fluid].to_numpy()
            actual[(win.variant, fluid)] = y
            candidates = []
            for mode, model, route in approaches:
                if route == "gor" and fluid == "oil":
                    continue
                if model in GRIDS:
                    params, inner, grid = tune(field, hist, fluid, mode, model, route)
                    grid = grid.assign(variant=win.variant, fluid=fluid, mode=mode, model=model, route=route)
                    tuning_rows.append(grid)
                else:
                    params, inner = {}, float("nan")
                cfg = (mode, model, params, route)
                p = forecast_fluid(hist, fut_hours, fluid, cfg)
                key = (win.variant, fluid, mode, model, route)
                preds[key] = p
                w = wape(y, p)
                rows.append({"variant": win.variant, "fluid": fluid, "mode": mode, "model": model,
                             "route": route, "params": str(params), "inner_wape": inner, "test_wape": w})
                if model in GRIDS and mode != "rate_noplan":
                    candidates.append((inner, cfg, w, key))
                if verbose:
                    print(f"w{win.variant} {fluid:3s} {mode:11s} {model:18s} {route:6s} inner={inner:6.1f} test={w:6.1f}", flush=True)
            # Ансамбль: медиана настроенных классических моделей (в каждом режиме)
            for mode in ("volume", "rate", "volume_hours"):
                for route in (("direct", "gor") if fluid == "gas" else ("direct",)):
                    ks = [(win.variant, fluid, mode, m, route) for m in CLASSIC]
                    if mode == "volume_hours":
                        ks += [(win.variant, fluid, "rate", m, route) for m in ("naive", "arps")]
                    ks = [k for k in ks if k in preds]
                    if not ks:
                        continue
                    p = np.median(np.vstack([preds[k] for k in ks]), axis=0)
                    inner = float(np.median([r["inner_wape"] for r in rows if r["variant"] == win.variant
                                             and r["fluid"] == fluid and r["mode"] == mode and r["route"] == route
                                             and r["model"] in CLASSIC]))
                    preds[(win.variant, fluid, mode, "ensemble", route)] = p
                    rows.append({"variant": win.variant, "fluid": fluid, "mode": mode, "model": "ensemble",
                                 "route": route, "params": "median(naive, arps, ridge, lgbm)",
                                 "inner_wape": float("nan"), "test_wape": wape(y, p)})
            # Первая версия решения (v1) — для сравнения
            if v1 is not None:
                p = v1[win.variant][fluid]
                preds[(win.variant, fluid, "volume", "v1", "direct")] = p
                rows.append({"variant": win.variant, "fluid": fluid, "mode": "volume", "model": "v1",
                             "route": "direct", "params": "Ridge alpha=5 / blend (v1)",
                             "inner_wape": float("nan"), "test_wape": wape(y, p)})
            # Авто-выбор подхода по внутренней валидации (без взгляда на тест)
            candidates = [c for c in candidates if np.isfinite(c[0])]
            best = min(candidates, key=lambda c: c[0])
            auto_rows.append({"variant": win.variant, "fluid": fluid, "mode": best[1][0], "model": best[1][1],
                              "route": best[1][3], "params": str(best[1][2]), "inner_wape": best[0], "test_wape": best[2]})
            preds[(win.variant, fluid, "auto", "auto", "auto")] = preds[best[3]]

            # Утечка: подбор alpha Ridge по самому контрольному окну (как было в v1)
            for mode in ("volume", "rate"):
                best_leak = None
                for params in GRIDS["ridge"]:
                    p = forecast_fluid(hist, fut_hours, fluid, (mode, "ridge", params, "direct"))
                    w = wape(y, p)
                    if best_leak is None or w < best_leak[0]:
                        best_leak = (w, params)
                honest = [r for r in rows if r["variant"] == win.variant and r["fluid"] == fluid
                          and r["mode"] == mode and r["model"] == "ridge" and r["route"] == "direct"][0]
                leak_rows.append({"variant": win.variant, "fluid": fluid, "mode": mode,
                                  "leaky_params": str(best_leak[1]), "leaky_test_wape": best_leak[0],
                                  "honest_params": honest["params"], "honest_test_wape": honest["test_wape"]})
    return Experiment(
        rows=pd.DataFrame(rows), preds=preds, actual=actual, dates=dates,
        tuning=pd.concat(tuning_rows, ignore_index=True) if tuning_rows else pd.DataFrame(),
        leak=pd.DataFrame(leak_rows), auto=pd.DataFrame(auto_rows),
    )


def summary_table(rows: pd.DataFrame) -> pd.DataFrame:
    """RMS WAPE по 5 окнам для каждого подхода."""
    out = []
    for (fluid, mode, model, route), g in rows.groupby(["fluid", "mode", "model", "route"]):
        d = {"fluid": fluid, "mode": mode, "model": model, "route": route}
        for _, r in g.iterrows():
            d[f"w{int(r['variant'])}"] = r["test_wape"]
        d["RMS_WAPE"] = rms_wape(g["test_wape"])
        d["inner_mean"] = float(g["inner_wape"].mean())
        out.append(d)
    return pd.DataFrame(out).sort_values(["fluid", "RMS_WAPE"]).reset_index(drop=True)


# ----------------------------------------------------- сохранение / загрузка

def save_experiment(exp: Experiment, folder) -> None:
    from pathlib import Path

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    exp.rows.to_csv(folder / "experiments_windows.csv", index=False)
    summary_table(exp.rows).to_csv(folder / "experiments_summary.csv", index=False)
    exp.tuning.to_csv(folder / "experiments_tuning.csv", index=False)
    exp.leak.to_csv(folder / "experiments_leakage.csv", index=False)
    exp.auto.to_csv(folder / "final_model_choice.csv", index=False)
    recs = []
    for (v, fluid, mode, model, route), p in exp.preds.items():
        for i, (d, val) in enumerate(zip(exp.dates[v], p)):
            recs.append({"variant": v, "fluid": fluid, "mode": mode, "model": model, "route": route,
                         "step": i + 1, "DATE": d, "pred": float(val), "actual": float(exp.actual[(v, fluid)][i])})
    pd.DataFrame(recs).to_csv(folder / "experiments_predictions.csv", index=False)


def load_experiment(folder) -> Experiment:
    from pathlib import Path

    folder = Path(folder)
    rows = pd.read_csv(folder / "experiments_windows.csv")
    pr = pd.read_csv(folder / "experiments_predictions.csv", parse_dates=["DATE"])
    preds, actual, dates = {}, {}, {}
    for (v, fluid, mode, model, route), g in pr.groupby(["variant", "fluid", "mode", "model", "route"]):
        g = g.sort_values("step")
        preds[(int(v), fluid, mode, model, route)] = g["pred"].to_numpy()
        actual[(int(v), fluid)] = g["actual"].to_numpy()
        dates[int(v)] = pd.DatetimeIndex(g["DATE"])
    return Experiment(rows=rows, preds=preds, actual=actual, dates=dates,
                      tuning=pd.read_csv(folder / "experiments_tuning.csv"),
                      leak=pd.read_csv(folder / "experiments_leakage.csv"),
                      auto=pd.read_csv(folder / "final_model_choice.csv"))
