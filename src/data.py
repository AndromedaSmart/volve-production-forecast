"""Загрузка и подготовка данных Volve."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import DATA_PATH, ROOT

CACHE_DIR = ROOT / "data" / ".cache"

DAILY_USECOLS = [
    "DATEPRD",
    "NPD_WELL_BORE_NAME",
    "ON_STREAM_HRS",
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_WHP_P",
    "BORE_OIL_VOL",
    "BORE_GAS_VOL",
    "BORE_WAT_VOL",
    "FLOW_KIND",
    "WELL_TYPE",
]

DAILY_NUMERIC = [
    "ON_STREAM_HRS",
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_WHP_P",
    "BORE_OIL_VOL",
    "BORE_GAS_VOL",
    "BORE_WAT_VOL",
]


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.replace({"NULL": np.nan, "null": np.nan, "": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def _cache_file(src: Path, name: str) -> Path:
    stamp = int(src.stat().st_mtime)
    return CACHE_DIR / f"{name}-{stamp}.pkl"


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_pickle(path)
    except Exception:
        return None


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    prefix = path.name.split("-")[0]
    for old in CACHE_DIR.glob(f"{prefix}-*.pkl"):
        if old != path:
            old.unlink()
    df.to_pickle(path, protocol=4)


def load_daily(path: Optional[Path] = None) -> pd.DataFrame:
    src = Path(path or DATA_PATH)
    cache = _cache_file(src, "daily")
    cached = _read_cache(cache)
    if cached is not None:
        return cached

    df = pd.read_excel(src, sheet_name="Daily Production Data", usecols=DAILY_USECOLS)
    df["DATEPRD"] = pd.to_datetime(df["DATEPRD"])
    for col in DAILY_NUMERIC:
        if col in df.columns:
            df[col] = _to_numeric(df[col])
    df["WELL"] = df["NPD_WELL_BORE_NAME"].astype(str).str.strip()
    df["FLOW_KIND"] = df["FLOW_KIND"].astype(str).str.strip().str.lower()
    df["WELL_TYPE"] = df["WELL_TYPE"].astype(str).str.strip().str.upper()
    df["YEAR_MONTH"] = df["DATEPRD"].dt.to_period("M").dt.to_timestamp()
    df = df.sort_values(["WELL", "DATEPRD"]).reset_index(drop=True)
    _write_cache(cache, df)
    return df


def load_monthly(path: Optional[Path] = None) -> pd.DataFrame:
    src = Path(path or DATA_PATH)
    cache = _cache_file(src, "monthly")
    cached = _read_cache(cache)
    if cached is not None:
        return cached

    raw = pd.read_excel(src, sheet_name="Monthly Production Data", header=None)
    columns = [
        "WELL",
        "NPD_CODE",
        "YEAR",
        "MONTH",
        "ON_STREAM_HRS",
        "OIL",
        "GAS",
        "WATER",
        "GI",
        "WI",
    ]
    df = raw.iloc[2:].copy()
    df.columns = columns
    df["WELL"] = df["WELL"].astype(str).str.strip()
    for col in ["NPD_CODE", "YEAR", "MONTH", "ON_STREAM_HRS", "OIL", "GAS", "WATER", "GI", "WI"]:
        df[col] = _to_numeric(df[col])
    df = df.dropna(subset=["YEAR", "MONTH", "WELL"])
    df["YEAR"] = df["YEAR"].astype(int)
    df["MONTH"] = df["MONTH"].astype(int)
    df["DATE"] = pd.to_datetime(dict(year=df["YEAR"], month=df["MONTH"], day=1))
    df[["OIL", "GAS", "WATER", "ON_STREAM_HRS", "GI", "WI"]] = df[
        ["OIL", "GAS", "WATER", "ON_STREAM_HRS", "GI", "WI"]
    ].fillna(0.0)
    df = df.sort_values(["WELL", "DATE"]).reset_index(drop=True)
    _write_cache(cache, df)
    return df


def monthly_field(monthly: pd.DataFrame) -> pd.DataFrame:
    """Месячный профиль месторождения: сумма по стволам."""
    grouped = (
        monthly.groupby("DATE", as_index=False)
        .agg(
            oil=("OIL", "sum"),
            gas=("GAS", "sum"),
            water=("WATER", "sum"),
            on_stream=("ON_STREAM_HRS", "sum"),
            gi=("GI", "sum"),
            wi=("WI", "sum"),
            n_wells=("WELL", "nunique"),
        )
        .sort_values("DATE")
        .reset_index(drop=True)
    )
    producers = monthly.loc[monthly["OIL"] + monthly["GAS"] > 0]
    n_prod = producers.groupby("DATE")["WELL"].nunique().rename("n_producers")
    grouped = grouped.merge(n_prod, on="DATE", how="left")
    grouped["n_producers"] = grouped["n_producers"].fillna(0).astype(int)
    grouped["gor"] = np.where(grouped["oil"] > 0, grouped["gas"] / grouped["oil"], np.nan)
    grouped["water_cut"] = np.where(
        grouped["oil"] + grouped["water"] > 0,
        grouped["water"] / (grouped["oil"] + grouped["water"]),
        np.nan,
    )
    return grouped


def well_summary(daily: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    daily_agg = daily.groupby("WELL", as_index=False).agg(
        daily_rows=("DATEPRD", "size"),
        date_min=("DATEPRD", "min"),
        date_max=("DATEPRD", "max"),
        flow_kinds=("FLOW_KIND", lambda s: ", ".join(sorted(pd.unique(s.dropna())))),
        well_types=("WELL_TYPE", lambda s: ", ".join(sorted(pd.unique(s.dropna())))),
    )
    monthly_agg = monthly.groupby("WELL", as_index=False).agg(
        oil_sm3=("OIL", "sum"),
        gas_sm3=("GAS", "sum"),
        water_sm3=("WATER", "sum"),
        wi_sm3=("WI", "sum"),
        on_stream_hrs=("ON_STREAM_HRS", "sum"),
    )
    out = daily_agg.merge(monthly_agg, left_on="WELL", right_on="WELL", how="left")
    out = out.rename(columns={"WELL": "well"})
    return out.sort_values("oil_sm3", ascending=False).reset_index(drop=True)


def slice_history(field: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    out = field.loc[(field["DATE"] >= start_ts) & (field["DATE"] <= end_ts)]
    return out.reset_index(drop=True)


def complete_month_index(df: pd.DataFrame, date_col: str = "DATE") -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.date_range(df[date_col].min(), df[date_col].max(), freq="MS")
    out = df.set_index(date_col).reindex(idx)
    out.index.name = date_col
    return out.reset_index()
