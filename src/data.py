"""Загрузка и подготовка данных Volve."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import DATA_PATH


DAILY_NUMERIC = [
    "ON_STREAM_HRS",
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_DOWNHOLE_TEMPERATURE",
    "AVG_DP_TUBING",
    "AVG_ANNULUS_PRESS",
    "AVG_CHOKE_SIZE_P",
    "AVG_WHP_P",
    "AVG_WHT_P",
    "DP_CHOKE_SIZE",
    "BORE_OIL_VOL",
    "BORE_GAS_VOL",
    "BORE_WAT_VOL",
    "BORE_WI_VOL",
]


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.replace({"NULL": np.nan, "null": np.nan, "": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def load_daily(path: Optional[Path] = None) -> pd.DataFrame:
    df = pd.read_excel(path or DATA_PATH, sheet_name="Daily Production Data")
    df["DATEPRD"] = pd.to_datetime(df["DATEPRD"])
    for col in DAILY_NUMERIC:
        if col in df.columns:
            df[col] = _to_numeric(df[col])
    df["WELL"] = df["NPD_WELL_BORE_NAME"].astype(str).str.strip()
    df["FLOW_KIND"] = df["FLOW_KIND"].astype(str).str.strip().str.lower()
    df["WELL_TYPE"] = df["WELL_TYPE"].astype(str).str.strip().str.upper()
    df["YEAR_MONTH"] = df["DATEPRD"].dt.to_period("M").dt.to_timestamp()
    return df.sort_values(["WELL", "DATEPRD"]).reset_index(drop=True)


def load_monthly(path: Optional[Path] = None) -> pd.DataFrame:
    raw = pd.read_excel(path or DATA_PATH, sheet_name="Monthly Production Data", header=None)
    # Первая строка — имена, вторая — единицы измерения.
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
    return df.sort_values(["WELL", "DATE"]).reset_index(drop=True)


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
    rows = []
    for well, part in daily.groupby("WELL"):
        mpart = monthly.loc[monthly["WELL"] == well]
        rows.append(
            {
                "well": well,
                "daily_rows": int(len(part)),
                "date_min": part["DATEPRD"].min(),
                "date_max": part["DATEPRD"].max(),
                "flow_kinds": ", ".join(sorted(part["FLOW_KIND"].dropna().unique())),
                "well_types": ", ".join(sorted(part["WELL_TYPE"].dropna().unique())),
                "oil_sm3": float(mpart["OIL"].sum()),
                "gas_sm3": float(mpart["GAS"].sum()),
                "water_sm3": float(mpart["WATER"].sum()),
                "wi_sm3": float(mpart["WI"].sum()),
                "on_stream_hrs": float(mpart["ON_STREAM_HRS"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("oil_sm3", ascending=False).reset_index(drop=True)


def slice_history(field: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    out = field.loc[(field["DATE"] >= start_ts) & (field["DATE"] <= end_ts)].copy()
    return out.reset_index(drop=True)


def complete_month_index(df: pd.DataFrame, date_col: str = "DATE") -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.date_range(df[date_col].min(), df[date_col].max(), freq="MS")
    out = df.set_index(date_col).reindex(idx)
    out.index.name = date_col
    return out.reset_index()
