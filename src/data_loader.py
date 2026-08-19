"""Fetches real day-ahead electricity prices (and grid load) for Germany/
Luxembourg from SMARD (Bundesnetzagentur's official market data platform).

SMARD serves data in weekly buckets keyed by a Monday-anchored timestamp.
There is no bulk-range endpoint, so building a multi-year series means
walking the weekly index and concatenating each week's response.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.smard.de/app/chart_data"
PRICE_FILTER = 4169  # Day-ahead auction price, EUR/MWh
LOAD_FILTER = 410  # Actual total grid load, MW
REGION = "DE-LU"

RAW_DIR = Path(__file__).resolve().parent.parent / "data"


def _fetch_index(filter_id: int) -> list[int]:
    url = f"{BASE}/{filter_id}/{REGION}/index_hour.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()["timestamps"]


def _fetch_week(filter_id: int, week_ts: int) -> list[tuple[int, float | None]]:
    url = f"{BASE}/{filter_id}/{REGION}/{filter_id}_{REGION}_hour_{week_ts}.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()["series"]


def fetch_series(filter_id: int, n_weeks: int, pause_s: float = 0.05) -> pd.Series:
    """Fetch the most recent n_weeks of hourly data for a given SMARD filter."""
    index = _fetch_index(filter_id)
    week_stamps = index[-n_weeks:]

    rows: list[tuple[int, float | None]] = []
    for i, ts in enumerate(week_stamps):
        rows.extend(_fetch_week(filter_id, ts))
        if pause_s:
            time.sleep(pause_s)
        if (i + 1) % 20 == 0:
            print(f"  fetched {i + 1}/{len(week_stamps)} weeks")

    df = pd.DataFrame(rows, columns=["timestamp_ms", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    return df["value"]


def load_or_fetch(n_weeks: int = 156, force_refresh: bool = False) -> pd.DataFrame:
    """Returns a DataFrame with columns [price_eur_mwh, load_mw], hourly index (UTC).

    Caches to data/electricity_raw.csv so repeated runs don't re-hit the API.
    """
    cache_path = RAW_DIR / "electricity_raw.csv"
    if cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    print(f"Fetching {n_weeks} weeks of day-ahead price data from SMARD...")
    price = fetch_series(PRICE_FILTER, n_weeks)
    print(f"Fetching {n_weeks} weeks of grid load data from SMARD...")
    load = fetch_series(LOAD_FILTER, n_weeks)

    df = pd.DataFrame({"price_eur_mwh": price, "load_mw": load})
    df = df.dropna(subset=["price_eur_mwh"])

    RAW_DIR.mkdir(exist_ok=True)
    df.to_csv(cache_path)
    print(f"Saved {len(df)} hourly rows to {cache_path}")
    return df


if __name__ == "__main__":
    df = load_or_fetch(n_weeks=156, force_refresh=True)
    print(df.describe())
    print(df.head())
    print(df.tail())
