"""
Build features for training and inference with simple.

We only use:
- calendar time features (hour, day_of_week, is_weekend, sin/cos)
- past observed temperature (lags and simple rolling stats) for TRAIN
- time features only for INFER (lags/rolling are handled later in make_forecast.py)

How to use:
1) Open this file.
2) Go to the bottom and set MODE to "train" or "infer".
3) Adjust the file paths and dates if needed.
4) Run:  python src/features/build_features.py
"""

import os
import pandas as pd
import numpy as np

# Time feature helpers
def add_time_features(df, ts_col="time"):
    """
    Add basic calendar and cyclical features.
    These features do not use any future information, so they are safe.
    """
    out = df.copy()
    ts = pd.to_datetime(out[ts_col])

    # basic integer features
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    # cyclical encoding for hour and day-of-year
    out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24.0)

    doy = ts.dt.dayofyear
    out["sin_doy"] = np.sin(2 * np.pi * doy / 365.0)
    out["cos_doy"] = np.cos(2 * np.pi * doy / 365.0)

    return out


def check_hourly(df, ts_col="time"):
    """
    Make sure the time column is sorted and roughly hourly.
    I allow a 0-hour gap because of possible DST duplicate hour.
    """
    ts = pd.to_datetime(df[ts_col])
    if not ts.is_monotonic_increasing:
        raise ValueError("Timestamps are not sorted ascending.")
    diffs = ts.diff().dropna().unique()
    allowed = {pd.Timedelta(hours=1), pd.Timedelta(0)}
    if not set(diffs).issubset(allowed):
        raise ValueError("Found non-hourly gaps in the time series.")

# TRAIN: build features from observed temperature
def build_train_features(staging_path, cutoff, out_path, timezone_note="America/New_York"):
    """
    Read the standardized base table from staging.
    It must have:
      - time
      - temp_obs (observed temperature in Celsius)

    We clip to the cutoff time to avoid any leakage.
    Then we add time features, and temperature lags/rolling.
    Finally we save a single CSV that the model can read.
    """
    # 1) Read and basic checks
    df = pd.read_csv(staging_path)
    if "time" not in df.columns or "temp_obs" not in df.columns:
        raise ValueError("Staging file must have 'time' and 'temp_obs' columns.")
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    check_hourly(df, "time")

    # 2) Clip to cutoff (no future info in train)
    cutoff_ts = pd.to_datetime(cutoff)
    df = df[df["time"] <= cutoff_ts].copy()
    if df.empty:
        raise ValueError("Training data is empty after applying the cutoff.")

    # 3) Add time features
    df = add_time_features(df, "time")

    # 4) Build temperature lags and simple rolling means/stds (from observed temp only)
    lags = [1, 24, 48, 72, 168]  # 1h, 1d, 2d, 3d, 7d
    for L in lags:
        df[f"temp_lag_{L}"] = df["temp_obs"].shift(L)

    # simple first difference (a tiny trend signal)
    df["temp_diff_1"] = df["temp_obs"].shift(1) - df["temp_obs"].shift(2)

    # rolling stats for recent day and recent week
    df["roll_mean_24"] = df["temp_obs"].rolling(24, min_periods=24).mean()
    df["roll_mean_168"] = df["temp_obs"].rolling(168, min_periods=24).mean()
    df["roll_std_24"] = df["temp_obs"].rolling(24, min_periods=24).std()
    df["roll_std_168"] = df["temp_obs"].rolling(168, min_periods=24).std()

    # 5) Drop rows that do not have enough history for these features
    df = df.dropna().reset_index(drop=True)

    # 6) Reorder columns so time is first (just for readability)
    cols = ["time"] + [c for c in df.columns if c != "time"]
    df = df[cols]

    # 7) Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[TRAIN] timezone note: {timezone_note}")
    print(f"[TRAIN] rows: {len(df)}, columns: {len(df.columns)}")
    print(f"[TRAIN] saved to: {out_path}")

# INFER: build day-0 legal feature frame (time only)
def build_infer_features(t0, t1, out_path, timezone="America/New_York"):
    """
    Create inference features for the forecast horizon.
    We only create time features here. The temperature lags/rolling
    will be computed during rolling prediction inside make_forecast.py
    using the seed history (t0- history + previous predictions).
    """
    # 1) Create an hourly range from t0 00:00 to t1 23:00
    start = pd.to_datetime(t0)
    end = pd.to_datetime(t1) + pd.Timedelta(hours=23)
    ts = pd.date_range(start=start, end=end, freq="h")
    df = pd.DataFrame({"time": ts})

    # 2) Add time features
    df = add_time_features(df, "time")

    # 3) Safety: inference features should NOT contain observed temperature
    for bad in ["temp_obs", "temperature_2m"]:
        if bad in df.columns:
            raise ValueError("Inference features should not contain observed temperature columns.")

    # 4) Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[INFER] timezone: {timezone}")
    print(f"[INFER] rows: {len(df)}, columns: {len(df.columns)}")
    print(f"[INFER] saved to: {out_path}")

# Main: set MODE and paths here
if __name__ == "__main__":
    # 1) Build training features
    STAGING_PATH = "data/staging/rdu_weather_standardized.csv"
    CUTOFF = "2025-09-16 23:00"
    TRAIN_OUT = "data/processed/xgb_train_full.csv"
    TIMEZONE_NOTE = "America/New_York"

    build_train_features(
        staging_path=STAGING_PATH,
        cutoff=CUTOFF,
        out_path=TRAIN_OUT,
        timezone_note=TIMEZONE_NOTE,
    )

    # 2) Build inference features
    T0 = "2025-09-17"
    T1 = "2025-09-30"
    INFER_OUT = "data/processed/infer_features_2025-09-17_to_09-30.csv"
    TIMEZONE = "America/New_York"

    build_infer_features(
        t0=T0,
        t1=T1,
        out_path=INFER_OUT,
        timezone=TIMEZONE,
    )

    print("[DONE] Created both training and inference features.")
