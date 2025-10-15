import os
import requests
import pandas as pd
import numpy as np
from datetime import timedelta

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
LAT, LON = 35.8776, -78.7875          # RDU
TZ = "America/New_York"

# 取更长窗口，确保有足够历史去构造 lag/rolling
RAW_START = "2025-08-01"
RAW_END   = "2025-09-30"

# 需要导出的“测试窗口”
SLICE_START = "2025-09-17"
SLICE_END   = "2025-09-30"

OUT_PATH = "data/processed/cheat_test.csv"

# API 拉数
def fetch_weather(lat, lon, start_date, end_date, timezone=TZ):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "timezone": timezone
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if "hourly" not in data or "time" not in data["hourly"] or "temperature_2m" not in data["hourly"]:
        raise RuntimeError("API response missing hourly.time or hourly.temperature_2m")
    df = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "temp_obs": data["hourly"]["temperature_2m"],
    }).sort_values("time").reset_index(drop=True)
    return df

# 时间特征（与训练完全一致）
def add_time_features(df, ts_col="time"):
    out = df.copy()
    ts = pd.to_datetime(out[ts_col])
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    doy = ts.dt.dayofyear
    out["sin_doy"] = np.sin(2 * np.pi * doy / 365.0)
    out["cos_doy"] = np.cos(2 * np.pi * doy / 365.0)
    return out

def check_hourly(df, ts_col="time"):
    ts = pd.to_datetime(df[ts_col])
    if not ts.is_monotonic_increasing:
        raise ValueError("Timestamps are not sorted ascending.")
    diffs = ts.diff().dropna().unique()
    allowed = {pd.Timedelta(hours=1), pd.Timedelta(0)}
    if not set(diffs).issubset(allowed):
        raise ValueError("Found non-hourly gaps in the time series.")

# 构造 lag / rolling（与训练完全一致）
def add_temp_lag_and_roll(df):
    out = df.copy()
    # lags
    lags = [1, 24, 48, 72, 168]
    for L in lags:
        out[f"temp_lag_{L}"] = out["temp_obs"].shift(L)

    # diff
    out["temp_diff_1"] = out["temp_obs"].shift(1) - out["temp_obs"].shift(2)

    # rolling（min_periods=24 与训练一致）
    out["roll_mean_24"]  = out["temp_obs"].rolling(24,  min_periods=24).mean()
    out["roll_mean_168"] = out["temp_obs"].rolling(168, min_periods=24).mean()
    out["roll_std_24"]   = out["temp_obs"].rolling(24,  min_periods=24).std()
    out["roll_std_168"]  = out["temp_obs"].rolling(168, min_periods=24).std()

    # 丢掉前期缺历史导致的 NA
    out = out.dropna().reset_index(drop=True)
    return out

# 主流程
def main():
    print(f"Fetching {RAW_START} → {RAW_END} from Open-Meteo (timezone={TZ}) ...")
    df = fetch_weather(LAT, LON, RAW_START, RAW_END, TZ)
    check_hourly(df, "time")

    # 时间特征
    df = add_time_features(df, "time")
    # 温度派生特征（注意：这一步用到了测试期真实温度 → 会泄漏）
    df = add_temp_lag_and_roll(df)

    # 只保留 9/17–9/30（含当日 00:00 到 23:00）
    t0 = pd.to_datetime(SLICE_START)
    t1 = pd.to_datetime(SLICE_END) + pd.Timedelta(hours=23)
    mask = (df["time"] >= t0) & (df["time"] <= t1)
    df_slice = df.loc[mask].reset_index(drop=True)

    # 列顺序（与训练 CSV 风格一致：time 放最前）
    first = ["time"]
    others = [c for c in df_slice.columns if c != "time"]
    df_slice = df_slice[first + others]

    # 保存
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    df_slice.to_csv(OUT_PATH, index=False)

    print("\n[cheat_test] DONE")
    print(f"- Full window fetched: {RAW_START} → {RAW_END} ({len(df)} rows after dropna)")
    print(f"- Sliced window:       {SLICE_START} → {SLICE_END} ({len(df_slice)} rows)")
    print(f"- Saved to: {OUT_PATH}")
    print("\nHead:")
    print(df_slice.head())
    print("\nTail:")
    print(df_slice.tail())

if __name__ == "__main__":
    main()