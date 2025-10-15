# import os
# import requests
# import pandas as pd

# OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
# LAT, LON = 35.8776, -78.7875  # RDU Airport
# TZ = "America/New_York"
# START_DATE = "2025-09-17"
# END_DATE   = "2025-09-30"
# OUT_PATH = "data/processed/xgb_test_truth_2025-09-17_to_09-30.csv"

# params = {
#     "latitude": LAT,
#     "longitude": LON,
#     "start_date": START_DATE,
#     "end_date": END_DATE,
#     "hourly": "temperature_2m",
#     "timezone": TZ,
# }

# print(f"Fetching hourly temperature from {START_DATE} to {END_DATE} ...")
# resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
# resp.raise_for_status()
# data = resp.json()

# hourly = data["hourly"]
# df = pd.DataFrame({
#     "time": pd.to_datetime(hourly["time"]),
#     "temp_obs": hourly["temperature_2m"],
# }).sort_values("time").reset_index(drop=True)

# os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
# df.to_csv(OUT_PATH, index=False)

# print(f"\n✓ Saved {len(df)} rows to {OUT_PATH}")
# print(df.head())

"""
准备test data with历史climatology作为lag features
策略：
1. 能用2025年真实lag就用真实lag
2. 不够的用2020-2024年对应日期时间的平均值
"""
import os
import requests
import pandas as pd
import numpy as np

# ============================================================================
# Part 1: 获取test period的真实温度数据（用于时间特征和真实值对比）
# ============================================================================
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
LAT, LON = 35.8776, -78.7875
TZ = "America/New_York"
TEST_START = "2025-09-17"
TEST_END = "2025-09-30"

params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": TEST_START,
    "end_date": TEST_END,
    "hourly": "temperature_2m",
    "timezone": TZ,
}

print(f"Fetching test period data: {TEST_START} to {TEST_END} ...")
resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
resp.raise_for_status()
data = resp.json()

hourly = data["hourly"]
test_df = pd.DataFrame({
    "time": pd.to_datetime(hourly["time"]),
    "temp_obs": hourly["temperature_2m"],
}).sort_values("time").reset_index(drop=True)

# ============================================================================
# Part 2: 添加时间特征（和build_features.py完全一样）
# ============================================================================
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

test_df = add_time_features(test_df, "time")

# ============================================================================
# Part 3: 读取历史数据用于构建climatology
# ============================================================================
# 读取你的完整历史数据（2020-2025-09-16）
HISTORY_PATH = "data/staging/rdu_weather_standardized.csv"
history_df = pd.read_csv(HISTORY_PATH)
history_df["time"] = pd.to_datetime(history_df["time"])

# 添加辅助列用于匹配
history_df["month"] = history_df["time"].dt.month
history_df["day"] = history_df["time"].dt.day
history_df["hour"] = history_df["time"].dt.hour
history_df["year"] = history_df["time"].dt.year

print(f"\nLoaded history data: {len(history_df)} rows from {history_df['time'].min()} to {history_df['time'].max()}")

# ============================================================================
# Part 4: 为test_df构建lag features
# ============================================================================
def get_lag_value(target_time, lag_hours, history_df, use_climatology_years=[2020, 2021, 2022, 2023, 2024]):
    """
    获取某个时间点的lag值
    优先用真实值，如果不存在则用历史同期均值
    
    target_time: 目标时间点（如2025-09-17 05:00）
    lag_hours: lag小时数（如24）
    history_df: 历史数据
    use_climatology_years: 用于计算climatology的年份
    """
    # 计算需要的历史时间点
    lag_time = target_time - pd.Timedelta(hours=lag_hours)
    
    # 先尝试找真实值
    real_value = history_df[history_df["time"] == lag_time]["temp_obs"]
    if len(real_value) > 0:
        return real_value.iloc[0], "real"
    
    # 如果没有真实值，用历史同期均值
    lag_month = lag_time.month
    lag_day = lag_time.day
    lag_hour = lag_time.hour
    
    climatology_data = history_df[
        (history_df["month"] == lag_month) &
        (history_df["day"] == lag_day) &
        (history_df["hour"] == lag_hour) &
        (history_df["year"].isin(use_climatology_years))
    ]["temp_obs"]
    
    if len(climatology_data) > 0:
        return climatology_data.mean(), "climatology"
    else:
        return np.nan, "missing"

# 为每一行构建lag features
print("\nBuilding lag features...")
lag_configs = [1, 24, 48, 72, 168]

for lag_h in lag_configs:
    values = []
    sources = []
    for idx, row in test_df.iterrows():
        val, source = get_lag_value(row["time"], lag_h, history_df)
        values.append(val)
        sources.append(source)
    
    test_df[f"temp_lag_{lag_h}"] = values
    test_df[f"temp_lag_{lag_h}_source"] = sources  # 记录数据来源，方便debug
    
    real_count = sum([1 for s in sources if s == "real"])
    clim_count = sum([1 for s in sources if s == "climatology"])
    print(f"  lag_{lag_h}: {real_count} real, {clim_count} climatology")

# ============================================================================
# Part 5: 构建diff和rolling features
# ============================================================================
print("\nBuilding diff and rolling features...")

# Diff feature: 用lag_1和lag_2
test_df["temp_diff_1"] = test_df["temp_lag_1"] - test_df.apply(
    lambda row: get_lag_value(row["time"], 2, history_df)[0], axis=1
)

# Rolling features: 用历史同期的rolling统计
def get_rolling_stats(target_time, window_hours, history_df, use_climatology_years=[2020, 2021, 2022, 2023, 2024]):
    """
    获取rolling统计量，基于历史同期数据
    """
    # 获取过去window_hours的所有时间点
    times_needed = [target_time - pd.Timedelta(hours=i) for i in range(1, window_hours + 1)]
    
    temps = []
    for t in times_needed:
        # 尝试获取真实值
        real = history_df[history_df["time"] == t]["temp_obs"]
        if len(real) > 0:
            temps.append(real.iloc[0])
        else:
            # 用climatology
            clim = history_df[
                (history_df["month"] == t.month) &
                (history_df["day"] == t.day) &
                (history_df["hour"] == t.hour) &
                (history_df["year"].isin(use_climatology_years))
            ]["temp_obs"]
            if len(clim) > 0:
                temps.append(clim.mean())
    
    if len(temps) >= max(24, int(window_hours * 0.5)):  # 至少要有一半数据
        return np.mean(temps), np.std(temps)
    else:
        return np.nan, np.nan

# 计算rolling_24
roll_24_means = []
roll_24_stds = []
for idx, row in test_df.iterrows():
    mean_val, std_val = get_rolling_stats(row["time"], 24, history_df)
    roll_24_means.append(mean_val)
    roll_24_stds.append(std_val)

test_df["roll_mean_24"] = roll_24_means
test_df["roll_std_24"] = roll_24_stds

# 计算rolling_168
print("  Computing roll_mean_168 and roll_std_168 (this may take a moment)...")
roll_168_means = []
roll_168_stds = []
for idx, row in test_df.iterrows():
    mean_val, std_val = get_rolling_stats(row["time"], 168, history_df)
    roll_168_means.append(mean_val)
    roll_168_stds.append(std_val)
    if (idx + 1) % 50 == 0:
        print(f"    Processed {idx + 1}/{len(test_df)} rows...")

test_df["roll_mean_168"] = roll_168_means
test_df["roll_std_168"] = roll_168_stds

# ============================================================================
# Part 6: 保存结果
# ============================================================================
OUT_PATH = "data/processed/xgb_test_with_climatology_2025-09-17_to_09-30.csv"
os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)

# 保存时可以选择保留或删除source列（用于debug）
# test_df_final = test_df.drop(columns=[c for c in test_df.columns if "_source" in c])
test_df.to_csv(OUT_PATH, index=False)

print(f"\n✓ Saved {len(test_df)} rows to {OUT_PATH}")
print("\nFeatures summary:")
print(f"  Time features: hour, day_of_week, is_weekend, sin/cos")
print(f"  Lag features: temp_lag_1, 24, 48, 72, 168")
print(f"  Diff feature: temp_diff_1")
print(f"  Rolling features: roll_mean_24, roll_std_24, roll_mean_168, roll_std_168")
print(f"  Target: temp_obs (actual temperature for evaluation)")

print("\nFirst few rows:")
print(test_df[["time", "temp_obs", "temp_lag_1", "temp_lag_24", "temp_lag_168"]].head(10))

print("\nData source statistics:")
for lag_h in lag_configs:
    source_col = f"temp_lag_{lag_h}_source"
    print(f"\n  temp_lag_{lag_h}:")
    print(test_df[source_col].value_counts())