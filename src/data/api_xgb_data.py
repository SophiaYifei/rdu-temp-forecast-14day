import os
import requests
import pandas as pd
from datetime import timedelta

# I keep a single base URL so it is easy to change later if needed.
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_weather(lat, lon, start_date, end_date, timezone="America/New_York"):
    """
    Call the Open-Meteo API and return the JSON.
    I only request hourly temperature to keep it simple.
    The API returns local-time strings when a timezone is provided.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "timezone": timezone
    }

    # Make the request. If it fails, raise an error with details.
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"API request failed: {resp.status_code}\n{resp.text}")

    data = resp.json()

    # Basic keys check to avoid silent errors.
    if "hourly" not in data or "time" not in data["hourly"] or "temperature_2m" not in data["hourly"]:
        raise RuntimeError("API response is missing 'hourly.time' or 'hourly.temperature_2m'.")

    return data


def json_to_dataframe(data):
    """
    Convert the JSON payload into a simple DataFrame with:
    - time: naive local timestamps (no tz info)
    - temp_obs: temperature in Celsius
    """
    hourly = data["hourly"]
    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),   # ✅ 改名：原来是 'timestamp'
        "temp_obs": hourly["temperature_2m"]
    })
    df = df.sort_values("time").reset_index(drop=True)
    return df


def clip_to_exact_range(df, start_date, end_date):
    """
    Keep rows from start 00:00 to end 23:00 inclusive.
    """
    t0 = pd.to_datetime(start_date)
    t1 = pd.to_datetime(end_date) + timedelta(hours=23)
    mask = (df["time"] >= t0) & (df["time"] <= t1)   # ✅ 改为 'time'
    df = df.loc[mask].reset_index(drop=True)
    return df


def save_csv(df, path):
    """
    Save a CSV. Create the folder if it does not exist.
    """
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    df.to_csv(path, index=False)


def make_year_chunks(start_date, end_date):
    """
    Make small yearly chunks from a big continuous range.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    chunks = []
    year = start.year
    while year <= end.year:
        y0 = pd.Timestamp(year=year, month=1, day=1)
        y1 = pd.Timestamp(year=year, month=12, day=31)
        s = max(start, y0)
        e = min(end, y1)
        chunks.append((s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
        year += 1
    return chunks


if __name__ == "__main__":
    LAT = 35.8776
    LON = -78.7875
    TZ = "America/New_York"

    START_DATE = "2020-01-01"
    END_DATE = "2025-09-16"

    RAW_OUT = "data/raw/openmeteo_rdu_2020-01-01_to_2025-09-16.csv"
    STANDARD_OUT = "data/staging/rdu_weather_standardized.csv"

    dfs = []
    year_chunks = make_year_chunks(START_DATE, END_DATE)
    for (start_d, end_d) in year_chunks:
        print(f"Fetching {start_d} to {end_d} ...")
        data = fetch_weather(LAT, LON, start_d, end_d, TZ)
        df_part = json_to_dataframe(data)
        df_part = clip_to_exact_range(df_part, start_d, end_d)
        dfs.append(df_part)

    if len(dfs) == 0:
        raise RuntimeError("No data fetched. Please check your ranges.")
    df_all = pd.concat(dfs, axis=0, ignore_index=True)
    df_all = df_all.sort_values("time").reset_index(drop=True)  # ✅ 改为 'time'

    before = len(df_all)
    df_all = df_all.drop_duplicates(subset=["time"], keep="first").reset_index(drop=True)
    dropped = before - len(df_all)
    if dropped > 0:
        print(f"Note: dropped {dropped} duplicate timestamps (likely DST fall-back).")

    # --- Sanity checks ---
    assert df_all["time"].is_monotonic_increasing, "Timestamps are not strictly increasing."
    assert df_all["time"].dt.hour.isin(range(24)).all(), "There are non-hourly timestamps."

    diffs = df_all["time"].diff().dropna()
    bad = (diffs != pd.Timedelta(hours=1)).sum()
    if bad > 0:
        print(f"Note: found {bad} non-1h steps (DST or missing hours). I will still save the data.")

    save_csv(df_all, RAW_OUT)
    save_csv(df_all, STANDARD_OUT)

    print(f"Configured range: {START_DATE} to {END_DATE}")
    print(f"\nSaved standardized base table to: {STANDARD_OUT}")
    print(f"Total rows: {len(df_all)}")
    print(f"From: {df_all['time'].iloc[0]}  To: {df_all['time'].iloc[-1]}")
    print("\nFirst 5 rows:")
    print(df_all.head())
    print("\nLast 5 rows:")
    print(df_all.tail())
