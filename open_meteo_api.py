import requests
import pandas as pd

def get_historical_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,pressure_msl",
        "timezone": "America/New_York"
    }
    response = requests.get(url, params=params)
    return response.json()

def save_to_csv(data, output_path):
    # Extract hourly data
    df = pd.DataFrame(data['hourly'])

    # Display summary
    print(f"\n{'='*60}")
    print(f"Data Retrieved: {len(df)} hourly records")
    print(f"Location: {data['latitude']}, {data['longitude']}")
    print(f"Timezone: {data['timezone']}")
    print(f"Date Range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    print(f"{'='*60}\n")

    print("First 5 rows:")
    print(df.head())
    print(f"\nLast 5 rows:")
    print(df.tail())
    print(f"\nSummary Statistics:")
    print(df.describe())

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\n✓ Data saved to: {output_path}")

# RDU coordinates
data = get_historical_weather(35.8776, -78.7875, "2025-08-17", "2025-08-30")
save_to_csv(data, "data/raw/rdu_historical_2024-01-01_to_2024-09-16.csv")