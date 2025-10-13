import requests
import pandas as pd

# Open meteo api for training and test set data retrieval
# https://open-meteo.com/

def get_test_weather_data(lat, lon, start_date, end_date):
    """
    Retrieve actual temperature observations for the test period.
    This serves as the ground truth for model evaluation.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"

    # Retrieve features you want to train your model with
    # Look at https://open-meteo.com/en/docs/historical-weather-api?latitude=35.8776&longitude=-78.7875&timezone=America%2FNew_York&start_date=2025-09-17&end_date=2025-09-30#api_response
    # for param types

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,dew_point_2m",  # Only temperature
        "timezone": "America/New_York"
    }
    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")

    return response.json()

def save_test_set(data, output_path):
    """
    Save test set data to CSV with summary information.
    """
    # Extract hourly data
    df = pd.DataFrame(data['hourly'])

    # Verify date range
    first_time = df['time'].iloc[0]
    last_time = df['time'].iloc[-1]

    # Display summary
    print(f"\n{'='*60}")
    print(f"TEST SET DATA RETRIEVED")
    print(f"{'='*60}")
    print(f"Records: {len(df)} hourly observations")
    print(f"Location: RDU Airport ({data['latitude']}, {data['longitude']})")
    print(f"Timezone: {data['timezone']}")
    print(f"First hour: {first_time}")
    print(f"Last hour:  {last_time}")
    print(f"{'='*60}\n")

    # Verify we have exactly 336 hours (14 days * 24 hours)
    # Sep 17 00:00 through Sep 30 23:00 = 14 days × 24 hours = 336 hours
    expected_hours = 14 * 24
    if len(df) == expected_hours:
        print(f"✓ Correct number of hours: {expected_hours}")
    else:
        print(f"⚠ Warning: Expected {expected_hours} hours, got {len(df)}")

    # Verify first and last timestamps
    if first_time == "2025-09-17T00:00" and last_time == "2025-09-30T23:00":
        print(f"✓ Date range verified: Sep 17 00:00 - Sep 30 23:00")
    else:
        print(f"⚠ Warning: Unexpected date range!")

    print(f"\nFirst 10 rows:")
    print(df.head(10))
    print(f"\nLast 10 rows:")
    print(df.tail(10))

    print(f"\n{'='*60}")
    print("Temperature Statistics (°C):")
    print(f"{'='*60}")
    print(df['temperature_2m'].describe())

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\n✓ Test set saved to: {output_path}")

if __name__ == "__main__":
    # RDU Airport coordinates
    RDU_LAT = 35.8776
    RDU_LON = -78.7875

    # Test period: Sep 17 00:00 - Sep 30 23:00, 2025
    START_DATE = "2025-09-17"
    END_DATE = "2025-09-30"

    print(f"\nRetrieving test set data from Open-Meteo API...")
    print(f"Target: Sep 17, 2025 00:00 through Sep 30, 2025 23:00 (336 hours)")

    data = get_test_weather_data(RDU_LAT, RDU_LON, START_DATE, END_DATE)
    save_test_set(data, "data/raw/rdu_test_set_2025-09-17_to_2025-09-30.csv")
