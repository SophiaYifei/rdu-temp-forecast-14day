import requests
import pandas as pd
from datetime import datetime

# API configuration
BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"
TOKEN = "TvwGuenbBWZMrbxmzPhfitsWDxKSwwnx" 

headers = {
    'token': TOKEN
}

# Rate limits: 5 requests/second, 10,000/day

# Search for RDU airport stations
def find_rdu_stations():
    url = f"{BASE_URL}/stations"
    params = {
        'locationid': 'FIPS:37',  # North Carolina
        'limit': 1000
    }
    response = requests.get(url, headers=headers, params=params)
    stations = response.json()
    
    # Filter for RDU area stations
    rdu_stations = [s for s in stations['results'] 
                   if 'RDU' in s['name'] or 'RALEIGH' in s['name']]
    return rdu_stations

def get_temperature_data(station_id, start_date, end_date):
    url = f"{BASE_URL}/data"
    params = {
        'datasetid': 'GHCND',  # Global Historical Climatology Network Daily
        'stationid': station_id,
        'datatypeid': 'TMAX,TMIN,TAVG',  # Max, Min, Average temp
        'startdate': start_date,
        'enddate': end_date,
        'limit': 1000,
        'units': 'standard'  # Fahrenheit
    }
    
    response = requests.get(url, headers=headers, params=params)
    return response.json()

# Example usage
data = get_temperature_data('GHCND:USW00013722', '2024-01-01', '2024-09-16')

print(data)