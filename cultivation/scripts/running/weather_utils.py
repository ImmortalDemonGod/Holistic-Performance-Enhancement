import requests
import time
from datetime import timedelta
import pandas as pd
from pathlib import Path

CACHE_PATH = Path(__file__).parents[2] / 'data' / 'weather_cache.parquet'

# Load cache if exists
if CACHE_PATH.exists():
    weather_cache = pd.read_parquet(CACHE_PATH)
else:
    weather_cache = pd.DataFrame(columns=['lat','lon','date','weather'])

def fetch_weather_open_meteo(lat, lon, dt, max_retries=6, max_backoff=2.0):
    global weather_cache
    """
    Robustly fetch weather for a given latitude, longitude, and datetime (UTC).
    Implements exponential backoff on network/API errors.
    Tries variations in time (±0 to ±5 hours) on the same day and location (rounded, nudged lat/lon) until data is found.
    Returns:
        tuple: (weather_dict, offset_hours) if successful, or (None, None) if all attempts fail.
    """
    # Check cache first
    date_str = pd.to_datetime(dt).strftime('%Y-%m-%d')
    cached = weather_cache[(weather_cache['lat']==lat)&(weather_cache['lon']==lon)&(weather_cache['date']==date_str)]
    if not cached.empty:
        return cached.iloc[0]['weather'], 0

    time_offsets = [timedelta(hours=h) for h in range(-5,6)]
    lat_variations = [lat, round(lat,3), round(lat,2), lat+0.01, lat-0.01, lat+0.05, lat-0.05, lat+0.1, lat-0.1]
    lon_variations = [lon, round(lon,3), round(lon,2), lon+0.01, lon-0.01, lon+0.05, lon-0.05, lon+0.1, lon-0.1]
    attempt = 0
    while attempt < max_retries:
        for lat_try in lat_variations:
            for lon_try in lon_variations:
                for offset in time_offsets:
                    hour_iso = (dt + offset).replace(minute=0, second=0, microsecond=0).isoformat()
                    try:
                        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat_try}&longitude={lon_try}&hourly=temperature_2m,precipitation,weathercode,windspeed_10m,windgusts_10m,winddirection_10m&start={hour_iso}&end={hour_iso}&timezone=UTC"
                        resp = requests.get(url, timeout=6)
                        if resp.status_code == 200:
                            weather = resp.json()
                            if 'hourly' in weather and weather['hourly']['temperature_2m']:
                                result = weather
                                # Save to cache
                                weather_cache = pd.concat([weather_cache, pd.DataFrame([{'lat':lat,'lon':lon,'date':date_str,'weather':result}])], ignore_index=True)
                                weather_cache.to_parquet(CACHE_PATH, index=False)
                                return result, offset.total_seconds() / 3600
                    except Exception as e:
                        print(f"[Weather] Error: {e} (lat={lat_try}, lon={lon_try}, time={hour_iso})")
        # If we reach here, all variations failed for this attempt
        backoff = min(max_backoff, 0.2 * (2 ** attempt))
        print(f"[Weather][Backoff] Attempt {attempt+1} failed, retrying in {backoff:.2f}s...")
        time.sleep(backoff)
        attempt += 1
    print(f"[Weather] Failed to fetch weather after {max_retries} retries for lat={lat}, lon={lon}, time={dt}")
    return None, None
