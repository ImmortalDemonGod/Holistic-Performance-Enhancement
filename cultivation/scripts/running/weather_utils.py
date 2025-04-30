import requests
import time
from datetime import timedelta

def fetch_weather_open_meteo(lat, lon, dt, max_retries=6, max_backoff=5.0):
    """
    Robustly fetch weather for a given latitude, longitude, and datetime (UTC).
    Implements exponential backoff on network/API errors.
    Tries variations in time (±0 to ±5 hours) on the same day and location (rounded, nudged lat/lon) until data is found.
    Returns dict or None if all reasonable attempts fail.
    """
    time_offsets = [timedelta(hours=h) for h in range(-5,6)]
    lat_variations = [lat, round(lat,3), round(lat,2), lat+0.01, lat-0.01, lat+0.05, lat-0.05, lat+0.1, lat-0.1]
    lon_variations = [lon, round(lon,3), round(lon,2), lon+0.01, lon-0.01, lon+0.05, lon-0.05, lon+0.1, lon-0.1]
    date_str = dt.strftime('%Y-%m-%d')
    tried = set()
    attempt = 0
    while attempt < max_retries:
        for offset in time_offsets:
            dt_try = dt + offset
            if dt_try.strftime('%Y-%m-%d') != date_str:
                continue
            hour_iso = dt_try.strftime('%Y-%m-%dT%H:00')
            for lat_try in lat_variations:
                for lon_try in lon_variations:
                    key = (hour_iso, lat_try, lon_try)
                    if key in tried:
                        continue
                    tried.add(key)
                    url = (
                        f"https://archive-api.open-meteo.com/v1/archive?"
                        f"latitude={lat_try:.4f}&longitude={lon_try:.4f}"
                        f"&start_date={date_str}&end_date={date_str}"
                        f"&hourly=temperature_2m,relative_humidity_2m,precipitation,apparent_temperature,wind_speed_10m,weather_code"
                        f"&timezone=auto"
                    )
                    try:
                        resp = requests.get(url, timeout=10)
                        if resp.status_code != 200:
                            continue
                        data = resp.json()
                        if 'hourly' not in data or 'time' not in data['hourly']:
                            continue
                        times = data['hourly']['time']
                        if hour_iso not in times:
                            continue
                        idx = times.index(hour_iso)
                        weather = {
                            'temperature_c': data['hourly']['temperature_2m'][idx],
                            'apparent_temp_c': data['hourly']['apparent_temperature'][idx],
                            'precipitation_mm': data['hourly']['precipitation'][idx],
                            'humidity_percent': data['hourly']['relative_humidity_2m'][idx],
                            'wind_speed_kmh': data['hourly']['wind_speed_10m'][idx],
                            'weather_code': data['hourly']['weather_code'][idx],
                        }
                        code_map = {0:'Clear sky',1:'Mainly clear',2:'Partly cloudy',3:'Overcast',45:'Fog',48:'Rime fog',51:'Light drizzle',53:'Moderate drizzle',55:'Dense drizzle',61:'Slight rain',63:'Moderate rain',65:'Heavy rain',71:'Slight snow',73:'Moderate snow',75:'Heavy snow',80:'Slight rain showers',81:'Moderate rain showers',82:'Violent rain showers',95:'Thunderstorm'}
                        weather['weather_description'] = code_map.get(weather['weather_code'], f"Code {weather['weather_code']}")
                        if weather['temperature_c'] is not None:
                            # Debug log for fallback: compare original vs. used lat/lon/time
                            if (lat_try != lat or lon_try != lon or hour_iso != dt.isoformat()[:13]+":00:00"):
                                print(f"[Weather][DEBUG] Fallback used. Original: lat={lat}, lon={lon}, time={dt.isoformat()}. Used: lat={lat_try}, lon={lon_try}, time={hour_iso}. Weather: {weather}")
                            else:
                                print(f"[Weather] Success: lat={lat_try:.4f}, lon={lon_try:.4f}, time={hour_iso}")
                            return weather
                    except Exception as e:
                        print(f"[Weather] Error: {e} (lat={lat_try}, lon={lon_try}, time={hour_iso})")
        # If we reach here, all variations failed for this attempt
        backoff = min(max_backoff, 0.2 * (2 ** attempt))
        print(f"[Weather][Backoff] Attempt {attempt+1} failed, retrying in {backoff:.2f}s...")
        time.sleep(backoff)
        attempt += 1
    print(f"[Weather] Failed to fetch weather after {max_retries} retries for lat={lat}, lon={lon}, time={dt}")
    return None
