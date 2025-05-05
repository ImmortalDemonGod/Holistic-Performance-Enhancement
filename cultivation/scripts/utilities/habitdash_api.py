import requests
import os
import time
import logging
from dotenv import load_dotenv
from typing import Optional, Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load API Key from .env
load_dotenv()
API_KEY = os.getenv("HABITDASH_API_KEY")

BASE_URL = "https://api.habitdash.com/v1"
REQUEST_DELAY_SECONDS = 0.6 # To stay under 2 requests/sec limit

# Field IDs for easy metric access
FIELD_IDS = {
    "whoop": {
        "hrv": 86,  # Heart Rate Variability
        "rhr": 87,  # Resting Heart Rate
        "recovery_score": 88,  # Recovery Score
        "sleep_score": 107,  # Sleep Score
        "sleep_total": 101,  # Total Sleep (seconds)
        "sleep_disturbances_per_hour": 93,  # Sleep Disturbances / Hour
        "sleep_consistency": 106,  # Sleep Consistency
        "strain_score": 112,  # Strain Score
        "skin_temp": 340,  # Skin Temperature (Celsius)
        "resp_rate": 105,  # Sleep Respiratory Rate (bpm)
    },
    "garmin": {
        "rhr": 162,  # Resting Heart Rate
        "body_battery": 188,  # Body Battery
        "steps": 170,  # Steps
        "total_activity": 173,  # Active Time (seconds)
        "avg_stress": 176,  # Avg Stress Level
        "resp_rate": 191,  # Sleep Respiratory Rate (bpm)
        "vo2max": 187,  # VO2 Max
    },
    # Extend with Withings and others as needed
}

class HabitDashClient:
    """Client for interacting with the Habit Dash API."""

    def __init__(self, api_key: str = API_KEY):
        if not api_key:
            raise ValueError("Habit Dash API key not found. Set HABITDASH_API_KEY in .env")
        self.api_key = api_key
        self.headers = {
            "accept": "application/json",
            "x-api-key": self.api_key
        }
        self.session = requests.Session() # Use session for potential connection pooling
        self.session.headers.update(self.headers)

    def _make_request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Makes a request to the API, handles basic errors and rate limiting."""
        url = f"{BASE_URL}/{endpoint}/"
        try:
            time.sleep(REQUEST_DELAY_SECONDS) # Basic rate limiting pre-request
            response = self.session.request(method, url, params=params)
            response.raise_for_status() # Raises HTTPError for bad responses (4XX, 5XX)
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed for {method} {url} with params {params}: {e}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred during API request: {e}")
            return None

    def get_fields(self, source: Optional[str] = None, field_id: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
        """Lists all available data fields, optionally filtered."""
        params = {}
        if source:
            params["source"] = source.lower()
        if field_id:
            params["field_id"] = field_id
        
        response_data = self._make_request("GET", "fields", params=params)
        # Assuming the actual list of fields is under a key like 'results' if paginated,
        # or the response is directly the list. Adjust based on actual API response structure.
        # For now, assume response is directly the list or contains a 'results' key.
        if response_data:
             return response_data.get('results', response_data) if isinstance(response_data, dict) else response_data
        return None

    def get_data(self,
                 field_id: Optional[int] = None,
                 source: Optional[str] = None,
                 date_start: Optional[str] = None, # Expects YYYY-MM-DD
                 date_end: Optional[str] = None    # Expects YYYY-MM-DD
                 ) -> Optional[List[Dict[str, Any]]]:
        """Lists user data, ordered by date descending, with filters."""
        params = {}
        if field_id:
            params["field_id"] = field_id
        if source:
            params["source"] = source.lower()
        if date_start:
            params["date_start"] = date_start
        if date_end:
            params["date_end"] = date_end

        # --- Pagination Handling Placeholder ---
        # The actual API might use 'page' query param or return 'next' URLs.
        # This needs implementation based on observed API behavior for /data endpoint.
        # For now, this fetches the first page only.
        # Example rough logic:
        # all_results = []
        # page = 1
        # while True:
        #     params['page'] = page
        #     response_data = self._make_request("GET", "data", params=params)
        #     if not response_data or not response_data.get('results'):
        #         break
        #     all_results.extend(response_data['results'])
        #     if not response_data.get('next'): # Check if there's a next page indicator
        #          break
        #     page += 1
        # return all_results
        # --- End Placeholder ---

        # Simplified: fetch first page for now
        response_data = self._make_request("GET", "data", params=params)
        if response_data:
             return response_data.get('results', response_data) if isinstance(response_data, dict) else response_data
        return None

    def get_metric_for_date(self, source: str, metric_name: str, date_str: str) -> Optional[Any]:
        """Fetches a specific metric for a single date."""
        field_id = FIELD_IDS.get(source, {}).get(metric_name)
        if not field_id:
            logging.warning(f"Field ID not found for {source} - {metric_name}")
            return None
        data = self.get_data(field_id=field_id, date_start=date_str, date_end=date_str)
        if data:
            return data[0].get('value') if data else None
        return None

    def get_metrics_for_date_range(self, source: str, metric_name: str, start_date: str, end_date: str) -> Optional[List[Dict[str, Any]]]:
        """Fetches a specific metric over a date range."""
        field_id = FIELD_IDS.get(source, {}).get(metric_name)
        if not field_id:
            logging.warning(f"Field ID not found for {source} - {metric_name}")
            return None
        return self.get_data(field_id=field_id, date_start=start_date, date_end=end_date)

# Example usage (can be removed or placed in a test/example script)
# if __name__ == "__main__":
#     client = HabitDashClient()
#     if client.api_key:
#         logging.info("Fetching fields...")
#         fields = client.get_fields(source="whoop")
#         if fields:
#             logging.info(f"Found {len(fields)} Whoop fields.")
#             # Find field_id for HRV (example)
#             hrv_field = next((f for f in fields if f.get('name') == 'Heart Rate Variability'), None)
#             if hrv_field:
#                 hrv_field_id = hrv_field.get('id')
#                 logging.info(f"HRV Field ID: {hrv_field_id}")
#                 logging.info("Fetching last 7 days of HRV data...")
#                 # Requires knowing today's date, e.g., using datetime
#                 # from datetime import date, timedelta
#                 # today = date.today()
#                 # start_date = today - timedelta(days=7)
#                 # data = client.get_data(field_id=hrv_field_id, date_start=start_date.isoformat(), date_end=today.isoformat())
#                 # if data:
#                 #     logging.info(f"Fetched {len(data)} HRV data points.")
#                 #     # print(data)
#                 # else:
#                 #     logging.warning("No HRV data fetched.")
#             else:
#                 logging.warning("Could not find field ID for Heart Rate Variability.")
#         else:
#             logging.warning("Could not fetch fields.")
