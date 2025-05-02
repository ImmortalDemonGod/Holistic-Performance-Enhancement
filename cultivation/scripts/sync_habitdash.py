import pandas as pd
from datetime import date, timedelta
import logging
from utilities.habitdash_api import HabitDashClient, FIELD_IDS

CACHE_FILE = "cultivation/data/daily_wellness.parquet"
# Define desired metrics and their sources/field IDs (using FIELD_IDS map)
METRICS_TO_FETCH = [
    ("whoop", "hrv"), ("whoop", "rhr"), ("whoop", "recovery_score"), ("whoop", "sleep_score"),
    ("whoop", "sleep_total"), ("whoop", "sleep_disturbances_per_hour"), ("whoop", "sleep_consistency"),
    ("whoop", "strain_score"), ("whoop", "skin_temp"), ("whoop", "resp_rate"),
    ("garmin", "body_battery"), ("garmin", "rhr"), ("garmin", "steps"), ("garmin", "total_activity"),
    ("garmin", "avg_stress"), ("garmin", "resp_rate"), ("garmin", "vo2max"),
    # Add more Tier 1/2 metrics as needed
]

def sync_data(days_to_sync=7):
    client = HabitDashClient()
    if not client.api_key:
        logging.error("API key not found, aborting sync.")
        return

    end_date = date.today()
    start_date = end_date - timedelta(days=days_to_sync)
    logging.info(f"Syncing Habit Dash data from {start_date} to {end_date}")

    all_data = []
    for source, metric in METRICS_TO_FETCH:
        field_id = FIELD_IDS.get(source, {}).get(metric)
        if field_id:
            logging.info(f"Fetching {source} - {metric} (ID: {field_id})")
            fetched = client.get_data(field_id=field_id, date_start=start_date.isoformat(), date_end=end_date.isoformat())
            if fetched:
                for item in fetched:
                    col_name = f"{metric}_{source}"
                    all_data.append({'date': item.get('date'), col_name: item.get('value')})
        else:
            logging.warning(f"Skipping {source} - {metric}, Field ID not found.")

    if not all_data:
        logging.warning("No data fetched from Habit Dash.")
        return

    new_df = pd.DataFrame(all_data)
    new_df['date'] = pd.to_datetime(new_df['date']).dt.date
    new_df_pivot = new_df.pivot_table(index='date', values=new_df.columns.drop('date'), aggfunc='first')

    try:
        existing_df = pd.read_parquet(CACHE_FILE)
        existing_df.index = pd.to_datetime(existing_df.index).date
        combined_df = pd.concat([existing_df, new_df_pivot])
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df.sort_index(inplace=True)
    except FileNotFoundError:
        logging.info("Cache file not found, creating new one.")
        combined_df = new_df_pivot
        combined_df.sort_index(inplace=True)

    combined_df.to_parquet(CACHE_FILE)
    logging.info(f"Successfully synced data and updated {CACHE_FILE}")

if __name__ == "__main__":
    sync_data(days_to_sync=7)  # Sync last 7 days by default
