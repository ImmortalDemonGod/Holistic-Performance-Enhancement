import pandas as pd
from datetime import date, timedelta
import logging
import time
from utilities.habitdash_api import HabitDashClient, FIELD_IDS

REQUEST_DELAY_SECONDS = 2.5  # Wait between API calls to avoid rate limiting
MAX_RETRIES = 3  # Retry up to 3 times on 429 errors
RETRY_WAIT_SECONDS = 10  # Wait this long after 429 before retrying

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

def sync_data(days_to_sync=7, specific_dates=None):
    client = HabitDashClient()
    if not client.api_key:
        logging.error("API key not found, aborting sync.")
        return

    all_data = []
    error_count = 0

    if specific_dates:
        # Parse and validate dates
        from datetime import datetime
        date_list = []
        for d in specific_dates:
            try:
                date_list.append(datetime.strptime(d, "%Y-%m-%d").date())
            except Exception:
                logging.error(f"Invalid date format: {d}. Use YYYY-MM-DD.")
        if not date_list:
            logging.error("No valid dates provided for --dates.")
            return
        logging.info(f"Syncing Habit Dash data for specific dates: {date_list}")
        for sync_date in date_list:
            for source, metric in METRICS_TO_FETCH:
                field_id = FIELD_IDS.get(source, {}).get(metric)
                if field_id:
                    logging.info(f"Fetching {source} - {metric} (ID: {field_id}) for {sync_date}")
                    # Retry logic for API calls
                    for attempt in range(1, MAX_RETRIES + 1):
                            try:
                                fetched = client.get_data(
                                    field_id=field_id,
                                    date_start=sync_date.isoformat(),
                                    date_end=sync_date.isoformat()
                                )
                                time.sleep(REQUEST_DELAY_SECONDS)
                                if fetched:
                                    for item in fetched:
                                        col_name = f"{metric}_{source}"
                                        all_data.append({'date': item.get('date'), col_name: item.get('value')})
                                    break
                                else:
                                    logging.warning(f"No data returned for {source} - {metric} on {sync_date}")
                                    break
                            except Exception as e:
                                if hasattr(e, 'response') and getattr(e.response, 'status_code', None) == 429:
                                    if attempt < MAX_RETRIES:
                                        logging.warning(f"429 Too Many Requests for {source}-{metric} on {sync_date}, retrying in {RETRY_WAIT_SECONDS}s (attempt {attempt}/{MAX_RETRIES})")
                                        time.sleep(RETRY_WAIT_SECONDS)
                                    else:
                                        logging.error(f"Max retries reached for {source}-{metric} on {sync_date} due to 429 errors.")
                                        error_count += 1
                                else:
                                    logging.error(f"Error fetching {source} - {metric} for {sync_date}: {e}")
                                    error_count += 1
                                    break
                else:
                    logging.warning(f"Skipping {source} - {metric}, Field ID not found.")
    else:
        end_date = date.today()
        start_date = end_date - timedelta(days=days_to_sync)
        logging.info(f"Syncing Habit Dash data from {start_date} to {end_date}")
        for source, metric in METRICS_TO_FETCH:
            field_id = FIELD_IDS.get(source, {}).get(metric)
            if field_id:
                logging.info(f"Fetching {source} - {metric} (ID: {field_id})")
                # Retry logic for API calls
                for attempt in range(1, MAX_RETRIES + 1):
                        try:
                            fetched = client.get_data(
                                field_id=field_id,
                                date_start=start_date.isoformat(),
                                date_end=end_date.isoformat()
                            )
                            time.sleep(REQUEST_DELAY_SECONDS)
                            if fetched:
                                for item in fetched:
                                    col_name = f"{metric}_{source}"
                                    all_data.append({'date': item.get('date'), col_name: item.get('value')})
                                break
                            else:
                                logging.warning(f"No data returned for {source} - {metric}")
                                break
                        except Exception as e:
                            if hasattr(e, 'response') and getattr(e.response, 'status_code', None) == 429:
                                if attempt < MAX_RETRIES:
                                    logging.warning(f"429 Too Many Requests for {source}-{metric}, retrying in {RETRY_WAIT_SECONDS}s (attempt {attempt}/{MAX_RETRIES})")
                                    time.sleep(RETRY_WAIT_SECONDS)
                                else:
                                    logging.error(f"Max retries reached for {source}-{metric} due to 429 errors.")
                                    error_count += 1
                            else:
                                logging.error(f"Error fetching {source} - {metric}: {e}")
                                error_count += 1
                                break
            else:
                logging.warning(f"Skipping {source} - {metric}, Field ID not found.")

    if error_count > 0:
        logging.warning(f"{error_count} errors occurred while fetching data")

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7, help="Number of days to sync (default: 7)")
    parser.add_argument("--dates", type=str, help="Comma-separated list of specific dates to sync (YYYY-MM-DD,YYYY-MM-DD,...)")
    args = parser.parse_args()
    if args.dates:
        date_list = [d.strip() for d in args.dates.split(",") if d.strip()]
        sync_data(specific_dates=date_list)
    else:
        sync_data(days_to_sync=args.days)
