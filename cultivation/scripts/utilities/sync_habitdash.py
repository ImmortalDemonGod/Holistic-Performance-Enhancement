import pandas as pd
from datetime import date, timedelta
import logging
import time
from utilities.habitdash_api import HabitDashClient, FIELD_IDS

REQUEST_DELAY_SECONDS = 15.0  # Increased delay to 15 seconds to avoid rate limiting
MAX_RETRIES = 3  # Retry up to 3 times on 429 errors
RETRY_WAIT_SECONDS = 10  # Wait this long after 429 before retrying

CACHE_FILE = "cultivation/data/daily_wellness.parquet"
# Define desired metrics and their sources/field IDs (using FIELD_IDS map)
# Canonical export metric names (source, canonical_metric_name)
METRICS_TO_FETCH = [
    ("whoop", "heart_rate_variability_whoop"),
    ("whoop", "resting_heart_rate_whoop"),
    ("whoop", "recovery_score_whoop"),
    ("whoop", "sleep_score_whoop"),
    ("whoop", "total_sleep_whoop"),
    ("whoop", "sleep_disturbances__per__hour_whoop"),
    ("whoop", "sleep_consistency_whoop"),
    ("whoop", "strain_score_whoop"),
    ("whoop", "skin_temperature_whoop"),
    ("whoop", "sleep_respiratory_rate_whoop"),
    ("garmin", "body_battery_garmin"),
    ("garmin", "resting_heart_rate_garmin"),
    ("garmin", "steps_garmin"),
    ("garmin", "total_activity_garmin"),
    ("garmin", "avg_stress_level_garmin"),
    ("garmin", "sleep_respiratory_rate_garmin"),
    ("garmin", "vo₂_max_garmin"),
    # Add more as needed
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
    parser.add_argument("--metrics-csv", type=str, help="CSV file with columns: date,missing_metric for targeted gap filling")
    args = parser.parse_args()

    if args.metrics_csv:
        import pandas as pd
        metrics_df = pd.read_csv(args.metrics_csv)
        # Group by date for efficient processing
        grouped = metrics_df.groupby('date')['missing_metric'].apply(list).to_dict()
        for sync_date, metrics in grouped.items():
            client = HabitDashClient()
            if not client.api_key:
                logging.error("API key not found, aborting sync.")
                continue
            all_data = []
            error_count = 0
            from datetime import datetime
            try:
                sync_date_obj = datetime.strptime(sync_date, "%Y-%m-%d").date()
            except Exception:
                logging.error(f"Invalid date format: {sync_date}. Use YYYY-MM-DD.")
                continue
            logging.info(f"Targeted sync for {sync_date}: metrics {metrics}")
            for metric_col in metrics:
                # Parse metric_col into metric and source
                if '_' not in metric_col:
                    logging.warning(f"Skipping malformed metric column: {metric_col}")
                    continue
                metric, source = metric_col.rsplit('_', 1)
                field_id = FIELD_IDS.get(source, {}).get(metric)
                if not field_id:
                    logging.warning(f"Field ID not found for {source} - {metric}")
                    continue
                logging.info(f"Fetching {source} - {metric} (ID: {field_id}) for {sync_date}")
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        fetched = client.get_data(
                            field_id=field_id,
                            date_start=sync_date_obj.isoformat(),
                            date_end=sync_date_obj.isoformat()
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
            # Merge results into the cache as usual
            if all_data:
                try:
                    df_new = pd.DataFrame(all_data)
                    df_new.set_index('date', inplace=True)
                    df_new.index = pd.to_datetime(df_new.index).strftime('%Y-%m-%d')
                    df_cache = pd.read_parquet(CACHE_FILE)
                    df_cache.index = pd.to_datetime(df_cache.index).strftime('%Y-%m-%d')
                    df_merged = df_cache.combine_first(df_new)
                    df_merged.to_parquet(CACHE_FILE)
                    logging.info(f"Updated {CACHE_FILE} with targeted metrics for {sync_date}")
                except Exception as e:
                    logging.error(f"Error updating cache for {sync_date}: {e}")
            if error_count > 0:
                logging.warning(f"{error_count} errors occurred while fetching targeted metrics for {sync_date}")
    elif args.dates:
        date_list = [d.strip() for d in args.dates.split(",") if d.strip()]
        sync_data(specific_dates=date_list)
    else:
        sync_data(days_to_sync=args.days)
