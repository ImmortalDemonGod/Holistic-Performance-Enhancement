"""
Script to discover and print field IDs for high-priority metrics from Habit Dash API.
Run this script to get the field_id values for Whoop, Garmin, Withings, etc.
"""
import logging
from utilities.habitdash_api import HabitDashClient

PRIORITY_METRICS = [
    # (source, metric_name_substring)
    ("whoop", ["hrv", "rhr", "sleep score", "recovery score"]),
    ("garmin", ["rhr", "body battery", "stress"]),
    ("withings", ["weight"]),
]

def discover_field_ids():
    client = HabitDashClient()
    for source, _ in PRIORITY_METRICS:
        print(f"\n--- RAW RESPONSE FOR {source.upper()} ---")
        fields = client.get_fields(source=source)
        print(fields)
        print(f"\n--- {source.upper()} FIELDS ---")
        if not fields:
            logging.warning(f"No fields found for {source}")
            print(f"No fields found for {source}")
            continue
        for field in fields:
            if field.get('field_id') is not None and field.get('field_name') is not None and field.get('unit_tooltip') is not None and field.get('category2') is not None:
                print(f"ID: {field.get('field_id')} | Name: {field.get('field_name')} | Unit: {field.get('unit_tooltip')} | Description: {field.get('category2')}")

if __name__ == "__main__":
    discover_field_ids()
