"""
config_loader.py -- Utility to load YAML config for Dev Daily Reflect
"""
import os
import yaml
from pathlib import Path

def load_config():
    # Locate config file relative to this script
    config_path = Path(__file__).parent / "config" / "daily_review.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    # Optionally allow environment variable overrides
    config["repository_path"] = os.environ.get("DEV_DAILY_REVIEW_REPO_PATH", config["repository_path"])
    config["lookback_days"] = int(os.environ.get("DEV_DAILY_REVIEW_LOOKBACK_DAYS", config["lookback_days"]))
    config["report_output_dir"] = os.environ.get("DEV_DAILY_REVIEW_REPORT_OUTPUT_DIR", config["report_output_dir"])
    return config
