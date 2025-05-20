"""
config_loader.py -- Utility to load YAML config for Dev Daily Reflect
"""
import os
import yaml
from pathlib import Path

def load_config():
    """
    Load configuration from YAML file with environment variable overrides.

    Returns:
        dict: Configuration dictionary with all settings

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ValueError: If required configuration is missing or invalid
    """
    # Default configuration
    default_config = {
        "repository_path": ".",
        "lookback_days": 7,
        "report_output_dir": "cultivation/outputs/software/dev_daily_reflect/reports",
        "rollup_dir": "cultivation/outputs/software/dev_daily_reflect/rollup",
        "raw_data_dir": "cultivation/outputs/software/dev_daily_reflect/raw",
    }

    # Locate config file relative to this script
    config_path = Path(__file__).parent / "config" / "daily_review.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            loaded_config = yaml.safe_load(f)
        if loaded_config is None:
            loaded_config = {}
        # Merge with defaults
        config = {**default_config, **loaded_config}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}")

    # Optionally allow environment variable overrides
    config["repository_path"] = os.environ.get("DEV_DAILY_REFLECT_REPO_PATH", config["repository_path"])
    try:
        lookback_env = os.environ.get("DEV_DAILY_REFLECT_LOOKBACK_DAYS")
        if lookback_env is not None:
            config["lookback_days"] = int(lookback_env)
    except ValueError:
        raise ValueError(f"Invalid lookback_days value: {lookback_env} - must be an integer")
    config["report_output_dir"] = os.environ.get("DEV_DAILY_REFLECT_REPORT_OUTPUT_DIR", config["report_output_dir"])

    # Validate required configuration
    if not config.get("repository_path"):
        raise ValueError("repository_path is required in configuration")
    if not isinstance(config.get("lookback_days"), int) or config["lookback_days"] < 1:
        raise ValueError("lookback_days must be a positive integer")

    return config
