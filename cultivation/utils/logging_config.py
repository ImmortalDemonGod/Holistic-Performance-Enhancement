import logging
import os
from typing import Optional

def setup_logging(log_file: Optional[str] = None):
    """Sets up a standardized logger for the project.

    The log level is configured via the `LOG_LEVEL` environment variable
    (defaulting to INFO). The format includes a timestamp, log level,
    and message.

    If a log_file path is provided, output will be written to that file
    in addition to the console.

    Using force=True ensures this configuration overrides any other
    logging configurations that may have been set.
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True
    )

if __name__ == '__main__':
    # Example usage
    print("Setting up standardized logger (console only)...")
    setup_logging()
    logging.info("This is an info message.")

    print("\nSetting up standardized logger (console and file)...")
    setup_logging(log_file="test_log.log")
    logging.info("This message goes to console and test_log.log")
    logging.warning("This is a warning message.")
    print("Demonstration complete. Check test_log.log.")
