import logging
import os
import sys
from typing import Optional, List # Removed TextIO

class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger_instance, level):
        self.logger = logger_instance
        self.level = level
        self.linebuf = ''
        # Provide an encoding attribute, as some libraries (e.g., PyTorch Lightning) expect it.
        # Try to get it from the original stdout, otherwise default to utf-8.
        try:
            self.encoding = sys.__stdout__.encoding
        except AttributeError:
            self.encoding = 'utf-8'
        print(f"StreamToLogger initialized for {logger_instance.name}. Encoding: {self.encoding}. ID: {id(self)}", file=sys.__stdout__)


    def write(self, buf: str) -> None:
        self.linebuf += buf
        while '\n' in self.linebuf:
            line, self.linebuf = self.linebuf.split('\n', 1)
            if line.rstrip(): # Avoid logging empty lines if multiple newlines
                self.logger.log(self.level, line.rstrip())

    def flush(self) -> None:
        if self.linebuf.rstrip(): # Log remaining buffer if not just whitespace
            self.logger.log(self.level, self.linebuf.rstrip())
            self.linebuf = ''

    def isatty(self) -> bool:
        # Some libraries check isatty. The original stdout/stderr might be a TTY.
        # We can try to delegate or default to False.
        original_stream = sys.__stdout__ if self.logger.name == 'STDOUT' else sys.__stderr__
        if hasattr(original_stream, 'isatty'):
            return original_stream.isatty()
        return False

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

    # Ensure original streams are used by the basic config handlers
    # to prevent recursion with StreamToLogger.
    # Mypy expects StreamHandler[TextIO] or compatible.
    # --- Root Logger Configuration --- 
    # These handlers will be used by the root logger, configured by basicConfig.
    # The console handler for the root logger should write to the *original* stderr.
    root_console_handler: logging.StreamHandler = logging.StreamHandler(sys.__stderr__)
    root_handlers: List[logging.StreamHandler] = [root_console_handler]
    if log_file:
        file_handler: logging.FileHandler = logging.FileHandler(log_file) # FileHandler is a StreamHandler
        root_handlers.append(file_handler) # type: ignore[arg-type] # Mypy struggles with FileHandler vs StreamHandler[TextIO]

    # --- Dedicated Loggers for STDOUT/STDERR Redirection --- 
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Configure STDOUT logger
    stdout_logger = logging.getLogger('STDOUT')
    stdout_logger.setLevel(log_level) # Use the global log level
    stdout_logger.propagate = False   # CRITICAL: Do not propagate to root
    stdout_direct_handler = logging.StreamHandler(sys.__stdout__) # Writes to original stdout
    stdout_direct_handler.setFormatter(formatter)
    stdout_logger.handlers = [stdout_direct_handler] # Replace any existing handlers

    # Configure STDERR logger
    stderr_logger = logging.getLogger('STDERR')
    stderr_logger.setLevel(log_level) # Use the global log level
    stderr_logger.propagate = False   # CRITICAL: Do not propagate to root
    stderr_direct_handler = logging.StreamHandler(sys.__stderr__) # Writes to original stderr
    stderr_direct_handler.setFormatter(formatter)
    stderr_logger.handlers = [stderr_direct_handler] # Replace any existing handlers

    # --- Redirect sys.stdout and sys.stderr --- 
    # These StreamToLogger instances will use the dedicated loggers configured above.
    sys.stdout = StreamToLogger(stdout_logger, logging.INFO) # Log actual stdout content as INFO
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR) # Log actual stderr content as ERROR

    # Diagnostic print (can be removed later)
    # print(f"setup_logging: type(sys.stdout) is {type(sys.stdout)}, hasattr(sys.stdout, 'encoding'): {hasattr(sys.stdout, 'encoding')}", file=sys.__stdout__)
    # if hasattr(sys.stdout, 'encoding'):
    #     print(f"setup_logging: sys.stdout.encoding is {sys.stdout.encoding}", file=sys.__stdout__)

    # --- Configure Root Logger (basicConfig) --- 
    # This configures the root logger and any other loggers that propagate to it.
    # It uses the root_handlers list defined earlier.
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=root_handlers, # Changed 'handlers' to 'root_handlers'
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
