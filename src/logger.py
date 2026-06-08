import logging
import sys
import time


def setup_logger(log_file: str) -> logging.Logger:
    """
    Configure and return a logger that writes to the given file path.

    Format: %(asctime)s %(levelname)s %(message)s
    Timestamps are UTC (ISO-8601).

    Args:
        log_file: Path to the log output file.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger("mlops_batch_job")
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers if called more than once
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s")
    # Force UTC timestamps
    formatter.converter = time.gmtime

    # File handler — writes all log output to the provided path
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Stream handler — mirrors output to stdout for visibility
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
