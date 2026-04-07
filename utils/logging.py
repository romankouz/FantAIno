import logging
import os

def create_logger(logger_name: str, log_file_name: str, logging_level: int = logging.DEBUG) -> logging.Logger:
    """Creates a logger for the Fantaino project."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging_level)
    handler = logging.FileHandler(os.path.join('logs', log_file_name))
    logger.addHandler(handler)
    logger.propagate = False # perhaps remove?
    return logger