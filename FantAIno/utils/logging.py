import logging
import os
import streamlit as st

from FantAIno.constants import LOGS_DIR

def create_logger(logger_name: str, log_file_name: str, logging_level: int = logging.DEBUG) -> logging.Logger:
    """Creates a logger for the Fantaino project."""

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging_level)

    if not logger.handlers:

        is_cloud = "OPENAI_API_KEY" in st.secrets

        if not is_cloud and log_file_name:
            handler = logging.FileHandler(os.path.join(LOGS_DIR, log_file_name), encoding='utf-8')
        else:
            handler = logging.StreamHandler()

        logger.addHandler(handler)
        logger.propagate = False # perhaps remove?

    return logger
