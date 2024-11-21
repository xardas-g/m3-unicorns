import logging
from logging import FileHandler
from os import path
from .config import LOG_PATH, LOG_FILE_NAME


def init_logger(logger_name: str, log_file=LOG_FILE_NAME):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # # create dir to ensure its there.
    LOG_PATH.mkdir(parents=True, exist_ok=True)

    file_handler = FileHandler(LOG_PATH.joinpath(log_file), encoding="utf-8")

    file_handler.setLevel(logging.INFO)
    #
    # # Define log format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    #
    # # Add the file handler to the logger
    logger.addHandler(file_handler)
    return logger


