import logging
from logging import FileHandler
from os import path
from .config import LOG_PATH, LOG_FILE_NAME


def init_logger(logger_name: str, log_file=LOG_FILE_NAME):
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # # create dir to ensure its there.
    LOG_PATH.mkdir(parents=True, exist_ok=True)



    # # Define File logger
    file_handler = FileHandler(LOG_PATH.joinpath(log_file), encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # # Define Stream logger
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # # Add the file handler to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


