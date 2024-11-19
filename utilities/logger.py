import logging
from logging.handlers import TimedRotatingFileHandler
from os import path
from pathlib import Path

from typeguard import config


def init_logger(logger_name: str, log_file="ml_run.log"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # # create dir to ensure its there.
    # Path(config.Ba).mkdir(parents=True, exist_ok=True)
    #
    # # Create a daily rotating file handler
    #
    # # file_handler = TimedRotatingFileHandler(path.join(conf, log_file),
    #                                         when='midnight', interval=1, backupCount=30)
    # file_handler.setLevel(logging.INFO)
    #
    # # Define log format
    # log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # formatter = logging.Formatter(log_format)
    # file_handler.setFormatter(formatter)
    #
    # # Add the file handler to the logger
    # logger.addHandler(file_handler)
    return logger


