import logging
from logging.handlers import TimedRotatingFileHandler
import sys
import os
from src import config


log_dir_path = 'logs/'
log_filename = 'human_gait_analysis.log'
# default logging level is warning
log_level = config.LOG_LEVELS.get(os.environ.get('LOG_LEVEL')) if os.environ.get('LOG_LEVEL') != None else logging.WARNING
formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
# formatter = logging.Formatter("[%(levelname)s] - %(pathname)s:%(lineno)d - %(asctime)s - %(name)s - : %(message)s")

def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    return console_handler

def get_file_handler(log_filename):
    file_handler = TimedRotatingFileHandler(os.path.join(log_dir_path, log_filename), backupCount = 5, when = 'midnight')
    file_handler.setFormatter(formatter)
    return file_handler

def get_logger(logger_name: str, log_filename: str = log_filename):
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler(log_filename))
    logger.propagate = False
    return logger