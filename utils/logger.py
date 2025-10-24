import logging
import os
from pathlib import Path


class LogHelper:
    NAME = 'BaseLogger'

    @staticmethod
    def get_logger(logger_name, file_logging=False, file_name=None, log_directory=None):

        # Create file handler
        if file_logging:
            if not log_directory or not file_name:
                raise ValueError("Please provide the log directory path and log file name")
            log_directory_path = Path(log_directory)
            if file_logging and not log_directory_path.exists():
                log_directory_path.mkdir(parents=True, exist_ok=True)            
            log_file = log_directory_path / f"{file_name}.log"
            log_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        else:
            log_handler = logging.StreamHandler()
        # Create formatter
        formatter = logging.Formatter("%(message)s")
        log_handler.setFormatter(formatter)
        logger = logging.getLogger(f"{logger_name}")
        logger.setLevel(logging.INFO)
        logger.addHandler(log_handler)

        return logger

    @staticmethod
    def setup_base_logger(file_logging=False, log_directory=None, file_name=None):
        logging.getLogger().handlers.clear()
        if file_logging:
            if file_logging:
                if not log_directory or not file_name:
                    raise ValueError("Please provide the log directory path and log file name")
            log_directory_path = Path(log_directory)
            if not log_directory_path.exists():
                log_directory_path.mkdir()
            log_file = log_directory_path / f"{file_name}.log"
            logging.basicConfig(filename=log_file, format='%(message)s', level=logging.INFO)
        else:
            logging.basicConfig(format='%(message)s', level=logging.INFO)

    @staticmethod
    def clear_log_handlers():
        logging.getLogger().handlers.clear()