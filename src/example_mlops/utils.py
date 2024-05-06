import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import hydra
from rich.logging import RichHandler


class HydraRichLogger(object):
    """A utility class to create a rich logger that works with hydra."""

    def __init__(self, level: str = "INFO") -> None:
        """Initialize the logger."""
        self.level = level
        self.logger = None

    def get_logger(self) -> logging.Logger:
        """Create a rich logger that logs to a file and the console."""
        try:
            hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            job_name = hydra.core.hydra_config.HydraConfig.get().job.name
        except ValueError:
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
            hydra_path = f"{os.getcwd()}/outputs/other/{formatted_datetime}"
            job_name = "example_mlops"
            os.mkdir(hydra_path)

        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "minimal": {"format": "%(message)s"},
                "detailed": {
                    "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                    "formatter": "minimal",
                    "level": logging.DEBUG,
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": Path(hydra_path, f"{job_name}.log"),
                    "maxBytes": 10485760,  # 1 MB
                    "backupCount": 10,
                    "formatter": "detailed",
                    "level": logging.INFO,
                },
            },
            "root": {
                "handlers": ["console", "file"],
                "level": logging.INFO,
                "propagate": True,
            },
        }
        logging.config.dictConfig(logging_config)
        logger = logging.getLogger()
        logger.handlers[0] = RichHandler(markup=True)  # set rich handler
        logger.debug("Successfully create rich logger")
        return logger

    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger = self.logger or self.get_logger()
        self.logger.info(message)

    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger = self.logger or self.get_logger()
        self.logger.debug(message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger = self.logger or self.get_logger()
        self.logger.error(message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger = self.logger or self.get_logger()
        self.logger.warning(message)

    def critical(self, message: str) -> None:
        """Log a critical message."""
        self.logger = self.logger or self.get_logger()
        self.logger.critical(message)

    def exception(self, message: str) -> None:
        """Log an exception message."""
        self.logger = self.logger or self.get_logger()
        self.logger.exception(message)
