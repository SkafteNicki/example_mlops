import logging
import os
from datetime import datetime

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

        file_handler = logging.FileHandler(os.path.join(hydra_path, f"{job_name}.log"))
        rich_handler = RichHandler()

        log = logging.getLogger("example_mlops")
        log.setLevel(self.level)
        log.addHandler(rich_handler)
        log.addHandler(file_handler)
        log.propagate = False
        log.debug("Successfully create rich logger")
        return log

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
