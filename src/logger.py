"""
MediChat Logging: A simple, shared logging setup to keep our console output 
clean and consistent across all modules.

Just import get_logger anywhere you need to track events or errors.
"""

import logging
import sys
from src.config import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to stdout with a clean format."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s – %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    return logger
