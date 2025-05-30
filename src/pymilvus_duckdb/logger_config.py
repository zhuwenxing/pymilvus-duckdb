#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from loguru import logger

# Default logger configuration
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",  # Default log level
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    backtrace=True,
    diagnose=True,
)

# You can add more sinks here, for example, to a file:
# logger.add(
#     "logs/app.log",
#     level="DEBUG",
#     rotation="10 MB",  # Rotate log file when it reaches 10 MB
#     retention="7 days", # Keep logs for 7 days
#     compression="zip", # Compress rotated files
#     format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
# )

# Export the configured logger instance
__all__ = ["logger"]
