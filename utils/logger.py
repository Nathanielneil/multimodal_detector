# -*- coding: utf-8 -*-
"""
Logging Module

Provides centralized logging configuration for the Multimodal Detector application.

Features:
- Console output with colored formatting
- File output with rotation
- Per-module loggers
- Configuration from config file

Usage:
    from utils import get_logger

    logger = get_logger(__name__)
    logger.info("Application started")
    logger.error("An error occurred", exc_info=True)
"""

import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Dict


# ANSI color codes for console output
class LogColors:
    """ANSI color codes for log level formatting"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Log level colors
    DEBUG = "\033[36m"      # Cyan
    INFO = "\033[32m"       # Green
    WARNING = "\033[33m"    # Yellow
    ERROR = "\033[31m"      # Red
    CRITICAL = "\033[35m"   # Magenta

    # Component colors
    TIMESTAMP = "\033[90m"  # Gray
    NAME = "\033[34m"       # Blue
    MESSAGE = "\033[0m"     # Default


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter with colored output for console.

    Adds color coding based on log level for better readability.
    """

    LEVEL_COLORS = {
        logging.DEBUG: LogColors.DEBUG,
        logging.INFO: LogColors.INFO,
        logging.WARNING: LogColors.WARNING,
        logging.ERROR: LogColors.ERROR,
        logging.CRITICAL: LogColors.CRITICAL,
    }

    def __init__(self, fmt: str = None, datefmt: str = None, use_colors: bool = True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if not self.use_colors:
            return super().format(record)

        # Get level color
        level_color = self.LEVEL_COLORS.get(record.levelno, LogColors.RESET)

        # Format the record
        record.levelname_colored = f"{level_color}{record.levelname:8}{LogColors.RESET}"
        record.name_colored = f"{LogColors.NAME}{record.name}{LogColors.RESET}"
        record.message_colored = f"{level_color}{record.getMessage()}{LogColors.RESET}"

        # Create formatted string
        formatted = (
            f"{LogColors.TIMESTAMP}[{self.formatTime(record, self.datefmt)}]{LogColors.RESET} "
            f"{record.levelname_colored} - "
            f"{record.name_colored} - "
            f"{record.message_colored}"
        )

        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{level_color}{self.formatException(record.exc_info)}{LogColors.RESET}"

        return formatted


class LogManager:
    """
    Centralized logging manager.

    Handles logging configuration, logger creation, and management.
    Implements singleton pattern for consistent logging across application.
    """

    _instance: Optional["LogManager"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if LogManager._initialized:
            return

        self._loggers: Dict[str, logging.Logger] = {}
        self._root_logger: Optional[logging.Logger] = None
        self._log_dir: Optional[Path] = None
        self._console_handler: Optional[logging.Handler] = None
        self._file_handler: Optional[logging.Handler] = None

        LogManager._initialized = True

    def setup(
        self,
        level: str = "INFO",
        console_enabled: bool = True,
        console_format: str = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        console_date_format: str = "%H:%M:%S",
        file_enabled: bool = True,
        file_path: str = "logs/multimodal_detector.log",
        file_format: str = "[%(asctime)s] %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s",
        file_date_format: str = "%Y-%m-%d %H:%M:%S",
        file_max_bytes: int = 10 * 1024 * 1024,  # 10MB
        file_backup_count: int = 5,
    ):
        """
        Setup logging configuration.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_enabled: Enable console output
            console_format: Console log format
            console_date_format: Console timestamp format
            file_enabled: Enable file output
            file_path: Log file path
            file_format: File log format
            file_date_format: File timestamp format
            file_max_bytes: Max file size before rotation
            file_backup_count: Number of backup files to keep
        """
        # Get numeric log level
        numeric_level = getattr(logging, level.upper(), logging.INFO)

        # Setup root logger
        self._root_logger = logging.getLogger("multimodal")
        self._root_logger.setLevel(numeric_level)

        # Remove existing handlers
        self._root_logger.handlers.clear()

        # Setup console handler
        if console_enabled:
            self._setup_console_handler(
                numeric_level, console_format, console_date_format
            )

        # Setup file handler
        if file_enabled:
            self._setup_file_handler(
                numeric_level,
                file_path,
                file_format,
                file_date_format,
                file_max_bytes,
                file_backup_count,
            )

        # Log startup message
        self._root_logger.info(
            f"Logging initialized - Level: {level}, "
            f"Console: {console_enabled}, File: {file_enabled}"
        )

    def _setup_console_handler(
        self, level: int, fmt: str, datefmt: str
    ):
        """Setup console logging handler"""
        self._console_handler = logging.StreamHandler(sys.stdout)
        self._console_handler.setLevel(level)

        # Use colored formatter for console
        formatter = ColoredFormatter(fmt, datefmt, use_colors=True)
        self._console_handler.setFormatter(formatter)

        self._root_logger.addHandler(self._console_handler)

    def _setup_file_handler(
        self,
        level: int,
        file_path: str,
        fmt: str,
        datefmt: str,
        max_bytes: int,
        backup_count: int,
    ):
        """Setup file logging handler with rotation"""
        # Create log directory
        log_path = Path(file_path)
        if not log_path.is_absolute():
            # Relative to project root
            project_root = Path(__file__).parent.parent
            log_path = project_root / file_path

        self._log_dir = log_path.parent
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Create rotating file handler
        self._file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        self._file_handler.setLevel(level)

        # Use standard formatter for file (no colors)
        formatter = logging.Formatter(fmt, datefmt)
        self._file_handler.setFormatter(formatter)

        self._root_logger.addHandler(self._file_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create a logger with the given name.

        Args:
            name: Logger name (typically __name__)

        Returns:
            Configured logger instance
        """
        if name in self._loggers:
            return self._loggers[name]

        # Create child logger under multimodal namespace
        if name.startswith("multimodal."):
            logger_name = name
        else:
            # Convert module path to logger name
            logger_name = f"multimodal.{name.split('.')[-1]}"

        logger = logging.getLogger(logger_name)
        self._loggers[name] = logger

        return logger

    def set_level(self, level: str, logger_name: Optional[str] = None):
        """
        Change log level at runtime.

        Args:
            level: New log level
            logger_name: Specific logger name, or None for root logger
        """
        numeric_level = getattr(logging, level.upper(), logging.INFO)

        if logger_name:
            logger = self._loggers.get(logger_name)
            if logger:
                logger.setLevel(numeric_level)
        else:
            if self._root_logger:
                self._root_logger.setLevel(numeric_level)
            if self._console_handler:
                self._console_handler.setLevel(numeric_level)
            if self._file_handler:
                self._file_handler.setLevel(numeric_level)

    @property
    def log_dir(self) -> Optional[Path]:
        """Get log directory path"""
        return self._log_dir


# Global log manager instance
_log_manager = LogManager()


def setup_logging(
    level: str = "INFO",
    console_enabled: bool = True,
    file_enabled: bool = True,
    file_path: str = "logs/multimodal_detector.log",
    **kwargs
):
    """
    Setup logging with specified configuration.

    This is a convenience function that should be called once at application startup.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_enabled: Enable console output
        file_enabled: Enable file output
        file_path: Log file path
        **kwargs: Additional arguments passed to LogManager.setup()

    Example:
        >>> from utils import setup_logging
        >>> setup_logging(level="DEBUG", file_path="logs/app.log")
    """
    _log_manager.setup(
        level=level,
        console_enabled=console_enabled,
        file_enabled=file_enabled,
        file_path=file_path,
        **kwargs
    )


def setup_logging_from_config(config):
    """
    Setup logging from configuration object.

    Args:
        config: Config instance with logging section
    """
    logging_config = config.get_section("logging")

    _log_manager.setup(
        level=logging_config.get("level", "INFO"),
        console_enabled=logging_config.get("console", {}).get("enabled", True),
        console_format=logging_config.get("console", {}).get(
            "format", "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
        ),
        console_date_format=logging_config.get("console", {}).get(
            "date_format", "%H:%M:%S"
        ),
        file_enabled=logging_config.get("file", {}).get("enabled", True),
        file_path=logging_config.get("file", {}).get(
            "path", "logs/multimodal_detector.log"
        ),
        file_format=logging_config.get("file", {}).get(
            "format",
            "[%(asctime)s] %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s"
        ),
        file_date_format=logging_config.get("file", {}).get(
            "date_format", "%Y-%m-%d %H:%M:%S"
        ),
        file_max_bytes=logging_config.get("file", {}).get("max_bytes", 10485760),
        file_backup_count=logging_config.get("file", {}).get("backup_count", 5),
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance

    Example:
        >>> from utils import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting detector...")
    """
    return _log_manager.get_logger(name)


def set_log_level(level: str, logger_name: Optional[str] = None):
    """
    Change log level at runtime.

    Args:
        level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger_name: Specific logger name, or None for all loggers
    """
    _log_manager.set_level(level, logger_name)
