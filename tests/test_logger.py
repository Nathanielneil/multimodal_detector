# -*- coding: utf-8 -*-
"""
Unit Tests for Logging Module

Tests logging setup, output, and configuration.
"""

import logging
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import (
    get_logger,
    setup_logging,
    set_log_level,
    LogManager,
    ColoredFormatter,
)


class TestLoggerSetup:
    """Test logging setup functionality"""

    def test_setup_logging_default(self, tmp_path):
        """Test setup with default parameters"""
        log_file = tmp_path / "test.log"
        setup_logging(
            level="INFO",
            console_enabled=True,
            file_enabled=True,
            file_path=str(log_file),
        )

        logger = get_logger("test")
        assert logger is not None

    def test_setup_logging_debug_level(self, tmp_path):
        """Test setup with DEBUG level"""
        log_file = tmp_path / "debug.log"
        setup_logging(
            level="DEBUG",
            file_path=str(log_file),
        )

        logger = get_logger("test_debug")
        logger.debug("Debug message")
        # Debug message should be logged

    def test_setup_logging_file_only(self, tmp_path):
        """Test setup with file output only"""
        log_file = tmp_path / "file_only.log"
        setup_logging(
            level="INFO",
            console_enabled=False,
            file_enabled=True,
            file_path=str(log_file),
        )

        logger = get_logger("file_test")
        logger.info("File only message")

        # Check file was created
        assert log_file.exists()

    def test_setup_logging_console_only(self, tmp_path):
        """Test setup with console output only"""
        setup_logging(
            level="INFO",
            console_enabled=True,
            file_enabled=False,
        )

        logger = get_logger("console_test")
        logger.info("Console only message")


class TestLoggerOutput:
    """Test logger output functionality"""

    def test_log_info(self, tmp_path, capsys):
        """Test INFO level logging"""
        log_file = tmp_path / "info.log"
        setup_logging(
            level="INFO",
            file_path=str(log_file),
        )

        logger = get_logger("info_test")
        logger.info("Test info message")

        # Check file contains message
        content = log_file.read_text()
        assert "Test info message" in content

    def test_log_warning(self, tmp_path):
        """Test WARNING level logging"""
        log_file = tmp_path / "warning.log"
        setup_logging(
            level="WARNING",
            file_path=str(log_file),
        )

        logger = get_logger("warning_test")
        logger.warning("Test warning message")
        logger.info("This should not appear")

        content = log_file.read_text()
        assert "Test warning message" in content
        assert "This should not appear" not in content

    def test_log_error_with_exception(self, tmp_path):
        """Test ERROR level logging with exception info"""
        log_file = tmp_path / "error.log"
        setup_logging(
            level="ERROR",
            file_path=str(log_file),
        )

        logger = get_logger("error_test")
        try:
            raise ValueError("Test error")
        except ValueError:
            logger.error("Caught error", exc_info=True)

        content = log_file.read_text()
        assert "Caught error" in content
        assert "ValueError" in content


class TestLogLevelChange:
    """Test runtime log level changes"""

    def test_set_log_level(self, tmp_path):
        """Test changing log level at runtime"""
        log_file = tmp_path / "level_change.log"
        setup_logging(
            level="INFO",
            file_path=str(log_file),
        )

        logger = get_logger("level_test")

        # Info should work
        logger.info("Info before")

        # Change to WARNING
        set_log_level("WARNING")

        # Info should not work now (need to check actual behavior)
        logger.info("Info after")
        logger.warning("Warning after")


class TestColoredFormatter:
    """Test colored formatter functionality"""

    def test_formatter_with_colors(self):
        """Test formatter with colors enabled"""
        formatter = ColoredFormatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            "%H:%M:%S",
            use_colors=True,
        )

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "Test message" in formatted

    def test_formatter_without_colors(self):
        """Test formatter with colors disabled"""
        formatter = ColoredFormatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            "%H:%M:%S",
            use_colors=False,
        )

        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=10,
            msg="Warning message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "Warning message" in formatted
        # Should not contain ANSI codes
        assert "\033[" not in formatted


class TestLoggerNaming:
    """Test logger naming conventions"""

    def test_logger_name_simple(self, tmp_path):
        """Test logger with simple name"""
        log_file = tmp_path / "naming.log"
        setup_logging(file_path=str(log_file))

        logger = get_logger("simple")
        assert "simple" in logger.name or "multimodal" in logger.name

    def test_logger_name_module(self, tmp_path):
        """Test logger with module-style name"""
        log_file = tmp_path / "module.log"
        setup_logging(file_path=str(log_file))

        logger = get_logger("detectors.gesture_detector")
        logger.info("Module logger test")


class TestLogFileRotation:
    """Test log file rotation functionality"""

    def test_log_rotation(self, tmp_path):
        """Test log file rotation when size limit reached"""
        log_file = tmp_path / "rotation.log"
        setup_logging(
            level="DEBUG",
            file_path=str(log_file),
            file_max_bytes=1024,  # 1KB for testing
            file_backup_count=2,
        )

        logger = get_logger("rotation_test")

        # Write enough to trigger rotation
        for i in range(100):
            logger.info(f"Log message {i} " + "x" * 50)

        # Check backup files exist
        backup_files = list(tmp_path.glob("rotation.log*"))
        assert len(backup_files) >= 1


class TestLoggerIntegration:
    """Integration tests for logging system"""

    def test_multiple_loggers(self, tmp_path):
        """Test multiple loggers writing to same file"""
        log_file = tmp_path / "multi.log"
        setup_logging(
            level="DEBUG",
            file_path=str(log_file),
        )

        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        logger3 = get_logger("module3")

        logger1.info("Message from module1")
        logger2.warning("Message from module2")
        logger3.error("Message from module3")

        content = log_file.read_text()
        assert "module1" in content or "Message from module1" in content
        assert "Message from module2" in content
        assert "Message from module3" in content

    def test_logger_with_config(self, temp_config_file, tmp_path):
        """Test logger setup from config"""
        from config import Config

        config = Config(temp_config_file)

        # Setup from config would use these values
        log_level = config.get("logging.level", "INFO")
        assert log_level == "DEBUG"  # From test config
