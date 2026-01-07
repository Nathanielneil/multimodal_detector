# -*- coding: utf-8 -*-
"""
Configuration Management Module

Provides centralized configuration management for the Multimodal Detector application.
Supports YAML configuration files with environment variable overrides.

Usage:
    from config import config

    # Access configuration values
    model_name = config.get("voice.model_name")

    # With default value
    threshold = config.get("gesture.threshold", 0.5)

    # Reload configuration
    config.reload()
"""

from .config import Config, get_config

# Global configuration instance
config = get_config()

__all__ = ["Config", "config", "get_config"]
