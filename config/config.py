# -*- coding: utf-8 -*-
"""
Configuration Manager

Handles loading, accessing, and managing application configuration.
Supports:
- YAML configuration files
- Environment variable overrides
- Nested key access with dot notation
- Configuration validation
- Hot reload capability
"""

import os
import yaml
from pathlib import Path
from typing import Any, Optional, Dict, List
from copy import deepcopy


class ConfigError(Exception):
    """Configuration related errors"""
    pass


class Config:
    """
    Configuration Manager Class

    Provides centralized configuration management with support for:
    - YAML file loading
    - Environment variable overrides (format: MULTIMODAL_SECTION_KEY)
    - Dot notation access (e.g., config.get("voice.model_name"))
    - Default values
    - Configuration validation

    Attributes:
        config_dir: Directory containing configuration files
        config_file: Path to the active configuration file
    """

    # Environment variable prefix
    ENV_PREFIX = "MULTIMODAL_"

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file. If None, uses default.yaml
        """
        self._config: Dict[str, Any] = {}
        self._config_dir = Path(__file__).parent
        self._config_file: Optional[Path] = None
        self._loaded = False

        # Load configuration
        self._load(config_path)

    def _load(self, config_path: Optional[str] = None):
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file

        Raises:
            ConfigError: If configuration file cannot be loaded
        """
        # Determine config file path
        if config_path:
            self._config_file = Path(config_path)
        else:
            # Check for user config first, then default
            user_config = self._config_dir / "config.yaml"
            default_config = self._config_dir / "default.yaml"

            if user_config.exists():
                self._config_file = user_config
            elif default_config.exists():
                self._config_file = default_config
            else:
                raise ConfigError(
                    f"No configuration file found. Expected at: {default_config}"
                )

        # Load YAML file
        try:
            with open(self._config_file, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse YAML configuration: {e}")
        except IOError as e:
            raise ConfigError(f"Failed to read configuration file: {e}")

        # Apply environment variable overrides
        self._apply_env_overrides()

        self._loaded = True

    def _apply_env_overrides(self):
        """
        Apply environment variable overrides to configuration.

        Environment variables are expected in format: MULTIMODAL_SECTION_KEY
        For nested keys: MULTIMODAL_SECTION_SUBSECTION_KEY

        Examples:
            MULTIMODAL_PROXY_ENABLED=true
            MULTIMODAL_VOICE_MODEL_NAME=medium
            MULTIMODAL_LOGGING_LEVEL=DEBUG
        """
        for key, value in os.environ.items():
            if not key.startswith(self.ENV_PREFIX):
                continue

            # Remove prefix and convert to lowercase
            config_key = key[len(self.ENV_PREFIX):].lower()

            # Split by underscore to get nested path
            parts = config_key.split("_")

            # Convert value to appropriate type
            typed_value = self._parse_env_value(value)

            # Set value in config
            self._set_nested(parts, typed_value)

    def _parse_env_value(self, value: str) -> Any:
        """
        Parse environment variable value to appropriate Python type.

        Args:
            value: String value from environment variable

        Returns:
            Parsed value (bool, int, float, None, or str)
        """
        # Boolean
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        if value.lower() in ("false", "no", "0", "off"):
            return False

        # None
        if value.lower() in ("null", "none", ""):
            return None

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # String
        return value

    def _set_nested(self, keys: List[str], value: Any):
        """
        Set a value in nested dictionary structure.

        Args:
            keys: List of keys representing the path
            value: Value to set
        """
        current = self._config

        # Navigate to parent
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set value
        if keys:
            current[keys[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., "voice.model_name")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Examples:
            >>> config.get("voice.model_name")
            "small"
            >>> config.get("nonexistent.key", "fallback")
            "fallback"
        """
        parts = key.split(".")
        current = self._config

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section as dictionary.

        Args:
            section: Section name (e.g., "voice", "gesture")

        Returns:
            Section dictionary (deep copy) or empty dict if not found
        """
        value = self.get(section, {})
        return deepcopy(value) if isinstance(value, dict) else {}

    def set(self, key: str, value: Any):
        """
        Set configuration value (runtime only, not persisted).

        Args:
            key: Configuration key (e.g., "voice.model_name")
            value: Value to set
        """
        parts = key.split(".")
        self._set_nested(parts, value)

    def reload(self):
        """
        Reload configuration from file.

        Useful for hot-reloading configuration changes.
        """
        self._load(str(self._config_file) if self._config_file else None)

    def save(self, path: Optional[str] = None):
        """
        Save current configuration to file.

        Args:
            path: Path to save to. If None, uses config/config.yaml
        """
        save_path = Path(path) if path else self._config_dir / "config.yaml"

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    self._config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False
                )
        except IOError as e:
            raise ConfigError(f"Failed to save configuration: {e}")

    def validate(self) -> List[str]:
        """
        Validate configuration values.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate voice model
        valid_voice_models = ["tiny", "base", "small", "medium", "large"]
        voice_model = self.get("voice.model_name")
        if voice_model and voice_model not in valid_voice_models:
            errors.append(
                f"Invalid voice.model_name '{voice_model}'. "
                f"Must be one of: {valid_voice_models}"
            )

        # Validate gesture model complexity
        gesture_complexity = self.get("gesture.model_complexity")
        if gesture_complexity is not None and gesture_complexity not in [0, 1, 2]:
            errors.append(
                f"Invalid gesture.model_complexity '{gesture_complexity}'. "
                f"Must be 0 (Lite), 1 (Full), or 2 (Heavy)"
            )

        # Validate logging level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        log_level = self.get("logging.level")
        if log_level and log_level.upper() not in valid_log_levels:
            errors.append(
                f"Invalid logging.level '{log_level}'. "
                f"Must be one of: {valid_log_levels}"
            )

        # Validate visualization settings
        drone_count = self.get("visualization.drone_count", 6)
        max_drones = self.get("visualization.max_drones", 12)
        min_drones = self.get("visualization.min_drones", 1)

        if not (min_drones <= drone_count <= max_drones):
            errors.append(
                f"Invalid drone_count {drone_count}. "
                f"Must be between {min_drones} and {max_drones}"
            )

        return errors

    @property
    def config_file(self) -> Optional[Path]:
        """Get path to loaded configuration file"""
        return self._config_file

    @property
    def is_loaded(self) -> bool:
        """Check if configuration is loaded"""
        return self._loaded

    def __repr__(self) -> str:
        return f"Config(file={self._config_file}, loaded={self._loaded})"


# Singleton instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get configuration singleton instance.

    Args:
        config_path: Path to configuration file (only used on first call)

    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance


def reset_config():
    """Reset configuration singleton (mainly for testing)"""
    global _config_instance
    _config_instance = None
