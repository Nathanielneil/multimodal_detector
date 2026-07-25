# -*- coding: utf-8 -*-
"""
Unit Tests for Configuration Module

Tests config loading, access, and validation.
"""

import os
import pytest
from pathlib import Path

# Import after path setup in conftest
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config, ConfigError, reset_config


class TestConfigLoading:
    """Test configuration loading functionality"""

    def setup_method(self):
        """Reset config singleton before each test"""
        reset_config()

    def test_load_default_config(self):
        """Test loading default configuration file"""
        config = Config()
        assert config.is_loaded
        assert config.config_file is not None

    def test_load_custom_config(self, temp_config_file):
        """Test loading a custom configuration file"""
        config = Config(temp_config_file)
        assert config.is_loaded
        assert config.get("app.name") == "Test App"

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading a non-existent configuration file"""
        nonexistent = tmp_path / "nonexistent.yaml"
        with pytest.raises(ConfigError):
            Config(str(nonexistent))

    def test_load_invalid_yaml(self, tmp_path):
        """Test loading an invalid YAML file"""
        invalid_file = tmp_path / "invalid.yaml"
        invalid_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ConfigError):
            Config(str(invalid_file))


class TestConfigAccess:
    """Test configuration value access"""

    def setup_method(self):
        """Reset config singleton before each test"""
        reset_config()

    def test_get_simple_key(self, temp_config_file):
        """Test getting a simple configuration value"""
        config = Config(temp_config_file)
        assert config.get("app.name") == "Test App"
        assert config.get("app.version") == "1.0.0"

    def test_get_nested_key(self, temp_config_file):
        """Test getting a nested configuration value"""
        config = Config(temp_config_file)
        assert config.get("voice.model_name") == "tiny"
        assert config.get("logging.console.enabled") is True

    def test_get_with_default(self, temp_config_file):
        """Test getting a value with default"""
        config = Config(temp_config_file)
        assert config.get("nonexistent.key", "default") == "default"
        assert config.get("voice.nonexistent", 42) == 42

    def test_get_section(self, temp_config_file):
        """Test getting an entire section"""
        config = Config(temp_config_file)
        voice_section = config.get_section("voice")

        assert isinstance(voice_section, dict)
        assert voice_section["model_name"] == "tiny"
        assert voice_section["language"] == "en"

    def test_get_section_nonexistent(self, temp_config_file):
        """Test getting a non-existent section returns empty dict"""
        config = Config(temp_config_file)
        section = config.get_section("nonexistent")
        assert section == {}


class TestConfigSet:
    """Test configuration value setting"""

    def setup_method(self):
        """Reset config singleton before each test"""
        reset_config()

    def test_set_simple_value(self, temp_config_file):
        """Test setting a simple value"""
        config = Config(temp_config_file)
        config.set("app.name", "New Name")
        assert config.get("app.name") == "New Name"

    def test_set_nested_value(self, temp_config_file):
        """Test setting a nested value"""
        config = Config(temp_config_file)
        config.set("voice.model_name", "medium")
        assert config.get("voice.model_name") == "medium"

    def test_set_new_key(self, temp_config_file):
        """Test setting a new key"""
        config = Config(temp_config_file)
        config.set("new.nested.key", "value")
        assert config.get("new.nested.key") == "value"


class TestEnvironmentOverrides:
    """Test environment variable overrides"""

    def setup_method(self):
        """Reset config singleton and clear env vars before each test"""
        reset_config()
        # Clear any test env vars
        for key in list(os.environ.keys()):
            if key.startswith("MULTIMODAL_"):
                del os.environ[key]

    def teardown_method(self):
        """Clean up env vars after each test"""
        for key in list(os.environ.keys()):
            if key.startswith("MULTIMODAL_"):
                del os.environ[key]

    def test_env_override_string(self, temp_config_file):
        """Test overriding a string value via environment variable"""
        os.environ["MULTIMODAL_APP_NAME"] = "Env Override"
        config = Config(temp_config_file)
        assert config.get("app.name") == "Env Override"

    def test_env_override_bool(self, temp_config_file):
        """Test overriding a boolean value"""
        os.environ["MULTIMODAL_VOICE_ENABLED"] = "false"
        config = Config(temp_config_file)
        assert config.get("voice.enabled") is False

    def test_env_override_int(self, temp_config_file):
        """Test overriding an integer value (single-word key)"""
        # Note: Use single-word keys to avoid underscore ambiguity
        os.environ["MULTIMODAL_GESTURE_ENABLED"] = "0"
        config = Config(temp_config_file)
        # 0 is parsed as integer, then compared
        assert config.get("gesture.enabled") == 0

    def test_env_override_null(self, temp_config_file):
        """Test setting a null value via environment variable"""
        os.environ["MULTIMODAL_VOICE_LANGUAGE"] = "null"
        config = Config(temp_config_file)
        assert config.get("voice.language") is None


class TestConfigValidation:
    """Test configuration validation"""

    def setup_method(self):
        """Reset config singleton before each test"""
        reset_config()

    def test_validate_valid_config(self, temp_config_file):
        """Test validation with valid configuration"""
        config = Config(temp_config_file)
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_invalid_voice_model(self, tmp_path):
        """Test validation catches invalid voice model"""
        config_content = """
voice:
  model_name: "invalid_model"
"""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text(config_content)

        config = Config(str(config_file))
        errors = config.validate()
        assert any("voice.model_name" in e for e in errors)

    def test_validate_invalid_asr_engine(self, tmp_path):
        """Test validation catches an unsupported ASR backend."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text('voice:\n  asr_engine: "unsupported"\n')

        config = Config(str(config_file))
        errors = config.validate()
        assert any("voice.asr_engine" in e for e in errors)

    def test_validate_invalid_log_level(self, tmp_path):
        """Test validation catches invalid log level"""
        config_content = """
logging:
  level: "INVALID"
"""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text(config_content)

        config = Config(str(config_file))
        errors = config.validate()
        assert any("logging.level" in e for e in errors)


class TestConfigReload:
    """Test configuration reload functionality"""

    def setup_method(self):
        """Reset config singleton before each test"""
        reset_config()

    def test_reload_config(self, tmp_path):
        """Test reloading configuration picks up changes"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("app:\n  name: Original")

        config = Config(str(config_file))
        assert config.get("app.name") == "Original"

        # Modify file
        config_file.write_text("app:\n  name: Updated")

        # Reload
        config.reload()
        assert config.get("app.name") == "Updated"


class TestConfigSave:
    """Test configuration save functionality"""

    def setup_method(self):
        """Reset config singleton before each test"""
        reset_config()

    def test_save_config(self, temp_config_file, tmp_path):
        """Test saving configuration to file"""
        config = Config(temp_config_file)
        config.set("app.name", "Saved Name")

        save_path = tmp_path / "saved_config.yaml"
        config.save(str(save_path))

        # Load saved config
        reset_config()
        loaded = Config(str(save_path))
        assert loaded.get("app.name") == "Saved Name"
