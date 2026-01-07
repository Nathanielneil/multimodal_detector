# -*- coding: utf-8 -*-
"""
Pytest Configuration and Fixtures

Provides common fixtures and configuration for all tests.
"""

import sys
import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary configuration file for testing"""
    config_content = """
app:
  name: "Test App"
  version: "1.0.0"

voice:
  enabled: true
  model_name: "tiny"
  language: "en"

gesture:
  enabled: true
  max_hands: 1
  stable_threshold: 3

logging:
  level: "DEBUG"
  console:
    enabled: true
  file:
    enabled: false
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return str(config_file)


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for tests"""
    return tmp_path


# =============================================================================
# Mock Fixtures for Detectors
# =============================================================================

@pytest.fixture
def mock_cv2():
    """Mock OpenCV module"""
    with patch.dict(sys.modules, {"cv2": MagicMock()}):
        yield sys.modules["cv2"]


@pytest.fixture
def mock_mediapipe():
    """Mock MediaPipe module"""
    mock_mp = MagicMock()
    mock_mp.solutions.hands.Hands.return_value = MagicMock()
    mock_mp.solutions.drawing_utils = MagicMock()

    with patch.dict(sys.modules, {"mediapipe": mock_mp}):
        yield mock_mp


@pytest.fixture
def mock_whisper():
    """Mock Whisper module"""
    mock_whisper = MagicMock()
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": "test transcription",
        "language": "zh",
        "segments": [{"avg_logprob": -0.3}]
    }
    mock_whisper.load_model.return_value = mock_model

    with patch.dict(sys.modules, {"whisper": mock_whisper}):
        yield mock_whisper


@pytest.fixture
def sample_frame():
    """Generate a sample BGR image frame for testing"""
    # Create a 480x640 BGR image with random noise
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def sample_audio():
    """Generate sample audio data for testing"""
    # Generate 1 second of random audio at 16000 Hz
    sample_rate = 16000
    duration = 1.0
    audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
    audio = audio / np.max(np.abs(audio))  # Normalize
    return audio


# =============================================================================
# Qt Application Fixture
# =============================================================================

@pytest.fixture(scope="session")
def qapp():
    """Create a Qt application instance for UI tests"""
    try:
        from PySide6.QtWidgets import QApplication

        # Check if app already exists
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app
    except ImportError:
        pytest.skip("PySide6 not available")


# =============================================================================
# Detector Fixtures
# =============================================================================

@pytest.fixture
def gesture_landmarks():
    """Sample hand landmarks for gesture testing"""
    # Create 21 landmarks with x, y, z coordinates
    # Simulating an open palm gesture
    landmarks = []
    for i in range(21):
        # Spread fingers upward for open palm
        x = 0.5 + (i % 4) * 0.05
        y = 0.5 - (i // 4) * 0.1
        z = 0.0
        landmarks.append((x, y, z))
    return landmarks


@pytest.fixture
def fist_landmarks():
    """Sample hand landmarks for closed fist gesture"""
    # Create 21 landmarks simulating a closed fist
    landmarks = []
    for i in range(21):
        # All fingers curled down
        x = 0.5 + (i % 4) * 0.02
        y = 0.5 + (i // 4) * 0.05  # Fingers down
        z = 0.0
        landmarks.append((x, y, z))
    return landmarks


# =============================================================================
# Utility Functions
# =============================================================================

def assert_valid_detection_result(result):
    """Assert that a detection result has required fields"""
    assert result is not None
    assert hasattr(result, "modal_type")
    assert hasattr(result, "command")
    assert hasattr(result, "confidence")
    assert 0.0 <= result.confidence <= 1.0
