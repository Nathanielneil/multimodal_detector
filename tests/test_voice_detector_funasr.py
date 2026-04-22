import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_worker():
    w = MagicMock()
    w.is_initialized = True
    return w


@pytest.fixture
def detector(qtbot, mock_worker):
    with patch("detectors.voice_detector.sounddevice"):
        from detectors.voice_detector import VoiceDetector
        d = VoiceDetector(funasr_worker=mock_worker)
        d._is_initialized = True
        yield d
        d.release()


def test_audio_chunk_signal_exists(detector):
    """VoiceDetector must have audio_chunk signal."""
    assert hasattr(detector, "audio_chunk")


def test_stop_recording_calls_send_eos(qtbot, detector, mock_worker):
    """stop_recording() must call funasr_worker.send_eos()."""
    detector._is_recording = True
    detector.stop_recording()
    mock_worker.send_eos.assert_called_once()


def test_stop_recording_returns_none(qtbot, detector, mock_worker):
    """stop_recording() no longer returns a DetectionResult."""
    detector._is_recording = True
    result = detector.stop_recording()
    assert result is None


def test_start_recording_blocked_when_worker_not_initialized(qtbot, mock_worker):
    with patch("detectors.voice_detector.sounddevice"):
        from detectors.voice_detector import VoiceDetector
        mock_worker.is_initialized = False
        d = VoiceDetector(funasr_worker=mock_worker)
        d._is_initialized = True
        errors = []
        d.error_occurred.connect(errors.append)
        d.start_recording()
        assert errors, "Should emit error when worker not initialized"
