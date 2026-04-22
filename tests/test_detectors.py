# -*- coding: utf-8 -*-
"""
Unit Tests for Detector Modules

Tests for voice, gesture, image, and touch detectors.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from detectors.base_detector import BaseDetector, DetectionResult


class TestDetectionResult:
    """Test DetectionResult dataclass"""

    def test_create_detection_result(self):
        """Test creating a basic detection result"""
        result = DetectionResult(
            modal_type="gesture",
            command="takeoff",
            confidence=0.95,
            details={},
        )

        assert result.modal_type == "gesture"
        assert result.command == "takeoff"
        assert result.confidence == 0.95
        assert result.details == {}
        assert result.timestamp is not None

    def test_detection_result_with_details(self):
        """Test detection result with extra details"""
        result = DetectionResult(
            modal_type="voice",
            command="land",
            confidence=0.8,
            details={"language": "zh", "duration": 2.5},
        )

        assert result.details["language"] == "zh"
        assert result.details["duration"] == 2.5

    def test_detection_result_to_dict(self):
        """Test converting detection result to dictionary"""
        result = DetectionResult(
            modal_type="image",
            command="person detected",
            confidence=0.75,
            details={"count": 1},
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["modal_type"] == "image"
        assert result_dict["command"] == "person detected"
        assert result_dict["confidence"] == 0.75


class TestGestureDetector:
    """Test GestureDetector functionality"""

    @pytest.fixture
    def gesture_detector(self):
        """Create a gesture detector with mocked dependencies"""
        with patch("detectors.gesture_detector.mp_hands", MagicMock()):
            with patch("detectors.gesture_detector.mp_drawing", MagicMock()):
                with patch("detectors.gesture_detector.cv2", MagicMock()):
                    from detectors.gesture_detector import GestureDetector
                    detector = GestureDetector(max_hands=1)
                    return detector

    def test_gesture_detector_init(self, gesture_detector):
        """Test gesture detector initialization"""
        assert gesture_detector is not None
        assert gesture_detector.enabled is True
        assert not gesture_detector.is_initialized

    def test_gesture_stable_threshold(self, gesture_detector):
        """Test gesture stability threshold"""
        assert gesture_detector._stable_threshold == 5

        gesture_detector._stable_threshold = 3
        assert gesture_detector._stable_threshold == 3

    def test_gesture_recognition_fist(self, gesture_detector, fist_landmarks):
        """Test recognizing closed fist gesture"""
        from detectors.gesture_detector import GestureType

        gesture, confidence = gesture_detector._recognize_gesture(fist_landmarks)

        # Fist should be recognized (all fingers down)
        assert gesture == GestureType.FIST
        assert confidence >= 0.8

    def test_gesture_recognition_open_palm(self, gesture_detector, gesture_landmarks):
        """Test recognizing open palm gesture"""
        from detectors.gesture_detector import GestureType

        # Create landmarks for open palm (all fingers extended up)
        open_palm_landmarks = []
        for i in range(21):
            x = 0.5 + (i % 5) * 0.03
            # Fingers pointing up (lower y = higher on screen)
            if i in [4, 8, 12, 16, 20]:  # fingertips
                y = 0.2
            elif i in [3, 7, 11, 15, 19]:  # DIP
                y = 0.3
            elif i in [2, 6, 10, 14, 18]:  # PIP/IP
                y = 0.4
            elif i in [1, 5, 9, 13, 17]:  # MCP/CMC
                y = 0.5
            else:  # wrist
                y = 0.6
            open_palm_landmarks.append((x, y, 0.0))

        gesture, confidence = gesture_detector._recognize_gesture(open_palm_landmarks)
        assert gesture == GestureType.OPEN_PALM
        assert confidence >= 0.8


class TestVoiceDetector:
    """Test VoiceDetector functionality"""

    @pytest.fixture
    def voice_detector(self):
        """Create a voice detector with mocked sounddevice"""
        with patch("detectors.voice_detector.sounddevice", MagicMock()):
            from detectors.voice_detector import VoiceDetector
            mock_worker = MagicMock()
            mock_worker.is_initialized = True
            detector = VoiceDetector(funasr_worker=mock_worker)
            return detector

    def test_voice_detector_init(self, voice_detector):
        """Test voice detector initialization"""
        assert voice_detector is not None
        assert not voice_detector.is_initialized

    def test_voice_detector_initialize(self, voice_detector):
        """Test voice detector initialize() marks ready"""
        result = voice_detector.initialize()
        assert result is True
        assert voice_detector.is_initialized

    def test_voice_detector_set_device(self, voice_detector):
        """Test setting audio device"""
        voice_detector.set_device(5)
        assert voice_detector._audio_device == 5

        voice_detector.set_device(None)
        assert voice_detector._audio_device is None


class TestTouchDetector:
    """Test TouchDetector functionality"""

    @pytest.fixture
    def touch_detector(self):
        """Create a touch detector"""
        with patch("detectors.touch_detector.cv2", MagicMock()):
            from detectors.touch_detector import TouchDetector
            detector = TouchDetector()
            return detector

    def test_touch_detector_init(self, touch_detector):
        """Test touch detector initialization"""
        assert touch_detector is not None
        assert not touch_detector.is_initialized

    def test_touch_detector_initialize(self, touch_detector):
        """Test touch detector initialization"""
        result = touch_detector.initialize()
        assert result is True
        assert touch_detector.is_initialized

    def test_touch_detector_set_video_size(self, touch_detector):
        """Test setting video size"""
        touch_detector.initialize()
        touch_detector.set_video_size(1280, 720)

        # Check the video size tuple
        assert touch_detector._video_size == (1280, 720)


class TestImageDetector:
    """Test ImageDetector functionality"""

    @pytest.fixture
    def image_detector(self):
        """Create an image detector with mocked YOLO"""
        # Mock the ultralytics global variable
        mock_yolo_class = MagicMock()
        mock_model = MagicMock()
        mock_model.names = {0: "person", 1: "car"}
        mock_yolo_class.return_value = mock_model

        with patch("detectors.image_detector.ultralytics", mock_yolo_class):
            with patch("detectors.image_detector.cv2", MagicMock()):
                from detectors.image_detector import ImageDetector
                detector = ImageDetector(model_name="yolov8n")
                return detector

    def test_image_detector_init(self, image_detector):
        """Test image detector initialization"""
        assert image_detector is not None
        assert image_detector._model_name == "yolov8n"
        assert not image_detector.is_initialized


class TestDetectorIntegration:
    """Integration tests for detector modules"""

    def test_detection_result_serialization(self):
        """Test that detection results can be serialized"""
        import json

        result = DetectionResult(
            modal_type="gesture",
            command="takeoff",
            confidence=0.95,
            details={"hands": 1},
        )

        result_dict = result.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(result_dict)
        assert "gesture" in json_str
        assert "takeoff" in json_str

    def test_multiple_detectors_coexist(self):
        """Test that multiple detectors can exist simultaneously"""
        with patch("detectors.gesture_detector.mp_hands", MagicMock()):
            with patch("detectors.gesture_detector.mp_drawing", MagicMock()):
                with patch("detectors.gesture_detector.cv2", MagicMock()):
                    with patch("detectors.touch_detector.cv2", MagicMock()):
                        from detectors.gesture_detector import GestureDetector
                        from detectors.touch_detector import TouchDetector

                        gesture = GestureDetector()
                        touch = TouchDetector()

                        # Both should be creatable
                        assert gesture is not None
                        assert touch is not None

                        # Initialize touch (gesture needs MediaPipe)
                        touch.initialize()
                        assert touch.is_initialized

                        touch.release()
                        assert not touch.is_initialized
