# -*- coding: utf-8 -*-
"""
Detectors Module - Multimodal detectors

Detectors:
- VoiceDetector: Voice recognition (Whisper)
- GestureDetector: Gesture recognition (MediaPipe)
- ImageDetector: Image recognition (YOLOv8)
- TouchDetector: Touch command detection

Base:
- BaseDetector: Detector base class
- DetectionResult: Detection result data class
"""

from .base_detector import BaseDetector, DetectionResult
from .voice_detector import VoiceDetector
from .gesture_detector import GestureDetector
from .image_detector import ImageDetector
from .touch_detector import TouchDetector

__all__ = [
    "BaseDetector",
    "DetectionResult",
    "VoiceDetector",
    "GestureDetector",
    "ImageDetector",
    "TouchDetector",
]
