# -*- coding: utf-8 -*-
"""
UI Module - All interface components

Components:
- MainWindow: Main window
- ControlPanel: Left control panel
- VideoWidget: Center video display
- HistoryTable: Right history table
- SwarmView3D: PyQtGraph 3D drone swarm visualization
- ProgressDialog: Progress dialog for loading operations
- VoiceOverlayWidget: Voice recognition visualization overlay
- styles: QSS style definitions
"""

from .main_window import MainWindow
from .control_panel import ControlPanel
from .video_widget import VideoWidget
from .history_table import HistoryTable
from .swarm_view_3d import SwarmView3D
from .progress_dialog import ProgressDialog, LoadingOverlay
from .voice_overlay import VoiceOverlayWidget
from .styles import MAIN_STYLESHEET, MODAL_COLORS

__all__ = [
    "MainWindow",
    "ControlPanel",
    "VideoWidget",
    "HistoryTable",
    "SwarmView3D",
    "ProgressDialog",
    "LoadingOverlay",
    "VoiceOverlayWidget",
    "MAIN_STYLESHEET",
    "MODAL_COLORS",
]
