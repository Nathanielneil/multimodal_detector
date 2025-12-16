# -*- coding: utf-8 -*-
"""
UI Module - All interface components

Components:
- MainWindow: Main window
- ControlPanel: Left control panel
- VideoWidget: Center video display
- HistoryTable: Right history table
- styles: QSS style definitions
"""

from .main_window import MainWindow
from .control_panel import ControlPanel
from .video_widget import VideoWidget
from .history_table import HistoryTable
from .styles import MAIN_STYLESHEET, MODAL_COLORS

__all__ = [
    "MainWindow",
    "ControlPanel",
    "VideoWidget",
    "HistoryTable",
    "MAIN_STYLESHEET",
    "MODAL_COLORS",
]
