"""
主窗口 - 整合所有组件，管理应用逻辑
"""

import cv2
import time
import numpy as np
from typing import Optional, Dict
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QMessageBox, QApplication, QFrame, QLabel, QScrollArea,
    QComboBox, QPushButton, QProgressBar, QFileDialog
)
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import (
    QCloseEvent, QKeyEvent, QShortcut, QKeySequence,
    QPixmap, QPainter, QColor, QPen, QBrush
)

from .styles import MAIN_STYLESHEET
from .control_panel import ControlPanel
from .video_widget import VideoWidget
from utils.logger import get_logger
from config import config

logger = get_logger(__name__)


class CollapsibleDroneCard(QFrame):
    """
    可展开/折叠的无人机状态卡片

    折叠时: 显示 ID、状态指示灯、简要状态
    展开时: 显示位置、电量、信号、高度、速度等详细信息
    """

    def __init__(self, drone_id: int, color: tuple, parent=None, label: str = None):
        super().__init__(parent)
        self._drone_id = drone_id
        self._color = color
        self._color_hex = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
        self._label = label or f"UAV-{self._drone_id}"
        self._is_expanded = False

        # 模拟数据
        self._data = {
            'status': '待机',
            'position': (0.0, 0.0, 0.0),
            'battery': 100,
            'signal': 100,
            'speed': 0.0,
            'altitude': 0.0,
        }

        self._setup_ui()

    def _setup_ui(self):
        """初始化UI"""
        self.setStyleSheet("""
            CollapsibleDroneCard {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
            }
            CollapsibleDroneCard:hover {
                border: 1px solid #1e88e5;
                background-color: #f8f9fa;
            }
        """)
        self.setCursor(Qt.PointingHandCursor)

        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(10, 8, 10, 8)
        self._main_layout.setSpacing(6)

        # ===== 头部（始终显示）=====
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        # 状态指示灯
        self._indicator = QLabel("●")
        self._indicator.setStyleSheet("color: #9e9e9e; font-size: 10px;")
        self._indicator.setFixedWidth(12)
        header_layout.addWidget(self._indicator)

        # 无人机ID
        self._id_label = QLabel(self._label)
        self._id_label.setStyleSheet(f"color: {self._color_hex}; font-size: 12px; font-weight: bold;")
        header_layout.addWidget(self._id_label)

        # 状态文字
        self._status_label = QLabel("待机")
        self._status_label.setStyleSheet("color: #757575; font-size: 11px;")
        header_layout.addWidget(self._status_label, 1)

        # 折叠态摘要：心跳 + 电量，便于全屏状态截图
        self._summary_label = QLabel("HB -- | 100%")
        self._summary_label.setStyleSheet("color: #757575; font-size: 10px;")
        self._summary_label.setFixedWidth(78)
        header_layout.addWidget(self._summary_label)

        # 展开/折叠图标
        self._expand_icon = QLabel("▶")
        self._expand_icon.setStyleSheet("color: #9e9e9e; font-size: 10px;")
        self._expand_icon.setFixedWidth(15)
        header_layout.addWidget(self._expand_icon)

        self._main_layout.addWidget(header)

        # ===== 详情区域（展开时显示）=====
        self._detail_widget = QWidget()
        self._detail_widget.setVisible(False)
        detail_layout = QVBoxLayout(self._detail_widget)
        detail_layout.setContentsMargins(20, 4, 0, 0)
        detail_layout.setSpacing(3)

        # 位置信息
        self._pos_label = QLabel("位置: (0.0, 0.0, 0.0)")
        self._pos_label.setStyleSheet("color: #757575; font-size: 10px;")
        detail_layout.addWidget(self._pos_label)

        # 高度信息
        self._alt_label = QLabel("高度: 0.0 m")
        self._alt_label.setStyleSheet("color: #757575; font-size: 10px;")
        detail_layout.addWidget(self._alt_label)

        # 速度信息
        self._speed_label = QLabel("速度: 0.0 m/s")
        self._speed_label.setStyleSheet("color: #757575; font-size: 10px;")
        detail_layout.addWidget(self._speed_label)

        # 电量和信号（水平排列）
        battery_signal_layout = QHBoxLayout()
        battery_signal_layout.setSpacing(15)

        self._battery_label = QLabel("电量: 100%")
        self._battery_label.setStyleSheet("color: #4caf50; font-size: 10px; font-weight: bold;")
        battery_signal_layout.addWidget(self._battery_label)

        self._signal_label = QLabel("信号: 100%")
        self._signal_label.setStyleSheet("color: #4caf50; font-size: 10px; font-weight: bold;")
        battery_signal_layout.addWidget(self._signal_label)

        battery_signal_layout.addStretch()
        detail_layout.addLayout(battery_signal_layout)

        self._main_layout.addWidget(self._detail_widget)

        # 设置初始高度
        self._update_height()

    def _update_height(self):
        """更新卡片高度"""
        if self._is_expanded:
            self.setFixedHeight(115)
        else:
            self.setFixedHeight(34)

    def mousePressEvent(self, event):
        """点击切换展开/折叠"""
        self._is_expanded = not self._is_expanded
        self._detail_widget.setVisible(self._is_expanded)
        self._expand_icon.setText("▼" if self._is_expanded else "▶")
        self._update_height()
        super().mousePressEvent(event)

    def update_data(self, data: dict):
        """
        更新显示数据

        Args:
            data: {'status': str, 'position': tuple, 'battery': int, 'signal': int, 'speed': float}
        """
        self._data.update(data)

        status = data.get('status', '待机')
        robot_id = data.get('robot_id')
        pos = data.get('position', (0, 0, 0))
        battery = data.get('battery', 100)
        signal = data.get('signal', 100)
        speed = data.get('speed', 0.0)
        online = data.get('online', status == '飞行中')
        heartbeat = data.get('heartbeat')

        # 更新状态指示灯颜色
        if online:
            self._indicator.setStyleSheet("color: #4caf50; font-size: 10px;")
            self._status_label.setStyleSheet("color: #4caf50; font-size: 11px; font-weight: bold;")
        else:
            self._indicator.setStyleSheet("color: #9e9e9e; font-size: 10px;")
            self._status_label.setStyleSheet("color: #757575; font-size: 11px;")

        # 更新文字
        if robot_id:
            self._id_label.setText(robot_id)
        self._status_label.setText(status)
        self._pos_label.setText(f"位置: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        self._alt_label.setText(f"高度: {pos[2]:.1f} m")
        self._speed_label.setText(f"速度: {speed:.1f} m/s")

        # 电量颜色
        if battery > 50:
            bat_color = "#4caf50"
        elif battery > 20:
            bat_color = "#ff9800"
        else:
            bat_color = "#f44336"
        self._battery_label.setText(f"电量: {battery}%")
        self._battery_label.setStyleSheet(f"color: {bat_color}; font-size: 10px; font-weight: bold;")

        if heartbeat is not None:
            sig_color = "#4caf50" if heartbeat else "#ff9800"
            hb_text = "跳动" if heartbeat else "等待"
            hb_summary = "ON" if heartbeat else "WAIT"
            self._signal_label.setText(f"心跳: {hb_text}")
            self._signal_label.setStyleSheet(f"color: {sig_color}; font-size: 10px; font-weight: bold;")
        else:
            # 信号颜色
            if signal > 60:
                sig_color = "#4caf50"
            elif signal > 30:
                sig_color = "#ff9800"
            else:
                sig_color = "#f44336"
            self._signal_label.setText(f"信号: {signal}%")
            self._signal_label.setStyleSheet(f"color: {sig_color}; font-size: 10px; font-weight: bold;")
            hb_summary = f"{signal}%"

        self._summary_label.setText(f"HB {hb_summary} | {battery}%")
        self._summary_label.setStyleSheet(f"color: {bat_color}; font-size: 10px;")
from .history_table import HistoryTable
from .progress_dialog import ProgressDialog
from .swarm_view_3d import SwarmView3D, FormationType as ViewFormationType, SwarmCommand as ViewSwarmCommand
from workers.camera_worker import CameraWorker
from detectors.base_detector import DetectionResult
from detectors.voice_detector import VoiceDetector
from detectors.gesture_detector import GestureDetector
from workers.gesture_worker import GestureWorker
from workers.image_worker import ImageWorker
from workers.funasr_worker import FunASRWorker
from workers.sensevoice_worker import SenseVoiceWorker
from detectors.image_detector import ImageDetector
from detectors.touch_detector import TouchDetector

# ROS Bridge (可选)
try:
    from ros_bridge import ROSBridge, SwarmCommand, FormationType
    _ros_available = True
except ImportError:
    _ros_available = False
    # Fallback 到 3D 可视化的内部枚举
    SwarmCommand = ViewSwarmCommand
    FormationType = ViewFormationType


class MainWindow(QMainWindow):
    """
    多模态检测器主窗口

    整合:
    - 左栏: 控制面板 (ControlPanel)
    - 中栏: 视频显示 (VideoWidget)
    - 右栏: 历史表格 (HistoryTable)

    管理:
    - 摄像头采集 (CameraWorker)
    - 四模态检测器 (Voice/Gesture/Image/Touch)
    - 帧更新定时器
    """

    def __init__(self):
        super().__init__()
        self._setup_window()
        self._init_components()
        self._setup_ui()
        self._connect_signals()
        self._setup_shortcuts()
        self._init_detectors()
        QTimer.singleShot(500, self._init_heavy_detectors)

    def _setup_window(self):
        """设置窗口属性"""
        self.setWindowTitle("Multimodal Detector")
        self.setMinimumSize(1280, 720)
        self.resize(1280, 720)

        # 应用样式表
        self.setStyleSheet(MAIN_STYLESHEET)

    def _init_components(self):
        """初始化组件"""
        # 摄像头工作类
        self._camera = CameraWorker(camera_id=0)

        # 帧更新定时器 (30 FPS)
        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(16)  # ~60 FPS 目标

        # 检测器实例
        self._voice_detector: Optional[VoiceDetector] = None
        self._funasr_worker: Optional[FunASRWorker] = None
        self._gesture_detector: Optional[GestureDetector] = None
        self._gesture_worker: Optional[GestureWorker] = None  # 异步手势检测
        self._image_detector: Optional[ImageDetector] = None  # 仅用于绘制（缓存结果）
        self._image_worker: Optional[ImageWorker] = None      # 异步 YOLO 推理
        self._touch_detector: Optional[TouchDetector] = None

        # 检测器初始化状态
        self._detectors_initialized = False

        # FPS 计算
        self._fps = 0.0
        self._frame_count = 0
        self._fps_start_time = time.time()

        # 跳帧优化 - 重型检测器不需要每帧运行 (从配置读取)
        self._detection_frame_count = 0
        self._gesture_skip_frames = config.get("gesture.skip_frames", 4)
        self._image_skip_frames = config.get("image.skip_frames", 5)
        logger.debug(f"跳帧设置: gesture={self._gesture_skip_frames}, image={self._image_skip_frames}")

        # 无人机状态更新独立计数器（不受 FPS 计算重置影响）
        self._drone_status_frame_count = 0

        # 各模态置信度
        self._confidences: Dict[str, float] = {
            "gesture": 0.0,
            "image": 0.0,
            "voice": 0.0,
            "touch": 0.0,
        }

        # 当前手势命令
        self._current_gesture_command = ""

        # 任务闭环状态
        self._target_goal = 10
        self._target_progress = 0
        self._target_event_index = 0
        self._task_completed = False

        # 手势到指令的映射 (使用 GestureType.value 中文)
        self._gesture_to_ros_command = {
            "集群起飞": SwarmCommand.TAKEOFF,        # 张开手掌 -> 起飞
            "集群降落": SwarmCommand.LAND,          # 握拳 -> 降落
            "集群悬停": SwarmCommand.HOVER,         # 食指向上 -> 悬停
            "编队飞行": SwarmCommand.FORMATION,     # 摇滚手势 -> 编队
            "指令确定": SwarmCommand.CONFIRM,       # OK手势 -> 确认
            "集群高度上升": SwarmCommand.ALTITUDE_UP,    # 竖起大拇指 -> 上升
            "集群高度下降": SwarmCommand.ALTITUDE_DOWN,  # 向下大拇指 -> 下降
            "向前飞行": SwarmCommand.MOVE_FORWARD,  # V形手势（剪刀手）-> 向前飞行
        }
        # 手绘形状到编队类型的映射
        self._shape_to_formation = {
            "triangle": FormationType.TRIANGLE,
            "square": FormationType.SQUARE,
            "circle": FormationType.CIRCLE,
            "star": FormationType.STAR,
            "line": FormationType.LINE,
        }

        # ROS Bridge (可选)
        self._ros_bridge = None
        if _ros_available and config.get("ros.enabled", False):
            self._ros_bridge = ROSBridge()

        # 错误处理
        self._error_counts: Dict[str, int] = {
            "gesture": 0,
            "image": 0,
            "voice": 0,
            "touch": 0,
        }
        self._max_errors_before_disable = 10  # 连续错误次数阈值
        self._error_cooldown: Dict[str, float] = {}  # 错误冷却时间

        # 中文字体 (跨平台查找)
        from utils.font_utils import find_cjk_font
        self._font = find_cjk_font(size=18)

    def _create_drone_status_panel(self) -> QWidget:
        """创建无人机状态监控面板（竖向排列，可展开卡片）"""
        panel = QWidget()

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # 标题（与命令历史标题风格一致）
        title = QLabel("设备状态")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title)

        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)

        # 状态容器
        status_container = QWidget()
        status_container.setStyleSheet("background-color: transparent;")
        self._drone_status_layout = QVBoxLayout(status_container)
        self._drone_status_layout.setContentsMargins(0, 0, 0, 0)
        self._drone_status_layout.setSpacing(4)

        # 创建每台设备的可展开状态卡片
        self._drone_status_cards = []
        if hasattr(self, '_swarm_view_3d'):
            status_list = self._swarm_view_3d.get_device_status_list()
        else:
            status_list = []
        for i, status in enumerate(status_list):
            color = status.get('color', (0.2, 0.5, 0.9))
            card = CollapsibleDroneCard(
                i,
                color,
                label=status.get('robot_id', f"DEV-{i + 1:02d}")
            )
            self._drone_status_layout.addWidget(card)
            self._drone_status_cards.append(card)

        self._drone_status_layout.addStretch()

        scroll.setWidget(status_container)
        layout.addWidget(scroll, 1)

        return panel

    def _create_swarm_control_card(self) -> QFrame:
        """创建集群控制卡片（风格与无人机状态卡片一致）"""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
            }
        """)
        card.setFixedHeight(42)

        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)

        title = QLabel("平台数量")
        title.setStyleSheet("font-weight: bold; font-size: 13px; color: #333;")
        layout.addWidget(title)

        self._drone_count_label = QLabel(self._platform_count_summary())
        self._drone_count_label.setWordWrap(False)
        self._drone_count_label.setStyleSheet("""
            font-size: 12px;
            color: #1e88e5;
            font-weight: bold;
        """)
        layout.addWidget(self._drone_count_label, 1)

        return card

    def _platform_count_summary(self) -> str:
        """返回右侧栏总平台数量摘要。"""
        if not hasattr(self, '_swarm_view_3d'):
            drone_count = config.get("visualization.drone_count", 8)
            dog_count = config.get("visualization.robot_dog_count", 1)
            ugv_count = config.get("visualization.ugv_count", 1)
        else:
            drone_count = self._swarm_view_3d.drone_count
            dog_count = len(self._swarm_view_3d.robot_dogs)
            ugv_count = len(self._swarm_view_3d.ugvs)

        total = drone_count + dog_count + ugv_count
        return f"{total} 台（无人机 {drone_count}、机器狗 {dog_count}、无人车 {ugv_count}）"

    def _create_platform_command_panel(self) -> QFrame:
        """创建三类平台的本地仿真指令面板。"""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
            }
        """)
        card.setFixedHeight(46)

        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 7, 12, 7)
        layout.setSpacing(6)

        title = QLabel("平台指令")
        title.setFixedWidth(58)
        title.setStyleSheet("font-weight: bold; font-size: 13px; color: #333;")
        layout.addWidget(title)

        self._platform_command_specs = [
            ("drone", "无人机", [
                ("起飞", "takeoff"),
                ("降落", "land"),
                ("编队", "formation"),
                ("上升", "altitude_up"),
                ("下降", "altitude_down"),
                ("前进", "move_forward"),
                ("急停", "emergency"),
            ]),
            ("robot_dog", "机器狗", [
                ("站立", "stand"),
                ("趴下", "lie_down"),
                ("坐下", "sit"),
                ("前进", "forward"),
                ("后退", "backward"),
                ("左转", "turn_left"),
                ("右转", "turn_right"),
                ("停止", "stop"),
            ]),
            ("ugv", "无人车", [
                ("启动", "start"),
                ("停车", "park"),
                ("前进", "forward"),
                ("后退", "backward"),
                ("左转", "turn_left"),
                ("右转", "turn_right"),
                ("加速", "speed_up"),
                ("减速", "speed_down"),
            ]),
        ]

        self._platform_selector = QComboBox()
        self._platform_selector.setFixedWidth(76)
        self._platform_selector.setFixedHeight(28)
        self._platform_selector.setStyleSheet("""
            QComboBox {
                padding: 0px 6px;
                min-height: 28px;
                max-height: 28px;
                border: 1px solid #d0d7de;
                border-radius: 4px;
                background-color: #ffffff;
            }
        """)
        for platform, platform_label, _ in self._platform_command_specs:
            self._platform_selector.addItem(platform_label, platform)
        layout.addWidget(self._platform_selector)

        self._platform_command_combo = QComboBox()
        self._platform_command_combo.setFixedHeight(28)
        self._platform_command_combo.setStyleSheet("""
            QComboBox {
                padding: 0px 6px;
                min-height: 28px;
                max-height: 28px;
                border: 1px solid #d0d7de;
                border-radius: 4px;
                background-color: #ffffff;
            }
        """)
        layout.addWidget(self._platform_command_combo, 1)

        execute_button = QPushButton("执行")
        execute_button.setFixedWidth(48)
        execute_button.setFixedHeight(28)
        execute_button.setStyleSheet("""
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #1e88e5;
                border-radius: 4px;
                color: #1e88e5;
                font-size: 12px;
                font-weight: bold;
                padding: 0px;
                min-height: 28px;
                max-height: 28px;
            }
            QPushButton:hover {
                background-color: #e3f2fd;
            }
        """)
        execute_button.clicked.connect(self._execute_selected_platform_command)
        layout.addWidget(execute_button)

        self._platform_selector.currentIndexChanged.connect(
            self._refresh_platform_command_combo
        )
        self._refresh_platform_command_combo()

        return card

    def _refresh_platform_command_combo(self):
        """根据平台选择刷新对应指令集。"""
        if not hasattr(self, '_platform_command_combo'):
            return

        platform = self._platform_selector.currentData()
        self._platform_command_combo.clear()
        for platform_id, _, commands in self._platform_command_specs:
            if platform_id == platform:
                for command_label, command_value in commands:
                    self._platform_command_combo.addItem(command_label, command_value)
                break

    def _execute_selected_platform_command(self):
        """执行当前平台下拉框选择的指令。"""
        platform = self._platform_selector.currentData()
        command = self._platform_command_combo.currentData()
        if not platform or not command:
            return

        self._execute_platform_panel_command(
            platform,
            self._platform_selector.currentText(),
            command,
            self._platform_command_combo.currentText(),
        )

    def _create_acceptance_demo_panel(self) -> QFrame:
        """创建任务闭环面板。"""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
            }
        """)
        card.setFixedHeight(267)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(5)

        self._command_popup_label = QLabel("指令: 待触发")
        self._command_popup_label.setWordWrap(True)
        self._command_popup_label.setFixedHeight(34)
        self._command_popup_label.setStyleSheet("""
            background-color: #e3f2fd;
            border: 1px solid #64b5f6;
            border-radius: 4px;
            padding: 4px 6px;
            color: #0d47a1;
            font-size: 11px;
            font-weight: bold;
        """)
        layout.addWidget(self._command_popup_label)

        self._target_progress_bar = QProgressBar()
        self._target_progress_bar.setRange(0, self._target_goal)
        self._target_progress_bar.setValue(0)
        self._target_progress_bar.setFormat("目标: %v/%m")
        self._target_progress_bar.setFixedHeight(18)
        layout.addWidget(self._target_progress_bar)

        self._target_event_card = QFrame()
        self._target_event_card.setFixedHeight(58)
        self._target_event_card.setStyleSheet("""
            QFrame {
                background-color: #fff8e1;
                border: 1px solid #ffb300;
                border-radius: 4px;
            }
        """)
        event_layout = QHBoxLayout(self._target_event_card)
        event_layout.setContentsMargins(6, 5, 6, 5)
        event_layout.setSpacing(6)

        self._target_image_label = QLabel()
        self._target_image_label.setFixedSize(64, 42)
        self._target_image_label.setPixmap(self._make_target_pixmap("TGT"))
        self._target_image_label.setScaledContents(True)
        event_layout.addWidget(self._target_image_label)

        self._target_event_text = QLabel("事件: 暂无")
        self._target_event_text.setWordWrap(True)
        self._target_event_text.setStyleSheet("font-size: 11px; color: #4e342e;")
        event_layout.addWidget(self._target_event_text, 1)
        layout.addWidget(self._target_event_card)

        button_row_1 = QHBoxLayout()
        button_row_1.setSpacing(5)
        self._btn_demo_recognition = QPushButton("指令")
        self._btn_demo_topology = QPushButton("拓扑")
        self._btn_demo_target = QPushButton("目标")
        compact_button_style = """
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #1e88e5;
                border-radius: 4px;
                color: #1e88e5;
                font-size: 12px;
                font-weight: bold;
                padding: 0px;
                min-height: 24px;
                max-height: 24px;
            }
            QPushButton:hover {
                background-color: #e3f2fd;
            }
            QPushButton:pressed {
                background-color: #bbdefb;
            }
        """
        for button in (self._btn_demo_recognition, self._btn_demo_topology, self._btn_demo_target):
            button.setFixedHeight(24)
            button.setStyleSheet(compact_button_style)
            button_row_1.addWidget(button)
        layout.addLayout(button_row_1)

        button_row_2 = QHBoxLayout()
        button_row_2.setSpacing(5)
        self._btn_demo_complete = QPushButton("10/10")
        self._btn_demo_rtl = QPushButton("归建")
        for button in (self._btn_demo_complete, self._btn_demo_rtl):
            button.setFixedHeight(24)
            button.setStyleSheet(compact_button_style)
            button_row_2.addWidget(button)
        layout.addLayout(button_row_2)

        self._closure_status_label = QLabel("状态: 待命")
        self._closure_status_label.setFixedHeight(16)
        self._closure_status_label.setStyleSheet("color: #757575; font-size: 11px;")
        layout.addWidget(self._closure_status_label)

        self._btn_demo_recognition.clicked.connect(self._simulate_recognition_popup)
        self._btn_demo_topology.clicked.connect(self._simulate_topology_demo)
        self._btn_demo_target.clicked.connect(self._simulate_target_event)
        self._btn_demo_complete.clicked.connect(self._complete_targets_demo)
        self._btn_demo_rtl.clicked.connect(self._trigger_rtl_demo)

        return card

    def _make_target_pixmap(self, label: str) -> QPixmap:
        """生成目标事件缩略图。"""
        pixmap = QPixmap(72, 48)
        pixmap.fill(QColor("#263238"))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#ffca28"), 2))
        painter.setBrush(QBrush(QColor(255, 202, 40, 80)))
        painter.drawEllipse(22, 10, 28, 28)
        painter.drawLine(36, 2, 36, 46)
        painter.drawLine(8, 24, 64, 24)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(4, 44, label[:8])
        painter.end()
        return pixmap

    def _record_demo_event(self, command: str, details: dict):
        """向历史表记录任务闭环事件。"""
        self._history_table.add_result(
            DetectionResult(
                modal_type="simulation",
                command=command,
                confidence=1.0,
                details=details,
            )
        )

    def _show_command_popup(self, text: str, source: str = "仿真识别"):
        """显示识别文字指令弹窗区域。"""
        if not hasattr(self, '_command_popup_label'):
            return

        self._command_popup_label.setText(f"指令: {source} -> {text}")
        self._command_popup_label.setStyleSheet("""
            background-color: #e8f5e9;
            border: 2px solid #43a047;
            border-radius: 4px;
            padding: 4px 6px;
            color: #1b5e20;
            font-size: 11px;
            font-weight: bold;
        """)

    def _set_target_progress(self, value: int):
        """更新目标进度。"""
        self._target_progress = max(0, min(self._target_goal, value))
        if hasattr(self, '_target_progress_bar'):
            self._target_progress_bar.setValue(self._target_progress)
            self._target_progress_bar.setFormat(
                f"目标: {self._target_progress}/{self._target_goal}"
            )

    def _simulate_recognition_popup(self):
        """模拟语音/手势识别成功后的文字指令弹窗。"""
        command_text = "建立网络拓扑，以1号车为中心"
        self._show_command_popup(command_text, "语音/手势")
        self._record_demo_event(
            "识别成功: 建立网络拓扑",
            {"source": "voice_or_gesture", "command_label": command_text}
        )
        self.statusBar().showMessage("识别成功，已转换为文字指令", 1500)

    def _simulate_topology_demo(self):
        """启动拓扑辐射动画。"""
        self._show_command_popup("建立网络拓扑，以1号车为中心", "指令执行")
        self._swarm_view_3d.start_topology_animation()
        self._closure_status_label.setText("状态: 拓扑建立中")
        self._record_demo_event(
            "拓扑动画: 1号车中心辐射",
            {"center": "UGV-01", "device_count": 10, "mode": "local_simulation"}
        )
        self.statusBar().showMessage("拓扑动画已启动", 1500)

    def _simulate_target_event(self):
        """模拟边缘设备发现目标并弹出事件卡片。"""
        devices = [
            device for device in self._swarm_view_3d.get_device_status_list()
            if device.get("robot_id") != "UGV-01"
        ]
        if not devices:
            return

        device = devices[self._target_event_index % len(devices)]
        self._target_event_index += 1
        self._set_target_progress(self._target_progress + 1)

        base_pos = device.get("position", (0.0, 0.0, 0.0))
        coords = (
            base_pos[0] + 1.2 + 0.15 * self._target_progress,
            base_pos[1] + 0.8,
            max(0.0, base_pos[2]),
        )
        robot_id = device.get("robot_id", "UNKNOWN")
        self._target_image_label.setPixmap(self._make_target_pixmap(robot_id))
        self._target_event_text.setText(
            f"图像已截获  {robot_id}\n坐标: ({coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f})"
        )
        self._target_event_card.setStyleSheet("""
            QFrame {
                background-color: #fff3e0;
                border: 2px solid #fb8c00;
                border-radius: 4px;
            }
        """)
        self._closure_status_label.setText("状态: 目标搜索中")
        self._record_demo_event(
            f"目标发现: {robot_id}",
            {
                "robot_id": robot_id,
                "absolute_position": coords,
                "progress": f"{self._target_progress}/{self._target_goal}",
            }
        )

        if self._target_progress >= self._target_goal:
            self._closure_status_label.setText("状态: 目标 10/10")

    def _complete_targets_demo(self):
        """直接推进到目标进度 10/10，便于截局部进度图。"""
        self._set_target_progress(self._target_goal)
        self._show_command_popup("目标进度已达到 10/10", "任务闭环")
        self._target_image_label.setPixmap(self._make_target_pixmap("10/10"))
        self._target_event_text.setText("目标计数已完成\n进度: 10/10")
        self._target_event_card.setStyleSheet("""
            QFrame {
                background-color: #e8f5e9;
                border: 2px solid #43a047;
                border-radius: 4px;
            }
        """)
        self._closure_status_label.setText("状态: 目标 10/10")
        self._record_demo_event(
            "目标进度: 10/10",
            {"progress": "10/10", "mode": "local_simulation"}
        )
        self.statusBar().showMessage("目标进度已达到 10/10", 1500)

    def _trigger_rtl_demo(self):
        """弹出任务完成并下发 RTL 归建指令。"""
        self._task_completed = True
        self._set_target_progress(self._target_goal)
        self._show_command_popup("任务完成，全员归建（RTL）", "全局闭环")
        self._closure_status_label.setText("状态: 任务完成，RTL 已下发")

        self._swarm_view_3d.execute_platform_command("drone", "rtl")
        self._swarm_view_3d.execute_platform_command("robot_dog", "stop")
        self._swarm_view_3d.execute_platform_command("ugv", "park")

        self._record_demo_event(
            "任务完成: 全员归建 RTL",
            {"progress": "10/10", "command": "rtl", "mode": "local_simulation"}
        )
        self.statusBar().showMessage("任务完成，全员归建 RTL 已下发", 2000)

    @Slot()
    def _on_add_drone(self):
        """增加无人机"""
        if not hasattr(self, '_swarm_view_3d'):
            return

        current_count = self._swarm_view_3d.drone_count
        max_count = config.get("visualization.max_drones", 8)

        if current_count >= max_count:
            self.statusBar().showMessage(f"已达到最大数量 ({max_count} 架)", 2000)
            return

        new_count = current_count + 1
        self._swarm_view_3d.set_drone_count(new_count)
        self._update_drone_count_display()
        self._rebuild_drone_status_cards()
        self.statusBar().showMessage(
            f"无人机数量: {new_count} 架，{self._platform_count_summary()}",
            1500
        )

    @Slot()
    def _on_remove_drone(self):
        """减少无人机"""
        if not hasattr(self, '_swarm_view_3d'):
            return

        current_count = self._swarm_view_3d.drone_count
        min_count = config.get("visualization.min_drones", 8)

        if current_count <= min_count:
            self.statusBar().showMessage(f"至少保留 {min_count} 架无人机", 2000)
            return

        new_count = current_count - 1
        self._swarm_view_3d.set_drone_count(new_count)
        self._update_drone_count_display()
        self._rebuild_drone_status_cards()
        self.statusBar().showMessage(
            f"无人机数量: {new_count} 架，{self._platform_count_summary()}",
            1500
        )

    def _update_drone_count_display(self):
        """更新右侧总平台数量显示。"""
        if hasattr(self, '_drone_count_label') and hasattr(self, '_swarm_view_3d'):
            self._drone_count_label.setText(self._platform_count_summary())

    def _rebuild_drone_status_cards(self):
        """重建设备状态卡片列表"""
        if not hasattr(self, '_drone_status_layout'):
            return

        # 清除现有卡片
        for card in self._drone_status_cards:
            card.deleteLater()
        self._drone_status_cards.clear()

        # 重新创建卡片
        status_list = self._swarm_view_3d.get_device_status_list()
        for i, status in enumerate(status_list):
            card = CollapsibleDroneCard(
                i,
                status.get('color', (0.2, 0.5, 0.9)),
                label=status.get('robot_id', f"DEV-{i + 1:02d}")
            )
            self._drone_status_layout.insertWidget(i, card)
            self._drone_status_cards.append(card)

    def _update_drone_status_display(self):
        """更新 10 台设备状态显示"""
        if not hasattr(self, '_swarm_view_3d'):
            return

        status_list = self._swarm_view_3d.get_device_status_list()

        if len(status_list) != len(self._drone_status_cards):
            self._rebuild_drone_status_cards()

        for idx, status in enumerate(status_list):
            if idx < len(self._drone_status_cards):
                card = self._drone_status_cards[idx]
                card.update_data(status)

    def _update_ground_platform_status_display(self):
        """更新机器狗/无人车的简要状态。"""
        if (not hasattr(self, '_swarm_view_3d') or
                not hasattr(self, '_ground_platform_status_label')):
            return

        status_list = self._swarm_view_3d.get_ground_platform_status()
        parts = []
        for status in status_list:
            parts.append(
                f"{status.get('name', '平台')}: {status.get('status', '待命')}"
            )
        self._ground_platform_status_label.setText(" | ".join(parts))

    def _execute_platform_panel_command(
        self,
        platform: str,
        platform_label: str,
        command: str,
        command_label: str
    ):
        """执行平台面板发出的本地仿真指令。"""
        self._execute_simulation_command(platform, platform_label, command, command_label)

    def _execute_simulation_command(
        self,
        platform: str,
        platform_label: str,
        command: str,
        command_label: str
    ):
        """统一执行并记录三平台本地仿真指令。"""
        if not hasattr(self, '_swarm_view_3d'):
            return

        self._swarm_view_3d.execute_platform_command(platform, command)

        result = DetectionResult(
            modal_type="simulation",
            command=f"{platform_label}: {command_label}",
            confidence=1.0,
            details={
                "platform": platform,
                "platform_label": platform_label,
                "command": command,
                "command_label": command_label,
                "mode": "local_simulation",
            }
        )
        self._history_table.add_result(result)
        self._update_ground_platform_status_display()
        self.statusBar().showMessage(f"{platform_label}指令: {command_label}", 1500)

    def _setup_ui(self):
        """设置界面布局"""
        # 中央容器
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 三列分割器 (水平)
        self._splitter = QSplitter(Qt.Horizontal)

        # 左栏: 控制面板
        self._control_panel = ControlPanel()
        self._control_panel.setMinimumWidth(200)
        self._control_panel.setMaximumWidth(280)

        # 中栏: 上下分割 (3D可视化 + 视频)
        self._center_splitter = QSplitter(Qt.Vertical)

        # 中栏上部: PyQtGraph 3D 可视化
        self._swarm_view_3d = SwarmView3D()
        self._swarm_view_3d.setMinimumHeight(200)

        # 中栏下部: 视频显示
        self._video_widget = VideoWidget()
        self._video_widget.setMinimumHeight(200)

        self._center_splitter.addWidget(self._swarm_view_3d)
        self._center_splitter.addWidget(self._video_widget)

        # 设置中栏上下比例 (1:1)
        self._center_splitter.setSizes([350, 350])
        self._center_splitter.setStretchFactor(0, 1)
        self._center_splitter.setStretchFactor(1, 1)

        # 右栏: 垂直布局 (命令历史 + 控制区 + 设备状态)
        right_widget = QWidget()
        right_widget.setMinimumWidth(320)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # 右栏上部: 命令历史表格
        self._history_table = HistoryTable()
        self._history_table.setMinimumWidth(200)
        self._history_table.setMinimumHeight(92)
        self._history_table.setMaximumHeight(130)

        # 右栏中部: 集群控制卡片 (新增)
        self._swarm_control_card = self._create_swarm_control_card()

        # 右栏中部: 三平台本地仿真指令
        self._platform_command_panel = self._create_platform_command_panel()

        # 右栏中部: 任务闭环触发面板
        self._acceptance_demo_panel = self._create_acceptance_demo_panel()

        # 右栏下部: 无人机状态监控面板
        self._drone_status_panel = self._create_drone_status_panel()

        # 按顺序添加: 命令历史 → 控制区 → 设备状态
        right_layout.addWidget(self._history_table)
        right_layout.addWidget(self._swarm_control_card)
        right_layout.addWidget(self._platform_command_panel)
        right_layout.addWidget(self._acceptance_demo_panel)
        right_layout.addWidget(self._drone_status_panel, 5)  # 展示 10 台设备并发状态

        # 添加到主分割器
        self._splitter.addWidget(self._control_panel)
        self._splitter.addWidget(self._center_splitter)
        self._splitter.addWidget(right_widget)

        # 设置初始比例 (右栏略宽，避免指令区文字换行压缩)
        self._splitter.setSizes([230, 690, 340])
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 3)
        self._splitter.setStretchFactor(2, 1)

        main_layout.addWidget(self._splitter)

        # 显示占位符
        self._video_widget.display_placeholder()

    def _connect_signals(self):
        """连接信号槽"""
        # 摄像头控制按钮
        self._video_widget.start_camera_clicked.connect(self._start_camera)
        self._video_widget.stop_camera_clicked.connect(self._stop_camera)

        # 帧更新
        self._frame_timer.timeout.connect(self._update_frame)

        # 摄像头事件
        self._camera.error_occurred.connect(self._on_camera_error)

        # 控制面板信号
        self._control_panel.modal_toggled.connect(self._on_modal_toggled)
        self._control_panel.threshold_changed.connect(self._on_threshold_changed)
        self._control_panel.reset_clicked.connect(self._on_reset)
        self._control_panel.export_clicked.connect(self._on_export)
        self._control_panel.record_clicked.connect(self._on_record_clicked)
        self._control_panel.camera_changed.connect(self._on_camera_changed)
        self._control_panel.refresh_cameras_clicked.connect(self._refresh_cameras)
        self._control_panel.load_map_clicked.connect(self._on_load_map_clicked)

        # 视频区域鼠标点击
        self._video_widget.mouse_clicked.connect(self._on_video_click)

        # 3D 可视化信号
        self._swarm_view_3d.command_executed.connect(self._on_swarm_command_executed)

    def _setup_shortcuts(self):
        """设置快捷键"""
        # S 键: 启动/停止摄像头
        shortcut_camera = QShortcut(QKeySequence(Qt.Key_S), self)
        shortcut_camera.activated.connect(self._toggle_camera)

        # R 键: 重置
        shortcut_reset = QShortcut(QKeySequence(Qt.Key_R), self)
        shortcut_reset.activated.connect(self._on_reset)

        # E 键: 导出
        shortcut_export = QShortcut(QKeySequence(Qt.Key_E), self)
        shortcut_export.activated.connect(self._on_export)

        # 数字键 1-4: 切换模态开关
        shortcut_1 = QShortcut(QKeySequence(Qt.Key_1), self)
        shortcut_1.activated.connect(lambda: self._toggle_modal("voice"))

        shortcut_2 = QShortcut(QKeySequence(Qt.Key_2), self)
        shortcut_2.activated.connect(lambda: self._toggle_modal("gesture"))

        shortcut_3 = QShortcut(QKeySequence(Qt.Key_3), self)
        shortcut_3.activated.connect(lambda: self._toggle_modal("image"))

        shortcut_4 = QShortcut(QKeySequence(Qt.Key_4), self)
        shortcut_4.activated.connect(lambda: self._toggle_modal("touch"))

        # F1: 显示快捷键帮助
        shortcut_help = QShortcut(QKeySequence(Qt.Key_F1), self)
        shortcut_help.activated.connect(self._show_shortcuts_help)

        # 更新窗口标题提示
        self.statusBar().showMessage("按 F1 查看快捷键帮助", 5000)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            if self._control_panel.is_modal_enabled("voice"):
                btn = self._control_panel.btn_record
                if not btn.isChecked():
                    btn.setChecked(True)
                    self._on_record_clicked(True)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            if self._control_panel.is_modal_enabled("voice"):
                btn = self._control_panel.btn_record
                if btn.isChecked():
                    btn.setChecked(False)
                    self._on_record_clicked(False)
        super().keyReleaseEvent(event)

    def _toggle_camera(self):
        """切换摄像头状态"""
        if self._camera.is_opened:
            self._stop_camera()
        else:
            self._start_camera()

    def _toggle_modal(self, modal_name: str):
        """切换指定模态的开关状态"""
        checkbox_map = {
            "voice": self._control_panel.cb_voice,
            "gesture": self._control_panel.cb_gesture,
            "image": self._control_panel.cb_image,
            "touch": self._control_panel.cb_touch,
        }
        checkbox = checkbox_map.get(modal_name)
        if checkbox:
            checkbox.setChecked(not checkbox.isChecked())

    def _show_shortcuts_help(self):
        """显示快捷键帮助"""
        help_text = """
<h3>快捷键列表</h3>
<table style="border-collapse: collapse; width: 100%;">
<tr><td style="padding: 5px;"><b>空格</b></td><td style="padding: 5px;">按住录音，松开识别（Push-to-Talk）</td></tr>
<tr><td style="padding: 5px;"><b>S</b></td><td style="padding: 5px;">启动/停止摄像头</td></tr>
<tr><td style="padding: 5px;"><b>R</b></td><td style="padding: 5px;">重置统计</td></tr>
<tr><td style="padding: 5px;"><b>E</b></td><td style="padding: 5px;">导出历史</td></tr>
<tr><td style="padding: 5px;"><b>1</b></td><td style="padding: 5px;">切换语音识别</td></tr>
<tr><td style="padding: 5px;"><b>2</b></td><td style="padding: 5px;">切换手势识别</td></tr>
<tr><td style="padding: 5px;"><b>3</b></td><td style="padding: 5px;">切换图像识别</td></tr>
<tr><td style="padding: 5px;"><b>4</b></td><td style="padding: 5px;">切换触屏指令</td></tr>
<tr><td style="padding: 5px;"><b>F1</b></td><td style="padding: 5px;">显示此帮助</td></tr>
</table>
"""
        QMessageBox.information(self, "快捷键帮助", help_text)

    def _init_detectors(self):
        """初始化检测器"""
        # 触屏检测器（不需要模型，立即初始化）
        self._touch_detector = TouchDetector()
        self._touch_detector.initialize()
        self._touch_detector.detection_ready.connect(self._on_detection_result)

        # 其他检测器延迟初始化（需要加载模型）
        # 将在后台线程中初始化，避免阻塞UI

    def _init_heavy_detectors(self):
        """初始化重型检测器（需要加载模型）"""
        if self._detectors_initialized:
            return

        # 创建进度对话框
        progress = ProgressDialog("初始化检测器", self)
        progress.show()
        QApplication.processEvents()

        total_steps = 4
        current_step = 0
        errors = []

        # 步骤1: 创建语音识别 Worker（可通过配置切换引擎）
        current_step += 1
        asr_engine = config.get("voice.asr_engine", "sensevoice")  # sensevoice | funasr
        asr_label = "FunASR" if asr_engine == "funasr" else "SenseVoice"
        if asr_engine == "funasr":
            progress.set_status("正在加载语音模型...")
            progress.set_progress(int(current_step / total_steps * 100))
            progress.set_detail(f"步骤 {current_step}/{total_steps} - 语音识别模块")
            self._funasr_worker = FunASRWorker()
        else:
            progress.set_status("正在加载语音模型...")
            progress.set_progress(int(current_step / total_steps * 100))
            progress.set_detail(f"步骤 {current_step}/{total_steps} - 语音识别模块")
            self._funasr_worker = SenseVoiceWorker()
        self._funasr_worker.detection_ready.connect(
            self._on_detection_result, Qt.QueuedConnection
        )
        self._funasr_worker.error_occurred.connect(
            self._on_detector_error, Qt.QueuedConnection
        )
        self._funasr_worker.status_changed.connect(
            lambda s: self._control_panel.set_record_status(s),
            Qt.QueuedConnection,
        )
        if not self._funasr_worker.initialize():
            errors.append(f"{asr_label} 模型加载失败")
        else:
            self._funasr_worker.start()

        self._voice_detector = VoiceDetector(funasr_worker=self._funasr_worker)
        self._voice_detector.status_changed.connect(
            lambda s: self._control_panel.set_record_status(s)
        )
        self._voice_detector.error_occurred.connect(self._on_detector_error)
        self._voice_detector.initialize()
        self._voice_detector.audio_chunk.connect(
            self._funasr_worker.enqueue, Qt.QueuedConnection
        )
        QTimer.singleShot(100, self._connect_voice_overlay_signals)

        # 步骤2: 创建并初始化手势检测器 (异步版本)
        current_step += 1
        progress.set_progress(int(current_step / total_steps * 100))
        progress.set_detail(f"步骤 {current_step}/{total_steps} - 手势识别模块")

        # 同步检测器 (仅用于绘制)
        self._gesture_detector = GestureDetector()
        self._gesture_detector.detection_ready.connect(self._on_detection_result)
        self._gesture_detector.error_occurred.connect(self._on_detector_error)
        try:
            if not self._gesture_detector.initialize():
                errors.append("手势检测器初始化失败")
        except Exception as e:
            errors.append(f"手势检测器: {str(e)}")

        # 异步检测工作线程
        self._gesture_worker = GestureWorker()
        self._gesture_worker.result_ready.connect(
            self._on_gesture_worker_result, Qt.QueuedConnection
        )
        try:
            if self._gesture_worker.initialize():
                self._gesture_worker.start()
            else:
                errors.append("手势异步工作线程初始化失败")
        except Exception as e:
            errors.append(f"手势工作线程: {str(e)}")

        # 步骤3: 创建并初始化图像检测器（异步工作线程）
        current_step += 1
        progress.set_progress(int(current_step / total_steps * 100))
        progress.set_detail(f"步骤 {current_step}/{total_steps} - 图像识别模块")

        # ImageDetector 仅保留绘制功能（draw_detections），推理由 ImageWorker 负责
        self._image_detector = ImageDetector()
        self._image_detector.detection_ready.connect(self._on_detection_result)
        self._image_detector.error_occurred.connect(self._on_detector_error)

        # 异步 YOLO 工作线程
        self._image_worker = ImageWorker()
        self._image_worker.initialize(
            model_name=config.get("image.model_name", "yolov8n"),
            conf_threshold=config.get("image.confidence_threshold", 0.5),
        )
        self._image_worker.result_ready.connect(
            self._on_image_worker_result, Qt.QueuedConnection
        )
        self._image_worker.start()

        # 步骤4: 完成
        current_step += 1
        progress.set_progress(100)
        progress.set_detail("初始化完成")

        self._detectors_initialized = True
        progress.close()

        # 显示错误信息（如果有）
        if errors:
            error_msg = "以下检测器初始化时出现问题:\n\n" + "\n".join(f"• {e}" for e in errors)
            error_msg += "\n\n相关功能可能不可用。"
            QMessageBox.warning(self, "初始化警告", error_msg)
            self.statusBar().showMessage("部分检测器初始化失败", 5000)
        else:
            self.statusBar().showMessage("所有检测器初始化完成", 3000)

    @Slot()
    def _start_camera(self):
        """启动摄像头"""
        # 初始化重型检测器
        if not self._detectors_initialized:
            self._init_heavy_detectors()

        # 初始化 ROS Bridge (可选)
        if self._ros_bridge and not self._ros_bridge.is_initialized:
            if self._ros_bridge.initialize():
                self.statusBar().showMessage("ROS 连接已建立", 2000)
            else:
                self.statusBar().showMessage("ROS 不可用，仅本地模式", 2000)

        if self._camera.open():
            self._frame_timer.start()
            self._video_widget.set_camera_running(True)
            self.statusBar().showMessage("摄像头已启动")

            # 更新触屏检测器的视频尺寸
            video_size = self._video_widget.get_video_size()
            if self._touch_detector:
                self._touch_detector.set_video_size(*video_size)
        else:
            QMessageBox.warning(
                self, "摄像头错误",
                "无法打开摄像头。请检查:\n"
                "1. 摄像头是否已连接\n"
                "2. 是否有其他程序正在使用摄像头\n"
                "3. 是否有访问摄像头的权限"
            )

    @Slot()
    def _stop_camera(self):
        """停止摄像头"""
        self._frame_timer.stop()
        self._camera.release()
        self._video_widget.set_camera_running(False)
        self.statusBar().showMessage("摄像头已停止")

    @Slot()
    def _update_frame(self):
        """更新视频帧"""
        frame = self._camera.read_frame()
        if frame is None:
            return

        # 计算 FPS
        self._frame_count += 1
        elapsed = time.time() - self._fps_start_time
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_start_time = time.time()

        # 运行启用的检测器
        processed_frame = frame.copy()

        # 更新检测帧计数
        self._detection_frame_count += 1

        # 手势检测 (异步模式)
        try:
            if self._control_panel.is_modal_enabled("gesture"):
                # 提交帧到异步工作线程 (非阻塞)
                if self._gesture_worker and self._gesture_worker.isRunning():
                    self._gesture_worker.submit_frame(frame)

                    # 使用工作线程缓存的结果绘制
                    cached = self._gesture_worker.get_cached_results()
                    if cached and cached.multi_hand_landmarks:
                        # 将缓存结果传给检测器用于绘制
                        self._gesture_detector._cached_results = cached
                        drawn = self._gesture_detector.draw_landmarks(processed_frame)
                        if drawn is not None:
                            processed_frame = drawn
        except Exception as e:
            self._handle_detection_error("gesture", e)

        # 图像检测 (异步 ImageWorker)
        try:
            if (self._control_panel.is_modal_enabled("image") and
                    self._image_worker and self._image_worker.isRunning()):
                # 按跳帧频率提交帧到工作线程（非阻塞）
                if self._detection_frame_count % (self._image_skip_frames + 1) == 0:
                    self._image_worker.submit_frame(frame)
                # 绘制缓存的检测结果（每帧都绘制，不阻塞）
                if self._image_detector:
                    cached = self._image_worker.get_cached_detections()
                    drawn = self._image_detector.draw_detections(processed_frame, cached)
                    if drawn is not None:
                        processed_frame = drawn
        except Exception as e:
            self._handle_detection_error("image", e)

        # 触屏轨迹绘制
        try:
            if (self._control_panel.is_modal_enabled("touch") and
                    self._touch_detector and self._touch_detector.is_initialized):
                drawn = self._touch_detector.draw_touch_overlay(processed_frame)
                if drawn is not None:
                    processed_frame = drawn
        except Exception as e:
            self._handle_detection_error("touch", e)

        # 在左上角绘制 FPS 和置信度
        processed_frame = self._draw_stats_overlay(processed_frame)

        # 显示处理后的帧
        self._video_widget.display_frame(processed_frame)

        # 更新无人机状态显示 (每5帧更新一次，使用独立计数器)
        self._drone_status_frame_count += 1
        if self._drone_status_frame_count >= 5:
            self._drone_status_frame_count = 0
            self._update_drone_status_display()
            self._update_ground_platform_status_display()

    def _draw_stats_overlay(self, frame):
        """
        在帧上绘制 FPS 和手势信息

        手势未启用时走 cv2.putText 快速路径（避免 PIL 全帧转换）。
        手势启用时使用 PIL 渲染中文文字。
        """
        h, w = frame.shape[:2]
        gesture_enabled = self._control_panel.is_modal_enabled("gesture")

        bg_width = 180
        bg_height = 70 if gesture_enabled else 30
        x_start = w - bg_width - 10
        y_start = 10

        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start),
                      (x_start + bg_width, y_start + bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        if not gesture_enabled:
            # 快速路径：仅 FPS，用 cv2.putText（无 PIL 转换）
            fps_text = f"FPS: {self._fps:.1f}"
            cv2.putText(frame, fps_text, (x_start + 8, y_start + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            return frame

        # 慢速路径：含中文，使用 PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        x_offset = x_start + 8
        y_offset = y_start + 5
        line_height = 20

        fps_text = f"FPS: {self._fps:.1f}"
        draw.text((x_offset, y_offset), fps_text, font=self._font, fill=(0, 255, 0))
        y_offset += line_height

        gesture_conf = self._confidences.get("gesture", 0.0)
        conf_text = f"置信度: {gesture_conf:.0%}"
        draw.text((x_offset, y_offset), conf_text, font=self._font, fill=(255, 165, 0))
        y_offset += line_height

        if self._current_gesture_command:
            cmd = self._current_gesture_command
            if len(cmd) > 8:
                cmd = cmd[:7] + "..."
            draw.text((x_offset, y_offset), f"命令: {cmd}", font=self._font, fill=(0, 255, 255))

        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return frame

    @Slot(str)
    def _on_camera_error(self, error_msg: str):
        """处理摄像头错误"""
        self._stop_camera()
        QMessageBox.warning(self, "摄像头错误", error_msg)

    @Slot(str)
    def _on_detector_error(self, error_msg: str):
        """处理检测器错误"""
        self.statusBar().showMessage(f"检测器错误: {error_msg}", 5000)

    @Slot(str, float, object)
    def _on_image_worker_result(self, command: str, confidence: float, detections):
        """处理异步图像检测结果"""
        if command and detections:
            self._confidences["image"] = confidence
            self._video_widget.update_modal_status("image", command, confidence)

            from detectors.base_detector import DetectionResult
            result = DetectionResult(
                modal_type="image",
                command=command,
                confidence=confidence,
                details={"async": True, "total_count": len(detections)},
            )
            self._on_detection_result(result)

    @Slot(str, float, object)
    def _on_gesture_worker_result(self, command: str, confidence: float, results):
        """处理异步手势检测结果"""
        if command and command != "无手势":
            self._confidences["gesture"] = confidence
            self._current_gesture_command = command
            self._video_widget.update_modal_status("gesture", command, confidence)

            # 创建检测结果并发送
            from detectors.base_detector import DetectionResult
            result = DetectionResult(
                modal_type="gesture",
                command=command,
                confidence=confidence,
                details={"async": True}
            )
            self._on_detection_result(result)

    def _handle_detection_error(self, modal_type: str, error: Exception):
        """
        处理检测过程中的错误

        Args:
            modal_type: 模态类型
            error: 异常对象
        """
        current_time = time.time()

        # 检查冷却时间（避免频繁提示）
        last_error_time = self._error_cooldown.get(modal_type, 0)
        if current_time - last_error_time < 5.0:  # 5秒内不重复提示
            return

        self._error_counts[modal_type] += 1
        self._error_cooldown[modal_type] = current_time

        modal_names = {
            "gesture": "手势检测",
            "image": "图像检测",
            "voice": "语音识别",
            "touch": "触屏检测",
        }
        modal_name = modal_names.get(modal_type, modal_type)

        # 在状态栏显示错误
        error_msg = str(error)[:50]  # 截断过长的错误信息
        self.statusBar().showMessage(f"{modal_name}错误: {error_msg}", 3000)

        # 如果连续错误次数过多，自动禁用该模态
        if self._error_counts[modal_type] >= self._max_errors_before_disable:
            self._error_counts[modal_type] = 0  # 重置计数
            # 禁用对应的检测器
            checkbox_map = {
                "gesture": self._control_panel.cb_gesture,
                "image": self._control_panel.cb_image,
                "voice": self._control_panel.cb_voice,
                "touch": self._control_panel.cb_touch,
            }
            checkbox = checkbox_map.get(modal_type)
            if checkbox and checkbox.isChecked():
                checkbox.setChecked(False)
                QMessageBox.warning(
                    self, "检测器异常",
                    f"{modal_name}连续出现多次错误，已自动禁用。\n\n"
                    f"错误信息: {str(error)[:100]}\n\n"
                    "您可以稍后在控制面板重新启用。"
                )

    @Slot(int)
    def _on_camera_changed(self, camera_id: int):
        """处理摄像头切换"""
        was_running = self._camera.is_opened
        if was_running:
            self._stop_camera()

        self._camera.set_camera_id(camera_id)
        self.statusBar().showMessage(f"已切换到摄像头 {camera_id}", 2000)

        if was_running:
            self._start_camera()

    @Slot()
    def _refresh_cameras(self):
        """刷新摄像头列表"""
        self.statusBar().showMessage("正在扫描摄像头设备...")
        QApplication.processEvents()

        from workers.camera_worker import CameraWorker
        available_cameras = CameraWorker.list_available_cameras()

        if available_cameras:
            self._control_panel.update_camera_list(available_cameras)
            self.statusBar().showMessage(
                f"找到 {len(available_cameras)} 个摄像头设备", 3000
            )
        else:
            self.statusBar().showMessage("未找到可用的摄像头设备", 3000)
            QMessageBox.warning(
                self, "摄像头扫描",
                "未找到可用的摄像头设备。\n请检查摄像头连接。"
            )

    @Slot()
    def _on_load_map_clicked(self):
        """处理加载点云地图按钮点击"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择点云地图文件", "", "点云文件 (*.pcd)"
        )
        if not file_path:
            return

        if not hasattr(self, '_swarm_view_3d'):
            return

        if self._swarm_view_3d.load_point_cloud_map(file_path):
            self.statusBar().showMessage(f"点云地图已加载: {file_path}", 3000)
        else:
            QMessageBox.warning(
                self, "地图加载失败",
                f"无法加载点云地图:\n{file_path}\n\n请检查文件格式是否为 PCD v0.7 (ASCII/binary)。"
            )
            self.statusBar().showMessage("点云地图加载失败", 3000)

    @Slot(str, bool)
    def _on_modal_toggled(self, modal_name: str, enabled: bool):
        """处理模态开关切换"""
        detector_map = {
            "voice": self._voice_detector,
            "gesture": self._gesture_detector,
            "image": self._image_detector,
            "touch": self._touch_detector,
        }
        detector = detector_map.get(modal_name)
        if detector:
            detector.enabled = enabled

        status = "启用" if enabled else "禁用"
        self.statusBar().showMessage(f"{modal_name} 检测已{status}", 2000)

    @Slot(str, float)
    def _on_threshold_changed(self, threshold_type: str, value: float):
        """处理阈值变更"""
        if threshold_type == "recall":
            # 更新所有检测器的阈值
            for detector in [self._voice_detector, self._gesture_detector,
                           self._image_detector, self._touch_detector]:
                if detector:
                    detector.threshold = value

    @Slot()
    def _on_reset(self):
        """处理重置操作"""
        reply = QMessageBox.question(
            self, "确认重置",
            "确定要重置所有统计数据吗？\n这将清空命令历史并重置模态状态。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self._history_table.clear()
            self._video_widget.reset_modal_status()
            self.statusBar().showMessage("已重置所有统计", 2000)

    @Slot()
    def _on_export(self):
        """处理导出操作"""
        self._history_table.export_to_json()

    @Slot(bool)
    def _on_record_clicked(self, is_recording: bool):
        if not self._control_panel.is_modal_enabled("voice"):
            self._control_panel.set_record_status("语音模态已禁用")
            return
        if self._voice_detector is None or not self._voice_detector.is_initialized:
            self._control_panel.set_record_status("语音模块未就绪")
            return

        if is_recording:
            self._video_widget.show_voice_overlay()
            self._video_widget.voice_overlay.start_recording()
            self._voice_detector.start_recording()
        else:
            self._video_widget.voice_overlay.stop_recording()
            self._voice_detector.stop_recording()

    def _connect_voice_overlay_signals(self):
        """连接语音检测器与可视化叠加层的信号"""
        if self._voice_detector and hasattr(self._video_widget, 'voice_overlay'):
            overlay = self._video_widget.voice_overlay
            logger.debug(f"连接语音可视化信号: overlay={overlay}")

            # 连接音量信号 (使用 QueuedConnection 确保线程安全)
            self._voice_detector.volume_changed.connect(
                overlay.update_volume, Qt.QueuedConnection
            )

            # 连接 FunASR 中间结果 → overlay 实时字幕
            if self._funasr_worker:
                self._funasr_worker.partial_result_ready.connect(
                    lambda text: overlay.set_result(text, 0.0, ""),
                    Qt.QueuedConnection,
                )
                # 未识别到语音时重置 overlay
                self._funasr_worker.status_changed.connect(
                    lambda s: overlay.set_idle() if s == "未识别到语音" else None,
                    Qt.QueuedConnection,
                )

            # 连接错误信号
            self._voice_detector.error_occurred.connect(
                overlay.set_error, Qt.QueuedConnection
            )
            logger.debug("语音可视化信号连接完成")
        else:
            logger.warning(f"无法连接语音可视化信号: detector={self._voice_detector}, has_overlay={hasattr(self._video_widget, 'voice_overlay')}")

    def _get_voice_command_text(self, text: str) -> str:
        """
        根据识别文本返回映射的指令名称

        Args:
            text: 识别文本

        Returns:
            str: 映射的指令名称，如果没有匹配返回空字符串
        """
        text_lower = text.lower()

        # 指令关键词映射
        command_keywords = {
            ("起飞", "takeoff", "take off", "升空"): "TAKEOFF",
            ("降落", "landing", "land", "着陆"): "LAND",
            ("悬停", "hover", "停住"): "HOVER",
            ("向前", "前进", "forward", "向前飞"): "MOVE_FORWARD",
            ("上升", "升高", "up"): "ALTITUDE_UP",
            ("下降", "降低", "down"): "ALTITUDE_DOWN",
            ("紧急停止", "急停", "emergency", "stop"): "EMERGENCY",
            # 编队
            ("三角", "triangle"): "编队:三角形",
            ("方形", "square", "正方形"): "编队:正方形",
            ("圆形", "circle", "圆"): "编队:圆形",
            ("五角星", "star", "星形"): "编队:五角星",
            ("线形", "line", "一字"): "编队:线形",
        }

        for keywords, command in command_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return command

        return ""

    @Slot(object)
    def _on_video_click(self, event):
        """处理视频区域鼠标点击"""
        if not self._control_panel.is_modal_enabled("touch"):
            return

        if self._touch_detector:
            result = self._touch_detector.detect(event)
            if result:
                self._video_widget.update_modal_status(
                    "touch", result.command, result.confidence
                )

    @Slot(object)
    def _on_detection_result(self, result: DetectionResult):
        """处理检测结果"""
        logger.debug(f"收到检测结果: 模态={result.modal_type}, 命令='{result.command}'")
        # 添加到历史表格
        self._history_table.add_result(result)

        # 更新置信度
        self._confidences[result.modal_type] = result.confidence

        # 更新状态显示
        self._video_widget.update_modal_status(
            result.modal_type,
            result.command,
            result.confidence
        )

        if result.modal_type in ("voice", "gesture"):
            source = "语音识别" if result.modal_type == "voice" else "手势识别"
            self._show_command_popup(result.command, source)

        # 发布到 ROS (如果可用)
        self._publish_to_ros(result)

    def _publish_to_ros(self, result: DetectionResult):
        """
        将检测结果发布到 3D 可视化和 ROS

        Args:
            result: 检测结果
        """
        # 只处理置信度超过阈值的结果。语音结果由 ASR worker 自行过滤。
        threshold = self._control_panel.get_recall_threshold()
        logger.debug(f"发布检测结果: 模态={result.modal_type}, 置信度={result.confidence:.2f}, 阈值={threshold:.2f}")
        if result.modal_type != "voice" and result.confidence < threshold:
            logger.debug("跳过发布：置信度低于阈值")
            return

        # ROS 是否可用
        ros_available = self._ros_bridge and self._ros_bridge.is_initialized

        # 根据模态类型处理
        if result.modal_type == "gesture":
            # 手势 -> 控制指令
            ros_cmd = self._gesture_to_ros_command.get(result.command)
            if ros_cmd:
                # 发送到 3D 可视化
                self._swarm_view_3d.execute_command(ros_cmd.value)
                # 发送到 ROS (如果可用)
                if ros_available:
                    self._ros_bridge.publish_command(ros_cmd)

        elif result.modal_type == "touch":
            # 触屏手绘形状 -> 编队变换
            formation = self._shape_to_formation.get(result.command.lower())
            if formation:
                # 发送到 3D 可视化
                self._swarm_view_3d.change_formation(formation.value)
                # 发送到 ROS (如果可用)
                if ros_available:
                    self._ros_bridge.publish_formation(formation, drone_count=6)

        elif result.modal_type == "voice":
            command_text = self._get_voice_command_text(result.command)
            if hasattr(self._video_widget, "voice_overlay"):
                self._video_widget.update_modal_status(
                    "voice", result.command, result.confidence
                )
                self._video_widget.voice_overlay.set_result(
                    result.command, result.confidence, command_text
                )
            self._process_voice_command(result.command)

    def _process_ground_voice_command(self, text_lower: str) -> bool:
        """处理带平台语义的地面平台语音指令。"""
        dog_target = any(keyword in text_lower for keyword in ("机器狗", "机器犬", "四足", "狗"))
        ugv_target = any(keyword in text_lower for keyword in ("无人车", "小车", "车辆", "ugv"))

        # "停车/启动车辆" 这类词天然指向无人车，方便演示时少说前缀。
        if not ugv_target and any(keyword in text_lower for keyword in ("停车", "启动车", "车辆")):
            ugv_target = True

        if dog_target:
            dog_commands = {
                ("站立", "起来", "stand"): ("stand", "站立"),
                ("趴下", "卧倒", "lie down", "lie_down"): ("lie_down", "趴下"),
                ("坐下", "sit"): ("sit", "坐下"),
                ("前进", "向前", "forward"): ("forward", "前进"),
                ("后退", "backward", "back"): ("backward", "后退"),
                ("左转", "turn left"): ("turn_left", "左转"),
                ("右转", "turn right"): ("turn_right", "右转"),
                ("停止", "停下", "stop"): ("stop", "停止"),
            }
            for keywords, (command, label) in dog_commands.items():
                if any(keyword in text_lower for keyword in keywords):
                    self._execute_simulation_command("robot_dog", "机器狗", command, label)
                    return True

        if ugv_target:
            ugv_commands = {
                ("启动", "start"): ("start", "启动"),
                ("停车", "停止", "停下", "park", "stop"): ("park", "停车"),
                ("前进", "向前", "forward"): ("forward", "前进"),
                ("后退", "backward", "back"): ("backward", "后退"),
                ("左转", "turn left"): ("turn_left", "左转"),
                ("右转", "turn right"): ("turn_right", "右转"),
                ("加速", "speed up", "speed_up"): ("speed_up", "加速"),
                ("减速", "speed down", "speed_down"): ("speed_down", "减速"),
            }
            for keywords, (command, label) in ugv_commands.items():
                if any(keyword in text_lower for keyword in keywords):
                    self._execute_simulation_command("ugv", "无人车", command, label)
                    return True

        return False

    def _process_voice_command(self, text: str):
        """
        处理语音指令并发送到 3D 可视化和 ROS

        Args:
            text: 语音识别文本
        """
        text_lower = text.lower()
        logger.debug(f"语音指令识别: '{text}' -> '{text_lower}'")

        if self._process_ground_voice_command(text_lower):
            return

        if any(keyword in text_lower for keyword in ("拓扑", "网络", "组网")):
            self._simulate_topology_demo()
            return

        if any(keyword in text_lower for keyword in ("发现目标", "目标发现", "检测目标")):
            self._simulate_target_event()
            return

        if any(keyword in text_lower for keyword in ("归建", "rtl", "返航", "任务完成")):
            self._trigger_rtl_demo()
            return

        # 中文关键词映射
        voice_commands = {
            # 起飞
            ("起飞", "takeoff", "take off", "升空", "飞", "起来"): SwarmCommand.TAKEOFF,
            # 降落
            ("降落", "land", "着陆", "落地", "落下", "下降落地"): SwarmCommand.LAND,
            # 悬停 (包含常见误识别)
            ("悬停", "旋停", "选停", "hover", "停", "暂停", "停住", "定住"): SwarmCommand.HOVER,
            # 高度控制
            ("上升", "升高", "go up", "higher", "高一点", "往上"): SwarmCommand.ALTITUDE_UP,
            ("下降", "降低", "go down", "lower", "低一点", "往下"): SwarmCommand.ALTITUDE_DOWN,
            # 向前飞行 (包含多种说法)
            ("向前", "前进", "forward", "向前飞", "往前飞", "往前", "前飞", "go forward"): SwarmCommand.MOVE_FORWARD,
            # 紧急停止
            ("紧急停止", "急停", "emergency", "stop", "紧急"): SwarmCommand.EMERGENCY_STOP,
        }

        for keywords, command in voice_commands.items():
            for keyword in keywords:
                if keyword in text_lower:
                    logger.info(f"语音指令匹配: '{keyword}' -> {command.value}")
                    # 发送到 3D 可视化
                    self._swarm_view_3d.execute_command(command.value)
                    # 发送到 ROS (如果可用)
                    if self._ros_bridge and self._ros_bridge.is_initialized:
                        self._ros_bridge.publish_command(command)
                    self.statusBar().showMessage(
                        f"语音指令: {command.value}", 1500
                    )
                    return

        # 编队指令 (包含常见误识别)
        formation_keywords = {
            ("三角形", "triangle", "三角", "三角型"): FormationType.TRIANGLE,
            ("正方形", "square", "方形", "四方形", "正方"): FormationType.SQUARE,
            ("圆形", "circle", "圆", "圆圈", "画圆"): FormationType.CIRCLE,
            ("五角星", "star", "星形", "星星", "五星", "武角星", "五角形", "五角"): FormationType.STAR,
            ("直线", "line", "一字", "一条线", "线形", "排成一排"): FormationType.LINE,
        }

        for keywords, formation in formation_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    logger.info(f"语音编队匹配: '{keyword}' -> {formation.value}")
                    # 发送到 3D 可视化
                    self._swarm_view_3d.change_formation(formation.value)
                    # 发送到 ROS (如果可用)
                    if self._ros_bridge and self._ros_bridge.is_initialized:
                        self._ros_bridge.publish_formation(formation, drone_count=6)
                    self.statusBar().showMessage(
                        f"语音编队: {formation.value}", 1500
                    )
                    return

        # 没有匹配到任何指令
        logger.debug(f"语音指令未匹配: '{text}'")

    @Slot(str)
    def _on_swarm_command_executed(self, command: str):
        """3D 可视化指令执行完成"""
        self._update_ground_platform_status_display()
        self.statusBar().showMessage(f"集群指令: {command}", 1500)

    def closeEvent(self, event: QCloseEvent):
        """窗口关闭事件 - 释放资源"""
        # 停止定时器
        self._frame_timer.stop()

        # 释放摄像头
        self._camera.release()

        # 停止异步工作线程
        if self._gesture_worker:
            self._gesture_worker.stop()
        if self._image_worker:
            self._image_worker.stop()

        # 释放检测器
        if self._voice_detector:
            self._voice_detector.release()
        if self._funasr_worker:
            self._funasr_worker.release()
            self._funasr_worker.wait(2000)
        if self._gesture_detector:
            self._gesture_detector.release()
        if self._image_detector:
            self._image_detector.release()
        if self._touch_detector:
            self._touch_detector.release()

        # 关闭 ROS 连接
        if self._ros_bridge:
            self._ros_bridge.shutdown()

        event.accept()
