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
    QMessageBox, QApplication, QFrame, QLabel, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QCloseEvent, QKeyEvent, QShortcut, QKeySequence

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

    def __init__(self, drone_id: int, color: tuple, parent=None):
        super().__init__(parent)
        self._drone_id = drone_id
        self._color = color
        self._color_hex = f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}"
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
        self._id_label = QLabel(f"UAV-{self._drone_id}")
        self._id_label.setStyleSheet(f"color: {self._color_hex}; font-size: 12px; font-weight: bold;")
        header_layout.addWidget(self._id_label)

        # 状态文字
        self._status_label = QLabel("待机")
        self._status_label.setStyleSheet("color: #757575; font-size: 11px;")
        header_layout.addWidget(self._status_label, 1)

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
            self.setFixedHeight(36)

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
        pos = data.get('position', (0, 0, 0))
        battery = data.get('battery', 100)
        signal = data.get('signal', 100)
        speed = data.get('speed', 0.0)

        # 更新状态指示灯颜色
        if status == '飞行中':
            self._indicator.setStyleSheet("color: #4caf50; font-size: 10px;")
            self._status_label.setStyleSheet("color: #4caf50; font-size: 11px; font-weight: bold;")
        else:
            self._indicator.setStyleSheet("color: #9e9e9e; font-size: 10px;")
            self._status_label.setStyleSheet("color: #757575; font-size: 11px;")

        # 更新文字
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

        # 信号颜色
        if signal > 60:
            sig_color = "#4caf50"
        elif signal > 30:
            sig_color = "#ff9800"
        else:
            sig_color = "#f44336"
        self._signal_label.setText(f"信号: {signal}%")
        self._signal_label.setStyleSheet(f"color: {sig_color}; font-size: 10px; font-weight: bold;")
from .history_table import HistoryTable
from .progress_dialog import ProgressDialog
from .swarm_view_3d import SwarmView3D, FormationType as ViewFormationType, SwarmCommand as ViewSwarmCommand
from workers.camera_worker import CameraWorker
from detectors.base_detector import DetectionResult
from detectors.voice_detector import VoiceDetector
from detectors.gesture_detector import GestureDetector
from workers.gesture_worker import GestureWorker
from detectors.image_detector import ImageDetector
from detectors.touch_detector import TouchDetector

# ROS Bridge (可选)
try:
    from ros_bridge import ROSBridge, SwarmCommand, FormationType
    _ros_available = True
except ImportError:
    _ros_available = False


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
        self._gesture_detector: Optional[GestureDetector] = None
        self._gesture_worker: Optional[GestureWorker] = None  # 异步手势检测
        self._image_detector: Optional[ImageDetector] = None
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

        # 各模态置信度
        self._confidences: Dict[str, float] = {
            "gesture": 0.0,
            "image": 0.0,
            "voice": 0.0,
            "touch": 0.0,
        }

        # 当前手势命令
        self._current_gesture_command = ""

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
        if _ros_available:
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

        # 中文字体 (使用系统字体)
        self._font = None
        try:
            # 尝试加载常见的中文字体
            font_paths = [
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            ]
            for path in font_paths:
                try:
                    self._font = ImageFont.truetype(path, 18)
                    break
                except OSError:
                    # Font file not found or cannot be loaded
                    continue
            if self._font is None:
                self._font = ImageFont.load_default()
        except Exception:
            # Fallback to default font on any unexpected error
            self._font = ImageFont.load_default()

    def _create_drone_status_panel(self) -> QWidget:
        """创建无人机状态监控面板（竖向排列，可展开卡片）"""
        panel = QWidget()

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # 标题（与命令历史标题风格一致）
        title = QLabel("无人机状态")
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

        # 创建每架无人机的可展开状态卡片
        self._drone_status_cards = []
        from .swarm_view_3d import DroneModel
        for i in range(6):  # 默认6架
            card = CollapsibleDroneCard(i, DroneModel.COLORS[i % len(DroneModel.COLORS)])
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
        card.setFixedHeight(80)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # 标题行
        title_layout = QHBoxLayout()
        title_layout.setSpacing(8)

        title = QLabel("集群数量")
        title.setStyleSheet("font-weight: bold; font-size: 13px; color: #333;")
        title_layout.addWidget(title)
        title_layout.addStretch()

        layout.addLayout(title_layout)

        # 控制行：数量显示 + 增减按钮
        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)

        # 当前数量标签
        self._drone_count_label = QLabel("当前: 6 架")
        self._drone_count_label.setStyleSheet("""
            font-size: 13px;
            color: #1e88e5;
            font-weight: bold;
        """)
        control_layout.addWidget(self._drone_count_label)

        control_layout.addStretch()

        # 减少按钮
        from PySide6.QtWidgets import QPushButton
        self._btn_remove_drone = QPushButton("-")
        self._btn_remove_drone.setFixedSize(36, 36)
        self._btn_remove_drone.setCursor(Qt.PointingHandCursor)
        self._btn_remove_drone.setStyleSheet("""
            QPushButton {
                background-color: #ffebee;
                border: 2px solid #ef9a9a;
                border-radius: 8px;
                font-size: 24px;
                font-weight: bold;
                color: #c62828;
                padding: 0px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #ffcdd2;
                border-color: #e57373;
            }
            QPushButton:pressed {
                background-color: #ef9a9a;
            }
        """)
        self._btn_remove_drone.clicked.connect(self._on_remove_drone)
        control_layout.addWidget(self._btn_remove_drone)

        # 增加按钮
        self._btn_add_drone = QPushButton("+")
        self._btn_add_drone.setFixedSize(36, 36)
        self._btn_add_drone.setCursor(Qt.PointingHandCursor)
        self._btn_add_drone.setStyleSheet("""
            QPushButton {
                background-color: #e8f5e9;
                border: 2px solid #a5d6a7;
                border-radius: 8px;
                font-size: 24px;
                font-weight: bold;
                color: #2e7d32;
                padding: 0px;
                text-align: center;
            }
            QPushButton:hover {
                background-color: #c8e6c9;
                border-color: #81c784;
            }
            QPushButton:pressed {
                background-color: #a5d6a7;
            }
        """)
        self._btn_add_drone.clicked.connect(self._on_add_drone)
        control_layout.addWidget(self._btn_add_drone)

        layout.addLayout(control_layout)

        return card

    @Slot()
    def _on_add_drone(self):
        """增加无人机"""
        if not hasattr(self, '_swarm_view_3d'):
            return

        current_count = self._swarm_view_3d.drone_count
        max_count = 12  # 最大数量限制

        if current_count >= max_count:
            self.statusBar().showMessage(f"已达到最大数量 ({max_count} 架)", 2000)
            return

        new_count = current_count + 1
        self._swarm_view_3d.set_drone_count(new_count)
        self._update_drone_count_display()
        self._rebuild_drone_status_cards()
        self.statusBar().showMessage(f"集群数量: {new_count} 架", 1500)

    @Slot()
    def _on_remove_drone(self):
        """减少无人机"""
        if not hasattr(self, '_swarm_view_3d'):
            return

        current_count = self._swarm_view_3d.drone_count
        min_count = 1  # 最小数量限制

        if current_count <= min_count:
            self.statusBar().showMessage(f"至少保留 {min_count} 架无人机", 2000)
            return

        new_count = current_count - 1
        self._swarm_view_3d.set_drone_count(new_count)
        self._update_drone_count_display()
        self._rebuild_drone_status_cards()
        self.statusBar().showMessage(f"集群数量: {new_count} 架", 1500)

    def _update_drone_count_display(self):
        """更新无人机数量显示"""
        if hasattr(self, '_drone_count_label') and hasattr(self, '_swarm_view_3d'):
            count = self._swarm_view_3d.drone_count
            self._drone_count_label.setText(f"当前: {count} 架")

    def _rebuild_drone_status_cards(self):
        """重建无人机状态卡片列表"""
        if not hasattr(self, '_drone_status_layout'):
            return

        # 清除现有卡片
        for card in self._drone_status_cards:
            card.deleteLater()
        self._drone_status_cards.clear()

        # 重新创建卡片
        from .swarm_view_3d import DroneModel
        count = self._swarm_view_3d.drone_count
        for i in range(count):
            card = CollapsibleDroneCard(i, DroneModel.COLORS[i % len(DroneModel.COLORS)])
            self._drone_status_layout.insertWidget(i, card)
            self._drone_status_cards.append(card)

    def _update_drone_status_display(self):
        """更新无人机状态显示"""
        if not hasattr(self, '_swarm_view_3d'):
            return

        status_list = self._swarm_view_3d.get_drone_status_list()

        for status in status_list:
            drone_id = status['id']
            if drone_id < len(self._drone_status_cards):
                card = self._drone_status_cards[drone_id]
                card.update_data(status)

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

        # 右栏: 垂直布局 (命令历史 + 集群控制 + 无人机状态)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # 右栏上部: 命令历史表格
        self._history_table = HistoryTable()
        self._history_table.setMinimumWidth(200)

        # 右栏中部: 集群控制卡片 (新增)
        self._swarm_control_card = self._create_swarm_control_card()

        # 右栏下部: 无人机状态监控面板
        self._drone_status_panel = self._create_drone_status_panel()

        # 按顺序添加: 命令历史 → 集群控制卡片 → 无人机状态
        right_layout.addWidget(self._history_table, 3)  # 弹性比例 3
        right_layout.addWidget(self._swarm_control_card)  # 固定高度，在中间
        right_layout.addWidget(self._drone_status_panel, 2)  # 弹性比例 2

        # 添加到主分割器
        self._splitter.addWidget(self._control_panel)
        self._splitter.addWidget(self._center_splitter)
        self._splitter.addWidget(right_widget)

        # 设置初始比例 (1:2.5:1)
        self._splitter.setSizes([240, 700, 280])
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

        # 视频区域鼠标点击
        self._video_widget.mouse_clicked.connect(self._on_video_click)

        # 3D 可视化信号
        self._swarm_view_3d.command_executed.connect(self._on_swarm_command_executed)

    def _setup_shortcuts(self):
        """设置快捷键"""
        # 空格键: 开始/停止录音
        shortcut_record = QShortcut(QKeySequence(Qt.Key_Space), self)
        shortcut_record.activated.connect(self._toggle_recording)

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

    def _toggle_recording(self):
        """切换录音状态"""
        if not self._control_panel.is_modal_enabled("voice"):
            self.statusBar().showMessage("语音模态已禁用，请先启用", 2000)
            return
        # 切换录音按钮状态
        btn = self._control_panel.btn_record
        btn.setChecked(not btn.isChecked())

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
<tr><td style="padding: 5px;"><b>空格</b></td><td style="padding: 5px;">开始/停止录音</td></tr>
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

        # 步骤1: 创建语音检测器
        current_step += 1
        progress.set_status("正在加载预训练模型...")
        progress.set_progress(int(current_step / total_steps * 100))
        progress.set_detail(f"步骤 {current_step}/{total_steps} - 语音识别模块")
        self._voice_detector = VoiceDetector()  # 从 config 读取参数
        self._voice_detector.status_changed.connect(
            lambda s: self._control_panel.set_record_status(s)
        )
        self._voice_detector.detection_ready.connect(self._on_detection_result)
        self._voice_detector.error_occurred.connect(self._on_detector_error)
        # 延迟连接语音可视化信号（等待 UI 初始化完成）
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
        self._gesture_worker.result_ready.connect(self._on_gesture_worker_result)
        try:
            if self._gesture_worker.initialize():
                self._gesture_worker.start()
            else:
                errors.append("手势异步工作线程初始化失败")
        except Exception as e:
            errors.append(f"手势工作线程: {str(e)}")

        # 步骤3: 创建并初始化图像检测器
        current_step += 1
        progress.set_progress(int(current_step / total_steps * 100))
        progress.set_detail(f"步骤 {current_step}/{total_steps} - 图像识别模块")
        self._image_detector = ImageDetector()  # 从 config 读取参数
        self._image_detector.detection_ready.connect(self._on_detection_result)
        self._image_detector.error_occurred.connect(self._on_detector_error)
        try:
            if not self._image_detector.initialize():
                errors.append("图像检测器初始化失败")
        except Exception as e:
            errors.append(f"图像检测器: {str(e)}")

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

        # 图像检测 (跳帧优化)
        try:
            if (self._control_panel.is_modal_enabled("image") and
                    self._image_detector and self._image_detector.is_initialized):
                # 只在特定帧运行检测
                if self._detection_frame_count % (self._image_skip_frames + 1) == 0:
                    result = self._image_detector.detect(frame)
                    if result:
                        self._confidences["image"] = result.confidence
                        self._video_widget.update_modal_status(
                            "image", result.command, result.confidence
                        )
                # 绘制检测框 (使用缓存的结果)
                drawn = self._image_detector.draw_detections(processed_frame)
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

        # 更新无人机状态显示 (每5帧更新一次)
        if self._frame_count % 5 == 0:
            self._update_drone_status_display()

    def _draw_stats_overlay(self, frame):
        """
        在帧上绘制 FPS 和手势信息（使用 PIL 渲染中文）

        OSD显示在右上角，减少对主要画面的遮挡
        """
        h, w = frame.shape[:2]

        # 判断是否启用手势模态
        gesture_enabled = self._control_panel.is_modal_enabled("gesture")

        # 计算背景尺寸
        bg_width = 180
        if gesture_enabled:
            bg_height = 70  # FPS + 置信度 + 命令
        else:
            bg_height = 30  # 仅 FPS

        # 右上角位置
        x_start = w - bg_width - 10
        y_start = 10

        # 背景半透明矩形（更透明）
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x_start, y_start),
            (x_start + bg_width, y_start + bg_height),
            (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # 转换为 PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        x_offset = x_start + 8
        y_offset = y_start + 5
        line_height = 20

        # FPS (绿色) - 更紧凑的显示
        fps_text = f"FPS: {self._fps:.1f}"
        draw.text((x_offset, y_offset), fps_text, font=self._font, fill=(0, 255, 0))
        y_offset += line_height

        # 手势模态启用时显示置信度和命令
        if gesture_enabled:
            # 置信度 (橙色)
            gesture_conf = self._confidences.get("gesture", 0.0)
            conf_text = f"置信度: {gesture_conf:.0%}"
            draw.text((x_offset, y_offset), conf_text, font=self._font, fill=(255, 165, 0))
            y_offset += line_height

            # 当前命令 (青色) - 截断过长的命令
            if self._current_gesture_command:
                cmd = self._current_gesture_command
                if len(cmd) > 8:
                    cmd = cmd[:7] + "..."
                cmd_text = f"命令: {cmd}"
                draw.text((x_offset, y_offset), cmd_text, font=self._font, fill=(0, 255, 255))

        # 转换回 OpenCV 格式
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
        """处理录音按钮点击"""
        if not self._control_panel.is_modal_enabled("voice"):
            self._control_panel.set_record_status("语音模态已禁用")
            return

        if self._voice_detector is None:
            # 正常情况下不会到这里，因为在初始化时已创建
            self._voice_detector = VoiceDetector()  # 从 config 读取参数
            self._voice_detector.status_changed.connect(
                lambda s: self._control_panel.set_record_status(s)
            )
            self._voice_detector.detection_ready.connect(self._on_detection_result)
            self._voice_detector.error_occurred.connect(self._on_detector_error)
            self._connect_voice_overlay_signals()

        if not self._voice_detector.is_initialized:
            self._control_panel.set_record_status("正在加载语音模型...")
            QApplication.processEvents()
            if not self._voice_detector.initialize():
                self._control_panel.set_record_status("语音模型加载失败")
                return

        if is_recording:
            # 显示语音叠加层并开始录音
            self._video_widget.show_voice_overlay()
            self._video_widget.voice_overlay.start_recording()
            self._voice_detector.start_recording()
        else:
            # 停止录音
            self._video_widget.voice_overlay.stop_recording()
            result = self._voice_detector.stop_recording()
            if result:
                self._video_widget.update_modal_status(
                    "voice", result.command, result.confidence
                )
                # 更新叠加层结果
                command_text = self._get_voice_command_text(result.command)
                self._video_widget.voice_overlay.set_result(
                    result.command, result.confidence, command_text
                )
            else:
                self._video_widget.voice_overlay.set_idle()

    def _connect_voice_overlay_signals(self):
        """连接语音检测器与可视化叠加层的信号"""
        if self._voice_detector and hasattr(self._video_widget, 'voice_overlay'):
            overlay = self._video_widget.voice_overlay
            logger.debug(f"连接语音可视化信号: overlay={overlay}")

            # 连接音量信号 (使用 QueuedConnection 确保线程安全)
            self._voice_detector.volume_changed.connect(
                overlay.update_volume, Qt.QueuedConnection
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

        # 发布到 ROS (如果可用)
        self._publish_to_ros(result)

    def _publish_to_ros(self, result: DetectionResult):
        """
        将检测结果发布到 3D 可视化和 ROS

        Args:
            result: 检测结果
        """
        # 只处理置信度超过阈值的结果 (语音指令跳过此检查，因为 Whisper 已内置过滤)
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
            # 语音指令 -> 解析并发送
            self._process_voice_command(result.command)

    def _process_voice_command(self, text: str):
        """
        处理语音指令并发送到 3D 可视化和 ROS

        Args:
            text: 语音识别文本
        """
        text_lower = text.lower()
        logger.debug(f"语音指令识别: '{text}' -> '{text_lower}'")

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

        # 释放检测器
        if self._voice_detector:
            self._voice_detector.release()
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
