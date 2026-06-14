"""
中栏视频显示组件 - 包含视频画面和四模态状态指示器
v2.12: 新增语音识别可视化叠加层
"""

import cv2
import numpy as np
from typing import Optional, Dict
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QPushButton, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QImage, QPixmap, QMouseEvent

from .styles import MODAL_COLORS
from .voice_overlay import VoiceOverlayWidget


class ModalStatusCard(QFrame):
    """
    模态状态卡片 - 显示单个模态的识别状态

    显示模态名称、最新识别结果和置信度
    """

    def __init__(self, modal_type: str, modal_name: str, parent=None):
        super().__init__(parent)
        self._modal_type = modal_type
        self._modal_name = modal_name
        self._setup_ui()

    def _setup_ui(self):
        """初始化UI"""
        # 设置样式类
        self.setProperty("class", f"modal-card modal-card-{self._modal_type}")
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # 模态名称和状态指示器
        header_layout = QHBoxLayout()

        # 状态指示灯
        self._indicator = QLabel()
        self._indicator.setFixedSize(10, 10)
        color = MODAL_COLORS.get(self._modal_type, "#1e88e5")
        self._indicator.setStyleSheet(f"""
            background-color: {color};
            border-radius: 5px;
            opacity: 0.5;
        """)
        header_layout.addWidget(self._indicator)

        # 模态名称
        self._name_label = QLabel(self._modal_name)
        self._name_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        header_layout.addWidget(self._name_label)
        header_layout.addStretch()

        # 置信度标签
        self._confidence_label = QLabel("--")
        self._confidence_label.setStyleSheet("color: #757575; font-size: 11px;")
        header_layout.addWidget(self._confidence_label)

        layout.addLayout(header_layout)

        # 识别结果
        self._result_label = QLabel("等待识别...")
        self._result_label.setStyleSheet("color: #424242; font-size: 11px;")
        self._result_label.setWordWrap(True)
        self._result_label.setMaximumHeight(30)
        layout.addWidget(self._result_label)

    def update_status(self, result: str, confidence: float, is_active: bool = True):
        """
        更新状态显示

        Args:
            result: 识别结果文本
            confidence: 置信度 (0-1)
            is_active: 是否激活状态
        """
        # 更新指示灯
        color = MODAL_COLORS.get(self._modal_type, "#1e88e5")
        opacity = "1.0" if is_active else "0.3"
        self._indicator.setStyleSheet(f"""
            background-color: {color};
            border-radius: 5px;
            opacity: {opacity};
        """)

        # 更新置信度
        self._confidence_label.setText(f"{confidence:.0%}")

        # 更新结果文本
        display_text = result if len(result) <= 30 else result[:27] + "..."
        self._result_label.setText(display_text)

    def reset(self):
        """重置状态"""
        self._confidence_label.setText("--")
        self._result_label.setText("等待识别...")
        color = MODAL_COLORS.get(self._modal_type, "#1e88e5")
        self._indicator.setStyleSheet(f"""
            background-color: {color};
            border-radius: 5px;
            opacity: 0.3;
        """)


class VideoWidget(QWidget):
    """
    视频显示组件

    包含:
    - 视频画面显示区域 (QLabel)
    - 四模态状态指示器 (侧边垂直列表)
    - 摄像头控制按钮

    Signals:
        start_camera_clicked: 启动摄像头按钮点击
        stop_camera_clicked: 停止摄像头按钮点击
        mouse_clicked: 视频区域鼠标点击 (event: QMouseEvent)
    """

    start_camera_clicked = Signal()
    stop_camera_clicked = Signal()
    mouse_clicked = Signal(object)  # QMouseEvent

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_aspect_ratio = 16 / 9
        self._setup_ui()
        self._setup_voice_overlay()

    def _setup_ui(self):
        """初始化UI - 简化版，仅包含视频和按钮"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(4)

        # 视频显示区域
        self._video_label = QLabel()
        self._video_label.setProperty("class", "video-display")
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setMinimumSize(480, 270)
        self._video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._video_label.setStyleSheet("""
            background-color: #1a1a1a;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
        """)

        # 启用鼠标追踪
        self._video_label.setMouseTracking(True)
        self._video_label.mousePressEvent = self._on_mouse_press
        self._video_label.mouseMoveEvent = self._on_mouse_move
        self._video_label.mouseReleaseEvent = self._on_mouse_release

        main_layout.addWidget(self._video_label, 1)

        # 语音叠加层占位（默认隐藏，录音时展开）
        self._voice_overlay_placeholder = QWidget()
        self._voice_overlay_placeholder.setVisible(False)
        self._voice_overlay_placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        overlay_layout = QVBoxLayout(self._voice_overlay_placeholder)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self._voice_overlay_placeholder)

        # 控制按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)

        self._btn_start = QPushButton("启动摄像头")
        self._btn_start.setMinimumHeight(36)
        self._btn_start.setMinimumWidth(120)

        self._btn_stop = QPushButton("停止摄像头")
        self._btn_stop.setMinimumHeight(36)
        self._btn_stop.setMinimumWidth(120)
        self._btn_stop.setEnabled(False)

        btn_layout.addStretch()
        btn_layout.addWidget(self._btn_start)
        btn_layout.addWidget(self._btn_stop)
        btn_layout.addStretch()

        main_layout.addLayout(btn_layout)

        # 连接信号
        self._btn_start.clicked.connect(self.start_camera_clicked.emit)
        self._btn_stop.clicked.connect(self.stop_camera_clicked.emit)

    def _on_mouse_press(self, event: QMouseEvent):
        """处理视频区域的鼠标按下"""
        self.mouse_clicked.emit(event)

    def _on_mouse_move(self, event: QMouseEvent):
        """处理视频区域的鼠标移动"""
        self.mouse_clicked.emit(event)

    def _on_mouse_release(self, event: QMouseEvent):
        """处理视频区域的鼠标释放"""
        self.mouse_clicked.emit(event)

    def display_frame(self, frame: np.ndarray):
        """
        显示视频帧

        Args:
            frame: BGR 格式的 numpy 数组
        """
        if frame is None:
            return

        # 转换 BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape

        # 创建 QImage
        bytes_per_line = ch * w
        q_image = QImage(
            rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
        )

        # 缩放到适合显示区域，保持 16:9 比例
        label_size = self._video_label.size()
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            label_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self._video_label.setPixmap(scaled_pixmap)

    def display_placeholder(self, message: str = None):
        """显示占位符画面（使用PIL渲染中文）"""
        from PIL import Image, ImageDraw, ImageFont

        # 创建占位符图像
        width = max(480, self._video_label.width())
        height = int(width / self._video_aspect_ratio)

        placeholder = np.full((height, width, 3), 30, dtype=np.uint8)

        # 绘制摄像头图标
        icon_y = height // 2 - 30
        icon_x = width // 2
        # 摄像头主体
        cv2.rectangle(placeholder, (icon_x - 30, icon_y - 20), (icon_x + 30, icon_y + 20), (80, 80, 80), 2)
        # 镜头
        cv2.circle(placeholder, (icon_x, icon_y), 12, (80, 80, 80), 2)
        cv2.circle(placeholder, (icon_x, icon_y), 5, (80, 80, 80), -1)
        # 闪光灯
        cv2.circle(placeholder, (icon_x + 20, icon_y - 12), 4, (80, 80, 80), -1)

        # 使用PIL渲染中文
        placeholder_rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(placeholder_rgb)
        draw = ImageDraw.Draw(pil_image)

        from utils.font_utils import find_cjk_font
        font = find_cjk_font(24)
        small_font = find_cjk_font(14)

        # 绘制主文字
        main_text = "摄像头已停止"
        bbox = draw.textbbox((0, 0), main_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (width - text_width) // 2
        text_y = height // 2 + 20
        draw.text((text_x, text_y), main_text, font=font, fill=(120, 120, 120))

        # 绘制提示文字
        hint_text = "点击 [启动摄像头] 开始"
        bbox = draw.textbbox((0, 0), hint_text, font=small_font)
        hint_width = bbox[2] - bbox[0]
        hint_x = (width - hint_width) // 2
        draw.text((hint_x, text_y + 40), hint_text, font=small_font, fill=(80, 80, 80))

        # 转回OpenCV格式
        placeholder = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        self.display_frame(placeholder)

    def update_modal_status(
        self, modal_type: str, result: str, confidence: float, is_active: bool = True
    ):
        """更新模态状态显示 (已移除识别状态卡片，保留接口兼容)"""
        pass

    def reset_modal_status(self):
        """重置所有模态状态 (已移除识别状态卡片，保留接口兼容)"""
        pass

    def set_camera_running(self, is_running: bool):
        """
        设置摄像头运行状态，更新按钮状态

        Args:
            is_running: 是否正在运行
        """
        self._btn_start.setEnabled(not is_running)
        self._btn_stop.setEnabled(is_running)

        if not is_running:
            self.display_placeholder()

    def get_video_size(self) -> tuple:
        """获取视频显示区域尺寸"""
        return (self._video_label.width(), self._video_label.height())

    def _setup_voice_overlay(self):
        """将语音叠加层放入布局占位容器，不遮挡视频。"""
        self._voice_overlay = VoiceOverlayWidget(self._voice_overlay_placeholder)
        self._voice_overlay_placeholder.layout().addWidget(self._voice_overlay)
        self._voice_overlay.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)

    def _update_voice_overlay_position(self):
        """占位容器方案下无需手动定位。"""
        pass

    @property
    def voice_overlay(self) -> VoiceOverlayWidget:
        """获取语音叠加层组件"""
        return self._voice_overlay

    def show_voice_overlay(self):
        """显示语音叠加层（在视频下方展开，不遮挡画面）。"""
        self._voice_overlay.show()
        self._voice_overlay_placeholder.setVisible(True)

    def hide_voice_overlay(self):
        """隐藏语音叠加层。"""
        self._voice_overlay.hide()
        self._voice_overlay_placeholder.setVisible(False)
