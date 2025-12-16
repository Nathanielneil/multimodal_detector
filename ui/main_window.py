"""
主窗口 - 整合所有组件，管理应用逻辑
"""

import cv2
import time
import numpy as np
from typing import Optional, Dict
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout,
    QMessageBox, QApplication
)
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QCloseEvent

from .styles import MAIN_STYLESHEET
from .control_panel import ControlPanel
from .video_widget import VideoWidget
from .history_table import HistoryTable
from workers.camera_worker import CameraWorker
from detectors.base_detector import DetectionResult
from detectors.voice_detector import VoiceDetector
from detectors.gesture_detector import GestureDetector
from detectors.image_detector import ImageDetector
from detectors.touch_detector import TouchDetector


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
        self._init_detectors()

    def _setup_window(self):
        """设置窗口属性"""
        self.setWindowTitle("多模态检测器 - Multimodal Detector")
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
        self._frame_timer.setInterval(33)  # ~30 FPS

        # 检测器实例
        self._voice_detector: Optional[VoiceDetector] = None
        self._gesture_detector: Optional[GestureDetector] = None
        self._image_detector: Optional[ImageDetector] = None
        self._touch_detector: Optional[TouchDetector] = None

        # 检测器初始化状态
        self._detectors_initialized = False

        # FPS 计算
        self._fps = 0.0
        self._frame_count = 0
        self._fps_start_time = time.time()

        # 各模态置信度
        self._confidences: Dict[str, float] = {
            "gesture": 0.0,
            "image": 0.0,
            "voice": 0.0,
            "touch": 0.0,
        }

        # 当前手势命令
        self._current_gesture_command = ""

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
                except:
                    continue
            if self._font is None:
                self._font = ImageFont.load_default()
        except:
            self._font = ImageFont.load_default()

    def _setup_ui(self):
        """设置界面布局"""
        # 中央容器
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 三列分割器
        self._splitter = QSplitter(Qt.Horizontal)

        # 左栏: 控制面板
        self._control_panel = ControlPanel()
        self._control_panel.setMinimumWidth(200)
        self._control_panel.setMaximumWidth(280)

        # 中栏: 视频显示
        self._video_widget = VideoWidget()

        # 右栏: 历史表格
        self._history_table = HistoryTable()
        self._history_table.setMinimumWidth(200)
        self._history_table.setMaximumWidth(350)

        # 添加到分割器
        self._splitter.addWidget(self._control_panel)
        self._splitter.addWidget(self._video_widget)
        self._splitter.addWidget(self._history_table)

        # 设置初始比例 (1:2:1)
        self._splitter.setSizes([256, 512, 256])
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 2)
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

        # 视频区域鼠标点击
        self._video_widget.mouse_clicked.connect(self._on_video_click)

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

        # 语音检测器
        self._voice_detector = VoiceDetector(model_name="base")
        self._voice_detector.status_changed.connect(
            lambda s: self._control_panel.set_record_status(s)
        )
        self._voice_detector.detection_ready.connect(self._on_detection_result)
        self._voice_detector.error_occurred.connect(self._on_detector_error)

        # 手势检测器
        self._gesture_detector = GestureDetector()
        self._gesture_detector.detection_ready.connect(self._on_detection_result)
        self._gesture_detector.error_occurred.connect(self._on_detector_error)

        # 图像检测器 (使用 yolov8n 更小更快)
        self._image_detector = ImageDetector(model_name="yolov8n")
        self._image_detector.detection_ready.connect(self._on_detection_result)
        self._image_detector.error_occurred.connect(self._on_detector_error)

        # 初始化检测器（这可能需要一些时间）
        self.statusBar().showMessage("正在初始化检测器...")
        QApplication.processEvents()

        # 初始化手势检测器
        try:
            if self._gesture_detector.initialize():
                self.statusBar().showMessage("手势检测器已就绪")
        except Exception as e:
            print(f"手势检测器初始化失败: {e}")

        # 初始化图像检测器 (可能需要下载模型)
        try:
            QApplication.processEvents()
            if self._image_detector.initialize():
                self.statusBar().showMessage("图像检测器已就绪")
        except Exception as e:
            print(f"图像检测器初始化失败: {e}")
            self.statusBar().showMessage("图像检测器初始化失败，跳过")

        self._detectors_initialized = True
        self.statusBar().showMessage("检测器初始化完成", 3000)

    @Slot()
    def _start_camera(self):
        """启动摄像头"""
        # 初始化重型检测器
        if not self._detectors_initialized:
            self._init_heavy_detectors()

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

        # 手势检测
        try:
            if (self._control_panel.is_modal_enabled("gesture") and
                    self._gesture_detector and self._gesture_detector.is_initialized):
                result = self._gesture_detector.detect(frame)
                if result:
                    self._confidences["gesture"] = result.confidence
                    self._current_gesture_command = result.command
                    self._video_widget.update_modal_status(
                        "gesture", result.command, result.confidence
                    )
                # 绘制手部关键点
                drawn = self._gesture_detector.draw_landmarks(processed_frame)
                if drawn is not None:
                    processed_frame = drawn
        except Exception as e:
            pass  # 忽略检测错误，继续显示视频

        # 图像检测
        try:
            if (self._control_panel.is_modal_enabled("image") and
                    self._image_detector and self._image_detector.is_initialized):
                result = self._image_detector.detect(frame)
                if result:
                    self._confidences["image"] = result.confidence
                    self._video_widget.update_modal_status(
                        "image", result.command, result.confidence
                    )
                # 绘制检测框
                drawn = self._image_detector.draw_detections(processed_frame)
                if drawn is not None:
                    processed_frame = drawn
        except Exception as e:
            pass  # 忽略检测错误，继续显示视频

        # 触屏轨迹绘制
        try:
            if (self._control_panel.is_modal_enabled("touch") and
                    self._touch_detector and self._touch_detector.is_initialized):
                drawn = self._touch_detector.draw_touch_overlay(processed_frame)
                if drawn is not None:
                    processed_frame = drawn
        except Exception as e:
            pass  # 忽略绘制错误，继续显示视频

        # 在左上角绘制 FPS 和置信度
        processed_frame = self._draw_stats_overlay(processed_frame)

        # 显示处理后的帧
        self._video_widget.display_frame(processed_frame)

    def _draw_stats_overlay(self, frame):
        """在帧左上角绘制 FPS 和手势信息（使用 PIL 渲染中文）"""
        # 判断是否启用手势模态
        gesture_enabled = self._control_panel.is_modal_enabled("gesture")

        # 计算背景高度
        if gesture_enabled:
            bg_height = 75  # FPS + 置信度 + 命令
        else:
            bg_height = 35  # 仅 FPS

        # 背景半透明矩形
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (200, 10 + bg_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # 转换为 PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        y_offset = 15
        line_height = 22

        # FPS (绿色)
        fps_text = f"FPS: {self._fps:.1f}"
        draw.text((15, y_offset), fps_text, font=self._font, fill=(0, 255, 0))
        y_offset += line_height

        # 手势模态启用时显示置信度和命令
        if gesture_enabled:
            # 置信度 (橙色)
            gesture_conf = self._confidences.get("gesture", 0.0)
            conf_text = f"置信度: {gesture_conf:.0%}"
            draw.text((15, y_offset), conf_text, font=self._font, fill=(255, 165, 0))
            y_offset += line_height

            # 当前命令 (青色)
            if self._current_gesture_command:
                cmd_text = f"命令: {self._current_gesture_command}"
                draw.text((15, y_offset), cmd_text, font=self._font, fill=(0, 255, 255))

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
            self._voice_detector = VoiceDetector(model_name="base")
            self._voice_detector.status_changed.connect(
                lambda s: self._control_panel.set_record_status(s)
            )
            self._voice_detector.detection_ready.connect(self._on_detection_result)
            self._voice_detector.error_occurred.connect(self._on_detector_error)

        if not self._voice_detector.is_initialized:
            self._control_panel.set_record_status("正在加载语音模型...")
            QApplication.processEvents()
            if not self._voice_detector.initialize():
                self._control_panel.set_record_status("语音模型加载失败")
                return

        if is_recording:
            self._voice_detector.start_recording()
        else:
            result = self._voice_detector.stop_recording()
            if result:
                self._video_widget.update_modal_status(
                    "voice", result.command, result.confidence
                )

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

    def closeEvent(self, event: QCloseEvent):
        """窗口关闭事件 - 释放资源"""
        # 停止定时器
        self._frame_timer.stop()

        # 释放摄像头
        self._camera.release()

        # 释放检测器
        if self._voice_detector:
            self._voice_detector.release()
        if self._gesture_detector:
            self._gesture_detector.release()
        if self._image_detector:
            self._image_detector.release()
        if self._touch_detector:
            self._touch_detector.release()

        event.accept()
