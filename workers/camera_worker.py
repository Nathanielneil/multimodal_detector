"""
摄像头工作类 - 管理视频捕获的打开、读取、释放
使用 OpenCV VideoCapture 进行摄像头操作
"""

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, QMutex, QMutexLocker
from typing import Optional, Tuple


class CameraWorker(QObject):
    """
    摄像头工作类

    负责管理 cv2.VideoCapture 的生命周期，
    提供帧读取功能，支持线程安全操作。

    Signals:
        frame_ready: 新帧就绪 (frame: np.ndarray)
        error_occurred: 发生错误 (error_msg: str)
        camera_opened: 摄像头已打开
        camera_closed: 摄像头已关闭
    """

    frame_ready = Signal(np.ndarray)
    error_occurred = Signal(str)
    camera_opened = Signal()
    camera_closed = Signal()

    def __init__(self, camera_id: int = 0, parent=None):
        """
        初始化摄像头工作类

        Args:
            camera_id: 摄像头设备ID，默认0 (通常是笔记本前置摄像头)
            parent: 父对象
        """
        super().__init__(parent)
        self._camera_id = camera_id
        self._capture: Optional[cv2.VideoCapture] = None
        self._mutex = QMutex()
        self._is_running = False

        # 视频属性
        self._frame_width = 640
        self._frame_height = 360  # 16:9 比例
        self._fps = 30

    @property
    def is_opened(self) -> bool:
        """检查摄像头是否已打开"""
        with QMutexLocker(self._mutex):
            return self._capture is not None and self._capture.isOpened()

    @property
    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self._is_running

    @property
    def frame_size(self) -> Tuple[int, int]:
        """获取帧尺寸 (width, height)"""
        return (self._frame_width, self._frame_height)

    def open(self) -> bool:
        """
        打开摄像头

        Returns:
            bool: 是否成功打开
        """
        with QMutexLocker(self._mutex):
            if self._capture is not None:
                self._capture.release()

            # 尝试打开摄像头
            self._capture = cv2.VideoCapture(self._camera_id)

            if not self._capture.isOpened():
                self.error_occurred.emit(
                    f"无法打开摄像头 (ID: {self._camera_id})。\n"
                    "请检查摄像头连接或权限设置。"
                )
                self._capture = None
                return False

            # 设置摄像头参数
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)
            self._capture.set(cv2.CAP_PROP_FPS, self._fps)

            # 获取实际参数
            self._frame_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._frame_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._fps = int(self._capture.get(cv2.CAP_PROP_FPS))

            self._is_running = True
            self.camera_opened.emit()
            return True

    def read_frame(self) -> Optional[np.ndarray]:
        """
        读取一帧图像

        Returns:
            np.ndarray: BGR格式的图像帧，失败返回None
        """
        with QMutexLocker(self._mutex):
            if self._capture is None or not self._capture.isOpened():
                return None

            ret, frame = self._capture.read()

            if not ret or frame is None:
                self.error_occurred.emit("读取视频帧失败")
                return None

            return frame

    def release(self):
        """释放摄像头资源"""
        with QMutexLocker(self._mutex):
            self._is_running = False
            if self._capture is not None:
                self._capture.release()
                self._capture = None
                self.camera_closed.emit()

    def set_camera_id(self, camera_id: int):
        """设置摄像头ID"""
        was_running = self._is_running
        if was_running:
            self.release()
        self._camera_id = camera_id
        if was_running:
            self.open()

    def set_resolution(self, width: int, height: int):
        """
        设置分辨率

        Args:
            width: 宽度
            height: 高度
        """
        self._frame_width = width
        self._frame_height = height

        with QMutexLocker(self._mutex):
            if self._capture is not None and self._capture.isOpened():
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    @staticmethod
    def get_placeholder_frame(width: int = 640, height: int = 360) -> np.ndarray:
        """
        生成占位符帧（摄像头不可用时显示）

        Args:
            width: 帧宽度
            height: 帧高度

        Returns:
            np.ndarray: 带提示文字的灰色占位符帧
        """
        # 创建深灰色背景
        frame = np.full((height, width, 3), 50, dtype=np.uint8)

        # 添加提示文字
        text = "Camera Not Available"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (150, 150, 150)

        # 计算文字位置（居中）
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2

        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

        # 添加摄像头图标
        icon_center = (width // 2, height // 2 - 50)
        cv2.circle(frame, icon_center, 30, (100, 100, 100), 2)
        cv2.circle(frame, icon_center, 10, (100, 100, 100), -1)

        return frame

    @staticmethod
    def list_available_cameras(max_cameras: int = 5) -> list:
        """
        列出可用的摄像头设备

        Args:
            max_cameras: 最大检测数量

        Returns:
            list: 可用摄像头ID列表
        """
        available = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
