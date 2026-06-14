"""
图像检测工作线程 - 异步处理 YOLOv8 推理，避免阻塞主线程
"""

import cv2
import numpy as np
from typing import Optional, List, Any
from threading import Lock
from PySide6.QtCore import QThread, Signal, QObject
from utils.logger import get_logger

logger = get_logger(__name__)


class ImageWorker(QThread):
    """
    YOLOv8 异步推理工作线程

    接收视频帧，在后台线程运行 YOLO 推理，结果通过信号发送回主线程。
    同时维护最近一次检测结果的缓存，供主线程绘制叠加层使用。
    """

    # 信号: (最高置信度目标名称, 置信度, 检测结果列表)
    result_ready = Signal(str, float, object)
    error_occurred = Signal(str)
    status_changed = Signal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._running = False
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = Lock()
        self._new_frame_available = False

        # YOLO 模型
        self._model = None
        self._is_initialized = False
        self._conf_threshold = 0.5
        self._iou_threshold = 0.45

        # 缓存的检测结果供主线程绘制
        self._cached_detections: List[Any] = []
        self._cached_lock = Lock()

    def initialize(self, model_name: str = "yolov8n", conf_threshold: float = 0.5) -> bool:
        """加载 YOLO 模型，在调用 start() 前执行。"""
        try:
            self.status_changed.emit(f"正在加载 {model_name} 模型...")
            from ultralytics import YOLO
            self._model = YOLO(f"{model_name}.pt")
            self._conf_threshold = conf_threshold
            self._is_initialized = True
            self.status_changed.emit(f"{model_name} 模型加载完成")
            return True
        except Exception as e:
            msg = f"YOLO 模型加载失败: {e}"
            logger.error(msg)
            self.error_occurred.emit(msg)
            return False

    def submit_frame(self, frame: np.ndarray) -> None:
        """提交新帧（非阻塞）。若上一帧尚未处理，新帧覆盖旧帧。"""
        with self._frame_lock:
            h, w = frame.shape[:2]
            if w > 640:
                self._frame = cv2.resize(frame, (640, int(h * 640 / w)))
            else:
                self._frame = frame.copy()
            self._new_frame_available = True

    def get_cached_detections(self) -> List[Any]:
        """获取缓存的检测结果（线程安全），供主线程绘制使用。"""
        with self._cached_lock:
            return list(self._cached_detections)

    def run(self) -> None:
        self._running = True
        while self._running:
            frame = None
            with self._frame_lock:
                if self._new_frame_available:
                    frame = self._frame
                    self._new_frame_available = False

            if frame is None:
                self.msleep(10)
                continue

            try:
                self._infer(frame)
            except Exception as e:
                logger.error(f"ImageWorker 推理错误: {e}")
                self.error_occurred.emit(str(e))

            self.msleep(5)

    def _infer(self, frame: np.ndarray) -> None:
        if not self._is_initialized or self._model is None:
            return

        results = self._model(
            frame,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                detections.append({
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                })

        with self._cached_lock:
            self._cached_detections = detections

        if detections:
            best = max(detections, key=lambda d: d["confidence"])
            self.result_ready.emit(
                f"检测到: {best['label']}",
                best["confidence"],
                detections,
            )

    def stop(self) -> None:
        """停止工作线程。"""
        self._running = False
        self.wait(1000)
        self._model = None
