"""
图像识别检测器 - 使用 YOLOv8 进行物体检测
"""

import numpy as np
from typing import Optional, Any, List, Dict, Tuple
from dataclasses import dataclass

from .base_detector import BaseDetector, DetectionResult
from PySide6.QtCore import QObject
from config import config
from utils.logger import get_logger

logger = get_logger(__name__)

# 延迟导入
ultralytics = None
cv2 = None


def _lazy_import():
    """延迟导入重型依赖"""
    global ultralytics, cv2
    if ultralytics is None:
        from ultralytics import YOLO
        import cv2 as _cv2
        ultralytics = YOLO
        cv2 = _cv2


@dataclass
class BoundingBox:
    """边界框数据类"""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class ImageDetector(BaseDetector):
    """
    图像识别检测器

    使用 YOLOv8 模型进行实时物体检测。

    Attributes:
        model_name: 模型名称 (yolov8n/yolov8s/yolov8m/yolov8l/yolov8x)
        conf_threshold: 置信度阈值
        iou_threshold: IoU 阈值（NMS用）
    """

    def __init__(self, model_name: Optional[str] = None, parent: Optional[QObject] = None):
        super().__init__(parent)
        # 从配置读取参数
        self._model_name = model_name or config.get("image.model_name", "yolov8n")
        self._model = None
        self._conf_threshold = config.get("image.confidence_threshold", 0.5)
        self._iou_threshold = 0.45
        self._target_classes: Optional[List[int]] = None  # 只检测特定类别
        self._last_detections: List[BoundingBox] = []

        logger.debug(f"ImageDetector 初始化: model={self._model_name}, conf_threshold={self._conf_threshold}")

    def initialize(self) -> bool:
        """
        初始化 YOLOv8 模型

        Returns:
            bool: 是否初始化成功
        """
        try:
            self.status_changed.emit(f"正在加载 {self._model_name} 模型...")
            _lazy_import()

            # 加载预训练模型
            self._model = ultralytics(f"{self._model_name}.pt")

            self._is_initialized = True
            self.status_changed.emit(f"{self._model_name} 模型加载完成")
            return True

        except Exception as e:
            self.error_occurred.emit(f"YOLO 模型加载失败: {str(e)}")
            return False

    def detect(self, data: Any) -> Optional[DetectionResult]:
        """
        检测图像中的物体

        Args:
            data: BGR 格式的图像帧 (np.ndarray)

        Returns:
            DetectionResult: 检测结果（最高置信度的物体）
        """
        if not self._enabled or not self._is_initialized:
            return None

        if not isinstance(data, np.ndarray):
            return None

        _lazy_import()

        try:
            # 运行推理
            results = self._model(
                data,
                conf=self._conf_threshold,
                iou=self._iou_threshold,
                classes=self._target_classes,
                verbose=False,
            )

            # 解析检测结果
            self._last_detections = []
            detections_info = []

            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = result.names[cls_id]

                    bbox = BoundingBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        label=label, confidence=conf
                    )
                    self._last_detections.append(bbox)
                    detections_info.append({
                        "label": label,
                        "confidence": round(conf, 2),
                        "bbox": [x1, y1, x2, y2],
                    })

            if not self._last_detections:
                return None

            # 取置信度最高的检测结果
            best = max(self._last_detections, key=lambda x: x.confidence)

            result = DetectionResult(
                modal_type="image",
                command=f"检测到: {best.label}",
                confidence=best.confidence,
                details={
                    "primary_object": best.label,
                    "bbox": [best.x1, best.y1, best.x2, best.y2],
                    "all_detections": detections_info,
                    "total_count": len(self._last_detections),
                }
            )

            self.emit_result(result)
            return result

        except Exception as e:
            self.error_occurred.emit(f"物体检测错误: {str(e)}")
            return None

    # 科研级配色方案 (BGR格式) - 基于 Matplotlib tab10 调色板
    CLASS_COLORS = [
        (214, 129, 31),   # #1F77B4 - 蓝色
        (39, 127, 255),   # #FF7F0E - 橙色
        (74, 175, 44),    # #2CA02C - 绿色
        (54, 54, 214),    # #D62728 - 红色
        (189, 103, 148),  # #9467BD - 紫色
        (96, 86, 140),    # #8C564B - 棕色
        (194, 119, 227),  # #E377C2 - 粉色
        (127, 127, 127),  # #7F7F7F - 灰色
        (34, 189, 188),   # #BCBD22 - 黄绿
        (207, 190, 23),   # #17BECF - 青色
        (66, 133, 244),   # Google Blue
        (52, 168, 83),    # Google Green
    ]

    def _get_color_for_class(self, class_name: str) -> tuple:
        """根据类别名获取颜色"""
        # 使用类别名的哈希值来选择颜色
        color_idx = hash(class_name) % len(self.CLASS_COLORS)
        return self.CLASS_COLORS[color_idx]

    def draw_detections(self, frame: np.ndarray,
                        detections: Optional[List[Dict]] = None) -> np.ndarray:
        """
        在图像上绘制检测结果。

        Args:
            frame: BGR 格式的图像帧
            detections: 外部传入的检测结果列表（来自 ImageWorker 缓存）。
                        为 None 时使用检测器自身的 _last_detections。

        Returns:
            绘制了边界框的图像帧
        """
        _lazy_import()

        # 支持两种来源：外部缓存（dict list）或本地 BoundingBox list
        if detections is not None:
            draw_list = detections
            use_dict = True
        else:
            draw_list = self._last_detections
            use_dict = False

        for det in draw_list:
            if use_dict:
                x1, y1, x2, y2 = det["bbox"]
                label = det["label"]
                conf = det["confidence"]
            else:
                x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
                label = det.label
                conf = det.confidence

            color = self._get_color_for_class(label)

            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 绘制角点装饰
            corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)

            # 绘制标签
            label_text = f"{label}: {conf:.2f}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

            # 标签背景 (半透明效果)
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0] + 10, y1),
                color,
                -1
            )
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # 标签文字
            cv2.putText(
                frame,
                label_text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

        return frame

    def get_detections(self) -> List[BoundingBox]:
        """获取最近一次的检测结果列表"""
        return self._last_detections.copy()

    def set_target_classes(self, classes: Optional[List[int]]):
        """
        设置目标检测类别

        Args:
            classes: 类别ID列表，None 表示检测所有类别
        """
        self._target_classes = classes

    def set_conf_threshold(self, threshold: float):
        """设置置信度阈值"""
        self._conf_threshold = max(0.0, min(1.0, threshold))

    def set_iou_threshold(self, threshold: float):
        """设置 IoU 阈值"""
        self._iou_threshold = max(0.0, min(1.0, threshold))

    def get_class_names(self) -> Dict[int, str]:
        """获取类别名称映射"""
        if self._model is None:
            return {}
        return self._model.names

    def release(self):
        """释放资源"""
        self._model = None
        self._last_detections = []
        self._is_initialized = False
        self.status_changed.emit("图像检测器已释放")
