"""
检测器基类 - 定义所有检测器的通用接口
"""

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any, Dict
from PySide6.QtCore import QObject, Signal


@dataclass
class DetectionResult:
    """
    检测结果数据类

    Attributes:
        modal_type: 模态类型 (voice/gesture/image/touch)
        command: 识别到的指令/内容
        confidence: 置信度 (0.0-1.0)
        details: 详细信息字典
        timestamp: 检测时间戳
    """
    modal_type: str
    command: str
    confidence: float
    details: Dict[str, Any]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> dict:
        """转换为字典格式（用于JSON导出）"""
        return {
            "modal_type": self.modal_type,
            "command": self.command,
            "confidence": round(self.confidence, 2),
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseDetector(QObject):
    """
    检测器基类

    所有具体检测器（语音、手势、图像、触屏）都应继承此类。

    Signals:
        detection_ready: 检测结果就绪 (result: DetectionResult)
        status_changed: 状态变更 (status: str)
        error_occurred: 发生错误 (error_msg: str)
    """

    detection_ready = Signal(object)  # DetectionResult
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._enabled = True
        self._threshold = 0.5
        self._is_initialized = False

    @property
    def enabled(self) -> bool:
        """检测器是否启用"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    @property
    def threshold(self) -> float:
        """检测阈值"""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        self._threshold = max(0.0, min(1.0, value))

    @property
    def is_initialized(self) -> bool:
        """是否已初始化"""
        return self._is_initialized

    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化检测器（加载模型等）

        Returns:
            bool: 是否初始化成功
        """
        pass

    @abstractmethod
    def detect(self, data: Any) -> Optional[DetectionResult]:
        """
        执行检测

        Args:
            data: 输入数据（图像帧、音频等）

        Returns:
            DetectionResult: 检测结果，无结果返回None
        """
        pass

    @abstractmethod
    def release(self):
        """释放检测器资源"""
        pass

    def emit_result(self, result: DetectionResult):
        """发送检测结果（仅当置信度超过阈值时）"""
        if result.confidence >= self._threshold:
            self.detection_ready.emit(result)
