"""
触屏指令检测器 - 检测视频区域的鼠标点击和拖动事件
支持手绘形状识别（三角形、正方形、圆形、五角星）
"""

import numpy as np
import math
from typing import Optional, Any, Tuple, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from PySide6.QtCore import QPoint, Qt, QEvent, QObject
from PySide6.QtGui import QMouseEvent

from .base_detector import BaseDetector, DetectionResult
from utils.logger import get_logger

logger = get_logger(__name__)

# 延迟导入
cv2 = None
Image = None
ImageDraw = None
ImageFont = None


def _lazy_import():
    global cv2, Image, ImageDraw, ImageFont
    if cv2 is None:
        import cv2 as _cv2
        from PIL import Image as _Image, ImageDraw as _ImageDraw, ImageFont as _ImageFont
        cv2 = _cv2
        Image = _Image
        ImageDraw = _ImageDraw
        ImageFont = _ImageFont


class ShapeType(Enum):
    """识别的形状类型"""
    NONE = "无"
    TRIANGLE = "三角形队形"
    SQUARE = "正方形队形"
    CIRCLE = "圆形队形"
    STAR = "五角星队形"


class TouchEventType(Enum):
    """触屏事件类型"""
    CLICK = "Click"
    DOUBLE_CLICK = "Double Click"
    RIGHT_CLICK = "Right Click"
    DRAG_START = "Drag Start"
    DRAGGING = "Dragging"
    DRAG_END = "Drag End"


@dataclass
class TouchEvent:
    """触屏事件数据"""
    event_type: TouchEventType
    position: Tuple[int, int]
    button: str
    timestamp: datetime
    relative_position: Tuple[float, float]


class TouchDetector(BaseDetector):
    """
    触屏/鼠标事件检测器

    监听视频显示区域的鼠标事件，并将其转换为触屏指令。
    支持点击、双击、拖动轨迹渲染。
    """

    # 轨迹颜色 (BGR)
    TRAJECTORY_COLOR = (255, 165, 0)  # 橙色
    CLICK_COLOR = (0, 255, 255)       # 黄色 - 点击
    DRAG_COLOR = (255, 0, 255)        # 品红 - 拖动
    TRAJECTORY_MAX_LEN = 100          # 轨迹最大长度

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._double_click_interval = 300  # ms
        self._last_click_time: Optional[datetime] = None
        self._last_click_pos: Optional[Tuple[int, int]] = None
        self._video_size = (640, 360)

        # 拖动状态
        self._is_dragging = False
        self._drag_trajectory: List[Tuple[float, float]] = []  # 存储归一化坐标 (0-1)

        # 点击标记 (用于显示点击位置) - 使用归一化坐标
        self._click_markers: List[dict] = []  # [{pos: (rel_x, rel_y), time, type}, ...]
        self._marker_duration = 3.0  # 点击标记显示时长（秒）

        # 形状识别相关
        self._recognized_shape: Optional[ShapeType] = None
        self._shape_display_time: Optional[datetime] = None
        self._shape_duration = 2.0  # 标准形状显示时长（秒）
        self._min_trajectory_points = 20  # 最小轨迹点数（用于形状识别）

        # 中文字体
        self._font = None

    def initialize(self) -> bool:
        """初始化触屏检测器"""
        self._is_initialized = True
        self.status_changed.emit("Touch detector ready")
        return True

    def detect(self, data: Any) -> Optional[DetectionResult]:
        """处理鼠标事件"""
        if not self._enabled or not self._is_initialized:
            return None

        if not isinstance(data, QMouseEvent):
            return None

        event = data
        pos = event.position().toPoint()
        current_time = datetime.now()
        event_type_val = event.type()

        # 获取鼠标按钮
        button = "left"
        if event.button() == Qt.RightButton:
            button = "right"
        elif event.button() == Qt.MiddleButton:
            button = "middle"

        # 计算相对位置
        rel_x = max(0.0, min(1.0, pos.x() / max(1, self._video_size[0])))
        rel_y = max(0.0, min(1.0, pos.y() / max(1, self._video_size[1])))

        result = None

        # 鼠标按下
        if event_type_val == QEvent.MouseButtonPress:
            if event.button() == Qt.RightButton:
                event_type = TouchEventType.RIGHT_CLICK
                self._add_click_marker(rel_x, rel_y, "right")
            elif self._check_double_click(pos, current_time):
                event_type = TouchEventType.DOUBLE_CLICK
                self._add_click_marker(rel_x, rel_y, "double")
            else:
                # 开始拖动
                self._is_dragging = True
                self._drag_trajectory = [(rel_x, rel_y)]  # 存储归一化坐标
                event_type = TouchEventType.DRAG_START
                self._add_click_marker(rel_x, rel_y, "drag_start")

            self._last_click_time = current_time
            self._last_click_pos = (pos.x(), pos.y())

            result = self._create_result(event_type, pos, rel_x, rel_y, button)

        # 鼠标移动（拖动中）
        elif event_type_val == QEvent.MouseMove:
            if self._is_dragging:
                self._drag_trajectory.append((rel_x, rel_y))  # 存储归一化坐标
                if len(self._drag_trajectory) > self.TRAJECTORY_MAX_LEN:
                    self._drag_trajectory.pop(0)
                # 拖动中不发送结果，只记录轨迹

        # 鼠标释放
        elif event_type_val == QEvent.MouseButtonRelease:
            if self._is_dragging:
                self._is_dragging = False
                # 判断是点击还是拖动
                if len(self._drag_trajectory) > 5:
                    event_type = TouchEventType.DRAG_END
                    self._add_click_marker(rel_x, rel_y, "drag_end")

                    # 尝试识别形状（轨迹点足够多时）
                    if len(self._drag_trajectory) >= self._min_trajectory_points:
                        shape = self._recognize_shape(self._drag_trajectory)
                        if shape != ShapeType.NONE:
                            self._recognized_shape = shape
                            self._shape_display_time = datetime.now()
                            self._drag_trajectory = []  # 清除手绘轨迹
                            self._click_markers = []  # 清除点击标记
                            result = self._create_result(event_type, pos, rel_x, rel_y, button,
                                                         shape=shape.value)
                        else:
                            result = self._create_result(event_type, pos, rel_x, rel_y, button,
                                                         trajectory_len=len(self._drag_trajectory))
                    else:
                        result = self._create_result(event_type, pos, rel_x, rel_y, button,
                                                     trajectory_len=len(self._drag_trajectory))
                else:
                    event_type = TouchEventType.CLICK
                    self._add_click_marker(rel_x, rel_y, "click")
                    result = self._create_result(event_type, pos, rel_x, rel_y, button)
                    self._drag_trajectory = []  # 清空短轨迹

        if result:
            self.emit_result(result)

        return result

    def _check_double_click(self, pos: QPoint, current_time: datetime) -> bool:
        """检查是否为双击"""
        if self._last_click_time is None or self._last_click_pos is None:
            return False

        time_diff = (current_time - self._last_click_time).total_seconds() * 1000
        dist = abs(pos.x() - self._last_click_pos[0]) + abs(pos.y() - self._last_click_pos[1])

        if time_diff < self._double_click_interval and dist < 10:
            self._last_click_time = None
            return True
        return False

    def _create_result(self, event_type: TouchEventType, pos: QPoint,
                       rel_x: float, rel_y: float, button: str, **kwargs) -> DetectionResult:
        """创建检测结果"""
        details = {
            "event_type": event_type.name,
            "position": (pos.x(), pos.y()),
            "relative_position": (round(rel_x, 3), round(rel_y, 3)),
            "button": button,
        }
        details.update(kwargs)

        return DetectionResult(
            modal_type="touch",
            command=f"{event_type.value} ({pos.x()}, {pos.y()})",
            confidence=1.0,
            details=details
        )

    def _add_click_marker(self, rel_x: float, rel_y: float, marker_type: str):
        """添加点击标记 (使用归一化坐标)"""
        self._click_markers.append({
            "pos": (rel_x, rel_y),  # 归一化坐标 (0-1)
            "time": datetime.now(),
            "type": marker_type
        })
        # 限制标记数量
        if len(self._click_markers) > 20:
            self._click_markers.pop(0)

    def _recognize_shape(self, trajectory: List[Tuple[float, float]]) -> ShapeType:
        """
        识别手绘轨迹的形状

        使用 OpenCV 轮廓分析和几何特征识别用户手绘的形状类型。
        分析过程包括:
        1. 将归一化坐标转换为图像坐标
        2. 检查轨迹是否闭合 (起点终点距离 < 0.15)
        3. 使用 Douglas-Peucker 算法简化轨迹获取顶点数
        4. 计算圆度 (circularity) 和凸包凹陷
        5. 根据几何特征匹配形状类型

        Args:
            trajectory: 归一化坐标 (0-1) 的轨迹点列表

        Returns:
            ShapeType: 识别到的形状类型，无法识别返回 ShapeType.NONE
        """
        _lazy_import()

        if len(trajectory) < self._min_trajectory_points:
            return ShapeType.NONE

        # 将归一化坐标转换为图像坐标（使用固定尺寸进行分析）
        scale = 500
        points = np.array([[int(x * scale), int(y * scale)] for x, y in trajectory], dtype=np.int32)

        # 创建轮廓图像
        img = np.zeros((scale, scale), dtype=np.uint8)
        cv2.polylines(img, [points], isClosed=False, color=255, thickness=3)

        # 检查轨迹是否闭合（起点和终点距离）
        start = np.array(trajectory[0])
        end = np.array(trajectory[-1])
        closure_dist = np.linalg.norm(start - end)
        is_closed = closure_dist < 0.15  # 闭合阈值

        # 简化轨迹获取顶点
        epsilon = 0.02 * cv2.arcLength(points, closed=is_closed)
        approx = cv2.approxPolyDP(points, epsilon, closed=is_closed)
        num_vertices = len(approx)

        # 计算凸包
        hull = cv2.convexHull(points)
        hull_area = cv2.contourArea(hull)

        # 计算圆度 (circularity)
        perimeter = cv2.arcLength(points, closed=is_closed)
        if perimeter > 0 and hull_area > 0:
            circularity = 4 * math.pi * hull_area / (perimeter * perimeter)
        else:
            circularity = 0

        # 检测五角星：通过检测凹凸性
        defects = None
        if len(hull) >= 3:
            hull_indices = cv2.convexHull(points, returnPoints=False)
            if len(hull_indices) >= 3:
                try:
                    defects = cv2.convexityDefects(points, hull_indices)
                except cv2.error:
                    # OpenCV may fail if contour is degenerate
                    defects = None

        # 统计显著的凹陷数量（五角星特征）
        significant_defects = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                _, _, _, d = defects[i, 0]
                if d > scale * 0.05:  # 显著凹陷
                    significant_defects += 1

        # 形状识别逻辑
        # 1. 五角星：有5个显著凹陷点，闭合
        if significant_defects >= 4 and is_closed:
            return ShapeType.STAR

        # 2. 圆形：圆度高，顶点多，闭合
        if circularity > 0.7 and num_vertices > 6 and is_closed:
            return ShapeType.CIRCLE

        # 3. 三角形：3个顶点，闭合
        if num_vertices == 3 and is_closed:
            return ShapeType.TRIANGLE

        # 4. 正方形/矩形：4个顶点，闭合
        if num_vertices == 4 and is_closed:
            return ShapeType.SQUARE

        # 5. 放宽条件再次尝试
        if is_closed:
            if num_vertices <= 4 and circularity < 0.5:
                return ShapeType.TRIANGLE
            if 4 <= num_vertices <= 6 and circularity < 0.7:
                return ShapeType.SQUARE
            if circularity > 0.5:
                return ShapeType.CIRCLE

        return ShapeType.NONE

    def _load_font(self) -> Any:
        """加载中文字体，跨平台查找可用字体文件。"""
        if self._font is not None:
            return self._font
        from utils.font_utils import find_cjk_font
        self._font = find_cjk_font(size=24)
        return self._font

    def draw_touch_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        在视频帧上绘制触屏轨迹和点击标记

        Args:
            frame: BGR 格式的图像帧

        Returns:
            绘制了触屏轨迹的图像帧
        """
        _lazy_import()

        h, w = frame.shape[:2]
        current_time = datetime.now()

        def to_frame_coords(rel_x: float, rel_y: float) -> Tuple[int, int]:
            """将归一化坐标转换为帧坐标"""
            return (int(rel_x * w), int(rel_y * h))

        # 绘制拖动轨迹
        if len(self._drag_trajectory) > 1:
            for i in range(1, len(self._drag_trajectory)):
                alpha = i / len(self._drag_trajectory)
                thickness = max(2, int(4 * alpha))

                # 根据是否正在拖动选择颜色
                if self._is_dragging:
                    color = self.DRAG_COLOR
                else:
                    color = tuple(int(c * alpha) for c in self.DRAG_COLOR)

                # 转换归一化坐标到帧坐标
                pt1 = to_frame_coords(*self._drag_trajectory[i - 1])
                pt2 = to_frame_coords(*self._drag_trajectory[i])
                cv2.line(frame, pt1, pt2, color, thickness)

            # 如果正在拖动，在当前位置绘制光标
            if self._is_dragging and self._drag_trajectory:
                last_pos = to_frame_coords(*self._drag_trajectory[-1])
                cv2.circle(frame, last_pos, 8, self.DRAG_COLOR, 2)
                cv2.circle(frame, last_pos, 3, self.DRAG_COLOR, -1)

        # 绘制点击标记（带淡出效果）
        markers_to_remove = []
        for i, marker in enumerate(self._click_markers):
            elapsed = (current_time - marker["time"]).total_seconds()
            if elapsed > self._marker_duration:
                markers_to_remove.append(i)
                continue

            # 计算透明度（淡出）
            alpha = 1.0 - (elapsed / self._marker_duration)
            rel_pos = marker["pos"]
            pos = to_frame_coords(*rel_pos)  # 转换到帧坐标
            marker_type = marker["type"]

            # 根据类型选择样式
            if marker_type == "click":
                color = self.CLICK_COLOR
                radius = int(15 * alpha) + 5
                cv2.circle(frame, pos, radius, color, 2)
            elif marker_type == "double":
                color = (0, 255, 0)  # 绿色
                radius = int(20 * alpha) + 5
                cv2.circle(frame, pos, radius, color, 2)
                cv2.circle(frame, pos, radius - 5, color, 2)
            elif marker_type == "right":
                color = (0, 0, 255)  # 红色
                size = int(10 * alpha) + 5
                cv2.rectangle(frame, (pos[0] - size, pos[1] - size),
                              (pos[0] + size, pos[1] + size), color, 2)
            elif marker_type in ["drag_start", "drag_end"]:
                color = self.DRAG_COLOR
                cv2.drawMarker(frame, pos, color, cv2.MARKER_CROSS, 15, 2)

        # 移除过期标记
        for i in reversed(markers_to_remove):
            self._click_markers.pop(i)

        # 如果轨迹不再活跃，逐渐清除（每10帧移除一个点，减慢消失速度）
        if not self._is_dragging and self._drag_trajectory:
            if not hasattr(self, '_fade_counter'):
                self._fade_counter = 0
            self._fade_counter += 1
            if self._fade_counter >= 10 and len(self._drag_trajectory) > 0:
                self._drag_trajectory.pop(0)
                self._fade_counter = 0

        # 绘制识别到的标准形状
        if self._recognized_shape and self._shape_display_time:
            elapsed = (current_time - self._shape_display_time).total_seconds()
            if elapsed < self._shape_duration:
                frame = self._draw_standard_shape(frame, self._recognized_shape)
            else:
                # 显示时间结束，清除形状
                self._recognized_shape = None
                self._shape_display_time = None

        return frame

    def _draw_standard_shape(self, frame: np.ndarray, shape: ShapeType) -> np.ndarray:
        """在视频中央绘制标准形状和文字标签"""
        _lazy_import()

        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        size = min(w, h) // 4  # 形状大小

        # 形状颜色 (BGR)
        shape_color = (0, 255, 255)  # 黄色
        thickness = 3

        if shape == ShapeType.TRIANGLE:
            # 等边三角形
            pts = np.array([
                [center_x, center_y - size],
                [center_x - int(size * 0.866), center_y + size // 2],
                [center_x + int(size * 0.866), center_y + size // 2]
            ], np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=shape_color, thickness=thickness)

        elif shape == ShapeType.SQUARE:
            # 正方形
            half = size // 2 + 20
            pts = np.array([
                [center_x - half, center_y - half],
                [center_x + half, center_y - half],
                [center_x + half, center_y + half],
                [center_x - half, center_y + half]
            ], np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=shape_color, thickness=thickness)

        elif shape == ShapeType.CIRCLE:
            # 圆形
            cv2.circle(frame, (center_x, center_y), size, shape_color, thickness)

        elif shape == ShapeType.STAR:
            # 五角星
            pts = []
            for i in range(5):
                # 外顶点
                angle_out = math.radians(-90 + i * 72)
                pts.append([
                    int(center_x + size * math.cos(angle_out)),
                    int(center_y + size * math.sin(angle_out))
                ])
                # 内顶点
                angle_in = math.radians(-90 + i * 72 + 36)
                pts.append([
                    int(center_x + size * 0.4 * math.cos(angle_in)),
                    int(center_y + size * 0.4 * math.sin(angle_in))
                ])
            pts = np.array(pts, np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=shape_color, thickness=thickness)

        # 使用 PIL 绘制中文文字
        font = self._load_font()
        text = shape.value  # 如 "三角形队形"

        # 转换为 PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        # 计算文字位置（形状正下方）
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
        except (AttributeError, TypeError):
            # Fallback for older PIL versions without textbbox
            text_width = len(text) * 24

        text_x = center_x - text_width // 2
        text_y = center_y + size + 20

        # 绘制文字背景
        padding = 5
        draw.rectangle(
            [text_x - padding, text_y - padding,
             text_x + text_width + padding, text_y + 30 + padding],
            fill=(0, 0, 0, 180)
        )

        # 绘制文字
        draw.text((text_x, text_y), text, font=font, fill=(0, 255, 255))

        # 转换回 OpenCV 格式
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return frame

    def handle_mouse_event(self, event: QMouseEvent) -> Optional[DetectionResult]:
        """处理鼠标事件的便捷方法"""
        return self.detect(event)

    def set_video_size(self, width: int, height: int):
        """设置视频区域尺寸"""
        self._video_size = (width, height)

    def clear_trajectory(self):
        """清空轨迹"""
        self._drag_trajectory = []
        self._click_markers = []

    def release(self):
        """释放资源"""
        self._is_initialized = False
        self._drag_trajectory = []
        self._click_markers = []
        self.status_changed.emit("Touch detector released")
