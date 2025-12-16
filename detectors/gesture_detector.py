"""
手势识别检测器 - 使用 MediaPipe Hands 进行手部关键点检测
"""

import numpy as np
from typing import Optional, Any, List, Tuple
from enum import Enum

from .base_detector import BaseDetector, DetectionResult

# 延迟导入
mp_hands = None
mp_drawing = None
cv2 = None


def _lazy_import():
    """延迟导入重型依赖"""
    global mp_hands, mp_drawing, cv2
    if mp_hands is None:
        import mediapipe as mp
        import cv2 as _cv2
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        cv2 = _cv2


class GestureType(Enum):
    """手势类型枚举 - 映射到无人机集群指令"""
    NONE = "无手势"
    FIST = "集群降落"           # 握拳
    OPEN_PALM = "集群起飞"      # 张开手掌
    POINTING = "集群悬停"       # 只出一个食指
    THUMBS_UP = "集群高度上升"  # 竖起大拇指
    THUMBS_DOWN = "集群高度下降"  # 向下大拇指
    OK = "指令确定"             # OK手势
    ROCK = "编队飞行"           # 三根手指（摇滚手势）


class GestureDetector(BaseDetector):
    """
    手势识别检测器

    使用 MediaPipe Hands 检测手部关键点，
    并基于关键点位置识别预定义的手势类型。

    Attributes:
        max_hands: 最大检测手数 (默认 2)
        min_detection_confidence: 最小检测置信度
        min_tracking_confidence: 最小跟踪置信度
    """

    # 手部轨迹颜色 (BGR)
    HAND_COLORS = [
        (0, 165, 255),   # 橙色 - 左手
        (255, 0, 128),   # 紫色 - 右手
    ]
    # 指尖颜色 (BGR) - 对应 thumb, index, middle, ring, pinky
    FINGERTIP_COLORS = [
        (244, 133, 66),   # 蓝色 - 大拇指
        (83, 168, 52),    # 绿色 - 食指
        (5, 188, 251),    # 黄色 - 中指
        (53, 67, 234),    # 红色 - 无名指
        (166, 160, 154),  # 灰色 - 小指
    ]
    TRAJECTORY_MAX_LEN = 5  # 轨迹最大长度
    # 指尖关键点索引
    FINGERTIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

    def __init__(self, max_hands: int = 2, parent=None):
        super().__init__(parent)
        self._max_hands = max_hands
        self._min_detection_confidence = 0.7
        self._min_tracking_confidence = 0.5
        self._hands = None
        self._last_gesture = GestureType.NONE
        self._gesture_stable_count = 0
        self._stable_threshold = 5  # 连续检测到相同手势的次数阈值
        # 指尖轨迹记录 {handedness: {fingertip_id: [(x, y), ...]}}
        # 使用 "Left"/"Right" 作为键，而不是检测索引
        self._trajectories = {
            "Left": {tip: [] for tip in self.FINGERTIP_IDS},
            "Right": {tip: [] for tip in self.FINGERTIP_IDS}
        }
        self._prev_hands_detected = set()  # 跟踪上一帧检测到的手

    def initialize(self) -> bool:
        """
        初始化 MediaPipe Hands

        Returns:
            bool: 是否初始化成功
        """
        try:
            self.status_changed.emit("正在初始化手势检测器...")
            _lazy_import()

            self._hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self._max_hands,
                min_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
            )

            self._is_initialized = True
            self.status_changed.emit("手势检测器初始化完成")
            return True

        except Exception as e:
            self.error_occurred.emit(f"手势检测器初始化失败: {str(e)}")
            return False

    def detect(self, data: Any) -> Optional[DetectionResult]:
        """
        检测图像中的手势

        Args:
            data: BGR 格式的图像帧 (np.ndarray)

        Returns:
            DetectionResult: 检测结果，无手势返回 None
        """
        if not self._enabled or not self._is_initialized:
            return None

        if not isinstance(data, np.ndarray):
            return None

        _lazy_import()

        # 转换 BGR -> RGB
        rgb_frame = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            self._reset_stable_count()
            return None

        # 分析每只检测到的手
        detected_gestures = []
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 获取手的类型（左/右）
            handedness = "Unknown"
            if results.multi_handedness:
                handedness = results.multi_handedness[hand_idx].classification[0].label

            # 提取关键点
            landmarks = self._extract_landmarks(hand_landmarks)

            # 识别手势
            gesture, confidence = self._recognize_gesture(landmarks)

            # 只有置信度 >= 0.8 才记录
            if gesture != GestureType.NONE and confidence >= 0.8:
                detected_gestures.append({
                    "gesture": gesture,
                    "confidence": confidence,
                    "handedness": handedness,
                    "landmarks": landmarks,
                })

        if not detected_gestures:
            self._reset_stable_count()
            return None

        # 取置信度最高的手势
        best = max(detected_gestures, key=lambda x: x["confidence"])

        # 稳定性检查
        if best["gesture"] == self._last_gesture:
            self._gesture_stable_count += 1
        else:
            self._gesture_stable_count = 1
            self._last_gesture = best["gesture"]

        # 只有稳定检测到才发送结果
        if self._gesture_stable_count < self._stable_threshold:
            return None

        result = DetectionResult(
            modal_type="gesture",
            command=best["gesture"].value,
            confidence=best["confidence"],
            details={
                "handedness": best["handedness"],
                "hands_count": len(results.multi_hand_landmarks),
                "gesture_type": best["gesture"].name,
            }
        )

        self.emit_result(result)
        return result

    def _extract_landmarks(self, hand_landmarks) -> List[Tuple[float, float, float]]:
        """提取手部关键点坐标"""
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append((lm.x, lm.y, lm.z))
        return landmarks

    def _recognize_gesture(
        self, landmarks: List[Tuple[float, float, float]]
    ) -> Tuple[GestureType, float]:
        """
        基于关键点识别手势

        MediaPipe 手部关键点索引:
        0: 手腕
        1-4: 大拇指 (CMC, MCP, IP, TIP)
        5-8: 食指 (MCP, PIP, DIP, TIP)
        9-12: 中指 (MCP, PIP, DIP, TIP)
        13-16: 无名指 (MCP, PIP, DIP, TIP)
        17-20: 小指 (MCP, PIP, DIP, TIP)

        Returns:
            (GestureType, confidence)
        """
        # 提取关键点
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]

        index_pip = landmarks[6]
        middle_pip = landmarks[10]
        ring_pip = landmarks[14]
        pinky_pip = landmarks[18]

        # 判断手指是否伸直
        def is_finger_extended(tip, pip, mcp) -> bool:
            return tip[1] < pip[1] < mcp[1]  # y 坐标：tip 在最上方

        def is_thumb_extended() -> bool:
            # 大拇指判断逻辑不同
            return abs(thumb_tip[0] - wrist[0]) > abs(thumb_mcp[0] - wrist[0])

        thumb_up = is_thumb_extended()
        index_up = is_finger_extended(index_tip, index_pip, index_mcp)
        middle_up = is_finger_extended(middle_tip, middle_pip, middle_mcp)
        ring_up = is_finger_extended(ring_tip, ring_pip, ring_mcp)
        pinky_up = is_finger_extended(pinky_tip, pinky_pip, pinky_mcp)

        fingers = [thumb_up, index_up, middle_up, ring_up, pinky_up]
        extended_count = sum(fingers)

        # 手势识别逻辑 (置信度 >= 0.8)
        # 握拳: 所有手指都弯曲 -> 集群降落
        if extended_count == 0:
            return GestureType.FIST, 0.9

        # 张开手掌: 所有手指都伸直 -> 集群起飞
        if extended_count == 5:
            return GestureType.OPEN_PALM, 0.9

        # 竖起大拇指: 只有大拇指伸直 -> 集群高度上升/下降
        if thumb_up and not any([index_up, middle_up, ring_up, pinky_up]):
            # 判断是向上还是向下
            if thumb_tip[1] < thumb_mcp[1]:
                return GestureType.THUMBS_UP, 0.85
            else:
                return GestureType.THUMBS_DOWN, 0.85

        # 指向: 只有食指伸直 -> 集群悬停
        if index_up and not any([middle_up, ring_up, pinky_up]):
            return GestureType.POINTING, 0.85

        # OK手势: 大拇指和食指接触，其他三指伸直 -> 指令确定
        thumb_index_dist = np.sqrt(
            (thumb_tip[0] - index_tip[0])**2 +
            (thumb_tip[1] - index_tip[1])**2
        )
        if thumb_index_dist < 0.05 and middle_up and ring_up and pinky_up:
            return GestureType.OK, 0.85

        # 三根手指: 大拇指、食指、小指伸直 -> 编队飞行
        if thumb_up and index_up and pinky_up and not middle_up and not ring_up:
            return GestureType.ROCK, 0.85

        return GestureType.NONE, 0.0

    def _reset_stable_count(self):
        """重置稳定计数"""
        self._gesture_stable_count = 0
        self._last_gesture = GestureType.NONE

    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """
        在图像上绘制手部关键点和运动轨迹

        Args:
            frame: BGR 格式的图像帧

        Returns:
            绘制了关键点和轨迹的图像帧
        """
        if not self._is_initialized:
            return frame

        _lazy_import()

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)

        # 科研级配色方案 (更柔和专业)
        JOINT_COLORS = {
            'thumb': (66, 133, 244),    # Google Blue
            'index': (52, 168, 83),     # Google Green
            'middle': (251, 188, 5),    # Google Yellow
            'ring': (234, 67, 53),      # Google Red
            'pinky': (154, 160, 166),   # Gray
            'palm': (255, 255, 255),    # White
        }
        CONNECTION_COLOR = (200, 200, 200)  # 浅灰连接线

        if results.multi_hand_landmarks and results.multi_handedness:
            # 获取当前帧检测到的手
            current_hands = set()
            for hand_idx in range(len(results.multi_hand_landmarks)):
                handedness = results.multi_handedness[hand_idx].classification[0].label
                current_hands.add(handedness)

            # 检测新出现的手，清空其旧轨迹
            new_hands = current_hands - self._prev_hands_detected
            for hand in new_hands:
                self._trajectories[hand] = {tip: [] for tip in self.FINGERTIP_IDS}

            # 更新上一帧检测到的手
            self._prev_hands_detected = current_hands

            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # 使用实际的左右手标识，而不是检测索引
                handedness = results.multi_handedness[hand_idx].classification[0].label
                hand_color = self.HAND_COLORS[0] if handedness == "Left" else self.HAND_COLORS[1]

                # 获取手腕位置
                wrist = hand_landmarks.landmark[0]
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

                # 更新所有指尖轨迹
                for tip_idx, tip_id in enumerate(self.FINGERTIP_IDS):
                    tip = hand_landmarks.landmark[tip_id]
                    tip_x, tip_y = int(tip.x * w), int(tip.y * h)

                    # 添加到轨迹（使用 handedness 作为键）
                    self._trajectories[handedness][tip_id].append((tip_x, tip_y))
                    if len(self._trajectories[handedness][tip_id]) > self.TRAJECTORY_MAX_LEN:
                        self._trajectories[handedness][tip_id].pop(0)

                    # 绘制该指尖的轨迹 (渐变透明度)
                    trajectory = self._trajectories[handedness][tip_id]
                    tip_color = self.FINGERTIP_COLORS[tip_idx]
                    for i in range(1, len(trajectory)):
                        alpha = i / len(trajectory)  # 越新越不透明
                        thickness = max(1, int(3 * alpha))
                        color = tuple(int(c * alpha) for c in tip_color)
                        cv2.line(frame, trajectory[i-1], trajectory[i], color, thickness)

                # 绘制连接线
                for connection in mp_hands.HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start = hand_landmarks.landmark[start_idx]
                    end = hand_landmarks.landmark[end_idx]
                    start_pt = (int(start.x * w), int(start.y * h))
                    end_pt = (int(end.x * w), int(end.y * h))
                    cv2.line(frame, start_pt, end_pt, CONNECTION_COLOR, 2)

                # 绘制关键点 (按手指分色)
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    x, y = int(landmark.x * w), int(landmark.y * h)

                    # 根据关键点索引确定颜色
                    if idx in [1, 2, 3, 4]:  # 大拇指
                        color = JOINT_COLORS['thumb']
                    elif idx in [5, 6, 7, 8]:  # 食指
                        color = JOINT_COLORS['index']
                    elif idx in [9, 10, 11, 12]:  # 中指
                        color = JOINT_COLORS['middle']
                    elif idx in [13, 14, 15, 16]:  # 无名指
                        color = JOINT_COLORS['ring']
                    elif idx in [17, 18, 19, 20]:  # 小指
                        color = JOINT_COLORS['pinky']
                    else:  # 手掌
                        color = JOINT_COLORS['palm']

                    # 指尖用大圆点
                    if idx in [4, 8, 12, 16, 20]:
                        cv2.circle(frame, (x, y), 6, color, -1)
                        cv2.circle(frame, (x, y), 8, hand_color, 2)
                    else:
                        cv2.circle(frame, (x, y), 4, color, -1)

                # 显示手的类型标签 (翻转左右，因为摄像头是镜像的)
                # MediaPipe 从摄像头视角判断，需要翻转
                display_hand = "Right" if handedness == "Left" else "Left"
                cv2.putText(frame, display_hand, (wrist_x - 25, wrist_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
        else:
            # 没有检测到手，清空轨迹和记录
            self._trajectories = {
                "Left": {tip: [] for tip in self.FINGERTIP_IDS},
                "Right": {tip: [] for tip in self.FINGERTIP_IDS}
            }
            self._prev_hands_detected = set()

        return frame

    def clear_trajectories(self):
        """清空轨迹记录"""
        self._trajectories = {
            "Left": {tip: [] for tip in self.FINGERTIP_IDS},
            "Right": {tip: [] for tip in self.FINGERTIP_IDS}
        }
        self._prev_hands_detected = set()

    def release(self):
        """释放资源"""
        if self._hands is not None:
            self._hands.close()
            self._hands = None
        self._is_initialized = False
        self._reset_stable_count()
        self.status_changed.emit("手势检测器已释放")
