"""
手势检测工作线程 - 异步处理手势识别，避免阻塞主线程
"""

import cv2
import numpy as np
from typing import Optional
from threading import Lock
from PySide6.QtCore import QThread, Signal


class GestureWorker(QThread):
    """
    手势检测工作线程

    异步处理 MediaPipe 手势识别，结果通过信号发送回主线程
    """

    # 信号: 检测结果 (command, confidence, landmarks_data)
    result_ready = Signal(str, float, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._frame = None
        self._frame_lock = Lock()
        self._new_frame_available = False

        # MediaPipe 相关
        self._hands = None
        self._is_initialized = False

        # 手势稳定性
        self._last_gesture = "无手势"
        self._gesture_stable_count = 0
        self._stable_threshold = 3  # 降低阈值以加快响应

        # 缓存的结果供绘制使用
        self.cached_results = None
        self.cached_results_lock = Lock()

    def initialize(self) -> bool:
        """初始化 MediaPipe"""
        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands

            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0,  # Lite 模型
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
            self._is_initialized = True
            return True
        except Exception as e:
            print(f"GestureWorker 初始化失败: {e}")
            return False

    def submit_frame(self, frame: np.ndarray):
        """提交新帧进行处理 (非阻塞)"""
        with self._frame_lock:
            # 降采样
            h, w = frame.shape[:2]
            if w > 480:
                self._frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            else:
                self._frame = frame.copy()
            self._new_frame_available = True

    def run(self):
        """工作线程主循环"""
        self._running = True

        while self._running:
            # 获取待处理的帧
            frame = None
            with self._frame_lock:
                if self._new_frame_available:
                    frame = self._frame
                    self._new_frame_available = False

            if frame is None:
                self.msleep(10)  # 没有新帧，短暂休眠
                continue

            # 处理帧
            try:
                result = self._process_frame(frame)
                if result:
                    command, confidence, results = result

                    # 缓存结果供绘制
                    with self.cached_results_lock:
                        self.cached_results = results

                    self.result_ready.emit(command, confidence, results)
            except Exception as e:
                print(f"GestureWorker 处理错误: {e}")

            self.msleep(5)  # 控制处理频率

    def _process_frame(self, frame: np.ndarray):
        """处理单帧"""
        if not self._is_initialized or self._hands is None:
            return None

        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb_frame)

        # 缓存原始结果
        with self.cached_results_lock:
            self.cached_results = results

        if not results.multi_hand_landmarks:
            self._gesture_stable_count = 0
            self._last_gesture = "无手势"
            return None

        # 识别手势
        hand_landmarks = results.multi_hand_landmarks[0]
        gesture, confidence = self._recognize_gesture(hand_landmarks)

        if gesture == "无手势" or confidence < 0.8:
            self._gesture_stable_count = 0
            return None

        # 稳定性检查
        if gesture == self._last_gesture:
            self._gesture_stable_count += 1
        else:
            self._gesture_stable_count = 1
            self._last_gesture = gesture

        if self._gesture_stable_count < self._stable_threshold:
            return None

        return (gesture, confidence, results)

    def _recognize_gesture(self, hand_landmarks) -> tuple:
        """识别手势"""
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

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
        def is_finger_extended(tip, pip, mcp):
            return tip[1] < pip[1] < mcp[1]

        def is_thumb_extended():
            return abs(thumb_tip[0] - wrist[0]) > abs(thumb_mcp[0] - wrist[0])

        thumb_up = is_thumb_extended()
        index_up = is_finger_extended(index_tip, index_pip, index_mcp)
        middle_up = is_finger_extended(middle_tip, middle_pip, middle_mcp)
        ring_up = is_finger_extended(ring_tip, ring_pip, ring_mcp)
        pinky_up = is_finger_extended(pinky_tip, pinky_pip, pinky_mcp)

        extended_count = sum([thumb_up, index_up, middle_up, ring_up, pinky_up])

        # 手势识别
        if extended_count == 0:
            return ("集群降落", 0.9)
        if extended_count == 5:
            return ("集群起飞", 0.9)
        if thumb_up and not any([index_up, middle_up, ring_up, pinky_up]):
            if thumb_tip[1] < thumb_mcp[1]:
                return ("集群高度上升", 0.85)
            else:
                return ("集群高度下降", 0.85)
        if index_up and not any([middle_up, ring_up, pinky_up]):
            return ("集群悬停", 0.85)
        if thumb_up and index_up and pinky_up and not middle_up and not ring_up:
            return ("编队飞行", 0.85)
        if index_up and middle_up and not thumb_up and not ring_up and not pinky_up:
            return ("向前飞行", 0.85)

        # OK 手势
        import numpy as np
        thumb_index_dist = np.sqrt(
            (thumb_tip[0] - index_tip[0])**2 +
            (thumb_tip[1] - index_tip[1])**2
        )
        if thumb_index_dist < 0.05 and middle_up and ring_up and pinky_up:
            return ("指令确定", 0.85)

        return ("无手势", 0.0)

    def get_cached_results(self):
        """获取缓存的结果 (线程安全)"""
        with self.cached_results_lock:
            return self.cached_results

    def stop(self):
        """停止工作线程"""
        self._running = False
        self.wait(1000)  # 等待线程结束

        if self._hands:
            self._hands.close()
            self._hands = None
