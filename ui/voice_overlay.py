"""
语音识别可视化叠加层 - 在视频区域底部显示
科技感设计: 霓虹波形 + 发光效果 + 动态粒子
"""

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QFrame, QSizePolicy, QProgressBar, QGraphicsDropShadowEffect
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QPointF
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QLinearGradient,
    QRadialGradient, QPainterPath, QFont
)
from collections import deque
import time
import math
import random
from utils.logger import get_logger

logger = get_logger(__name__)


class NeonWaveformWidget(QWidget):
    """
    霓虹风格音频波形 - 科技感设计
    特点: 渐变色、发光效果、平滑曲线、网格背景
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(55)

        # 波形数据 (双缓冲平滑)
        self._waveform_data = deque([0.0] * 64, maxlen=64)
        self._smooth_data = [0.0] * 64
        self._is_active = False

        # 动画相关
        self._phase = 0.0
        self._glow_intensity = 0.0

        # 颜色主题 (霓虹蓝紫)
        self._primary_color = QColor("#00D4FF")    # 青色
        self._secondary_color = QColor("#7B2FFF")  # 紫色
        self._accent_color = QColor("#FF00FF")     # 品红
        self._bg_color = QColor(10, 12, 18)        # 深蓝黑

        # 平滑动画定时器
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._animate)
        self._anim_timer.start(30)  # ~33fps

    def add_sample(self, amplitude: float):
        """添加振幅采样"""
        self._waveform_data.append(min(1.0, max(0.0, amplitude)))

    def set_active(self, active: bool):
        """设置激活状态"""
        self._is_active = active
        if not active:
            # 渐出效果
            pass

    def clear(self):
        """清空数据"""
        self._waveform_data = deque([0.0] * 64, maxlen=64)
        self._smooth_data = [0.0] * 64

    def _animate(self):
        """动画更新"""
        self._phase += 0.15

        # 平滑插值
        for i, val in enumerate(self._waveform_data):
            target = val if self._is_active else 0.0
            self._smooth_data[i] += (target - self._smooth_data[i]) * 0.3

        # 发光强度
        if self._is_active:
            avg = sum(self._smooth_data) / len(self._smooth_data)
            self._glow_intensity += (avg * 1.5 - self._glow_intensity) * 0.2
        else:
            self._glow_intensity *= 0.9

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        center_y = h / 2

        # 1. 深色背景
        painter.fillRect(0, 0, w, h, self._bg_color)

        # 2. 网格线 (科技感)
        self._draw_grid(painter, w, h)

        # 3. 扫描线动画
        self._draw_scanline(painter, w, h)

        # 4. 波形 (发光 + 主体)
        if any(v > 0.01 for v in self._smooth_data):
            self._draw_glow_wave(painter, w, h, center_y)
            self._draw_main_wave(painter, w, h, center_y)
            self._draw_mirror_wave(painter, w, h, center_y)

        # 5. 边框发光
        self._draw_border_glow(painter, w, h)

    def _draw_grid(self, painter, w, h):
        """绘制网格背景"""
        pen = QPen(QColor(30, 40, 60, 80))
        pen.setWidth(1)
        painter.setPen(pen)

        # 水平线
        for y in range(0, h, 10):
            painter.drawLine(0, y, w, y)

        # 垂直线
        for x in range(0, w, 20):
            painter.drawLine(x, 0, x, h)

        # 中心线 (高亮)
        pen.setColor(QColor(0, 212, 255, 60))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(0, h // 2, w, h // 2)

    def _draw_scanline(self, painter, w, h):
        """绘制扫描线动画"""
        scan_x = int((self._phase * 50) % (w + 100)) - 50

        gradient = QLinearGradient(scan_x - 50, 0, scan_x + 50, 0)
        gradient.setColorAt(0.0, QColor(0, 212, 255, 0))
        gradient.setColorAt(0.5, QColor(0, 212, 255, 40))
        gradient.setColorAt(1.0, QColor(0, 212, 255, 0))

        painter.fillRect(scan_x - 50, 0, 100, h, gradient)

    def _draw_glow_wave(self, painter, w, h, center_y):
        """绘制发光层"""
        if self._glow_intensity < 0.01:
            return

        data_len = len(self._smooth_data)
        spacing = w / data_len

        # 多层发光
        for glow_size in [12, 8, 4]:
            path = QPainterPath()
            first = True

            for i, amp in enumerate(self._smooth_data):
                x = i * spacing + spacing / 2
                wave_offset = math.sin(self._phase + i * 0.2) * 2
                bar_height = amp * (h - 12) / 2 + wave_offset

                if first:
                    path.moveTo(x, center_y - bar_height)
                    first = False
                else:
                    path.lineTo(x, center_y - bar_height)

            # 返回路径
            for i in range(data_len - 1, -1, -1):
                x = i * spacing + spacing / 2
                amp = self._smooth_data[i]
                wave_offset = math.sin(self._phase + i * 0.2) * 2
                bar_height = amp * (h - 12) / 2 + wave_offset
                path.lineTo(x, center_y + bar_height)

            path.closeSubpath()

            # 发光颜色
            glow_alpha = int(30 * self._glow_intensity * (12 / glow_size))
            pen = QPen(QColor(0, 212, 255, glow_alpha))
            pen.setWidth(glow_size)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)

    def _draw_main_wave(self, painter, w, h, center_y):
        """绘制主波形"""
        data_len = len(self._smooth_data)
        spacing = w / data_len
        bar_width = max(3, spacing - 2)

        for i, amp in enumerate(self._smooth_data):
            if amp < 0.01:
                continue

            x = i * spacing + spacing / 2
            wave_offset = math.sin(self._phase + i * 0.2) * 2
            bar_height = max(2, amp * (h - 12) / 2 + wave_offset)

            # 颜色渐变 (根据振幅)
            t = amp
            r = int(0 + t * 123)      # 0 -> 123 (紫)
            g = int(212 - t * 165)    # 212 -> 47
            b = int(255)              # 保持 255

            # 垂直渐变
            gradient = QLinearGradient(x, center_y - bar_height, x, center_y + bar_height)
            gradient.setColorAt(0.0, QColor(r, g, b, 255))
            gradient.setColorAt(0.5, QColor(0, 212, 255, 255))
            gradient.setColorAt(1.0, QColor(r, g, b, 255))

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(gradient))

            # 绘制圆角条
            rect_x = x - bar_width / 2
            rect_y = center_y - bar_height
            rect_h = bar_height * 2

            painter.drawRoundedRect(
                int(rect_x), int(rect_y),
                int(bar_width), int(rect_h),
                bar_width / 2, bar_width / 2
            )

            # 顶部亮点
            if amp > 0.3:
                highlight = QRadialGradient(x, center_y - bar_height, bar_width)
                highlight.setColorAt(0.0, QColor(255, 255, 255, int(150 * amp)))
                highlight.setColorAt(1.0, QColor(255, 255, 255, 0))
                painter.setBrush(QBrush(highlight))
                painter.drawEllipse(
                    int(x - bar_width/2), int(center_y - bar_height - bar_width/2),
                    int(bar_width), int(bar_width)
                )

    def _draw_mirror_wave(self, painter, w, h, center_y):
        """绘制镜像反射 (下半部分淡化)"""
        # 底部渐变遮罩
        mask_gradient = QLinearGradient(0, center_y, 0, h)
        mask_gradient.setColorAt(0.0, QColor(10, 12, 18, 0))
        mask_gradient.setColorAt(0.3, QColor(10, 12, 18, 180))
        mask_gradient.setColorAt(1.0, QColor(10, 12, 18, 255))
        painter.fillRect(0, int(center_y), w, int(h - center_y), mask_gradient)

    def _draw_border_glow(self, painter, w, h):
        """绘制边框发光"""
        # 顶部边框
        gradient = QLinearGradient(0, 0, w, 0)
        gradient.setColorAt(0.0, QColor(0, 212, 255, 0))
        gradient.setColorAt(0.3, QColor(0, 212, 255, 150))
        gradient.setColorAt(0.7, QColor(123, 47, 255, 150))
        gradient.setColorAt(1.0, QColor(123, 47, 255, 0))

        pen = QPen(QBrush(gradient), 2)
        painter.setPen(pen)
        painter.drawLine(0, 1, w, 1)

        # 底部边框
        gradient2 = QLinearGradient(0, 0, w, 0)
        gradient2.setColorAt(0.0, QColor(123, 47, 255, 0))
        gradient2.setColorAt(0.5, QColor(0, 212, 255, 100))
        gradient2.setColorAt(1.0, QColor(123, 47, 255, 0))

        pen2 = QPen(QBrush(gradient2), 1)
        painter.setPen(pen2)
        painter.drawLine(0, h - 1, w, h - 1)


class CyberProgressBar(QWidget):
    """科技风格音量条"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(130, 18)
        self._value = 0.0
        self._target = 0.0

        # 动画
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.start(30)

    def setValue(self, value: float):
        """设置值 (0-100)"""
        self._target = min(100, max(0, value)) / 100.0

    def _animate(self):
        self._value += (self._target - self._value) * 0.3
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()

        # 背景
        painter.fillRect(0, 0, w, h, QColor(15, 20, 30))

        # 边框
        pen = QPen(QColor(0, 212, 255, 100))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRoundedRect(0, 0, w - 1, h - 1, 3, 3)

        # 填充
        fill_w = int((w - 4) * self._value)
        if fill_w > 0:
            # 渐变色
            gradient = QLinearGradient(2, 0, w - 2, 0)
            gradient.setColorAt(0.0, QColor(0, 212, 255))
            gradient.setColorAt(0.5, QColor(0, 255, 200))
            gradient.setColorAt(0.8, QColor(255, 200, 0))
            gradient.setColorAt(1.0, QColor(255, 50, 50))

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawRoundedRect(2, 2, fill_w, h - 4, 2, 2)

            # 发光效果
            glow_gradient = QLinearGradient(2, 0, 2 + fill_w, 0)
            glow_gradient.setColorAt(0.0, QColor(0, 212, 255, 0))
            glow_gradient.setColorAt(0.8, QColor(0, 212, 255, 80))
            glow_gradient.setColorAt(1.0, QColor(255, 255, 255, 150))
            painter.setBrush(QBrush(glow_gradient))
            painter.drawRoundedRect(2, 2, fill_w, h - 4, 2, 2)

        # 刻度线
        painter.setPen(QPen(QColor(0, 212, 255, 40)))
        for i in range(1, 10):
            x = int(2 + (w - 4) * i / 10)
            painter.drawLine(x, 4, x, h - 4)


class VoiceOverlayWidget(QWidget):
    """
    语音识别可视化叠加层 - 科技感设计
    """

    recognition_complete = Signal(str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_recording = False
        self._record_start_time = None
        self._last_result = ""
        self._last_confidence = 0.0
        self._last_command = ""

        self._init_ui()
        self._init_timers()

    def _init_ui(self):
        """初始化界面"""
        self.setFixedHeight(115)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 波形区域
        self._waveform = NeonWaveformWidget()
        main_layout.addWidget(self._waveform)

        # 信息栏
        info_widget = QWidget()
        info_widget.setStyleSheet("background-color: rgba(10, 12, 18, 240);")
        info_widget.setFixedHeight(55)

        info_layout = QHBoxLayout(info_widget)
        info_layout.setContentsMargins(15, 5, 15, 5)
        info_layout.setSpacing(15)

        # 左侧: 状态
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(8)

        self._indicator = QLabel("◉")
        self._indicator.setStyleSheet("""
            color: #444444;
            font-size: 20px;
            font-weight: bold;
        """)
        status_layout.addWidget(self._indicator)

        self._status_label = QLabel("STANDBY")
        self._status_label.setStyleSheet("""
            color: #00D4FF;
            font-size: 12px;
            font-weight: bold;
            font-family: 'Consolas', 'Monaco', monospace;
            letter-spacing: 2px;
        """)
        status_layout.addWidget(self._status_label)

        self._duration_label = QLabel("")
        self._duration_label.setStyleSheet("""
            color: #00D4FF;
            font-size: 14px;
            font-weight: bold;
            font-family: 'Consolas', monospace;
        """)
        self._duration_label.setFixedWidth(55)
        status_layout.addWidget(self._duration_label)

        info_layout.addWidget(status_widget)

        # 中间: 音量
        volume_widget = QWidget()
        volume_layout = QHBoxLayout(volume_widget)
        volume_layout.setContentsMargins(0, 0, 0, 0)
        volume_layout.setSpacing(8)

        vol_label = QLabel("VOL")
        vol_label.setStyleSheet("""
            color: #666666;
            font-size: 10px;
            font-family: 'Consolas', monospace;
        """)
        volume_layout.addWidget(vol_label)

        self._volume_bar = CyberProgressBar()
        volume_layout.addWidget(self._volume_bar)

        info_layout.addWidget(volume_widget)

        info_layout.addStretch()

        # 右侧: 结果
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(0)

        self._result_label = QLabel("")
        self._result_label.setStyleSheet("""
            color: #00FF88;
            font-size: 13px;
            font-weight: bold;
        """)
        self._result_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._result_label.setWordWrap(True)
        self._result_label.setMinimumWidth(180)
        result_layout.addWidget(self._result_label)

        self._command_label = QLabel("")
        self._command_label.setStyleSheet("""
            color: #7B2FFF;
            font-size: 11px;
            font-family: 'Consolas', monospace;
        """)
        self._command_label.setAlignment(Qt.AlignRight)
        result_layout.addWidget(self._command_label)

        info_layout.addWidget(result_widget)

        main_layout.addWidget(info_widget)

    def _init_timers(self):
        """初始化定时器"""
        self._duration_timer = QTimer(self)
        self._duration_timer.timeout.connect(self._update_duration)
        self._duration_timer.setInterval(100)

        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._toggle_indicator)
        self._blink_timer.setInterval(300)
        self._blink_on = True

    def paintEvent(self, event):
        """绘制背景"""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 12, 18, 245))

    @Slot()
    def start_recording(self):
        """开始录音"""
        self._is_recording = True
        self._record_start_time = time.time()

        self._status_label.setText("● REC")
        self._status_label.setStyleSheet("""
            color: #FF3366;
            font-size: 12px;
            font-weight: bold;
            font-family: 'Consolas', monospace;
            letter-spacing: 2px;
        """)
        self._indicator.setStyleSheet("color: #FF3366; font-size: 20px;")

        self._duration_timer.start()
        self._blink_timer.start()
        self._waveform.set_active(True)

        self._result_label.setText("")
        self._command_label.setText("")
        self._volume_bar.setValue(0)

        self.show()
        self.raise_()
        logger.debug(f"显示叠加层: geometry={self.geometry()}, visible={self.isVisible()}")

    @Slot()
    def stop_recording(self):
        """停止录音"""
        self._is_recording = False

        self._status_label.setText("PROCESSING")
        self._status_label.setStyleSheet("""
            color: #FFD700;
            font-size: 12px;
            font-weight: bold;
            font-family: 'Consolas', monospace;
            letter-spacing: 2px;
        """)
        self._indicator.setStyleSheet("color: #FFD700; font-size: 20px;")
        self._blink_timer.stop()

        self._duration_timer.stop()
        self._waveform.set_active(False)

    @Slot(float)
    def update_volume(self, volume: float):
        """更新音量"""
        self._volume_bar.setValue(int(volume * 100))
        if self._is_recording:
            self._waveform.add_sample(volume)
            # 调试：每10次打印一次
            if not hasattr(self, '_vol_count'):
                self._vol_count = 0
            self._vol_count += 1
            if self._vol_count % 10 == 0:
                logger.debug(f"音量更新: {volume:.2f}, waveform激活={self._waveform._is_active}")

    @Slot(str, float, str)
    def set_result(self, text: str, confidence: float, command: str = ""):
        """设置结果"""
        self._last_result = text
        self._last_confidence = confidence
        self._last_command = command

        self._status_label.setText("COMPLETE")
        self._status_label.setStyleSheet("""
            color: #00FF88;
            font-size: 12px;
            font-weight: bold;
            font-family: 'Consolas', monospace;
            letter-spacing: 2px;
        """)
        self._indicator.setStyleSheet("color: #00FF88; font-size: 20px;")

        display_text = text if len(text) <= 20 else text[:18] + ".."
        conf_str = f" {int(confidence*100)}%" if confidence > 0 else ""
        self._result_label.setText(f'"{display_text}"{conf_str}')

        if command:
            self._command_label.setText(f"→ {command}")
            self._command_label.setStyleSheet("""
                color: #00D4FF;
                font-size: 11px;
                font-family: 'Consolas', monospace;
            """)
        else:
            self._command_label.setText("NO MATCH")
            self._command_label.setStyleSheet("""
                color: #FF6600;
                font-size: 11px;
                font-family: 'Consolas', monospace;
            """)

        self.recognition_complete.emit(text, confidence)

    @Slot()
    def set_idle(self):
        """待命状态"""
        self._is_recording = False
        self._duration_timer.stop()
        self._blink_timer.stop()

        self._status_label.setText("STANDBY")
        self._status_label.setStyleSheet("""
            color: #00D4FF;
            font-size: 12px;
            font-weight: bold;
            font-family: 'Consolas', monospace;
            letter-spacing: 2px;
        """)
        self._indicator.setStyleSheet("color: #444444; font-size: 20px;")
        self._duration_label.setText("")
        self._volume_bar.setValue(0)

        self._waveform.set_active(False)
        self._waveform.clear()

    @Slot(str)
    def set_error(self, message: str):
        """错误状态"""
        self._blink_timer.stop()
        self._duration_timer.stop()

        self._status_label.setText("ERROR")
        self._status_label.setStyleSheet("""
            color: #FF3366;
            font-size: 12px;
            font-weight: bold;
            font-family: 'Consolas', monospace;
            letter-spacing: 2px;
        """)
        self._indicator.setStyleSheet("color: #FF3366; font-size: 20px;")

        self._result_label.setText(message[:20])
        self._result_label.setStyleSheet("""
            color: #FF3366;
            font-size: 13px;
            font-family: 'Consolas', monospace;
        """)

    def _update_duration(self):
        """更新时长"""
        if self._record_start_time:
            elapsed = time.time() - self._record_start_time
            self._duration_label.setText(f"{elapsed:.1f}s")

    def _toggle_indicator(self):
        """闪烁"""
        if self._is_recording:
            self._blink_on = not self._blink_on
            color = "#FF3366" if self._blink_on else "#661133"
            self._indicator.setStyleSheet(f"color: {color}; font-size: 20px;")

    @property
    def is_recording(self) -> bool:
        return self._is_recording
