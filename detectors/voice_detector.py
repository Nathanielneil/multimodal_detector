"""
语音识别检测器 - 使用 OpenAI Whisper 进行语音转文字
支持点击录音模式
"""

import numpy as np
import threading
import tempfile
import os
from typing import Optional, Any

from .base_detector import BaseDetector, DetectionResult
from PySide6.QtCore import Signal, QObject
from utils.logger import get_logger
from config import config

logger = get_logger(__name__)

# 延迟导入，避免启动时加载
whisper = None
sounddevice = None
scipy_io = None


def _lazy_import():
    """延迟导入重型依赖"""
    global whisper, sounddevice, scipy_io
    if whisper is None:
        import whisper as _whisper
        import sounddevice as _sd
        import scipy.io.wavfile as _scipy_io
        whisper = _whisper
        sounddevice = _sd
        scipy_io = _scipy_io


class VoiceDetector(BaseDetector):
    """
    语音识别检测器

    使用 OpenAI Whisper 模型进行语音转文字。
    支持点击录音模式：点击开始录音，再次点击停止并识别。

    Attributes:
        model_name: Whisper 模型名称 (tiny/base/small/medium/large)
        sample_rate: 采样率 (默认 16000)
        language: 识别语言 (默认 zh 中文)

    Signals:
        volume_changed: 实时音量变化信号 (volume: float, 0.0-1.0)
        recording_started: 录音开始信号
        recording_stopped: 录音停止信号
    """

    # 新增信号
    volume_changed = Signal(float)      # 实时音量 (0.0-1.0)
    recording_started = Signal()        # 录音开始
    recording_stopped = Signal()        # 录音停止

    def __init__(self, model_name: Optional[str] = None, parent: Optional[QObject] = None):
        super().__init__(parent)
        # 语音检测器不使用置信度过滤 (Whisper 已有内置过滤)
        self._threshold = config.get("voice.threshold", 0.0)

        # 从配置读取参数，允许构造函数参数覆盖
        self._model_name = model_name or config.get("voice.model_name", "small")
        self._model = None
        self._whisper_sample_rate = config.get("voice.sample_rate", 16000)
        self._language = config.get("voice.language", "zh")

        # 音频设备 (None = 系统默认, 数字 = 指定设备ID)
        self._audio_device = config.get("voice.audio_device", None)
        self._device_sample_rate = 48000  # 会在录音时动态更新

        logger.debug(f"VoiceDetector 初始化: model={self._model_name}, device={self._audio_device}, lang={self._language}")

        # 录音相关
        self._is_recording = False
        self._audio_buffer = []
        self._record_thread: Optional[threading.Thread] = None
        self._stream = None

    def initialize(self) -> bool:
        """
        初始化 Whisper 模型

        Returns:
            bool: 是否初始化成功
        """
        try:
            self.status_changed.emit("正在加载 Whisper 模型...")
            _lazy_import()

            # 加载模型 (使用 GPU 如果可用)
            self._model = whisper.load_model(self._model_name)
            self._is_initialized = True
            self.status_changed.emit("Whisper 模型加载完成")
            return True

        except Exception as e:
            self.error_occurred.emit(f"Whisper 模型加载失败: {str(e)}")
            return False

    def start_recording(self):
        """开始录音"""
        if not self._is_initialized:
            self.error_occurred.emit("语音检测器未初始化")
            return

        if self._is_recording:
            return

        _lazy_import()
        self._is_recording = True
        self._audio_buffer = []

        # 显示使用的设备
        device_info = self.get_device_info()
        self.status_changed.emit(f"正在录音... ({device_info})")

        # 发出录音开始信号
        self.recording_started.emit()

        # 在后台线程中录音
        self._record_thread = threading.Thread(target=self._record_audio)
        self._record_thread.start()

    def stop_recording(self) -> Optional[DetectionResult]:
        """
        停止录音并进行识别

        Returns:
            DetectionResult: 识别结果
        """
        if not self._is_recording:
            return None

        self._is_recording = False

        # 发出录音停止信号
        self.recording_stopped.emit()

        # 等待录音线程结束
        if self._record_thread is not None:
            self._record_thread.join(timeout=1.0)
            self._record_thread = None

        # 停止音频流
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._audio_buffer:
            self.status_changed.emit("录音为空")
            return None

        self.status_changed.emit("正在识别...")

        # 合并音频数据
        audio_data = np.concatenate(self._audio_buffer, axis=0)
        audio_data = audio_data.flatten().astype(np.float32)

        # 如果采样率不是 16000 Hz，需要重采样
        if self._device_sample_rate != self._whisper_sample_rate:
            from scipy import signal
            # 计算重采样后的长度
            num_samples = int(len(audio_data) * self._whisper_sample_rate / self._device_sample_rate)
            audio_data = signal.resample(audio_data, num_samples).astype(np.float32)
            self.status_changed.emit(f"重采样: {self._device_sample_rate}Hz → {self._whisper_sample_rate}Hz")

        # 识别
        result = self._transcribe(audio_data)
        self._audio_buffer = []

        return result

    def _record_audio(self):
        """后台录音线程"""
        try:
            _lazy_import()

            # 用于自适应增益
            rms_history = [0.01]
            rms_min = [1.0]
            rms_max = [0.0]

            def callback(indata, frames, time, status):
                if self._is_recording:
                    self._audio_buffer.append(indata.copy())

                    # 计算音量 (RMS)
                    rms = np.sqrt(np.mean(indata ** 2))

                    # 更新动态范围
                    rms_min[0] = min(rms_min[0], rms)
                    rms_max[0] = max(rms_max[0], rms)

                    # 使用动态范围归一化
                    range_val = max(0.01, rms_max[0] - rms_min[0])
                    volume = (rms - rms_min[0]) / range_val
                    volume = min(1.0, max(0.0, volume))

                    # 保存历史用于平滑
                    rms_history.append(rms)
                    if len(rms_history) > 50:
                        rms_history.pop(0)
                        rms_min[0] = min(rms_history) * 0.95
                        rms_max[0] = max(rms_history) * 1.05

                    # 发出音量信号
                    self.volume_changed.emit(volume)

            # 尝试使用指定设备，失败则回退到默认设备
            device_to_use = self._audio_device
            # 动态获取设备采样率
            self._device_sample_rate = self._get_device_sample_rate(device_to_use)
            try:
                self._stream = sounddevice.InputStream(
                    device=device_to_use,
                    samplerate=self._device_sample_rate,
                    channels=1,
                    dtype='float32',
                    callback=callback,
                    blocksize=1024
                )
            except Exception:
                # 指定设备不可用，回退到默认设备和默认采样率
                self.status_changed.emit(f"设备 {device_to_use} 不可用，使用默认设备")
                self._audio_device = None
                self._device_sample_rate = self._get_device_sample_rate(None)
                self._stream = sounddevice.InputStream(
                    device=None,  # 使用默认设备
                    samplerate=self._device_sample_rate,
                    channels=1,
                    dtype='float32',
                    callback=callback,
                    blocksize=1024
                )

            self._stream.start()

            # 保持录音直到停止
            while self._is_recording:
                sounddevice.sleep(50)

        except Exception as e:
            self.error_occurred.emit(f"录音错误: {str(e)}")

    def _transcribe(self, audio_data: np.ndarray) -> Optional[DetectionResult]:
        """
        使用 Whisper 进行语音转文字

        Args:
            audio_data: 音频数据数组

        Returns:
            DetectionResult: 识别结果
        """
        try:
            # 保存临时 WAV 文件 (使用 Whisper 采样率 16000 Hz)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                scipy_io.write(temp_path, self._whisper_sample_rate, audio_data)

            # 使用 Whisper 识别
            result = self._model.transcribe(
                temp_path,
                language=self._language,
                fp16=False  # CPU 模式下关闭 fp16
            )

            # 删除临时文件
            os.unlink(temp_path)

            text = result.get("text", "").strip()
            if not text:
                self.status_changed.emit("未识别到语音")
                return None

            # 计算置信度（基于分段的平均概率）
            segments = result.get("segments", [])
            if segments:
                avg_prob = np.mean([s.get("avg_logprob", -1) for s in segments])
                # 将 log 概率转换为 0-1 范围的置信度
                confidence = min(1.0, max(0.0, (avg_prob + 1) / 1))
            else:
                confidence = 0.5

            detection = DetectionResult(
                modal_type="voice",
                command=text,
                confidence=confidence,
                details={
                    "language": result.get("language", self._language),
                    "duration": len(audio_data) / self._whisper_sample_rate,
                    "segments_count": len(segments),
                }
            )

            self.status_changed.emit(f"识别完成: {text[:20]}...")
            logger.debug(f"发射识别结果: '{text}', 置信度={confidence:.2f}")
            self.emit_result(detection)
            return detection

        except Exception as e:
            self.error_occurred.emit(f"语音识别错误: {str(e)}")
            self.status_changed.emit("识别失败")
            return None

    def detect(self, data: Any) -> Optional[DetectionResult]:
        """
        检测接口（用于直接传入音频数据）

        Args:
            data: 音频数据 (np.ndarray)

        Returns:
            DetectionResult: 识别结果
        """
        if not self._enabled or not self._is_initialized:
            return None

        if isinstance(data, np.ndarray):
            return self._transcribe(data)
        return None

    def release(self):
        """释放资源"""
        self._is_recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._record_thread is not None:
            self._record_thread.join(timeout=1.0)
            self._record_thread = None

        self._model = None
        self._is_initialized = False
        self.status_changed.emit("语音检测器已释放")

    def set_language(self, language: str):
        """设置识别语言"""
        self._language = language

    def set_model(self, model_name: str):
        """设置模型（需要重新初始化）"""
        self._model_name = model_name
        if self._is_initialized:
            self.release()
            self.initialize()

    def set_device(self, device_id: Optional[int]):
        """
        设置音频输入设备

        Args:
            device_id: 设备ID (None = 系统默认)
        """
        self._audio_device = device_id

    def get_device_info(self) -> str:
        """获取当前使用的音频设备信息"""
        _lazy_import()
        try:
            if self._audio_device is None:
                device = sounddevice.query_devices(kind='input')
                return f"默认设备: {device['name']}"
            else:
                device = sounddevice.query_devices(self._audio_device)
                return f"设备 {self._audio_device}: {device['name']}"
        except Exception as e:
            return f"设备信息获取失败: {str(e)}"

    def _get_device_sample_rate(self, device_id: Optional[int]) -> int:
        """获取设备支持的采样率"""
        _lazy_import()
        try:
            if device_id is None:
                device = sounddevice.query_devices(kind='input')
            else:
                device = sounddevice.query_devices(device_id)
            return int(device['default_samplerate'])
        except Exception:
            return 16000  # 回退到默认值
