"""
语音识别检测器 - 使用 OpenAI Whisper 进行语音转文字
支持点击录音模式
"""

import numpy as np
import threading
import tempfile
import os
from typing import Optional, Any
from datetime import datetime

from .base_detector import BaseDetector, DetectionResult

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
    """

    def __init__(self, model_name: str = "base", parent=None):
        super().__init__(parent)
        self._model_name = model_name
        self._model = None
        self._sample_rate = 16000
        self._language = "zh"

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
        self.status_changed.emit("正在录音...")

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

        # 识别
        result = self._transcribe(audio_data)
        self._audio_buffer = []

        return result

    def _record_audio(self):
        """后台录音线程"""
        try:
            _lazy_import()

            def callback(indata, frames, time, status):
                if self._is_recording:
                    self._audio_buffer.append(indata.copy())

            self._stream = sounddevice.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype='float32',
                callback=callback,
                blocksize=1024
            )
            self._stream.start()

            # 保持录音直到停止
            while self._is_recording:
                sounddevice.sleep(100)

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
            # 保存临时 WAV 文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                scipy_io.write(temp_path, self._sample_rate, audio_data)

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
                    "duration": len(audio_data) / self._sample_rate,
                    "segments_count": len(segments),
                }
            )

            self.status_changed.emit(f"识别完成: {text[:20]}...")
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
