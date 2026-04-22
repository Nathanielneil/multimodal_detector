"""
语音录音检测器 - 负责麦克风录音，将音频块发送给 FunASRWorker
"""

import numpy as np
import threading
from typing import Optional, Any

from .base_detector import BaseDetector, DetectionResult
from PySide6.QtCore import Signal, QObject
from utils.logger import get_logger
from config import config

logger = get_logger(__name__)

sounddevice = None


def _lazy_import():
    global sounddevice
    if sounddevice is None:
        import sounddevice as _sd
        sounddevice = _sd


class VoiceDetector(BaseDetector):
    """
    薄录音层：采集麦克风音频，通过 audio_chunk 信号发送给 FunASRWorker。
    不做任何推理。
    """

    volume_changed = Signal(float)
    recording_started = Signal()
    recording_stopped = Signal()
    audio_chunk = Signal(object)   # np.ndarray, float32 @ device sample rate

    def __init__(self, funasr_worker=None, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._threshold = 0.0
        self._funasr_worker = funasr_worker
        self._audio_device = config.get("voice.audio_device", None)
        self._device_sample_rate = 48000

        self._is_recording = False
        self._recording_lock = threading.Lock()
        self._record_thread: Optional[threading.Thread] = None
        self._stream = None

    def initialize(self) -> bool:
        self._is_initialized = True
        self.status_changed.emit("语音录音模块就绪")
        return True

    def start_recording(self):
        if not self._is_initialized:
            self.error_occurred.emit("语音检测器未初始化")
            return
        if self._funasr_worker is None or not self._funasr_worker.is_initialized:
            self.error_occurred.emit("FunASR 模型未就绪，请等待加载完成")
            return

        with self._recording_lock:
            if self._is_recording:
                return
            self._is_recording = True

        device_info = self.get_device_info()
        self.status_changed.emit(f"正在录音... ({device_info})")
        self.recording_started.emit()

        self._record_thread = threading.Thread(target=self._record_audio, daemon=True)
        self._record_thread.start()

    def stop_recording(self) -> None:
        with self._recording_lock:
            if not self._is_recording:
                return None
            self._is_recording = False

        self.recording_stopped.emit()

        if self._record_thread is not None:
            self._record_thread.join(timeout=1.0)
            self._record_thread = None

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._funasr_worker is not None:
            self._funasr_worker.send_eos()

        self.status_changed.emit("录音结束，识别中...")
        return None

    def _record_audio(self):
        try:
            _lazy_import()

            rms_history = [0.01]
            rms_min = [1.0]
            rms_max = [0.0]

            def callback(indata, frames, time, status):
                with self._recording_lock:
                    is_rec = self._is_recording
                if not is_rec:
                    return

                chunk = indata.copy().flatten().astype(np.float32)
                self.audio_chunk.emit(chunk)

                rms = float(np.sqrt(np.mean(indata ** 2)))
                rms_min[0] = min(rms_min[0], rms)
                rms_max[0] = max(rms_max[0], rms)
                range_val = max(0.01, rms_max[0] - rms_min[0])
                volume = min(1.0, max(0.0, (rms - rms_min[0]) / range_val))
                rms_history.append(rms)
                if len(rms_history) > 50:
                    rms_history.pop(0)
                    rms_min[0] = min(rms_history) * 0.95
                    rms_max[0] = max(rms_history) * 1.05
                self.volume_changed.emit(volume)

            device_to_use = self._audio_device
            self._device_sample_rate = self._get_device_sample_rate(device_to_use)
            try:
                self._stream = sounddevice.InputStream(
                    device=device_to_use,
                    samplerate=self._device_sample_rate,
                    channels=1,
                    dtype="float32",
                    callback=callback,
                    blocksize=1024,
                )
            except Exception:
                self.status_changed.emit(f"设备 {device_to_use} 不可用，使用默认设备")
                self._audio_device = None
                self._device_sample_rate = self._get_device_sample_rate(None)
                self._stream = sounddevice.InputStream(
                    device=None,
                    samplerate=self._device_sample_rate,
                    channels=1,
                    dtype="float32",
                    callback=callback,
                    blocksize=1024,
                )

            self._stream.start()
            while True:
                with self._recording_lock:
                    if not self._is_recording:
                        break
                sounddevice.sleep(50)

        except Exception as e:
            logger.error(f"录音线程异常: {e}")
            self.error_occurred.emit(f"录音错误: {e}")
        finally:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

    def detect(self, data: Any) -> Optional[DetectionResult]:
        return None

    def release(self):
        with self._recording_lock:
            self._is_recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._record_thread is not None:
            self._record_thread.join(timeout=1.0)
            self._record_thread = None
        self._is_initialized = False
        self.status_changed.emit("语音检测器已释放")

    def set_device(self, device_id: Optional[int]):
        self._audio_device = device_id

    def get_device_info(self) -> str:
        _lazy_import()
        try:
            if self._audio_device is None:
                device = sounddevice.query_devices(kind="input")
                return f"默认设备: {device['name']}"
            else:
                device = sounddevice.query_devices(self._audio_device)
                return f"设备 {self._audio_device}: {device['name']}"
        except Exception as e:
            return f"设备信息获取失败: {e}"

    def _get_device_sample_rate(self, device_id: Optional[int]) -> int:
        _lazy_import()
        try:
            if device_id is None:
                device = sounddevice.query_devices(kind="input")
            else:
                device = sounddevice.query_devices(device_id)
            return int(device["default_samplerate"])
        except Exception:
            return 16000
