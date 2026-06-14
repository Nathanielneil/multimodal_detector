"""
FunASR 流式推理工作线程 - 接收音频块，运行 Paraformer-zh-streaming 推理，发送识别结果
"""
import queue
import threading
import time
import numpy as np
from typing import Optional
from PySide6.QtCore import QThread, Signal, QObject
from utils.logger import get_logger
from detectors.base_detector import DetectionResult

logger = get_logger(__name__)

_STOP = object()

_MODEL_RATE = 16000
_CHUNK_SAMPLES = 7200   # ~450ms @ 16kHz
_QUEUE_MAX = 20


def _resample(data: np.ndarray, src_rate: int, dst_rate: int = _MODEL_RATE) -> np.ndarray:
    """高质量重采样，优先使用 soxr，回退到 scipy。"""
    if src_rate == dst_rate:
        return data
    try:
        import soxr
        return soxr.resample(data, src_rate, dst_rate, quality="HQ").astype(np.float32)
    except ImportError:
        from scipy import signal as scipy_signal
        n_out = int(len(data) * dst_rate / src_rate)
        return scipy_signal.resample(data, n_out).astype(np.float32)


class FunASRWorker(QThread):
    partial_result_ready = Signal(str)
    detection_ready = Signal(object)   # DetectionResult
    error_occurred = Signal(str)
    status_changed = Signal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._model = None
        self._is_initialized = False
        self._accepting = False
        self._queue: queue.Queue = queue.Queue()
        self._queue_lock = threading.Lock()
        self._cache: dict = {}
        self._device_rate: int = 48000  # updated by VoiceDetector before recording

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @staticmethod
    def _select_device() -> str:
        """按 CUDA → MPS → CPU 优先级选择推理设备。"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def initialize(self) -> bool:
        try:
            self.status_changed.emit("正在加载语音模型...")
            from funasr import AutoModel as _AutoModel
            device = self._select_device()
            self._model = _AutoModel(
                model="paraformer-zh-streaming",
                disable_update=True,
                device=device,
            )
            self._is_initialized = True
            self.status_changed.emit("语音模型加载完成")
            return True
        except Exception as e:
            self.error_occurred.emit(f"FunASR 模型加载失败: {e}")
            return False

    def enqueue(self, chunk: np.ndarray) -> bool:
        if not self._is_initialized or not self._accepting:
            return False
        with self._queue_lock:
            if self._queue.qsize() >= _QUEUE_MAX:
                items = []
                while True:
                    try:
                        items.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
                dropped = False
                kept = []
                for item in items:
                    if not dropped and item is not None and item is not _STOP:
                        logger.warning(f"队列积压，丢弃最旧音频块 @ {time.time():.3f}")
                        dropped = True
                    else:
                        kept.append(item)
                for item in kept:
                    self._queue.put(item)
                if not dropped:
                    # Queue full of sentinels — discard new chunk silently
                    return True
            self._queue.put(chunk)
        return True

    def send_eos(self):
        """Push EOS sentinel — triggers final inference."""
        self._accepting = False
        with self._queue_lock:
            self._queue.put(None)

    def set_device_rate(self, rate: int):
        """录音前由 VoiceDetector 调用，告知实际设备采样率。"""
        self._device_rate = rate

    def start_session(self):
        """Allow audio chunks to be enqueued (call before start_recording)."""
        self._accepting = True

    def run(self):
        buf = np.array([], dtype=np.float32)
        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is _STOP:
                break

            is_final = item is None
            final_emitted = False

            if not is_final:
                chunk_16k = _resample(item, self._device_rate)
                buf = np.concatenate([buf, chunk_16k])

            while len(buf) >= _CHUNK_SAMPLES or (is_final and len(buf) > 0):
                if is_final:
                    chunk = buf
                    buf = np.array([], dtype=np.float32)
                else:
                    chunk = buf[:_CHUNK_SAMPLES]
                    buf = buf[_CHUNK_SAMPLES:]

                try:
                    if self._model is None:
                        break
                    res = self._model.generate(
                        input=chunk,
                        cache=self._cache,
                        is_final=is_final and len(buf) == 0,
                        chunk_size=[0, 10, 5],
                    )
                    text = res[0]["text"].strip() if res else ""
                    if text:
                        if is_final and len(buf) == 0:
                            result = DetectionResult(
                                modal_type="voice",
                                command=text,
                                confidence=1.0,
                                details={"model": "paraformer-zh-streaming"},
                            )
                            self.detection_ready.emit(result)
                            final_emitted = True
                        else:
                            self.partial_result_ready.emit(text)
                except Exception as e:
                    logger.error(f"FunASR 推理异常: {e}")
                    self.error_occurred.emit(f"推理错误: {e}")

                if not is_final:
                    break

            if is_final:
                if not final_emitted:
                    self.status_changed.emit("未识别到语音")
                self._cache = {}
                buf = np.array([], dtype=np.float32)

        # Clean up model after thread exits cleanly
        self._model = None

    def release(self):
        self._is_initialized = False
        self._queue.put(_STOP)   # exits run() loop without triggering inference
        # self._model is cleaned up by run() after _STOP is processed
        self.status_changed.emit("FunASRWorker 已释放")
