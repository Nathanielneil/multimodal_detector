"""
SenseVoice-Small 推理工作线程 - 替代 FunASRWorker

使用 sherpa-onnx 运行 SenseVoice-Small ONNX 模型，纯 CPU 推理，
模型体积 ~250MB（vs Paraformer-zh-streaming 的 1.5GB）。

安装依赖：
    pip install sherpa-onnx

模型下载（首次运行自动下载，也可手动）：
    python -c "from workers.sensevoice_worker import _download_model; _download_model()"
"""

import queue
import threading
import time
import numpy as np
from pathlib import Path
from typing import Optional
from PySide6.QtCore import QThread, Signal, QObject
from utils.logger import get_logger
from detectors.base_detector import DetectionResult

logger = get_logger(__name__)

_STOP = object()
_MODEL_RATE = 16000
_QUEUE_MAX = 20

# 模型缓存目录和下载地址（GitHub Releases，无需登录）
_MODEL_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
_MODEL_DIR = Path.home() / ".cache" / "sensevoice_small"
_RELEASE_BASE = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
_TAR_NAME = f"{_MODEL_NAME}.tar.bz2"


def _resample(data: np.ndarray, src_rate: int, dst_rate: int = _MODEL_RATE) -> np.ndarray:
    if src_rate == dst_rate:
        return data
    try:
        import soxr
        return soxr.resample(data, src_rate, dst_rate, quality="HQ").astype(np.float32)
    except ImportError:
        from scipy import signal as scipy_signal
        n_out = int(len(data) * dst_rate / src_rate)
        return scipy_signal.resample(data, n_out).astype(np.float32)


def _download_model() -> Path:
    """下载并解压 SenseVoice-Small ONNX 模型（首次使用时自动调用）。"""
    import urllib.request
    import tarfile

    extracted_dir = _MODEL_DIR / _MODEL_NAME
    # 已解压则直接返回
    if (extracted_dir / "model.onnx").exists():
        return extracted_dir

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = _MODEL_DIR / _TAR_NAME

    if not tar_path.exists():
        url = f"{_RELEASE_BASE}/{_TAR_NAME}"
        logger.info(f"正在下载语音模型压缩包...")
        try:
            urllib.request.urlretrieve(url, tar_path)
            logger.info("模型下载完成，正在解压...")
        except Exception as e:
            tar_path.unlink(missing_ok=True)
            raise RuntimeError(f"模型下载失败: {e}") from e

    # 解压
    try:
        with tarfile.open(tar_path, "r:bz2") as tf:
            tf.extractall(_MODEL_DIR)
        tar_path.unlink(missing_ok=True)  # 解压后删除压缩包节省空间
        logger.info("模型解压完成")
    except Exception as e:
        raise RuntimeError(f"模型解压失败: {e}") from e

    return extracted_dir


class SenseVoiceWorker(QThread):
    """
    SenseVoice-Small 语音识别工作线程。

    信号接口与 FunASRWorker 完全兼容，可直接替换。
    """

    partial_result_ready = Signal(str)
    detection_ready = Signal(object)   # DetectionResult
    error_occurred = Signal(str)
    status_changed = Signal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._recognizer = None
        self._stream = None
        self._is_initialized = False
        self._accepting = False
        self._queue: queue.Queue = queue.Queue()
        self._queue_lock = threading.Lock()
        self._device_rate: int = 48000

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def initialize(self) -> bool:
        try:
            self.status_changed.emit("正在加载语音模型...")
            import sherpa_onnx

            model_dir = _download_model()

            self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model_dir / "model.onnx"),
                tokens=str(model_dir / "tokens.txt"),
                num_threads=2,
                language="auto",
                use_itn=True,
                debug=False,
            )

            self._is_initialized = True
            self.status_changed.emit("语音模型加载完成")
            return True

        except ImportError:
            msg = "缺少 sherpa-onnx，请运行: pip install sherpa-onnx"
            logger.error(msg)
            self.error_occurred.emit(msg)
            return False
        except Exception as e:
            msg = f"SenseVoice 模型加载失败: {e}"
            logger.error(msg)
            self.error_occurred.emit(msg)
            return False

    def set_device_rate(self, rate: int):
        """录音前由 VoiceDetector 调用，告知实际设备采样率。"""
        self._device_rate = rate

    def start_session(self):
        self._accepting = True

    def send_eos(self):
        self._accepting = False
        with self._queue_lock:
            self._queue.put(None)

    def enqueue(self, chunk: np.ndarray) -> bool:
        if not self._is_initialized or not self._accepting:
            return False
        with self._queue_lock:
            if self._queue.qsize() >= _QUEUE_MAX:
                # 丢弃最旧的普通块
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
                    return True
            self._queue.put(chunk)
        return True

    def run(self):
        """工作线程主循环：收集音频直到 EOS，然后一次性推理。"""
        buf = np.array([], dtype=np.float32)

        while True:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is _STOP:
                break

            if item is None:
                # EOS — 执行推理
                if len(buf) > 0 and self._recognizer is not None:
                    try:
                        stream = self._recognizer.create_stream()
                        stream.accept_waveform(_MODEL_RATE, buf)
                        self._recognizer.decode_stream(stream)
                        text = _strip_tags(stream.result.text.strip())

                        if text:
                            result = DetectionResult(
                                modal_type="voice",
                                command=text,
                                confidence=1.0,
                                details={"model": "sensevoice-small"},
                            )
                            self.detection_ready.emit(result)
                        else:
                            self.status_changed.emit("未识别到语音")

                    except Exception as e:
                        logger.error(f"SenseVoice 推理异常: {e}")
                        self.error_occurred.emit(f"推理错误: {e}")
                else:
                    self.status_changed.emit("未识别到语音")

                buf = np.array([], dtype=np.float32)
                continue

            # 普通音频块 — 重采样后累积到缓冲区
            chunk_16k = _resample(item, self._device_rate)
            buf = np.concatenate([buf, chunk_16k])

            # 流式中间结果（每积累约 1.5 秒给一次预览）
            if len(buf) >= _MODEL_RATE * 1 and self._recognizer is not None:
                try:
                    stream = self._recognizer.create_stream()
                    stream.accept_waveform(_MODEL_RATE, buf)
                    self._recognizer.decode_stream(stream)
                    partial = _strip_tags(stream.result.text.strip())
                    if partial:
                        self.partial_result_ready.emit(partial)
                except Exception:
                    pass

        self._recognizer = None

    def release(self):
        self._is_initialized = False
        self._queue.put(_STOP)
        self.status_changed.emit("SenseVoiceWorker 已释放")


def _strip_tags(text: str) -> str:
    """移除 SenseVoice 输出中的语言/情感标签，如 <|zh|><|NEUTRAL|><|Speech|>。"""
    import re
    return re.sub(r"<\|[^|]+\|>", "", text).strip()
