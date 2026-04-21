# FunASR Streaming Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace OpenAI Whisper with FunASR Paraformer-zh-streaming, adding a FunASRWorker QThread and Push-to-Talk recording mode with real-time subtitle display.

**Architecture:** VoiceDetector becomes a thin recording layer that emits `audio_chunk` signals. FunASRWorker(QThread) owns the FunASR model, buffers chunks, runs streaming inference, and emits `partial_result_ready` (for overlay subtitles) and `detection_ready` (for command dispatch). main_window wires Push-to-Talk key events to start/stop recording.

**Tech Stack:** FunASR>=1.0.0, PySide6 QThread/Signal, sounddevice, scipy (resample), numpy

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `workers/funasr_worker.py` | Create | FunASRWorker QThread: model load, chunk queue, streaming inference, signal emit |
| `detectors/voice_detector.py` | Modify | Remove Whisper; add `audio_chunk` signal; `stop_recording()` calls `send_eos()` |
| `ui/main_window.py` | Modify | Instantiate FunASRWorker; Push-to-Talk key events; connect `partial_result_ready` |
| `requirements.txt` | Modify | Replace `openai-whisper` with `funasr>=1.0.0` |
| `tests/test_funasr_worker.py` | Create | Unit tests for FunASRWorker |
| `tests/test_voice_detector_funasr.py` | Create | Unit tests for refactored VoiceDetector |

---

## Task 1: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Edit requirements.txt**

Replace `openai-whisper>=20231117` with `funasr>=1.0.0`. Keep all other lines unchanged.

```
# 语音识别 (FunASR)
funasr>=1.0.0
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: replace openai-whisper with funasr>=1.0.0 in requirements"
```

---

## Task 2: Create FunASRWorker

**Files:**
- Create: `workers/funasr_worker.py`
- Create: `tests/test_funasr_worker.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_funasr_worker.py`:

```python
import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import queue

sys.path.insert(0, str(Path(__file__).parent.parent))

from detectors.base_detector import DetectionResult


@pytest.fixture
def mock_funasr_model():
    model = MagicMock()
    model.generate.return_value = [{"text": "起飞"}]
    return model


@pytest.fixture
def worker(qtbot, mock_funasr_model):
    with patch("workers.funasr_worker.AutoModel") as MockAutoModel:
        MockAutoModel.return_value = mock_funasr_model
        from workers.funasr_worker import FunASRWorker, _STOP
        w = FunASRWorker()
        assert w.initialize() is True
        w.start()
        yield w
        w.release()
        w.wait(2000)  # run() exits on _STOP sentinel


def test_eos_triggers_detection_ready(qtbot, worker, mock_funasr_model):
    """EOS sentinel causes detection_ready to fire with final text."""
    chunk = np.zeros(7200, dtype=np.float32)
    mock_funasr_model.generate.return_value = [{"text": "起飞"}]

    with qtbot.waitSignal(worker.detection_ready, timeout=3000) as blocker:
        worker.enqueue(chunk)
        worker.send_eos()

    result = blocker.args[0]
    assert isinstance(result, DetectionResult)
    assert result.modal_type == "voice"
    assert "起飞" in result.command


def test_partial_result_emitted(qtbot, worker, mock_funasr_model):
    """Non-final chunk emits partial_result_ready."""
    chunk = np.zeros(7200, dtype=np.float32)
    mock_funasr_model.generate.return_value = [{"text": "起"}]

    with qtbot.waitSignal(worker.partial_result_ready, timeout=3000) as blocker:
        worker.enqueue(chunk)

    assert isinstance(blocker.args[0], str)


def test_queue_overflow_drops_oldest_not_eos(qtbot, worker):
    """Queue overflow drops audio chunks but never the EOS sentinel."""
    chunk = np.zeros(100, dtype=np.float32)
    # Fill beyond limit (21 items)
    for _ in range(22):
        worker.enqueue(chunk)
    # EOS must still be accepted and processed
    with qtbot.waitSignal(worker.detection_ready, timeout=3000):
        worker.send_eos()


def test_initialize_failure_returns_false(qtbot):
    with patch("workers.funasr_worker.AutoModel", side_effect=Exception("load fail")):
        from workers.funasr_worker import FunASRWorker
        w = FunASRWorker()
        assert w.initialize() is False


def test_enqueue_before_init_returns_false(qtbot):
    from workers.funasr_worker import FunASRWorker
    w = FunASRWorker()
    chunk = np.zeros(100, dtype=np.float32)
    assert w.enqueue(chunk) is False


def test_is_initialized_property(qtbot, mock_funasr_model):
    with patch("workers.funasr_worker.AutoModel") as MockAutoModel:
        MockAutoModel.return_value = mock_funasr_model
        from workers.funasr_worker import FunASRWorker
        w = FunASRWorker()
        assert w.is_initialized is False
        w.initialize()
        assert w.is_initialized is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/ubuntu/NGW/intern/multimodal_detector
python -m pytest tests/test_funasr_worker.py -v 2>&1 | head -30
```

Expected: ImportError or ModuleNotFoundError for `workers.funasr_worker`

- [ ] **Step 3: Create workers/funasr_worker.py**

```python
import queue
import time
import numpy as np
from typing import Optional
from PySide6.QtCore import QThread, Signal, QObject
from scipy import signal as scipy_signal
from utils.logger import get_logger
from detectors.base_detector import DetectionResult

logger = get_logger(__name__)

AutoModel = None

_STOP = object()   # sentinel to exit run() loop — distinct from EOS (None)

def _lazy_import():
    global AutoModel
    if AutoModel is None:
        from funasr import AutoModel as _AutoModel
        AutoModel = _AutoModel


_DEVICE_RATE = 48000
_MODEL_RATE = 16000
_CHUNK_SAMPLES = 7200   # ~450ms @ 16kHz
_QUEUE_MAX = 20


class FunASRWorker(QThread):
    partial_result_ready = Signal(str)
    detection_ready = Signal(object)   # DetectionResult
    error_occurred = Signal(str)
    status_changed = Signal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._model = None
        self._is_initialized = False
        self._queue: queue.Queue = queue.Queue()
        self._cache: dict = {}

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def initialize(self) -> bool:
        try:
            self.status_changed.emit("正在加载 FunASR 模型...")
            _lazy_import()
            self._model = AutoModel(
                model="paraformer-zh-streaming",
                disable_update=True,
            )
            self._is_initialized = True
            self.status_changed.emit("FunASR 模型加载完成")
            return True
        except Exception as e:
            self.error_occurred.emit(f"FunASR 模型加载失败: {e}")
            return False

    def enqueue(self, chunk: np.ndarray) -> bool:
        if not self._is_initialized:
            return False
        if self._queue.qsize() >= _QUEUE_MAX:
            try:
                dropped = self._queue.get_nowait()
                if dropped is None or dropped is _STOP:
                    # Never drop EOS or STOP sentinels
                    self._queue.put(dropped)
                else:
                    logger.warning(f"队列积压，丢弃最旧音频块 @ {time.time():.3f}")
            except queue.Empty:
                pass
        self._queue.put(chunk)
        return True

    def send_eos(self):
        """Push EOS sentinel — triggers final inference."""
        self._queue.put(None)

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

            if not is_final:
                # Resample device rate (48kHz) → model rate (16kHz)
                n_out = int(len(item) * _MODEL_RATE / _DEVICE_RATE)
                chunk_16k = scipy_signal.resample(item, n_out).astype(np.float32)
                buf = np.concatenate([buf, chunk_16k])

            while len(buf) >= _CHUNK_SAMPLES or (is_final and len(buf) > 0):
                if is_final:
                    chunk = buf
                    buf = np.array([], dtype=np.float32)
                else:
                    chunk = buf[:_CHUNK_SAMPLES]
                    buf = buf[_CHUNK_SAMPLES:]

                try:
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
                        else:
                            self.partial_result_ready.emit(text)
                except Exception as e:
                    logger.error(f"FunASR 推理异常: {e}")
                    self.error_occurred.emit(f"推理错误: {e}")

                if not is_final:
                    break

            if is_final:
                self._cache = {}
                buf = np.array([], dtype=np.float32)

    def release(self):
        self._is_initialized = False
        self._queue.put(_STOP)   # exits run() loop without triggering inference
        self._model = None
        self.status_changed.emit("FunASRWorker 已释放")
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_funasr_worker.py -v
```

Expected: all 5 tests PASS (FunASR model is mocked)

- [ ] **Step 5: Commit**

```bash
git add workers/funasr_worker.py tests/test_funasr_worker.py
git commit -m "feat: add FunASRWorker with streaming inference and queue overflow protection"
```

---

## Task 3: Refactor VoiceDetector

**Files:**
- Modify: `detectors/voice_detector.py`
- Create: `tests/test_voice_detector_funasr.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_voice_detector_funasr.py`:

```python
import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_worker():
    w = MagicMock()
    w.is_initialized = True
    return w


@pytest.fixture
def detector(qtbot, mock_worker):
    with patch("detectors.voice_detector.sounddevice"):
        from detectors.voice_detector import VoiceDetector
        d = VoiceDetector(funasr_worker=mock_worker)
        d._is_initialized = True
        yield d
        d.release()


def test_audio_chunk_signal_exists(detector):
    """VoiceDetector must have audio_chunk signal."""
    assert hasattr(detector, "audio_chunk")


def test_stop_recording_calls_send_eos(qtbot, detector, mock_worker):
    """stop_recording() must call funasr_worker.send_eos()."""
    detector._is_recording = True
    detector.stop_recording()
    mock_worker.send_eos.assert_called_once()


def test_stop_recording_returns_none(qtbot, detector, mock_worker):
    """stop_recording() no longer returns a DetectionResult."""
    detector._is_recording = True
    result = detector.stop_recording()
    assert result is None


def test_start_recording_blocked_when_worker_not_initialized(qtbot, mock_worker):
    with patch("detectors.voice_detector.sounddevice"):
        from detectors.voice_detector import VoiceDetector
        mock_worker.is_initialized = False
        d = VoiceDetector(funasr_worker=mock_worker)
        d._is_initialized = True
        errors = []
        d.error_occurred.connect(errors.append)
        d.start_recording()
        assert errors, "Should emit error when worker not initialized"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_voice_detector_funasr.py -v 2>&1 | head -30
```

Expected: ImportError or AttributeError (audio_chunk signal missing)

- [ ] **Step 3: Rewrite detectors/voice_detector.py**

Replace the entire file content. Key changes:
- Remove all `whisper`/`_transcribe`/`_model` references
- Add `audio_chunk = Signal(object)` signal
- Constructor takes `funasr_worker` parameter
- `stop_recording()` calls `self._funasr_worker.send_eos()`, returns None
- `start_recording()` checks `self._funasr_worker.is_initialized`
- `initialize()` no longer loads any model (just marks ready)
- Remove `detect()` Whisper path; keep signature for BaseDetector compliance

```python
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
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_voice_detector_funasr.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add detectors/voice_detector.py tests/test_voice_detector_funasr.py
git commit -m "feat: refactor VoiceDetector - remove Whisper, add audio_chunk signal, Push-to-Talk stop_recording"
```

---

## Task 4: Wire FunASRWorker into main_window.py

**Files:**
- Modify: `ui/main_window.py`

This task has no new tests — the integration is covered by the existing `_on_detection_result` path and the unit tests above.

- [ ] **Step 1: Add FunASRWorker import**

At the top of `ui/main_window.py`, near the other worker imports (around line 212), add:

```python
from workers.funasr_worker import FunASRWorker
```

Also update the VoiceDetector import line — it now requires a `funasr_worker` argument.

- [ ] **Step 2: Add `_funasr_worker` attribute in `__init__`**

In `MainWindow.__init__`, near `self._voice_detector: Optional[VoiceDetector] = None` (line ~270), add:

```python
self._funasr_worker: Optional[FunASRWorker] = None
```

- [ ] **Step 3: Update `_init_heavy_detectors` — replace VoiceDetector creation block**

Find the block starting at line ~786 (`# 步骤1: 创建语音检测器`). Replace it with:

```python
# 步骤1: 创建 FunASRWorker 和语音检测器
current_step += 1
progress.set_status("正在加载 FunASR 语音模型...")
progress.set_progress(int(current_step / total_steps * 100))
progress.set_detail(f"步骤 {current_step}/{total_steps} - 语音识别模块")

self._funasr_worker = FunASRWorker()
self._funasr_worker.detection_ready.connect(self._on_detection_result)
self._funasr_worker.error_occurred.connect(self._on_detector_error)
self._funasr_worker.status_changed.connect(
    lambda s: self._control_panel.set_record_status(s)
)
if not self._funasr_worker.initialize():
    errors.append("FunASR 模型加载失败")
else:
    self._funasr_worker.start()

self._voice_detector = VoiceDetector(funasr_worker=self._funasr_worker)
self._voice_detector.status_changed.connect(
    lambda s: self._control_panel.set_record_status(s)
)
self._voice_detector.error_occurred.connect(self._on_detector_error)
self._voice_detector.initialize()
# Wire audio chunks from recorder → inference worker
self._voice_detector.audio_chunk.connect(
    self._funasr_worker.enqueue, Qt.QueuedConnection
)
QTimer.singleShot(100, self._connect_voice_overlay_signals)
```

- [ ] **Step 4: Update `_connect_voice_overlay_signals` — add partial_result_ready**

In `_connect_voice_overlay_signals` (line ~1258), after connecting `volume_changed`, add:

```python
# 连接 FunASR 中间结果 → overlay 实时字幕
if self._funasr_worker:
    self._funasr_worker.partial_result_ready.connect(
        lambda text: overlay.set_result(text, 0.0, ""),
        Qt.QueuedConnection,
    )
```

- [ ] **Step 5: Update `_on_record_clicked` — remove result handling, add Push-to-Talk**

Replace the `_on_record_clicked` method body (line ~1214). The method now only starts/stops recording; result arrives asynchronously via `detection_ready`:

```python
@Slot(bool)
def _on_record_clicked(self, is_recording: bool):
    if not self._control_panel.is_modal_enabled("voice"):
        self._control_panel.set_record_status("语音模态已禁用")
        return
    if self._voice_detector is None or not self._voice_detector.is_initialized:
        self._control_panel.set_record_status("语音模块未就绪")
        return

    if is_recording:
        self._video_widget.show_voice_overlay()
        self._video_widget.voice_overlay.start_recording()
        self._voice_detector.start_recording()
    else:
        self._video_widget.voice_overlay.stop_recording()
        self._voice_detector.stop_recording()
```

- [ ] **Step 6: Replace Space shortcut with Push-to-Talk key events**

In `_setup_shortcuts` (line ~680), remove:

```python
shortcut_record = QShortcut(QKeySequence(Qt.Key_Space), self)
shortcut_record.activated.connect(self._toggle_recording)
```

Instead, override `keyPressEvent` and `keyReleaseEvent` on `MainWindow`:

```python
def keyPressEvent(self, event: QKeyEvent):
    if event.key() == Qt.Key_Space and not event.isAutoRepeat():
        if self._control_panel.is_modal_enabled("voice"):
            btn = self._control_panel.btn_record
            if not btn.isChecked():
                btn.setChecked(True)
                self._on_record_clicked(True)
    super().keyPressEvent(event)

def keyReleaseEvent(self, event: QKeyEvent):
    if event.key() == Qt.Key_Space and not event.isAutoRepeat():
        if self._control_panel.is_modal_enabled("voice"):
            btn = self._control_panel.btn_record
            if btn.isChecked():
                btn.setChecked(False)
                self._on_record_clicked(False)
    super().keyReleaseEvent(event)
```

Also update `_show_shortcuts_help` to reflect Push-to-Talk:

```python
<tr><td style="padding: 5px;"><b>空格</b></td><td style="padding: 5px;">按住录音，松开识别（Push-to-Talk）</td></tr>
```

- [ ] **Step 7: Update `_on_detection_result` — handle async voice result**

Find the block around line ~1384 that handles `result.modal_type == "voice"`. Results now arrive asynchronously from `FunASRWorker.detection_ready` — this is the single place that updates the overlay with the final result (the old synchronous path in `_on_record_clicked` has been removed in Step 5):

```python
elif result.modal_type == "voice":
    command_text = self._get_voice_command_text(result.command)
    if hasattr(self._video_widget, "voice_overlay"):
        self._video_widget.update_modal_status(
            "voice", result.command, result.confidence
        )
        self._video_widget.voice_overlay.set_result(
            result.command, result.confidence, command_text
        )
    self._process_voice_command(result.command)
```

- [ ] **Step 8: Update `closeEvent` / `release` — add FunASRWorker teardown**

Near line ~1475 where `self._voice_detector.release()` is called, add:

```python
if self._funasr_worker:
    self._funasr_worker.release()
    self._funasr_worker.wait(2000)
```

- [ ] **Step 9: Commit**

```bash
git add ui/main_window.py
git commit -m "feat: wire FunASRWorker into main_window - Push-to-Talk, partial subtitles, async detection_ready"
```

---

## Task 5: Run full test suite and verify

- [ ] **Step 1: Run all tests**

```bash
cd /home/ubuntu/NGW/intern/multimodal_detector
python -m pytest tests/ -v --ignore=tests/intent_engine 2>&1 | tail -30
```

Expected: all existing tests still pass, new tests pass

- [ ] **Step 2: Verify no Whisper imports remain**

```bash
grep -r "import whisper\|openai.whisper\|openai-whisper" detectors/ workers/ ui/
```

Expected: no output

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: verify FunASR migration complete - all tests pass, Whisper removed"
```
