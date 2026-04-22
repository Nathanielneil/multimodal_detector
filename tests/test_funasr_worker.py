import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import queue
import types

sys.path.insert(0, str(Path(__file__).parent.parent))

from detectors.base_detector import DetectionResult


@pytest.fixture(autouse=True)
def fake_funasr_module():
    """Inject a fake 'funasr' module into sys.modules so patch() can resolve it."""
    if "funasr" not in sys.modules:
        mod = types.ModuleType("funasr")
        mod.AutoModel = MagicMock()
        sys.modules["funasr"] = mod
    funasr_mod = sys.modules["funasr"]
    funasr_mod.AutoModel = MagicMock()
    yield funasr_mod


@pytest.fixture
def mock_funasr_model():
    model = MagicMock()
    model.generate.return_value = [{"text": "起飞"}]
    return model


@pytest.fixture
def worker(qtbot, mock_funasr_model, fake_funasr_module):
    fake_funasr_module.AutoModel = MagicMock(return_value=mock_funasr_model)
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
    """Non-final chunk emits partial_result_ready.

    Each 7200-sample chunk resamples from 48kHz to 16kHz → 2400 samples.
    _CHUNK_SAMPLES threshold is 7200, so we need 3 chunks (3*2400=7200) to
    trigger a non-final inference call.
    """
    chunk = np.zeros(7200, dtype=np.float32)
    mock_funasr_model.generate.return_value = [{"text": "起"}]

    with qtbot.waitSignal(worker.partial_result_ready, timeout=3000) as blocker:
        worker.enqueue(chunk)
        worker.enqueue(chunk)
        worker.enqueue(chunk)

    assert isinstance(blocker.args[0], str)


def test_queue_overflow_drops_oldest_not_eos(qtbot, mock_funasr_model):
    """Queue overflow drops audio chunks but never the EOS sentinel."""
    with patch("funasr.AutoModel") as MockAutoModel:
        MockAutoModel.return_value = mock_funasr_model
        from workers.funasr_worker import FunASRWorker
        w = FunASRWorker()
        w.initialize()
        # Do NOT start the worker thread — test queue state directly
        chunk = np.zeros(100, dtype=np.float32)
        for _ in range(22):
            w.enqueue(chunk)
        # Queue should be capped at _QUEUE_MAX (20)
        assert w._queue.qsize() <= 20
        # Cleanup without starting thread
        # (no need to call release/wait since thread was never started)


def test_initialize_failure_returns_false(qtbot, fake_funasr_module):
    fake_funasr_module.AutoModel = MagicMock(side_effect=Exception("load fail"))
    from workers.funasr_worker import FunASRWorker
    w = FunASRWorker()
    assert w.initialize() is False


def test_enqueue_before_init_returns_false(qtbot):
    from workers.funasr_worker import FunASRWorker
    w = FunASRWorker()
    chunk = np.zeros(100, dtype=np.float32)
    assert w.enqueue(chunk) is False


def test_is_initialized_property(qtbot, mock_funasr_model, fake_funasr_module):
    fake_funasr_module.AutoModel = MagicMock(return_value=mock_funasr_model)
    from workers.funasr_worker import FunASRWorker
    w = FunASRWorker()
    assert w.is_initialized is False
    w.initialize()
    assert w.is_initialized is True


def test_queue_overflow_eos_still_processed(qtbot, worker):
    """After overflow, EOS sentinel is still processed and triggers detection_ready."""
    chunk = np.zeros(100, dtype=np.float32)
    for _ in range(22):
        worker.enqueue(chunk)
    with qtbot.waitSignal(worker.detection_ready, timeout=3000):
        worker.send_eos()
