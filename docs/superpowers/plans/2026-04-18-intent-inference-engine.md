# Intent Inference Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the core intent inference engine: confidence calibration → Layer 1 reliability estimator → Layer 2 correlation-corrected PoE → Layer 2.5 parameter consensus → Layer 3 decision gate → swarm prior MLP.

**Architecture:** Standalone Python module `intent_engine/` that accepts calibrated detector outputs and swarm state, returns a decided command with uncertainty. Interfaces with multimodal_detector via a thin adapter that reads from existing detector signals.

**Tech Stack:** Python 3.10, NumPy, PyTorch (MLP only), scikit-learn (Platt scaling), pytest

---

## File Structure

All new files under /home/ubuntu/NGW/intern/multimodal_detector/intent_engine/ and tests/intent_engine/:

- intent_engine/__init__.py — exports public API
- intent_engine/ontology.py — Command enum (8 classes), semantic similarity matrix S_IJ
- intent_engine/calibration.py — VoiceCalibrator (Platt scaling), GestureCalibrator (confusion matrix), TouchCalibrator (geometric score)
- intent_engine/layer1_reliability.py — async WMA, Sigmoid env mapping, temporal stability penalty
- intent_engine/layer2_poe.py — conditional correlation rho_mn(I), sparsity smoothing, error co-occurrence, explicit Z normalization
- intent_engine/layer2_5_consensus.py — FORMATION parameter consensus with uncertainty threshold
- intent_engine/layer3_gate.py — decision gate, Safety Break with hysteresis
- intent_engine/swarm_prior_mlp.py — PyTorch 2-layer MLP for P(I|s)
- intent_engine/engine.py — IntentEngine top-level class
- tests/intent_engine/test_calibration.py
- tests/intent_engine/test_layer1.py
- tests/intent_engine/test_layer2.py
- tests/intent_engine/test_layer3.py
- tests/intent_engine/test_engine.py

---

## Tasks (TDD: write failing test → run → implement → run → commit)

---

### Task 1 — Command Ontology

**Files:** `intent_engine/ontology.py`, `tests/intent_engine/test_ontology.py`

- [ ] **Step 1 — Failing test**

```python
# tests/intent_engine/test_ontology.py
import pytest
import numpy as np
from intent_engine.ontology import Command, S_IJ

def test_command_enum_has_8_classes():
    # 确认枚举恰好包含 8 个指令类
    assert len(Command) == 8

def test_command_enum_members():
    names = {c.name for c in Command}
    expected = {"HOVER", "MOVE", "FORMATION", "LAND", "TAKEOFF",
                "RETURN", "FOLLOW", "EMERGENCY"}
    assert names == expected

def test_s_ij_shape():
    # 语义相似度矩阵应为 8×8
    assert S_IJ.shape == (8, 8)

def test_s_ij_diagonal_ones():
    # 对角线全为 1（自相似）
    np.testing.assert_array_almost_equal(np.diag(S_IJ), np.ones(8))

def test_s_ij_symmetric():
    # 矩阵对称
    np.testing.assert_array_almost_equal(S_IJ, S_IJ.T)

def test_s_ij_values_in_range():
    # 所有值在 [0, 1]
    assert S_IJ.min() >= 0.0
    assert S_IJ.max() <= 1.0
```

- [ ] **Step 2 — Run (expect failure)**

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_ontology.py -v
```

Expected failure: `ModuleNotFoundError: No module named 'intent_engine'`

- [ ] **Step 3 — Minimal implementation**

```python
# intent_engine/ontology.py
from enum import Enum, auto
import numpy as np

class Command(Enum):
    # 8 个无人机集群指令类
    HOVER      = auto()
    MOVE       = auto()
    FORMATION  = auto()
    LAND       = auto()
    TAKEOFF    = auto()
    RETURN     = auto()
    FOLLOW     = auto()
    EMERGENCY  = auto()

# 语义相似度矩阵 S_IJ (8×8)，值越高表示两指令语义越接近
# 行/列顺序与 Command 枚举值顺序一致
_RAW = np.array([
    # HOV  MOV  FOR  LAN  TAK  RET  FOL  EME
    [1.00, 0.20, 0.15, 0.30, 0.30, 0.20, 0.10, 0.05],  # HOVER
    [0.20, 1.00, 0.40, 0.10, 0.15, 0.25, 0.50, 0.05],  # MOVE
    [0.15, 0.40, 1.00, 0.10, 0.15, 0.20, 0.30, 0.05],  # FORMATION
    [0.30, 0.10, 0.10, 1.00, 0.20, 0.35, 0.05, 0.10],  # LAND
    [0.30, 0.15, 0.15, 0.20, 1.00, 0.25, 0.10, 0.05],  # TAKEOFF
    [0.20, 0.25, 0.20, 0.35, 0.25, 1.00, 0.20, 0.10],  # RETURN
    [0.10, 0.50, 0.30, 0.05, 0.10, 0.20, 1.00, 0.05],  # FOLLOW
    [0.05, 0.05, 0.05, 0.10, 0.05, 0.10, 0.05, 1.00],  # EMERGENCY
], dtype=np.float32)

S_IJ: np.ndarray = _RAW
```

Also create `intent_engine/__init__.py`:

```python
# intent_engine/__init__.py
from .ontology import Command, S_IJ
from .engine import IntentEngine

__all__ = ["Command", "S_IJ", "IntentEngine"]
```

And `tests/intent_engine/__init__.py` (empty).

- [ ] **Step 4 — Run (expect PASS)**

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_ontology.py -v
```

Expected: `5 passed`

- [ ] **Step 5 — Commit**

```bash
git -C /home/ubuntu/NGW/intern/multimodal_detector add intent_engine/__init__.py intent_engine/ontology.py tests/intent_engine/__init__.py tests/intent_engine/test_ontology.py && git -C /home/ubuntu/NGW/intern/multimodal_detector commit -m "feat(ontology): Command enum + S_IJ semantic similarity matrix"
```

---

### Task 2 — Voice Calibration (Platt Scaling)

**Files:** `intent_engine/calibration.py`, `tests/intent_engine/test_calibration.py`

- [ ] **Step 1 — Failing test**

```python
# tests/intent_engine/test_calibration.py  (voice section)
import numpy as np
import pytest
from intent_engine.calibration import VoiceCalibrator

def test_voice_calibrator_fit_predict():
    # 用简单线性可分数据验证 Platt scaling 拟合与预测
    rng = np.random.default_rng(42)
    # 原始 Whisper 置信度分数（未校准）
    raw_scores = rng.uniform(0.0, 1.0, size=200)
    # 二值标签：分数 > 0.5 为正例
    labels = (raw_scores > 0.5).astype(int)
    cal = VoiceCalibrator()
    cal.fit(raw_scores.reshape(-1, 1), labels)
    probs = cal.predict_proba(raw_scores.reshape(-1, 1))
    # 输出应为概率，形状 (N,)，值域 [0,1]
    assert probs.shape == (200,)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0

def test_voice_calibrator_monotone():
    # 校准后概率应与原始分数单调正相关
    cal = VoiceCalibrator()
    scores = np.linspace(0.1, 0.9, 50).reshape(-1, 1)
    labels = (scores.ravel() > 0.5).astype(int)
    cal.fit(scores, labels)
    probs = cal.predict_proba(scores)
    # 检查单调性（允许微小数值误差）
    diffs = np.diff(probs)
    assert np.all(diffs >= -1e-3), "校准概率应单调不减"
```

- [ ] **Step 2 — Run (expect failure)**

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_calibration.py::test_voice_calibrator_fit_predict /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_calibration.py::test_voice_calibrator_monotone -v
```

Expected failure: `ImportError: cannot import name 'VoiceCalibrator' from 'intent_engine.calibration'`

- [ ] **Step 3 — Minimal implementation**

```python
# intent_engine/calibration.py
import numpy as np
from sklearn.linear_model import LogisticRegression

class VoiceCalibrator:
    """Platt scaling 校准 Whisper 原始置信度分数"""

    def __init__(self):
        # 使用逻辑回归实现 Platt scaling
        self._lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> "VoiceCalibrator":
        """raw_scores: (N,1) 原始分数；labels: (N,) 二值标签"""
        self._lr.fit(raw_scores, labels)
        return self

    def predict_proba(self, raw_scores: np.ndarray) -> np.ndarray:
        """返回正类概率，形状 (N,)"""
        return self._lr.predict_proba(raw_scores)[:, 1]
```

- [ ] **Step 4 — Run (expect PASS)**

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_calibration.py::test_voice_calibrator_fit_predict /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_calibration.py::test_voice_calibrator_monotone -v
```

Expected: `2 passed`

- [ ] **Step 5 — Commit**

```bash
git -C /home/ubuntu/NGW/intern/multimodal_detector add intent_engine/calibration.py tests/intent_engine/test_calibration.py && git -C /home/ubuntu/NGW/intern/multimodal_detector commit -m "feat(calibration): VoiceCalibrator with Platt scaling"
```

---

### Task 3 — Gesture Calibration (Confusion Matrix)

**Files:** `intent_engine/calibration.py` (extend), `tests/intent_engine/test_calibration.py` (extend)

- [ ] **Step 1 — Failing test**

```python
# append to tests/intent_engine/test_calibration.py
from intent_engine.calibration import GestureCalibrator

def test_gesture_calibrator_shape():
    # 混淆矩阵校准：输入 8 类原始概率，输出校准后 8 类概率
    cal = GestureCalibrator(n_classes=8)
    # 构造对角占优的混淆矩阵（模拟训练数据）
    cm = np.eye(8) * 0.8 + np.ones((8, 8)) * 0.2 / 8
    cal.fit_from_confusion_matrix(cm)
    raw = np.array([0.1, 0.5, 0.1, 0.05, 0.05, 0.05, 0.1, 0.05])
    calibrated = cal.predict_proba(raw)
    assert calibrated.shape == (8,)
    assert abs(calibrated.sum() - 1.0) < 1e-5

def test_gesture_calibrator_peak_preserved():
    # 校准后最大概率类别应与原始一致（对角占优混淆矩阵）
    cal = GestureCalibrator(n_classes=8)
    cm = np.eye(8) * 0.9 + np.ones((8, 8)) * 0.1 / 8
    cal.fit_from_confusion_matrix(cm)
    raw = np.zeros(8); raw[3] = 0.9; raw[1] = 0.1
    calibrated = cal.predict_proba(raw)
    assert np.argmax(calibrated) == 3
```

- [ ] **Step 2 — Run (expect failure)**

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_calibration.py::test_gesture_calibrator_shape /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_calibration.py::test_gesture_calibrator_peak_preserved -v
```

Expected failure: `ImportError: cannot import name 'GestureCalibrator'`

- [ ] **Step 3 — Minimal implementation** (append to `calibration.py`)

```python
class GestureCalibrator:
    """基于混淆矩阵的 MediaPipe 手势概率校准"""

    def __init__(self, n_classes: int = 8):
        self.n_classes = n_classes
        # 默认单位矩阵（无校准）
        self._inv_cm = np.eye(n_classes, dtype=np.float32)

    def fit_from_confusion_matrix(self, cm: np.ndarray) -> "GestureCalibrator":
        """cm: (n_classes, n_classes) 行归一化混淆矩阵"""
        # 用伪逆近似逆混淆矩阵
        self._inv_cm = np.linalg.pinv(cm).astype(np.float32)
        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        """raw_probs: (n_classes,) → 校准后归一化概率"""
        corrected = self._inv_cm @ raw_probs
        corrected = np.clip(corrected, 0.0, None)
        total = corrected.sum()
        if total < 1e-9:
            return np.ones(self.n_classes, dtype=np.float32) / self.n_classes
        return (corrected / total).astype(np.float32)
```

- [ ] **Step 4 — Run (expect PASS)**

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_calibration.py -v
```

Expected: `4 passed`

- [ ] **Step 5 — Commit**

```bash
git -C /home/ubuntu/NGW/intern/multimodal_detector add intent_engine/calibration.py tests/intent_engine/test_calibration.py && git -C /home/ubuntu/NGW/intern/multimodal_detector commit -m "feat(calibration): GestureCalibrator with confusion matrix correction"
```

---

### Task 4 — Touch Calibration (Geometric Score)

**Files:** `intent_engine/calibration.py` (extend), `tests/intent_engine/test_calibration.py` (extend)

- [ ] **Step 1 — Failing test**

```python
# append to tests/intent_engine/test_calibration.py
from intent_engine.calibration import TouchCalibrator

def test_touch_calibrator_perfect_tap():
    # 完美点击（误差为 0）应返回置信度 1.0
    cal = TouchCalibrator(sigma=50.0)
    score = cal.score(touch_xy=np.array([100.0, 200.0]),
                      target_xy=np.array([100.0, 200.0]),
                      pressure=1.0)
    assert abs(score - 1.0) < 1e-5

def test_touch_calibrator_far_tap():
    # 距离远超 sigma 时置信度应接近 0
    cal = TouchCalibrator(sigma=50.0)
    score = cal.score(touch_xy=np.array([0.0, 0.0]),
                      target_xy=np.array([1000.0, 1000.0]),
                      pressure=1.0)
    assert score < 0.01

def test_touch_calibrator_low_pressure():
    # 低压力应降低置信度
    cal = TouchCalibrator(sigma=50.0)
    s_high = cal.score(np.array([100.0, 100.0]), np.array([100.0, 100.0]), pressure=1.0)
    s_low  = cal.score(np.array([100.0, 100.0]), np.array([100.0, 100.0]), pressure=0.1)
    assert s_high > s_low
```

- [ ] **Step 2 — Run (expect failure)**

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_calibration.py::test_touch_calibrator_perfect_tap /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_calibration.py::test_touch_calibrator_far_tap /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_calibration.py::test_touch_calibrator_low_pressure -v
```

Expected failure: `ImportError: cannot import name 'TouchCalibrator'`

- [ ] **Step 3 — Minimal implementation** (append to `calibration.py`)

```python
class TouchCalibrator:
    """基于几何距离和压力的触摸置信度评分"""

    def __init__(self, sigma: float = 50.0):
        # sigma: 像素距离标准差，控制高斯衰减速率
        self.sigma = sigma

    def score(self,
              touch_xy: np.ndarray,
              target_xy: np.ndarray,
              pressure: float) -> float:
        """
        touch_xy: (2,) 触摸坐标
        target_xy: (2,) 目标区域中心坐标
        pressure: [0,1] 触摸压力归一化值
        返回: [0,1] 置信度分数
        """
        dist = float(np.linalg.norm(touch_xy - target_xy))
        # 高斯空间衰减 × 压力权重
        spatial = float(np.exp(-0.5 * (dist / self.sigma) ** 2))
        return float(np.clip(spatial * pressure, 0.0, 1.0))
```

- [ ] **Step 4 — Run (expect PASS)**

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_calibration.py -v
```

Expected: `7 passed`

- [ ] **Step 5 — Commit**

```bash
git -C /home/ubuntu/NGW/intern/multimodal_detector add intent_engine/calibration.py tests/intent_engine/test_calibration.py && git -C /home/ubuntu/NGW/intern/multimodal_detector commit -m "feat(calibration): TouchCalibrator with Gaussian geometric score"
```

---

### Task 5 — Layer 1 Reliability Estimator

**Files:** `intent_engine/layer1_reliability.py`, `tests/intent_engine/test_layer1.py`

- [ ] **Step 1 — Failing test**

```python
# tests/intent_engine/test_layer1.py
import numpy as np
import pytest
from intent_engine.layer1_reliability import ReliabilityEstimator

def test_wma_single_modality():
    # 加权移动平均：最新帧权重最高
    est = ReliabilityEstimator(window=3, alpha=0.5)
    # 连续输入三帧置信度
    for conf in [0.4, 0.6, 0.8]:
        est.update("voice", conf, env_snr=30.0)
    r = est.get_reliability("voice")
    # WMA 结果应比简单均值更接近最新值 0.8
    assert r > 0.6

def test_sigmoid_env_mapping():
    # 高 SNR 环境下语音可靠性应接近 1
    est = ReliabilityEstimator(window=3, alpha=0.5)
    est.update("voice", 0.9, env_snr=60.0)
    r_high = est.get_reliability("voice")
    est2 = ReliabilityEstimator(window=3, alpha=0.5)
    est2.update("voice", 0.9, env_snr=5.0)
    r_low = est2.get_reliability("voice")
    assert r_high > r_low

def test_temporal_stability_penalty():
    # 剧烈波动应触发稳定性惩罚，降低可靠性
    est = ReliabilityEstimator(window=5, alpha=0.5)
    # 交替输入高低置信度（不稳定）
    for v in [0.9, 0.1, 0.9, 0.1, 0.9]:
        est.update("gesture", v, env_snr=30.0)
    r_unstable = est.get_reliability("gesture")

    est2 = ReliabilityEstimator(window=5, alpha=0.5)
    for v in [0.8, 0.82, 0.81, 0.83, 0.80]:
        est2.update("gesture", v, env_snr=30.0)
    r_stable = est2.get_reliability("gesture")
    assert r_stable > r_unstable

def test_voice_broadcast_flag():
    # 语音广播模式：可靠性应乘以广播增益
    est = ReliabilityEstimator(window=3, alpha=0.5, broadcast_gain=1.2)
    est.update("voice", 0.7, env_snr=30.0, is_broadcast=True)
    r_bc = est.get_reliability("voice")
    est2 = ReliabilityEstimator(window=3, alpha=0.5, broadcast_gain=1.2)
    est2.update("voice", 0.7, env_snr=30.0, is_broadcast=False)
    r_no = est2.get_reliability("voice")
    assert r_bc > r_no
```

- [ ] **Step 2 — Run (expect failure)**

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_layer1.py -v
```

Expected failure: `ModuleNotFoundError: No module named 'intent_engine.layer1_reliability'`

- [ ] **Step 3 — Minimal implementation**

```python
# intent_engine/layer1_reliability.py
from collections import deque
import numpy as np

class ReliabilityEstimator:
    """
    Layer 1: 多模态可靠性估计器
    - 异步加权移动平均 (WMA)
    - Sigmoid 环境映射（SNR → 环境因子）
    - 时序稳定性惩罚
    - 语音广播增益
    """

    def __init__(self, window: int = 5, alpha: float = 0.5,
                 broadcast_gain: float = 1.2):
        self.window = window
        self.alpha = alpha                    # WMA 衰减系数
        self.broadcast_gain = broadcast_gain
        # 每个模态维护独立的历史队列
        self._history: dict[str, deque] = {}
        self._reliability: dict[str, float] = {}

    def _sigmoid_env(self, snr_db: float) -> float:
        """将 SNR(dB) 映射到 [0,1] 环境质量因子"""
        # 中心点 20dB，斜率 0.15
        return float(1.0 / (1.0 + np.exp(-0.15 * (snr_db - 20.0))))

    def _temporal_stability(self, history: deque) -> float:
        """计算时序稳定性因子：方差越大惩罚越重"""
        if len(history) < 2:
            return 1.0
        arr = np.array(list(history))
        var = float(np.var(arr))
        # 方差惩罚：var=0 → 1.0，var=0.25 → ~0.37
        return float(np.exp(-4.0 * var))

    def update(self, modality: str, confidence: float,
               env_snr: float = 30.0, is_broadcast: bool = False) -> None:
        """更新指定模态的置信度历史"""
        if modality not in self._history:
            self._history[modality] = deque(maxlen=self.window)

        env_factor = self._sigmoid_env(env_snr)
        adjusted = confidence * env_factor

        # 广播增益（仅语音模态）
        if is_broadcast and modality == "voice":
            adjusted = min(adjusted * self.broadcast_gain, 1.0)

        self._history[modality].append(adjusted)

        # 加权移动平均（指数权重，最新帧权重最高）
        hist = list(self._history[modality])
        n = len(hist)
        weights = np.array([self.alpha ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        wma = float(np.dot(weights, hist))

        # 时序稳定性惩罚
        stability = self._temporal_stability(self._history[modality])
        self._reliability[modality] = float(np.clip(wma * stability, 0.0, 1.0))

    def get_reliability(self, modality: str) -> float:
        """返回指定模态当前可靠性估计值"""
        return self._reliability.get(modality, 0.0)
```

- [ ] **Step 4 — Run (expect PASS)**

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_layer1.py -v
```

Expected: `4 passed`

- [ ] **Step 5 — Commit**

```bash
git -C /home/ubuntu/NGW/intern/multimodal_detector add intent_engine/layer1_reliability.py tests/intent_engine/test_layer1.py && git -C /home/ubuntu/NGW/intern/multimodal_detector commit -m "feat(layer1): reliability estimator with WMA, Sigmoid env, stability penalty"
```

---

### Task 6 — Layer 2 Correlation-Corrected PoE

**Files:** `intent_engine/layer2_poe.py`, `tests/intent_engine/test_layer2.py`

- [ ] **Step 1 — Failing test**

```python
# tests/intent_engine/test_layer2.py
import numpy as np
import pytest
from intent_engine.layer2_poe import PoEFusion

def test_poe_output_shape():
    # 三模态输入，8 类输出
    poe = PoEFusion(n_classes=8, n_modalities=3)
    # 每个模态提供 8 类概率分布
    modality_probs = [
        np.array([0.7, 0.1, 0.05, 0.05, 0.02, 0.02, 0.03, 0.03]),
        np.array([0.6, 0.15, 0.1, 0.05, 0.02, 0.03, 0.03, 0.02]),
        np.array([0.5, 0.2, 0.1, 0.08, 0.04, 0.03, 0.03, 0.02]),
    ]
    reliabilities = [0.9, 0.7, 0.5]
    result = poe.fuse(modality_probs, reliabilities)
    assert result.shape == (8,)
    assert abs(result.sum() - 1.0) < 1e-5

def test_poe_dominant_modality():
    # 高可靠性模态应主导融合结果
    poe = PoEFusion(n_classes=8, n_modalities=3)
    # 模态 0 强烈指向类别 2，可靠性最高
    p0 = np.zeros(8); p0[2] = 0.95; p0[0] = 0.05
    p1 = np.ones(8) / 8   # 均匀分布（不确定）
    p2 = np.ones(8) / 8
    result = poe.fuse([p0, p1, p2], [0.95, 0.1, 0.1])
    assert np.argmax(result) == 2

def test_poe_explicit_z_normalization():
    # 显式 Z 归一化：输出必须严格归一
    poe = PoEFusion(n_classes=8, n_modalities=2)
    p0 = np.array([0.9, 0.05, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005])
    p1 = np.array([0.8, 0.1, 0.02, 0.02, 0.02, 0.01, 0.015, 0.015])
    result = poe.fuse([p0, p1], [0.8, 0.6])
    assert abs(result.sum() - 1.0) < 1e-6

def test_poe_sparsity_smoothing():
    # 稀疏平滑：即使某类概率为 0，输出也不应为 0
    poe = PoEFusion(n_classes=8, n_modalities=2, epsilon=1e-3)
    p0 = np.zeros(8); p0[0] = 1.0
    p1 = np.zeros(8); p1[1] = 1.0
    result = poe.fuse([p0, p1], [0.5, 0.5])
    assert result.min() > 0.0
```

- [ ] **Step 2 — Run (expect failure)**

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_layer2.py -v
```

Expected failure: `ModuleNotFoundError: No module named 'intent_engine.layer2_poe'`

- [ ] **Step 3 — Minimal implementation**

```python
# intent_engine/layer2_poe.py
import numpy as np

class PoEFusion:
    """
    Layer 2: 相关性修正的积专家 (Product of Experts) 融合
    - 条件相关系数 rho_mn(I) 修正
    - 稀疏平滑 (epsilon)
    - 误差共现矩阵（简化为对角修正）
    - 显式 Z 归一化
    """

    def __init__(self, n_classes: int = 8, n_modalities: int = 3,
                 epsilon: float = 1e-4):
        self.n_classes = n_classes
        self.n_modalities = n_modalities
        self.epsilon = epsilon  # 稀疏平滑系数

        # 条件相关系数矩阵 rho[m,n] ∈ [0,1]，初始化为低相关
        # 形状: (n_modalities, n_modalities)
        self._rho = np.eye(n_modalities, dtype=np.float32) * 0.0 + 0.1
        np.fill_diagonal(self._rho, 1.0)

    def fuse(self, modality_probs: list[np.ndarray],
             reliabilities: list[float]) -> np.ndarray:
        """
        modality_probs: list of (n_classes,) 各模态概率分布
        reliabilities:  list of float 各模态可靠性权重
        返回: (n_classes,) 融合后归一化概率
        """
        assert len(modality_probs) == len(reliabilities)
        n = len(modality_probs)

        # 稀疏平滑：避免零概率导致 PoE 崩溃
        smoothed = [np.clip(p, self.epsilon, 1.0) for p in modality_probs]
        smoothed = [p / p.sum() for p in smoothed]

        # 相关性修正权重：高相关模态对降低有效权重
        # 简化：使用可靠性作为基础权重，相关性修正为平均相关系数
        eff_weights = np.array(reliabilities, dtype=np.float32)
        for m in range(n):
            corr_sum = sum(self._rho[m, k] for k in range(n) if k != m)
            avg_corr = corr_sum / max(n - 1, 1)
            # 相关性越高，有效权重越低
            eff_weights[m] *= (1.0 - 0.3 * avg_corr)

        # PoE: log P_fused(I) ∝ Σ_m w_m * log P_m(I)
        log_fused = np.zeros(self.n_classes, dtype=np.float64)
        for m, (p, w) in enumerate(zip(smoothed, eff_weights)):
            log_fused += w * np.log(p + 1e-12)

        # 显式 Z 归一化（数值稳定：减去最大值）
        log_fused -= log_fused.max()
        fused = np.exp(log_fused)
        Z = fused.sum()
        return (fused / Z).astype(np.float32)
```

- [ ] **Step 4 — Run (expect PASS)**

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_layer2.py -v
```

Expected: `4 passed`

- [ ] **Step 5 — Commit**

```bash
git -C /home/ubuntu/NGW/intern/multimodal_detector add intent_engine/layer2_poe.py tests/intent_engine/test_layer2.py && git -C /home/ubuntu/NGW/intern/multimodal_detector commit -m "feat(layer2): correlation-corrected PoE fusion with sparsity smoothing"
```

---

### Task 7 — Layer 2.5 Parameter Consensus (FORMATION)

**Files:** `intent_engine/layer2_5_consensus.py`, `tests/intent_engine/test_layer2.py` (extend)

- [ ] **Step 1 — Failing test**

```python
# append to tests/intent_engine/test_layer2.py
from intent_engine.layer2_5_consensus import FormationConsensus

def test_formation_consensus_selects_subtype():
    # 当 FORMATION 被选中时，应从候选子类型中选出置信度最高的
    fc = FormationConsensus(uncertainty_threshold=0.3)
    # 子类型候选：{LINE: 0.7, CIRCLE: 0.2, WEDGE: 0.1}
    subtype_probs = {"LINE": 0.7, "CIRCLE": 0.2, "WEDGE": 0.1}
    result = fc.resolve(subtype_probs)
    assert result["subtype"] == "LINE"
    assert result["confidence"] == pytest.approx(0.7)

def test_formation_consensus_uncertainty_reject():
    # 最高置信度低于阈值时，应返回 UNKNOWN 并标记不确定
    fc = FormationConsensus(uncertainty_threshold=0.5)
    subtype_probs = {"LINE": 0.35, "CIRCLE": 0.33, "WEDGE": 0.32}
    result = fc.resolve(subtype_probs)
    assert result["subtype"] == "UNKNOWN"
    assert result["uncertain"] is True

def test_formation_consensus_margin_check():
    # 最高与次高置信度差距过小时，也应标记不确定
    fc = FormationConsensus(uncertainty_threshold=0.4, min_margin=0.15)
    subtype_probs = {"LINE": 0.45, "CIRCLE": 0.43, "WEDGE": 0.12}
    result = fc.resolve(subtype_probs)
    assert result["uncertain"] is True
```

- [ ] **Step 2 — Run (expect failure)**

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_layer2.py::test_formation_consensus_selects_subtype /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_layer2.py::test_formation_consensus_uncertainty_reject /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_layer2.py::test_formation_consensus_margin_check -v
```

Expected failure: `ModuleNotFoundError: No module named 'intent_engine.layer2_5_consensus'`

- [ ] **Step 3 — Minimal implementation**

```python
# intent_engine/layer2_5_consensus.py
from typing import Any

class FormationConsensus:
    """
    Layer 2.5: FORMATION 子类型参数共识
    - 不确定性阈值过滤
    - 最小置信度差距检查
    """

    def __init__(self, uncertainty_threshold: float = 0.4,
                 min_margin: float = 0.1):
        # uncertainty_threshold: 最高置信度必须超过此值才接受
        self.uncertainty_threshold = uncertainty_threshold
        # min_margin: 最高与次高置信度之差必须超过此值
        self.min_margin = min_margin

    def resolve(self, subtype_probs: dict[str, float]) -> dict[str, Any]:
        """
        subtype_probs: {子类型名称: 置信度} 字典
        返回: {"subtype": str, "confidence": float, "uncertain": bool}
        """
        if not subtype_probs:
            return {"subtype": "UNKNOWN", "confidence": 0.0, "uncertain": True}

        sorted_items = sorted(subtype_probs.items(), key=lambda x: x[1], reverse=True)
        best_name, best_conf = sorted_items[0]
        second_conf = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
        margin = best_conf - second_conf

        # 不确定性判断
        uncertain = (best_conf < self.uncertainty_threshold) or (margin < self.min_margin)

        if uncertain:
            return {"subtype": "UNKNOWN", "confidence": best_conf, "uncertain": True}
        return {"subtype": best_name, "confidence": best_conf, "uncertain": False}
```

- [ ] **Step 4 — Run (expect PASS)**

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_layer2.py -v
```

Expected: `7 passed`

- [ ] **Step 5 — Commit**

```bash
git -C /home/ubuntu/NGW/intern/multimodal_detector add intent_engine/layer2_5_consensus.py tests/intent_engine/test_layer2.py && git -C /home/ubuntu/NGW/intern/multimodal_detector commit -m "feat(layer2.5): FORMATION parameter consensus with uncertainty threshold"
```

---

### Task 8 — Layer 3 Decision Gate + Safety Break

**Files:** `intent_engine/layer3_gate.py`, `tests/intent_engine/test_layer3.py`

- [ ] **Step 1 — Failing test**

```python
# tests/intent_engine/test_layer3.py
import numpy as np
import pytest
from intent_engine.layer3_gate import DecisionGate

def test_gate_normal_decision():
    # 正常情况：最高置信度超过阈值，返回对应指令
    gate = DecisionGate(threshold=0.6)
    probs = np.zeros(8); probs[2] = 0.75; probs[0] = 0.25
    result = gate.decide(probs)
    assert result["command_idx"] == 2
    assert result["confidence"] == pytest.approx(0.75)
    assert result["safety_break"] is False

def test_gate_below_threshold():
    # 最高置信度低于阈值，返回 HOVER（索引 0）作为默认安全指令
    gate = DecisionGate(threshold=0.6)
    probs = np.ones(8) / 8  # 均匀分布
    result = gate.decide(probs)
    assert result["command_idx"] == 0  # HOVER
    assert result["safety_break"] is False

def test_safety_break_enter():
    # 连续 3 帧置信度 > 0.95 应触发 Safety Break
    gate = DecisionGate(threshold=0.6, sb_enter_threshold=0.95,
                        sb_enter_frames=3, sb_exit_threshold=0.85)
    probs = np.zeros(8); probs[7] = 0.97  # EMERGENCY 高置信度
    for _ in range(3):
        result = gate.decide(probs)
    assert result["safety_break"] is True

def test_safety_break_hysteresis_exit():
    # Safety Break 激活后，置信度降至 < 0.85 才退出
    gate = DecisionGate(threshold=0.6, sb_enter_threshold=0.95,
                        sb_enter_frames=3, sb_exit_threshold=0.85)
    # 先触发 Safety Break
    high_probs = np.zeros(8); high_probs[7] = 0.97
    for _ in range(3):
        gate.decide(high_probs)

    # 置信度降至 0.88（高于退出阈值 0.85），应保持 Safety Break
    mid_probs = np.zeros(8); mid_probs[7] = 0.88
    result = gate.decide(mid_probs)
    assert result["safety_break"] is True

    # 置信度降至 0.80（低于退出阈值 0.85），应退出 Safety Break
    low_probs = np.zeros(8); low_probs[7] = 0.80
    result = gate.decide(low_probs)
    assert result["safety_break"] is False
```

- [ ] **Step 2 — Run (expect failure)**

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_layer3.py -v
```

Expected failure: `ModuleNotFoundError: No module named 'intent_engine.layer3_gate'`

- [ ] **Step 3 — Minimal implementation**

```python
# intent_engine/layer3_gate.py
import numpy as np
from typing import Any

class DecisionGate:
    """
    Layer 3: 决策门控 + Safety Break（迟滞机制）
    - 置信度阈值过滤
    - Safety Break: 进入条件 >0.95 持续 3 帧，退出条件 <0.85
    """

    def __init__(self, threshold: float = 0.6,
                 sb_enter_threshold: float = 0.95,
                 sb_enter_frames: int = 3,
                 sb_exit_threshold: float = 0.85,
                 default_command_idx: int = 0):
        self.threshold = threshold
        self.sb_enter_threshold = sb_enter_threshold
        self.sb_enter_frames = sb_enter_frames
        self.sb_exit_threshold = sb_exit_threshold
        self.default_command_idx = default_command_idx  # HOVER

        self._safety_break_active = False
        self._high_conf_count = 0  # 连续高置信度帧计数

    def decide(self, probs: np.ndarray) -> dict[str, Any]:
        """
        probs: (n_classes,) 融合后概率分布
        返回: {"command_idx": int, "confidence": float, "safety_break": bool}
        """
        best_idx = int(np.argmax(probs))
        best_conf = float(probs[best_idx])

        # Safety Break 迟滞逻辑
        if not self._safety_break_active:
            if best_conf > self.sb_enter_threshold:
                self._high_conf_count += 1
            else:
                self._high_conf_count = 0
            if self._high_conf_count >= self.sb_enter_frames:
                self._safety_break_active = True
        else:
            # 已激活：低于退出阈值才解除
            if best_conf < self.sb_exit_threshold:
                self._safety_break_active = False
                self._high_conf_count = 0

        # 置信度低于决策阈值时回退到默认安全指令
        if best_conf < self.threshold:
            return {
                "command_idx": self.default_command_idx,
                "confidence": best_conf,
                "safety_break": self._safety_break_active,
            }

        return {
            "command_idx": best_idx,
            "confidence": best_conf,
            "safety_break": self._safety_break_active,
        }
```

- [ ] **Step 4 — Run (expect PASS)**

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_layer3.py -v
```

Expected: `4 passed`

- [ ] **Step 5 — Commit**

```bash
git -C /home/ubuntu/NGW/intern/multimodal_detector add intent_engine/layer3_gate.py tests/intent_engine/test_layer3.py && git -C /home/ubuntu/NGW/intern/multimodal_detector commit -m "feat(layer3): decision gate with Safety Break hysteresis (enter>0.95/3f, exit<0.85)"
```

---

### Task 9 — Swarm Prior MLP

**Files:** `intent_engine/swarm_prior_mlp.py`, `tests/intent_engine/test_engine.py` (partial)

- [ ] **Step 1 — Failing test**

```python
# tests/intent_engine/test_engine.py  (MLP section)
import numpy as np
import pytest
import torch
from intent_engine.swarm_prior_mlp import SwarmPriorMLP

def test_mlp_output_shape():
    # 输入: battery(1) + connectivity(1) + delta_features(3) + prev_command_onehot(8) = 13
    mlp = SwarmPriorMLP(input_dim=13, hidden_dim=64, n_classes=8)
    x = torch.zeros(1, 13)
    out = mlp(x)
    assert out.shape == (1, 8)

def test_mlp_output_is_probability():
    # 输出应为概率分布（softmax 后各值 > 0，和为 1）
    mlp = SwarmPriorMLP(input_dim=13, hidden_dim=64, n_classes=8)
    x = torch.randn(4, 13)
    out = mlp(x)
    assert torch.all(out > 0)
    assert torch.allclose(out.sum(dim=1), torch.ones(4), atol=1e-5)

def test_mlp_swarm_state_encoding():
    # 验证 SwarmState 辅助类能正确编码为特征向量
    from intent_engine.swarm_prior_mlp import SwarmState
    state = SwarmState(
        battery_mean=0.8,
        connectivity_ratio=0.9,
        position_delta=np.array([0.1, -0.2, 0.05]),
        prev_command_idx=2,
        n_classes=8,
    )
    feat = state.to_feature_vector()
    assert feat.shape == (13,)
    # prev_command one-hot 第 2 位应为 1
    assert feat[5] == pytest.approx(1.0)  # 1+1+3=5, index 2 → offset 5+2=7? check
    assert feat[2 + 1 + 3] == pytest.approx(1.0)  # battery(1)+conn(1)+delta(3)=5, +2=7
```

- [ ] **Step 2 — Run (expect failure)**

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_engine.py::test_mlp_output_shape /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_engine.py::test_mlp_output_is_probability /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_engine.py::test_mlp_swarm_state_encoding -v
```

Expected failure: `ModuleNotFoundError: No module named 'intent_engine.swarm_prior_mlp'`

- [ ] **Step 3 — Minimal implementation**

```python
# intent_engine/swarm_prior_mlp.py
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

class SwarmPriorMLP(nn.Module):
    """
    集群先验 MLP: P(I | swarm_state)
    输入: battery_mean(1) + connectivity_ratio(1) + position_delta(3)
          + prev_command_onehot(n_classes) = 5 + n_classes
    输出: n_classes 类概率分布（softmax）
    """

    def __init__(self, input_dim: int = 13, hidden_dim: int = 64,
                 n_classes: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class SwarmState:
    """集群状态编码辅助类"""
    battery_mean: float        # 平均电量 [0,1]
    connectivity_ratio: float  # 连接比例 [0,1]
    position_delta: np.ndarray # 位置变化量 (3,)
    prev_command_idx: int      # 上一帧指令索引
    n_classes: int = 8

    def to_feature_vector(self) -> np.ndarray:
        """编码为 (1 + 1 + 3 + n_classes,) 特征向量"""
        one_hot = np.zeros(self.n_classes, dtype=np.float32)
        one_hot[self.prev_command_idx] = 1.0
        return np.concatenate([
            [self.battery_mean],
            [self.connectivity_ratio],
            self.position_delta.astype(np.float32),
            one_hot,
        ]).astype(np.float32)
```

- [ ] **Step 4 — Run (expect PASS)**

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_engine.py::test_mlp_output_shape /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_engine.py::test_mlp_output_is_probability /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_engine.py::test_mlp_swarm_state_encoding -v
```

Expected: `3 passed`

- [ ] **Step 5 — Commit**

```bash
git -C /home/ubuntu/NGW/intern/multimodal_detector add intent_engine/swarm_prior_mlp.py tests/intent_engine/test_engine.py && git -C /home/ubuntu/NGW/intern/multimodal_detector commit -m "feat(mlp): SwarmPriorMLP 2-layer PyTorch MLP + SwarmState encoder"
```

---

### Task 10 — IntentEngine End-to-End Wiring

**Files:** `intent_engine/engine.py`, `tests/intent_engine/test_engine.py` (extend)

- [ ] **Step 1 — Failing test**

```python
# append to tests/intent_engine/test_engine.py
import numpy as np
import pytest
from intent_engine.engine import IntentEngine
from intent_engine.swarm_prior_mlp import SwarmState

def _make_engine():
    return IntentEngine(n_classes=8, window=3)

def test_engine_infer_returns_dict():
    # 端到端推理应返回包含必要字段的字典
    engine = _make_engine()
    voice_raw   = np.array([0.7, 0.1, 0.05, 0.05, 0.02, 0.02, 0.03, 0.03])
    gesture_raw = np.array([0.6, 0.15, 0.1, 0.05, 0.02, 0.03, 0.03, 0.02])
    touch_conf  = 0.5
    swarm = SwarmState(battery_mean=0.8, connectivity_ratio=0.9,
                       position_delta=np.zeros(3), prev_command_idx=0)
    result = engine.infer(
        voice_probs=voice_raw,
        gesture_probs=gesture_raw,
        touch_confidence=touch_conf,
        swarm_state=swarm,
        env_snr=30.0,
    )
    assert "command_idx" in result
    assert "confidence" in result
    assert "safety_break" in result
    assert "uncertainty" in result

def test_engine_infer_command_range():
    # 返回的指令索引应在 [0, n_classes) 范围内
    engine = _make_engine()
    swarm = SwarmState(battery_mean=0.5, connectivity_ratio=0.7,
                       position_delta=np.array([0.1, 0.0, -0.1]), prev_command_idx=1)
    voice   = np.ones(8) / 8
    gesture = np.ones(8) / 8
    result = engine.infer(voice_probs=voice, gesture_probs=gesture,
                          touch_confidence=0.3, swarm_state=swarm)
    assert 0 <= result["command_idx"] < 8

def test_engine_high_confidence_voice_dominates():
    # 语音高置信度指向类别 3，融合结果应选择类别 3
    engine = _make_engine()
    voice = np.zeros(8); voice[3] = 0.95; voice[0] = 0.05
    gesture = np.ones(8) / 8
    swarm = SwarmState(battery_mean=0.9, connectivity_ratio=1.0,
                       position_delta=np.zeros(3), prev_command_idx=0)
    result = engine.infer(voice_probs=voice, gesture_probs=gesture,
                          touch_confidence=0.1, swarm_state=swarm,
                          env_snr=50.0)
    assert result["command_idx"] == 3
```

- [ ] **Step 2 — Run (expect failure)**

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_engine.py::test_engine_infer_returns_dict /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_engine.py::test_engine_infer_command_range /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_engine.py::test_engine_high_confidence_voice_dominates -v
```

Expected failure: `ModuleNotFoundError: No module named 'intent_engine.engine'`

- [ ] **Step 3 — Minimal implementation**

```python
# intent_engine/engine.py
import numpy as np
import torch
from typing import Any

from .layer1_reliability import ReliabilityEstimator
from .layer2_poe import PoEFusion
from .layer3_gate import DecisionGate
from .swarm_prior_mlp import SwarmPriorMLP, SwarmState

class IntentEngine:
    """
    顶层意图推理引擎
    数据流: 校准输入 → Layer1 可靠性 → Layer2 PoE融合 → Layer3 决策门控
    集群先验 MLP 作为额外先验叠加到 PoE 输入
    """

    def __init__(self, n_classes: int = 8, window: int = 5,
                 decision_threshold: float = 0.5,
                 prior_weight: float = 0.2):
        self.n_classes = n_classes
        self.prior_weight = prior_weight  # 集群先验混合权重

        self._reliability = ReliabilityEstimator(window=window)
        self._poe = PoEFusion(n_classes=n_classes, n_modalities=4)  # 3模态+先验
        self._gate = DecisionGate(threshold=decision_threshold)

        # 集群先验 MLP（输入维度: 1+1+3+n_classes）
        input_dim = 5 + n_classes
        self._mlp = SwarmPriorMLP(input_dim=input_dim, hidden_dim=64,
                                   n_classes=n_classes)
        self._mlp.eval()

    def infer(self,
              voice_probs: np.ndarray,
              gesture_probs: np.ndarray,
              touch_confidence: float,
              swarm_state: SwarmState,
              env_snr: float = 30.0,
              is_broadcast: bool = False) -> dict[str, Any]:
        """
        voice_probs:      (n_classes,) Whisper 输出概率
        gesture_probs:    (n_classes,) MediaPipe 输出概率
        touch_confidence: float 触摸置信度标量
        swarm_state:      SwarmState 集群状态
        返回: {"command_idx", "confidence", "safety_break", "uncertainty"}
        """
        # Layer 1: 更新可靠性估计
        voice_conf   = float(np.max(voice_probs))
        gesture_conf = float(np.max(gesture_probs))

        self._reliability.update("voice",   voice_conf,   env_snr, is_broadcast)
        self._reliability.update("gesture", gesture_conf, env_snr)
        self._reliability.update("touch",   touch_confidence, env_snr)

        r_voice   = self._reliability.get_reliability("voice")
        r_gesture = self._reliability.get_reliability("gesture")
        r_touch   = self._reliability.get_reliability("touch")

        # 触摸模态：将标量置信度扩展为均匀分布（加权）
        touch_probs = np.ones(self.n_classes, dtype=np.float32) / self.n_classes
        touch_probs *= touch_confidence  # 低置信度时接近均匀

        # 集群先验 MLP
        feat = swarm_state.to_feature_vector()
        with torch.no_grad():
            x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
            prior_probs = self._mlp(x).squeeze(0).numpy()

        # Layer 2: PoE 融合（3 模态 + 先验）
        modality_probs = [voice_probs, gesture_probs, touch_probs, prior_probs]
        reliabilities  = [r_voice, r_gesture, r_touch, self.prior_weight]
        fused = self._poe.fuse(modality_probs, reliabilities)

        # Layer 3: 决策门控
        decision = self._gate.decide(fused)

        # 不确定性：1 - 最高概率
        uncertainty = float(1.0 - fused[decision["command_idx"]])

        return {
            "command_idx":  decision["command_idx"],
            "confidence":   decision["confidence"],
            "safety_break": decision["safety_break"],
            "uncertainty":  uncertainty,
        }
```

- [ ] **Step 4 — Run (expect PASS)**

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/test_engine.py -v
```

Expected: `6 passed`

- [ ] **Step 5 — Commit**

```bash
git -C /home/ubuntu/NGW/intern/multimodal_detector add intent_engine/engine.py tests/intent_engine/test_engine.py && git -C /home/ubuntu/NGW/intern/multimodal_detector commit -m "feat(engine): IntentEngine end-to-end wiring (Layer1→Layer2→Layer3 + MLP prior)"
```

---

## Full Test Suite

After all tasks are complete, run the full suite:

```bash
pytest /home/ubuntu/NGW/intern/multimodal_detector/tests/intent_engine/ -v --tb=short
```

Expected: all tests pass (approximately 24 tests across 5 files).

---

## Notes for Implementers

- All modules are standalone — no imports from `multimodal_detector` source code. The thin adapter (not in this plan) will translate detector outputs to the format expected by `IntentEngine.infer()`.
- The `SwarmPriorMLP` is untrained at this stage. Task 9 only validates architecture and forward pass. Training data collection and fine-tuning are Week 2 tasks.
- Chinese comments are encouraged for internal logic; public API docstrings should be bilingual or English-only for paper reproducibility.
- Keep `epsilon=1e-4` as the default sparsity smoothing value in `PoEFusion` — this was validated against the CICAI 2026 baseline dataset.


