---
title: FunASR Paraformer-zh-streaming 迁移设计
date: 2026-04-21
status: approved
---

# FunASR 流式语音识别迁移设计

## 背景

当前 `VoiceDetector` 使用 OpenAI Whisper，采用"录完再识别"模式，延迟高且不支持实时字幕。迁移至 FunASR Paraformer-zh-streaming，实现逐块流式推理（每 450ms 输出中间结果），提升中文短指令识别效果。

## 目标

- 替换 Whisper，使用 FunASR Paraformer-zh-streaming
- 实现 Push-to-Talk 触发模式（按住说话，松开停止）
- 中间结果显示为 voice_overlay 实时字幕
- 最终结果触发指令解析（接口不变）
- 不保留 Whisper 回退逻辑

## 架构

```
main_window.py
    │  keyPressEvent(Space) / button.pressed  → start_recording()
    │  keyReleaseEvent(Space) / button.released → stop_recording()
    ▼
VoiceDetector（薄层，只管录音）
    │  sounddevice InputStream callback
    │  audio_chunk(np.ndarray) 信号
    │  volume_changed(float) 信号（不变）
    ▼
FunASRWorker(QThread)
    │  Queue[np.ndarray | None]（None = EOS）
    │  partial_result_ready(str) → voice_overlay 字幕
    │  detection_ready(DetectionResult) → main_window 指令解析
    ▼
FunASR Paraformer-zh-streaming
```

## 文件变更

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `workers/funasr_worker.py` | 新建 | FunASRWorker(QThread)，流式推理 |
| `detectors/voice_detector.py` | 修改 | 移除 Whisper，新增 audio_chunk 信号，stop_recording() 不返回结果 |
| `ui/main_window.py` | 修改 | 实例化 FunASRWorker，Push-to-Talk 按键绑定，连接 partial_result_ready |

**不变：** `voice_overlay.py`、`DetectionResult` 数据结构、`detection_ready` 信号接口

## 数据流

### 按住按键（开始录音）

1. `main_window` 调用 `voice_detector.start_recording()`
2. sounddevice InputStream callback 每块（约 450ms，7200 samples @ 16kHz）：
   - 发出 `volume_changed(float)` → overlay 波形
   - 发出 `audio_chunk(np.ndarray)` → FunASRWorker 队列
3. FunASRWorker 从队列取块，调用 `model.generate(chunk, is_final=False)`
4. 收到中间文本 → 发出 `partial_result_ready(str)` → overlay 实时字幕

### 松开按键（停止录音）

1. `main_window` 调用 `voice_detector.stop_recording()`
2. VoiceDetector 关闭 InputStream，向队列推入 `None`（EOS 哨兵）
3. FunASRWorker 收到 `None`，调用 `model.generate(is_final=True)`
4. 构造 `DetectionResult(modal_type="voice", command=text, ...)` 发出 `detection_ready`

### Push-to-Talk 按键绑定

- `Space` 键：`keyPressEvent`（忽略 `isAutoRepeat()`）→ 开始；`keyReleaseEvent` → 停止
- UI 按钮：`pressed` 信号 → 开始；`released` 信号 → 停止

## FunASRWorker 接口

```python
class FunASRWorker(QThread):
    partial_result_ready = Signal(str)        # 中间识别文本
    detection_ready = Signal(DetectionResult) # 最终结果
    error_occurred = Signal(str)              # 错误信息
    status_changed = Signal(str)              # 状态文本

    def initialize(self) -> bool: ...         # 加载 FunASR 模型
    def enqueue(self, chunk: np.ndarray): ... # 送入音频块
    def send_eos(self): ...                   # 推入 None 触发最终识别
    def release(self): ...                    # 停止线程，释放模型
```

## VoiceDetector 变更摘要

- 移除：`whisper` 依赖、`_transcribe()`、`_model`、`stop_recording()` 返回值
- 新增：`audio_chunk = Signal(np.ndarray)` 信号
- `stop_recording()`：关闭 stream，不再做推理，调用方负责发 EOS

## 错误处理

| 场景 | 处理方式 |
|------|----------|
| FunASR 加载失败 | `initialize()` 返回 False，发出 `error_occurred`，语音模态禁用 |
| 推理异常 | worker 捕获，发出 `error_occurred`，清空队列，状态重置 |
| 队列积压（>20块，约9秒） | 丢弃最旧块，记录 warning |
| 按键重入 / autoRepeat | `_recording_lock` 保护，`isAutoRepeat()` 直接忽略 |

## 测试策略

- **FunASRWorker 单元测试**：mock FunASR model，验证 EOS 触发 `detection_ready`，验证队列积压丢弃
- **VoiceDetector 单元测试**：mock sounddevice，验证 `audio_chunk` 信号，验证 `stop_recording()` 推入 None
- **集成测试**：预录 WAV（"起飞"、"降落"）注入队列，验证端到端 `DetectionResult.command`

## 依赖

```
pip install funasr
# 模型首次运行时自动下载 paraformer-zh-streaming
```

GPU 显存需求：约 1-2GB（RTX 3080 可用）
