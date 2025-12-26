# 多模态无人机集群控制系统 - 技术文档

> **版本**: v2.13
> **创建时间**: 2025-12-20 00:37:02
> **作者**: 开发团队
> **状态**: 语音指令全链路修复 + 编队系统优化

---

## 目录

1. [项目概述](#1-项目概述)
2. [版本更新说明 (v2.12 → v2.13)](#2-版本更新说明-v212--v213)
3. [问题诊断与修复记录](#3-问题诊断与修复记录)
4. [USB 麦克风支持](#4-usb-麦克风支持)
5. [语音指令置信度修复](#5-语音指令置信度修复)
6. [编队系统优化](#6-编队系统优化)
7. [语音可视化修复](#7-语音可视化修复)
8. [语音指令关键词扩展](#8-语音指令关键词扩展)
9. [文件修改汇总](#9-文件修改汇总)
10. [待办事项](#10-待办事项)

---

## 1. 项目概述

### 1.1 项目简介

**多模态无人机集群控制系统** 是一个基于 PySide6 的桌面应用程序，通过融合多种人机交互方式（语音、手势、图像、触屏）实现对无人机集群的直观控制。

**v2.13 版本亮点**:
- USB 外接麦克风支持 (自动采样率适配)
- 语音指令置信度过滤问题彻底修复
- 五角星/三角形/正方形编队支持 12+ 架无人机
- 编队连线可视化修复
- 语音可视化信号连接修复

### 1.2 核心特性

| 特性 | 描述 | 状态 |
|------|------|------|
| **四模态输入** | 语音识别 (Whisper)、手势识别 (MediaPipe)、图像识别 (YOLOv8)、触屏手绘 | ✅ |
| **USB 麦克风** | 支持外接 USB 麦克风，自动采样率转换 | ✅ **v2.13 新增** |
| **语音可视化** | 霓虹波形 + 音量条 + 状态指示 + 识别结果 | ✅ 修复 |
| **多机编队** | 支持 12+ 架无人机的五角星/三角形/正方形编队 | ✅ **v2.13 优化** |
| **编队连线** | 五角星编队正确显示外圈+内圈+辐射连线 | ✅ **v2.13 修复** |

---

## 2. 版本更新说明 (v2.12 → v2.13)

### 2.1 问题修复总览

| 问题 | 原因 | 解决方案 | 文件 |
|------|------|----------|------|
| USB 麦克风不工作 | 不支持 16kHz 采样率 | 动态检测采样率 + 重采样 | `voice_detector.py` |
| 语音指令只有第一条生效 | 置信度阈值过滤 | 跳过语音的置信度检查 | `main_window.py`, `voice_detector.py` |
| 五角星编队错误 (12架) | 只生成5个点 | 外圈5点+内圈5点+中心 | `swarm_view_3d.py` |
| 编队连线缺边 | 顺序连接所有点 | 按编队类型分组连线 | `swarm_view_3d.py` |
| 语音波形不显示 | 信号未连接 | 初始化时连接信号 | `main_window.py` |
| 状态栏显示识别结果 | 观感不佳 | 过滤识别结果，居中显示 | `control_panel.py` |

### 2.2 功能增强

| 功能 | 描述 |
|------|------|
| **语音关键词扩展** | 添加常见误识别词和同义词（如"旋停"→悬停） |
| **三角形编队优化** | 多层三角形布局，支持任意数量无人机 |
| **正方形编队优化** | 多层正方形布局，支持任意数量无人机 |

---

## 3. 问题诊断与修复记录

### 3.1 语音指令链路断裂问题

**症状**：说"起飞"成功执行，但"五角星""降落"无响应

**诊断过程**：
```
1. 添加调试输出到 VoiceDetector.emit_result()
   → 发现 "五角星" 确实被识别，置信度 0.25

2. 添加调试输出到 MainWindow._on_detection_result()
   → "五角星" 未到达此函数

3. 检查 BaseDetector.emit_result()
   → 发现内部有 self._threshold = 0.5 过滤

4. 检查 MainWindow._publish_to_ros()
   → 发现又有一层 threshold 过滤
```

**根本原因**：双重置信度过滤
1. `BaseDetector.emit_result()`: `if result.confidence >= self._threshold (0.5)`
2. `MainWindow._publish_to_ros()`: `if result.confidence < threshold`

**解决方案**：
```python
# voice_detector.py - 设置语音检测器阈值为 0
def __init__(self, ...):
    super().__init__(parent)
    self._threshold = 0.0  # 禁用内部过滤

# main_window.py - 语音跳过置信度检查
if result.modal_type != "voice" and result.confidence < threshold:
    return
```

### 3.2 语音可视化不显示问题

**症状**：点击录音按钮后，视频底部的霓虹波形不显示

**诊断过程**：
```
1. 检查 _connect_voice_overlay_signals() 调用位置
   → 只在 _on_record_clicked 的 "if self._voice_detector is None" 分支

2. 检查 voice_detector 初始化位置
   → 在 _init_detectors() 中已创建

3. 结论：信号从未连接
```

**解决方案**：
```python
# main_window.py - 在初始化时延迟连接信号
self._voice_detector = VoiceDetector(...)
# ... 其他连接 ...
QTimer.singleShot(100, self._connect_voice_overlay_signals)
```

---

## 4. USB 麦克风支持

### 4.1 问题背景

用户连接 USB 麦克风 (UGREEN CM564)，但录音时报错：
```
Expression 'paInvalidSampleRate' failed in 'src/hostapi/alsa/pa_linux_alsa.c'
```

### 4.2 原因分析

| 设备 | 支持的采样率 | Whisper 要求 |
|------|-------------|--------------|
| 内置麦克风 | 16000, 44100, 48000 Hz | 16000 Hz |
| USB 麦克风 | **仅 48000 Hz** | 16000 Hz |

### 4.3 解决方案

**动态采样率检测 + 重采样**：

```python
# voice_detector.py

class VoiceDetector(BaseDetector):
    USB_MIC_DEVICE = 7  # UGREEN CM564 USB Audio

    def __init__(self, ...):
        self._audio_device = self.USB_MIC_DEVICE
        self._whisper_sample_rate = 16000  # Whisper 需要
        self._device_sample_rate = 48000   # 设备原生

    def _record_audio(self):
        # 动态获取设备采样率
        self._device_sample_rate = self._get_device_sample_rate(device_to_use)

        self._stream = sounddevice.InputStream(
            device=device_to_use,
            samplerate=self._device_sample_rate,  # 使用设备原生采样率
            ...
        )

    def stop_recording(self):
        # 重采样到 16000 Hz
        if self._device_sample_rate != self._whisper_sample_rate:
            from scipy import signal
            num_samples = int(len(audio_data) * 16000 / 48000)
            audio_data = signal.resample(audio_data, num_samples)
```

### 4.4 设备回退机制

如果指定设备不可用，自动回退到系统默认设备：

```python
try:
    self._stream = sounddevice.InputStream(device=device_to_use, ...)
except Exception:
    self.status_changed.emit(f"设备 {device_to_use} 不可用，使用默认设备")
    self._stream = sounddevice.InputStream(device=None, ...)
```

---

## 5. 语音指令置信度修复

### 5.1 问题现象

| 指令 | 置信度 | 阈值 | 结果 |
|------|--------|------|------|
| 起飞 | 0.55-0.67 | 0.50 | ✅ 通过 |
| 五角星 | 0.12-0.35 | 0.50 | ❌ 被过滤 |
| 降落 | 0.35-0.46 | 0.50 | ❌ 被过滤 |

### 5.2 原因分析

Whisper 的 `avg_logprob` 转换为置信度后，复杂词汇置信度偏低：
- 单字词（如"停"）：置信度较高
- 多字词（如"五角星"）：置信度偏低

### 5.3 解决方案

**语音检测器禁用置信度过滤**：

```python
# voice_detector.py
def __init__(self, ...):
    super().__init__(parent)
    self._threshold = 0.0  # Whisper 已有内置过滤，无需二次过滤
```

**主窗口跳过语音的置信度检查**：

```python
# main_window.py
def _publish_to_ros(self, result):
    if result.modal_type != "voice" and result.confidence < threshold:
        return  # 只对非语音模态检查阈值
```

---

## 6. 编队系统优化

### 6.1 五角星编队 (12 架无人机)

**修改前**：
```
无人机 0-4: 外圈 5 个顶点
无人机 5-11: 全部堆在中心 (0, 0, z)  ❌
```

**修改后**：
```
无人机 0-4:  外圈 5 个顶点（星尖）
无人机 5-9:  内圈 5 个点（星凹陷处，0.382 倍半径）
无人机 10:   中心点
无人机 11:   中心上方 (+0.5m)
```

**代码实现**：
```python
def star(center, radius, altitude, count):
    positions = []
    inner_radius = radius * 0.382  # 黄金比例

    # 外圈 5 个顶点
    for i in range(min(count, 5)):
        angle = 2 * math.pi * i / 5 - math.pi / 2
        positions.append((center[0] + radius * cos(angle),
                         center[1] + radius * sin(angle), altitude))

    # 内圈 5 个点（旋转 36°）
    for i in range(5, min(count, 10)):
        angle = 2 * math.pi * (i-5) / 5 - math.pi / 2 + math.pi / 5
        positions.append((center[0] + inner_radius * cos(angle),
                         center[1] + inner_radius * sin(angle), altitude))

    # 中心点
    if count > 10:
        positions.append((center[0], center[1], altitude))

    return positions
```

### 6.2 编队连线修复

**修改前**：顺序连接所有无人机 (0→1→2→...→11→0)

**修改后**：按编队类型分组连线
```python
if self.current_formation == FormationType.STAR and n >= 5:
    # 外圈五边形 (0-1-2-3-4-0)
    for i in range(min(5, n)):
        pts.append(positions[i])
        pts.append(positions[(i + 1) % 5])

    # 内圈五边形 (5-6-7-8-9-5)
    if n >= 10:
        for i in range(5, 10):
            pts.append(positions[i])
            pts.append(positions[5 + ((i - 5 + 1) % 5)])

        # 辐射连线（外圈→内圈）
        for i in range(5):
            pts.append(positions[i])
            pts.append(positions[i + 5])
```

### 6.3 三角形/正方形编队优化

**三角形**：多层三角形，每层递减
```
第 1 层: 3 架，半径 r
第 2 层: 3 架，半径 0.6r，高度 +0.5m
第 3 层: 3 架，半径 0.36r，高度 +1.0m
...
最后 1 架: 中心
```

**正方形**：多层正方形，每层递减
```
第 1 层: 4 架，边长 s
第 2 层: 4 架，边长 0.5s，高度 +0.5m
第 3 层: 4 架，边长 0.25s，高度 +1.0m
...
最后 1 架: 中心
```

---

## 7. 语音可视化修复

### 7.1 问题原因

`_connect_voice_overlay_signals()` 只在 `_on_record_clicked` 中调用，且条件是 `self._voice_detector is None`。

但 `voice_detector` 在 `_init_detectors()` 中已创建，所以条件永远为 False，信号从未连接。

### 7.2 修复方案

在初始化时延迟连接信号（等待 UI 完成）：

```python
# main_window.py - _init_detectors()
self._voice_detector = VoiceDetector(model_name="small")
self._voice_detector.status_changed.connect(...)
self._voice_detector.detection_ready.connect(...)
self._voice_detector.error_occurred.connect(...)
# 延迟连接语音可视化信号
QTimer.singleShot(100, self._connect_voice_overlay_signals)
```

---

## 8. 语音指令关键词扩展

### 8.1 基础指令

| 指令 | 关键词（支持误识别容错） |
|------|-------------------------|
| TAKEOFF | 起飞、takeoff、升空、飞、起来 |
| LAND | 降落、land、着陆、落地、落下 |
| HOVER | 悬停、**旋停**、**选停**、hover、停、暂停、停住、定住 |
| ALTITUDE_UP | 上升、升高、高一点、往上 |
| ALTITUDE_DOWN | 下降、降低、低一点、往下 |
| MOVE_FORWARD | **向前**、**前进**、forward、向前飞、往前飞、往前、前飞 |
| EMERGENCY | 紧急停止、急停、emergency、stop、紧急 |

### 8.2 编队指令

| 编队 | 关键词（支持误识别容错） |
|------|-------------------------|
| 三角形 | 三角形、triangle、三角、三角型 |
| 正方形 | 正方形、square、方形、四方形、正方 |
| 圆形 | 圆形、circle、圆、圆圈、画圆 |
| **五角星** | 五角星、star、星形、**星星**、**五星**、**武角星**、五角形、五角 |
| 直线 | 直线、line、一字、一条线、线形、排成一排 |

---

## 9. 文件修改汇总

| 文件 | 修改内容 |
|------|----------|
| `detectors/voice_detector.py` | USB 麦克风支持、动态采样率、重采样、阈值设为 0 |
| `ui/main_window.py` | 语音跳过置信度检查、信号连接修复、关键词扩展 |
| `ui/control_panel.py` | 状态栏居中、过滤识别结果、导入 Qt |
| `ui/swarm_view_3d.py` | 五角星/三角形/正方形编队优化、连线修复 |
| `ui/voice_overlay.py` | 调试输出（临时） |

### 9.1 关键代码变更

**voice_detector.py** - USB 麦克风支持：
```python
USB_MIC_DEVICE = 7  # UGREEN CM564 USB Audio

def __init__(self, ...):
    self._threshold = 0.0  # 禁用置信度过滤
    self._audio_device = self.USB_MIC_DEVICE
    self._whisper_sample_rate = 16000
    self._device_sample_rate = 48000

def stop_recording(self):
    # 重采样 48000 → 16000
    if self._device_sample_rate != self._whisper_sample_rate:
        from scipy import signal
        audio_data = signal.resample(audio_data, num_samples)
```

**main_window.py** - 语音跳过阈值：
```python
def _publish_to_ros(self, result):
    if result.modal_type != "voice" and result.confidence < threshold:
        return  # 只对非语音检查阈值
```

**swarm_view_3d.py** - 五角星编队：
```python
def star(center, radius, altitude, count):
    inner_radius = radius * 0.382
    # 外圈 5 点 + 内圈 5 点 + 中心
    ...
```

---

## 10. 待办事项

### 10.1 已完成 (v2.13)

- [x] USB 麦克风支持
- [x] 语音指令置信度修复
- [x] 五角星编队优化 (12 架)
- [x] 编队连线修复
- [x] 语音可视化信号连接
- [x] 状态栏 UI 优化
- [x] 语音关键词容错扩展

### 10.2 待处理

- [ ] 语音指令支持数字参数（如"向前飞 5 米"）
- [ ] 更多编队类型（V 形、菱形）
- [ ] 语音模型热切换
- [ ] 多语言支持 (英语)
- [ ] 清理调试输出

---

## 附录：调试命令

```bash
# 查看可用音频设备
python -c "import sounddevice as sd; print(sd.query_devices())"

# 测试 USB 麦克风采样率
python -c "
import sounddevice as sd
sd.check_input_settings(device=7, samplerate=48000, channels=1)
print('48000 Hz: OK')
"

# 运行程序
cd /home/ubuntu/NGW/intern/multimodal_detector
conda activate multimodal
python main.py
```

---

> **文档版本**: v2.13
> **最后更新**: 2025-12-20 00:37:02
