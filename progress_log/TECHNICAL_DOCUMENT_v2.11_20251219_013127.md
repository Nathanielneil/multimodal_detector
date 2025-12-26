# 多模态无人机集群控制系统 - 技术文档

> **版本**: v2.11
> **创建时间**: 2025-12-19 01:31:27
> **作者**: 开发团队
> **状态**: 性能优化专项 - 帧率提升与轨迹修复

---

## 目录

1. [项目概述](#1-项目概述)
2. [版本更新说明 (v2.10 → v2.11)](#2-版本更新说明-v210--v211)
3. [轨迹降落修复](#3-轨迹降落修复)
4. [动画帧率独立性](#4-动画帧率独立性)
5. [手势检测性能优化](#5-手势检测性能优化)
6. [异步检测架构](#6-异步检测架构)
7. [系统架构](#7-系统架构)
8. [开发日志](#8-开发日志)
9. [待办事项](#9-待办事项)

---

## 1. 项目概述

### 1.1 项目简介

**多模态无人机集群控制系统** 是一个基于 PySide6 的桌面应用程序，通过融合多种人机交互方式（语音、手势、图像、触屏）实现对无人机集群的直观控制。

**v2.11 版本亮点**:
- 修复降落时轨迹消失问题
- 实现帧率独立的动画系统
- 手势检测性能优化，帧率从 13 FPS 提升至 30 FPS

### 1.2 核心特性

| 特性 | 描述 | 状态 |
|------|------|------|
| **四模态输入** | 语音识别 (Whisper)、手势识别 (MediaPipe)、图像识别 (YOLOv8)、触屏手绘 | ✅ |
| **无人机指令映射** | 手势→起飞/降落/悬停/编队/高度控制/向前飞行 | ✅ |
| **手绘编队控制** | 绘制三角形/正方形/圆形/五角星切换编队 | ✅ |
| **PyQtGraph 3D 可视化** | 原生嵌入的无人机集群 3D 场景渲染 | ✅ |
| **APF 避障算法** | 人工势场法实现自主避障飞行 | ✅ |
| **L1 动力学模型** | 6DOF 位置+速度, PD控制 | ✅ |
| **轨迹历史显示** | 渐变色飞行轨迹可视化 | ✅ **v2.11 修复** |
| **异步手势检测** | 独立线程处理 MediaPipe 推理 | ✅ **v2.11 新增** |
| **帧率独立动画** | 使用实际 dt 计算物理 | ✅ **v2.11 新增** |

---

## 2. 版本更新说明 (v2.10 → v2.11)

### 2.1 问题与解决方案总览

| 问题 | 原因 | 解决方案 | 效果 |
|------|------|----------|------|
| 降落时轨迹消失 | `is_flying=False` 时跳过轨迹更新 | 地面状态也更新轨迹 | ✅ 轨迹完整显示 |
| 开启摄像头后动画变慢 | dt 硬编码为 0.033s | 使用实际时间差 | ✅ 速度恒定 |
| 手势检测帧率低 (~13 FPS) | MediaPipe 推理阻塞主线程 | 异步检测 + 多项优化 | ✅ 30 FPS |

### 2.2 性能提升数据

| 指标 | v2.10 | v2.11 | 提升 |
|------|-------|-------|------|
| 手势检测帧率 | ~13 FPS | **~30 FPS** | **+130%** |
| MediaPipe 推理/帧 | 2 次 | **0-1 次** | **-50%+** |
| 主线程阻塞时间 | ~60ms | **<5ms** | **-90%** |
| 动画帧率稳定性 | 受负载影响 | **恒定速度** | ✅ |

---

## 3. 轨迹降落修复

### 3.1 问题描述

**用户反馈**: 无人机降落时，飞行轨迹突然消失。

### 3.2 根因分析

```python
# 原代码 (ui/swarm_view_3d.py - _update_animation)
for drone in self.drones:
    if self.is_flying:
        # ... 飞行物理 ...
        if self.trajectory_enabled:
            drone.update_trajectory()  # ← 只有飞行时才更新
    else:
        # 地面状态 - 轨迹更新被跳过！
        drone.update_animation(0.5)
```

**问题链**:
1. `_execute_land()` 立即设置 `is_flying = False`
2. 无人机实际还在下降中
3. 动画循环进入 `else` 分支，跳过 `update_trajectory()`
4. 轨迹停止更新，视觉上"消失"

### 3.3 解决方案

```python
# 修复后代码
for drone in self.drones:
    if self.is_flying:
        # ... 飞行物理 ...
        if self.trajectory_enabled:
            drone.update_trajectory()
    else:
        drone.update_animation(0.5)
        # 降落过程中也要更新轨迹 (z > 0.05 时继续记录)
        if self.trajectory_enabled and drone.position[2] > 0.05:
            drone.update_trajectory()
```

同时修改 `update_trajectory()` 的高度阈值:
```python
# 只在有高度时记录轨迹 (z > 0.05)
if self.position[2] < 0.05:  # 原来是 0.1
    return
```

### 3.4 代码位置

| 文件 | 方法 | 修改内容 |
|------|------|----------|
| `ui/swarm_view_3d.py` | `_update_animation()` | 添加地面状态轨迹更新 |
| `ui/swarm_view_3d.py` | `update_trajectory()` | 高度阈值 0.1→0.05 |

---

## 4. 动画帧率独立性

### 4.1 问题描述

**用户反馈**: 开启摄像头和手势识别后，动态障碍物移动速度明显变慢。

### 4.2 根因分析

```python
# 原代码
def _update_animation(self):
    dt = 0.033  # 硬编码！假设 30 FPS

    for obs in self.dynamic_obstacles:
        obs.update(dt)  # 移动距离 = speed * dt
```

**问题**:
- 定时器间隔 33ms，但实际帧间隔可能因 CPU 负载变大
- dt 固定为 0.033，导致帧率下降时移动距离减少
- 帧率 15 FPS 时，速度降为原来的 50%

### 4.3 解决方案

```python
# 修复后代码
def _update_animation(self):
    import time
    current_time = time.time()
    if self._last_animation_time is None:
        dt = 0.033  # 首次调用使用默认值
    else:
        dt = current_time - self._last_animation_time
        # 限制 dt 范围，防止极端情况
        dt = max(0.01, min(dt, 0.1))  # 10-100ms
    self._last_animation_time = current_time

    for obs in self.dynamic_obstacles:
        obs.update(dt)  # 使用实际时间差
```

### 4.4 效果对比

| 场景 | 修改前 | 修改后 |
|------|--------|--------|
| 帧率 30 FPS | dt=0.033, 正常速度 | dt≈0.033, 正常速度 |
| 帧率 15 FPS (高负载) | dt=0.033, **速度减半** | dt≈0.066, **速度正常** |
| 帧率 10 FPS (极端) | dt=0.033, **速度降 2/3** | dt≈0.1, **速度正常** |

---

## 5. 手势检测性能优化

### 5.1 问题描述

**用户反馈**: 开启手势检测后，视频流帧率从 23 FPS 降至 13 FPS。

### 5.2 优化措施汇总

| 优化项 | 修改前 | 修改后 | 效果 |
|--------|--------|--------|------|
| MediaPipe 推理次数/帧 | 2 次 | **1 次** | -50% CPU |
| 模型复杂度 | 1 (Full) | **0 (Lite)** | ~2x 快 |
| 检测手数 | 2 只 | **1 只** | ~30% 快 |
| 输入分辨率 | 640×360 | **320×180** | ~2x 快 |
| 检测频率 | 每帧 | **每 5 帧** | -80% 推理 |
| 检测线程 | 主线程 | **独立线程** | 不阻塞 UI |

### 5.3 优化详解

#### 5.3.1 消除重复推理

**问题**: `detect()` 和 `draw_landmarks()` 各调用一次 `self._hands.process()`

```python
# 修复: 缓存推理结果
def detect(self, data):
    results = self._hands.process(rgb_frame)
    self._cached_results = results  # 缓存
    ...

def draw_landmarks(self, frame):
    results = self._cached_results  # 复用缓存
    if results is None:
        return frame  # 无缓存时直接返回
    ...
```

#### 5.3.2 使用轻量模型

```python
# 修改前
self._hands = mp_hands.Hands(
    max_num_hands=2,
    # model_complexity 默认为 1
)

# 修改后
self._hands = mp_hands.Hands(
    max_num_hands=1,          # 只检测一只手
    model_complexity=0,        # Lite 模型
)
```

#### 5.3.3 输入降采样

```python
def detect(self, data):
    # 降采样以提升性能
    h, w = data.shape[:2]
    scale = 0.5 if w > 480 else 1.0
    if scale < 1.0:
        small_frame = cv2.resize(data, None, fx=scale, fy=scale)
    else:
        small_frame = data

    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    results = self._hands.process(rgb_frame)
```

### 5.4 代码位置

| 文件 | 修改内容 |
|------|----------|
| `detectors/gesture_detector.py` | 结果缓存、轻量模型、降采样 |
| `ui/main_window.py` | 跳帧逻辑、异步调用 |

---

## 6. 异步检测架构

### 6.1 架构设计

**修改前 (同步模式)**:
```
主线程: 读帧 → MediaPipe推理(阻塞30ms) → 绘制 → 显示
                    ↑
                阻塞主线程
```

**修改后 (异步模式)**:
```
主线程:     读帧 → 提交帧(非阻塞) → 使用缓存绘制 → 显示
                      │
工作线程:             ↳ MediaPipe推理 → 缓存结果 → 发信号
```

### 6.2 新增文件

**`workers/gesture_worker.py`**:

```python
class GestureWorker(QThread):
    """手势检测工作线程"""

    result_ready = Signal(str, float, object)

    def __init__(self):
        self._frame = None
        self._frame_lock = Lock()
        self.cached_results = None

    def submit_frame(self, frame):
        """提交新帧 (非阻塞)"""
        with self._frame_lock:
            self._frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
            self._new_frame_available = True

    def run(self):
        """工作线程主循环"""
        while self._running:
            # 获取帧
            frame = self._get_frame()
            if frame is None:
                self.msleep(10)
                continue

            # 处理
            results = self._hands.process(rgb_frame)
            self.cached_results = results

            # 发送结果
            if gesture_detected:
                self.result_ready.emit(command, confidence, results)
```

### 6.3 主窗口集成

```python
# main_window.py

def _init_components(self):
    self._gesture_worker = GestureWorker()
    self._gesture_worker.result_ready.connect(self._on_gesture_worker_result)
    self._gesture_worker.initialize()
    self._gesture_worker.start()

def _update_frame(self):
    # 提交帧到异步工作线程 (非阻塞)
    if self._gesture_worker.isRunning():
        self._gesture_worker.submit_frame(frame)

        # 使用缓存结果绘制
        cached = self._gesture_worker.get_cached_results()
        if cached and cached.multi_hand_landmarks:
            self._gesture_detector._cached_results = cached
            drawn = self._gesture_detector.draw_landmarks(processed_frame)
```

### 6.4 数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                         主线程 (UI)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   摄像头 ──→ 读帧 ──→ submit_frame() ──→ [帧队列]              │
│                           │                   │                 │
│                           ↓                   ↓                 │
│              使用 cached_results ←── [结果缓存]                 │
│                           │                   ↑                 │
│                           ↓                   │                 │
│                    draw_landmarks()           │                 │
│                           │                   │                 │
│                           ↓                   │                 │
│                     显示画面                   │                 │
│                                               │                 │
├───────────────────────────────────────────────│─────────────────┤
│                    工作线程                    │                 │
├───────────────────────────────────────────────│─────────────────┤
│                                               │                 │
│   [帧队列] ──→ MediaPipe推理 ──→ 缓存结果 ────┘                 │
│                     │                                           │
│                     ↓                                           │
│              result_ready.emit() ──→ 主线程处理手势指令          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 系统架构

### 7.1 整体架构图 (v2.11 更新)

```
+-------------+-----------------------------------+-------------------+
|             |        中央显示区域                |                   |
|   控制面板   |  +-----------------------------+  |   命令历史        |
|             |  |    PyQtGraph 3D 可视化       |  |                   |
|  - 模态开关  |  |    无人机集群实时状态         |  +-------------------+
|  - 阈值设置  |  |    帧率独立动画 (v2.11)      |  | 集群数量          |
|  - 摄像头    |  +-----------------------------+  | 当前: 6架 [-] [+] |
|  - 录音按钮  |  +-----------------------------+  +-------------------+
|             |  |    视频流 + 检测叠加          |  |   无人机状态      |
|             |  |    异步手势检测 (v2.11)       |  |   UAV-0 ...       |
+-------------+-----------------------------------+-------------------+
                              │
                              ↓
              ┌───────────────────────────────┐
              │      GestureWorker (独立线程)   │
              │      MediaPipe Lite 模型       │
              │      320×180 输入分辨率         │
              └───────────────────────────────┘
```

### 7.2 线程模型

| 线程 | 职责 | 优先级 |
|------|------|--------|
| **主线程** | UI 渲染、帧显示、用户交互 | 高 |
| **GestureWorker** | MediaPipe 手势推理 | 中 |
| **动画定时器** | 3D 场景更新 (16ms 间隔) | 中 |

---

## 8. 开发日志

### v1.0 - v2.4 (2025-12-16 ~ 12-18)
- 项目初始化、四模态检测器
- 手势识别、UI优化
- ROS集成、3D渲染、APF避障
- 场景扩展、边界围栏

### v2.5 (2025-12-18)
- 轨迹历史显示
- 颜色渐变效果

### v2.6 (2025-12-18)
- L1 基础动力学模型
- Z轴围栏 (天花板)

### v2.7 (2025-12-18) - ❌ 已回滚
- L2 姿态动力学模型 (12 DOF)
- 问题: 姿态无法稳定，持续震荡

### v2.8 (2025-12-18)
- 回滚至 L1 动力学
- 记录 L2 失败原因和改进建议

### v2.9 (2025-12-18)
- 新增集群数量动态控制功能 (1-12架)
- 右侧栏新增"集群控制"卡片
- 增减时保持现有无人机位置不变

### v2.10 (2025-12-19)
- APF 避障算法优化
- 引力增益 2.0→4.0，编队恢复速度提升 50%+

### v2.11 (2025-12-19 01:31:27) - 当前版本

**Bug 修复**:
- 修复降落时轨迹消失问题
  - 原因: `is_flying=False` 时跳过轨迹更新
  - 方案: 地面状态也更新轨迹 (z > 0.05)

**性能优化**:
- 帧率独立动画系统
  - 原因: dt 硬编码导致高负载时动画变慢
  - 方案: 使用 `time.time()` 计算实际 dt

- 手势检测性能优化 (帧率 13 FPS → 30 FPS)
  - MediaPipe 结果缓存 (避免重复推理)
  - 轻量模型 (model_complexity=0)
  - 单手检测 (max_num_hands=1)
  - 输入降采样 (640×360 → 320×180)
  - 跳帧检测 (每 5 帧检测一次)

- 异步检测架构
  - 新增 `GestureWorker` 工作线程
  - 主线程不再阻塞于 MediaPipe 推理
  - 定时器间隔 33ms → 16ms (目标 60 FPS)

**新增文件**:
- `workers/gesture_worker.py` - 异步手势检测工作线程

---

## 9. 待办事项

### 高优先级 (Sim2Real)
- [ ] 重新设计 L2 姿态动力学 (参考 v2.8 文档建议)
- [ ] 参数辨识工具
- [ ] 单元测试框架

### 中优先级
- [ ] 边界接近警告
- [ ] 多方向移动指令
- [ ] 编队旋转指令
- [x] ~~轨迹降落显示~~ (v2.11 已完成)
- [x] ~~手势检测性能优化~~ (v2.11 已完成)

### 低优先级
- [ ] 轨迹回放功能
- [ ] Gazebo 仿真集成
- [ ] 配置文件支持
- [ ] GPU 加速 (MediaPipe GPU)

---

## 附录

### A. 文件结构 (v2.11 更新)

```
multimodal_detector/
├── main.py
├── PROGRESS.md
├── detectors/
│   ├── gesture_detector.py    # 结果缓存、轻量模型
│   └── ...
├── workers/
│   ├── camera_worker.py
│   └── gesture_worker.py      # v2.11 新增: 异步检测
├── ui/
│   ├── main_window.py         # 异步检测集成
│   ├── swarm_view_3d.py       # 帧率独立动画、轨迹修复
│   └── ...
├── progress_log/
│   ├── TECHNICAL_DOCUMENT_v2.10_20251219_002022.md
│   └── TECHNICAL_DOCUMENT_v2.11_20251219_013127.md  # 当前
└── ros_bridge/
```

### B. 版本演进总结

| 版本 | 主要功能 | 状态 |
|------|----------|------|
| v2.6 | L1 动力学 + Z轴围栏 | ✅ |
| v2.7 | L2 姿态动力学 | ❌ 已回滚 |
| v2.8 | 回滚 + 失败分析 | ✅ |
| v2.9 | 集群数量控制 | ✅ |
| v2.10 | APF 避障优化 | ✅ |
| **v2.11** | **性能优化专项** | **✅ 当前** |

### C. 性能优化记录

| 版本 | 手势帧率 | 主要优化 |
|------|----------|----------|
| v2.10 | ~13 FPS | 无 |
| v2.11 (缓存) | ~20 FPS | 结果缓存 |
| v2.11 (跳帧) | ~23 FPS | 每 5 帧检测 |
| v2.11 (轻量) | ~25 FPS | Lite 模型 + 降采样 |
| **v2.11 (异步)** | **~30 FPS** | 独立线程 |

### D. MediaPipe 配置对比

| 参数 | v2.10 | v2.11 |
|------|-------|-------|
| model_complexity | 1 (Full) | **0 (Lite)** |
| max_num_hands | 2 | **1** |
| 输入分辨率 | 640×360 | **320×180** |
| 推理位置 | 主线程 | **工作线程** |
| 推理次数/帧 | 2 | **0-1** |

---

*文档创建时间: 2025-12-19 01:31:27*
*版本: v2.11*
*状态: 性能优化专项 - 帧率提升与轨迹修复*
