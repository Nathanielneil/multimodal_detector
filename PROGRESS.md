# 多模态检测器 - 开发进展记录

## 项目概述

| 项目名称 | 多模态检测器 (Multimodal Detector) |
|----------|-----------------------------------|
| 开发日期 | 2025-12-16 ~ 2025-12-17 |
| 技术栈 | PySide6 + OpenCV + Whisper + MediaPipe + YOLOv8 + PIL |
| Python 版本 | 3.10 |
| 运行环境 | Ubuntu 20.04, CUDA 11.8 |
| 当前状态 | **功能开发完成，已通过测试** |

---

## 需求确认阶段

### [2025-12-16 需求讨论]

**用户原始需求:**
- 三列布局桌面应用 (QSplitter)
- 左栏: 检测器控制面板
- 中栏: 实时摄像头视频流
- 右栏: 命令历史表
- 四模态检测: 语音/手势/图像/触屏

**需求澄清结果:**

| 问题 | 用户选择 |
|------|----------|
| 模拟数据方式 | 接入真实识别 (非模拟) |
| 导出格式 | JSON 文件 |
| 可视化样式 | 侧边垂直列表 |
| 窗口布局 | 1280x720, 比例 1:2:1 |
| 语音识别 | Whisper (OpenAI) |
| 手势识别 | MediaPipe Hands |
| 图像识别 | YOLOv8n |
| 触屏指令 | 视频区域鼠标点击 + 手绘形状识别 |
| 语音输入方式 | 点击录音 |
| 重置功能 | 全部重置 (历史+状态) |
| 参数列内容 | 识别详情 |
| 代码结构 | 模块化多文件 |
| 环境管理 | Conda (multimodal) |
| GPU 支持 | CUDA 加速 |

---

## 开发进展时间线

### [2025-12-16 T1] 项目初始化
- [x] 创建项目目录结构 `multimodal_detector/`
- [x] 创建子目录: `ui/`, `workers/`, `detectors/`
- [x] 生成 `environment.yml` (Conda 环境配置)
  - PyTorch + CUDA 11.8
  - PySide6, OpenCV, Whisper, MediaPipe, Ultralytics

### [2025-12-16 T2] UI 样式模块
- [x] 完成 `ui/styles.py`
  - 定义主题颜色常量 (主色 #1e88e5)
  - 编写完整 QSS 样式表
  - 四模态颜色映射 (绿/橙/蓝/紫)

### [2025-12-16 T3] 左栏控制面板
- [x] 完成 `ui/control_panel.py`
  - 4个 QCheckBox (模态开关)
  - QComboBox (跟踪算法: highest/medium/low)
  - 2个 QDoubleSpinBox (融合/召回阈值)
  - 录音按钮 + 状态标签
  - 重置/导出按钮
  - 信号槽机制

### [2025-12-16 T4] 摄像头工作类
- [x] 完成 `workers/camera_worker.py`
  - cv2.VideoCapture 封装
  - 线程安全 (QMutex)
  - 占位符帧生成
  - 摄像头枚举功能

### [2025-12-16 T5] 检测器模块
- [x] 完成 `detectors/base_detector.py`
  - DetectionResult 数据类
  - BaseDetector 抽象基类
  - 信号定义 (detection_ready, status_changed, error_occurred)

- [x] 完成 `detectors/voice_detector.py`
  - Whisper 模型加载 (延迟导入)
  - sounddevice 录音
  - 点击录音/停止识别流程
  - 置信度计算

- [x] 完成 `detectors/gesture_detector.py`
  - MediaPipe Hands 集成
  - 手势识别逻辑
  - 手势稳定性检测
  - 关键点绘制

- [x] 完成 `detectors/image_detector.py`
  - YOLOv8 模型加载
  - BoundingBox 数据类
  - 检测框绘制
  - 类别筛选支持

- [x] 完成 `detectors/touch_detector.py`
  - Qt 鼠标事件处理
  - 单击/双击/右键检测
  - 归一化坐标计算

### [2025-12-16 T6] 中栏视频组件
- [x] 完成 `ui/video_widget.py`
  - QLabel 视频显示 (16:9 适配)
  - ModalStatusCard 状态卡片
  - 四模态垂直状态列表
  - 启动/停止按钮
  - 鼠标点击事件转发

### [2025-12-16 T7] 右栏历史表格
- [x] 完成 `ui/history_table.py`
  - QTableWidget (6列)
  - 不可编辑 + 表头固定
  - JSON 导出功能
  - 示例数据方法

### [2025-12-16 T8] 主窗口整合
- [x] 完成 `ui/main_window.py`
  - QSplitter 三栏布局
  - QTimer 帧更新 (30 FPS)
  - 检测器延迟初始化
  - 信号槽连接
  - closeEvent 资源释放

### [2025-12-16 T9] 入口文件
- [x] 完成 `main.py`
  - QApplication 配置
  - 高 DPI 支持
  - 字体设置

### [2025-12-16 T10] 文档
- [x] 完成 `README.md`
- [x] 完成 `PROGRESS.md` (本文档)

---

### [2025-12-17 T11] 手势识别优化

#### 指尖轨迹优化
- [x] 轨迹长度调整: 25帧 → 5帧 (~0.17秒)
- [x] 修复左右手轨迹重合 Bug
  - **问题**: 两只手同时出现时轨迹混乱
  - **原因**: 使用检测索引 `hand_idx` 而非实际左右手标识
  - **解决**: 使用 `multi_handedness` 返回的 "Left"/"Right" 作为轨迹字典键
  - **新增**: `_prev_hands_detected` 集合跟踪手部状态，新手出现时清空旧轨迹

#### 手势映射更新 (无人机集群指令)
- [x] 更新 `GestureType` 枚举映射:

| 手势 | 指令 | 描述 |
|------|------|------|
| 握拳 | 集群降落 | 所有手指弯曲 |
| 张开手掌 | 集群起飞 | 所有手指伸直 |
| 食指指向 | 集群悬停 | 仅食指伸直 |
| 大拇指+食指+小指 | 编队飞行 | 三指伸直 (ILY手势) |
| OK手势 | 指令确定 | 拇指食指接触 |
| 竖起大拇指 | 集群高度上升 | 拇指向上 |
| 向下大拇指 | 集群高度下降 | 拇指向下 |

#### 识别参数调整
- [x] 稳定阈值: 3帧 → 5帧 (连续5帧相同手势才确认)
- [x] 置信度阈值: ≥ 0.8 (低于此值不记录)

---

### [2025-12-17 T12] 视频信息叠加层

#### 左上角 OSD 显示
- [x] 实现 `_draw_stats_overlay()` 方法
- [x] **始终显示**: FPS (绿色)
- [x] **仅手势模态启用时显示**:
  - 置信度: XX% (橙色)
  - 命令: 集群起飞 (青色)
- [x] 半透明黑色背景
- [x] **中文渲染**: 使用 PIL + 系统中文字体 (wqy-zenhei)

#### 显示效果
```
┌──────────────────┐
│ FPS: 30.0        │  ← 始终显示
│ 置信度: 85%      │  ← 仅手势模态
│ 命令: 集群起飞   │  ← 仅手势模态
└──────────────────┘
```

---

### [2025-12-17 T13] 触屏指令优化

#### 轨迹渲染时长调整
- [x] 点击标记显示: 1秒 → 3秒
- [x] 拖动轨迹消失: 每帧移除1点 → 每10帧移除1点 (速度减慢10倍)

---

### [2025-12-17 T14] 手绘形状识别功能 (重要新功能)

#### 功能描述
用户在视频区域手绘形状，系统识别后清除手绘轨迹，在视频中央显示标准化形状。

#### 支持的形状

| 手绘形状 | 识别结果 | 无人机编队含义 |
|----------|----------|----------------|
| 三角形 | 三角形队形 | 三机编队 |
| 正方形 | 正方形队形 | 四机方阵 |
| 圆形 | 圆形队形 | 环绕编队 |
| 五角星 | 五角星队形 | 星形展开 |

#### 技术实现

**1. 形状识别算法** (`_recognize_shape()`)
```python
# 使用 OpenCV 轮廓分析
- 闭合检测: 起点终点距离 < 0.15 (归一化)
- 顶点计算: cv2.approxPolyDP() 多边形近似
- 圆度计算: 4πA/P² (A=面积, P=周长)
- 凹陷检测: cv2.convexityDefects() 凸包缺陷
```

**2. 识别逻辑**
```
五角星: 凹陷点 ≥ 4 且闭合
圆形:   圆度 > 0.7 且顶点 > 6 且闭合
三角形: 顶点 = 3 且闭合
正方形: 顶点 = 4 且闭合
```

**3. 标准形状绘制** (`_draw_standard_shape()`)
- 形状大小: `min(w, h) // 4` 固定尺寸
- 形状颜色: 黄色 (0, 255, 255)
- 显示时长: 2秒
- 文字标签: 形状正下方，使用 PIL 渲染中文

#### 使用流程
```
1. 用户在视频区域手绘形状 (需闭合)
2. 松开鼠标时触发识别 (轨迹点 ≥ 20)
3. 识别成功 → 清除手绘轨迹
4. 视频中央显示标准形状 + "XX队形" 标签
5. 2秒后自动消失
```

---

### [2025-12-17 T15] UI 优化

- [x] 控制面板复选框简化:
  - "语音识别 (Whisper)" → "语音识别"
  - "手势识别 (MediaPipe)" → "手势识别"
  - "图像识别 (YOLOv8)" → "图像识别"

---

## 技术决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| GUI 框架 | PySide6 | Qt 官方绑定，商业友好 |
| 模型延迟加载 | 是 | 避免启动时卡顿 |
| 手势稳定阈值 | 5帧 | 避免误触，提高准确性 |
| 手势置信度阈值 | 0.8 | 过滤低置信度结果 |
| Whisper 模型 | base | 速度与精度平衡 |
| YOLO 模型 | yolov8n | 轻量快速 |
| 视频 FPS | 30 | 流畅度与性能平衡 |
| 中文渲染 | PIL | OpenCV 不支持中文 |
| 形状识别 | OpenCV 轮廓分析 | 成熟稳定，无需额外模型 |
| 轨迹存储 | 归一化坐标 (0-1) | 适应不同分辨率 |

---

## 文件清单

```
multimodal_detector/
├── main.py                 # 入口文件
├── environment.yml         # Conda 环境配置
├── requirements.txt        # pip 依赖
├── README.md               # 项目说明
├── PROGRESS.md             # 进展记录 (本文档)
├── yolov8n.pt              # YOLOv8 模型权重
├── ui/
│   ├── __init__.py
│   ├── styles.py           # QSS 样式表
│   ├── control_panel.py    # 左栏控制面板
│   ├── video_widget.py     # 中栏视频组件
│   ├── history_table.py    # 右栏历史表格
│   └── main_window.py      # 主窗口 (含 OSD 绘制)
├── workers/
│   ├── __init__.py
│   └── camera_worker.py    # 摄像头管理
└── detectors/
    ├── __init__.py
    ├── base_detector.py    # 检测器基类
    ├── voice_detector.py   # 语音识别 (Whisper)
    ├── gesture_detector.py # 手势识别 (MediaPipe) + 无人机指令映射
    ├── image_detector.py   # 图像识别 (YOLOv8)
    └── touch_detector.py   # 触屏指令 + 手绘形状识别
```

---

## 核心类说明

### GestureDetector (手势检测器)

```python
class GestureDetector(BaseDetector):
    """
    手势识别检测器 - 无人机集群控制

    关键属性:
        TRAJECTORY_MAX_LEN = 5      # 指尖轨迹长度
        _stable_threshold = 5        # 连续帧阈值
        _min_detection_confidence = 0.7

    手势映射:
        握拳 → 集群降落
        张开手掌 → 集群起飞
        食指指向 → 集群悬停
        大拇指+食指+小指 → 编队飞行
        OK手势 → 指令确定
        竖起大拇指 → 集群高度上升
        向下大拇指 → 集群高度下降
    """
```

### TouchDetector (触屏检测器)

```python
class TouchDetector(BaseDetector):
    """
    触屏指令检测器 - 含手绘形状识别

    关键属性:
        _marker_duration = 3.0       # 点击标记显示时长
        _shape_duration = 2.0        # 标准形状显示时长
        _min_trajectory_points = 20  # 形状识别最小点数

    形状识别:
        ShapeType.TRIANGLE → 三角形队形
        ShapeType.SQUARE → 正方形队形
        ShapeType.CIRCLE → 圆形队形
        ShapeType.STAR → 五角星队形

    核心方法:
        _recognize_shape()      # 形状识别算法
        _draw_standard_shape()  # 标准形状绘制
    """
```

### MainWindow (主窗口)

```python
class MainWindow(QMainWindow):
    """
    主窗口 - 整合所有组件

    新增功能:
        _draw_stats_overlay()  # 左上角 OSD (FPS/置信度/命令)
        _confidences           # 各模态置信度记录
        _current_gesture_command  # 当前手势命令
        _font                  # PIL 中文字体
    """
```

---

## 运行说明

### 环境准备
```bash
cd /home/ubuntu/NGW/intern/multimodal_detector
conda activate multimodal
```

### 启动应用
```bash
python main.py
```

### 功能测试

1. **手势识别测试**
   - 勾选"手势识别"
   - 启动摄像头
   - 对摄像头做手势，观察左上角置信度和命令显示

2. **形状识别测试**
   - 勾选"触屏指令"
   - 在视频区域用鼠标手绘闭合形状
   - 观察视频中央标准形状显示

---

## 已完成功能

- [x] 四模态检测 (语音/手势/图像/触屏)
- [x] 实时视频流显示
- [x] 手势 → 无人机指令映射
- [x] 手绘形状识别 (三角形/正方形/圆形/五角星)
- [x] 指尖轨迹渲染 (左右手分离)
- [x] 视频 OSD 信息显示 (FPS/置信度/命令)
- [x] 中文渲染支持 (PIL)
- [x] 检测历史记录 + JSON 导出
- [x] 模态开关控制

---

## 待办事项

- [ ] 性能优化 (检测器多线程)
- [ ] 模型下载进度显示
- [ ] 配置文件支持 (阈值持久化)
- [ ] 多语言 UI
- [ ] 更多形状支持 (箭头、菱形等)
- [ ] 手势组合指令

---

## 备注

- 首次运行需下载 Whisper 模型 (自动)
- 语音识别需要麦克风权限
- GPU 加速需正确安装 CUDA 驱动
- 形状识别需要手绘轨迹闭合 (起点终点接近)
- 中文字体依赖系统字体 (wqy-zenhei/wqy-microhei)

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2025-12-16 | 基础框架搭建，四模态检测实现 |
| v1.1 | 2025-12-17 | 手势优化，无人机指令映射，手绘形状识别 |
