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

## [2025-12-17 T16] UI 交互优化 (v1.2)

### 新增功能

#### 1. 模型加载进度对话框
- [x] 新增 `ui/progress_dialog.py` 进度对话框组件
- [x] 检测器初始化时显示进度条和当前步骤
- [x] 支持确定进度和不确定进度两种模式

#### 2. 检测错误处理机制
- [x] 添加错误计数和冷却机制
- [x] 状态栏显示错误信息
- [x] 连续错误自动禁用对应模态并提示用户

#### 3. 快捷键支持
| 快捷键 | 功能 |
|--------|------|
| 空格 | 开始/停止录音 |
| S | 启动/停止摄像头 |
| R | 重置统计 |
| E | 导出历史 |
| 1-4 | 切换语音/手势/图像/触屏模态 |
| F1 | 显示快捷键帮助 |

#### 4. 摄像头切换功能
- [x] 控制面板添加摄像头选择下拉框
- [x] 添加"刷新设备列表"按钮
- [x] 支持运行时切换摄像头

#### 5. 历史记录筛选
- [x] 添加搜索框（关键词搜索指令）
- [x] 添加模态类型筛选下拉框
- [x] 添加清除筛选按钮
- [x] 筛选状态下显示"显示 X/Y 条记录"

#### 6. 界面本地化
- [x] 视频占位符文字改为中文
- [x] 使用 PIL 渲染中文，避免乱码

#### 7. OSD 显示优化
- [x] OSD 移至右上角，减少遮挡
- [x] 背景透明度调整（40%→60%）
- [x] 尺寸缩小，更紧凑

### 修改的文件

| 文件 | 修改内容 |
|------|----------|
| `ui/progress_dialog.py` | 新增 - 进度对话框和加载遮罩层 |
| `ui/main_window.py` | 进度对话框、错误处理、快捷键、摄像头切换、OSD优化 |
| `ui/control_panel.py` | 摄像头选择组件、信号定义 |
| `ui/video_widget.py` | 占位符中文渲染 |
| `ui/history_table.py` | 筛选功能组件 |
| `ui/__init__.py` | 导出新组件 |

### 技术细节

**错误处理机制**:
- 5秒冷却时间避免重复提示
- 连续10次错误自动禁用模态
- 错误信息截断显示（最长50字符）

**快捷键实现**:
- 使用 `QShortcut` + `QKeySequence`
- 支持单键快捷键（无需 Ctrl/Alt 组合）

---

## [2025-12-18 T17] 3D可视化与ROS集成 (v2.3)

### 核心新功能

#### 1. 无人机集群3D可视化

新增 `ui/swarm_view_3d.py` 模块，基于 PyQtGraph OpenGL 实现：

| 组件 | 功能 |
|------|------|
| SwarmView3D | 3D可视化主组件 |
| Drone3D | 单个无人机3D模型 |
| DynamicObstacle | 动态障碍物模型 |
| APFController | 人工势场避障控制器 |

**场景参数**:
- 地图大小: 16×16米 (原8×8)
- 安全飞行区: 中心4米半径
- 障碍物数量: 2个球体
- 无人机数量: 6架

#### 2. ROS Bridge 通信模块

新增 `ros_bridge/ros_bridge.py` 实现ROS1通信：

**发布话题**:
| 话题 | 类型 | 用途 |
|------|------|------|
| `/swarm/command` | String (JSON) | 集群控制指令 |
| `/swarm/formation` | String (JSON) | 编队类型指令 |
| `/swarm/visualization` | MarkerArray | RViz可视化 |

**订阅话题**:
| 话题 | 类型 | 用途 |
|------|------|------|
| `/swarm/status` | String (JSON) | 集群状态反馈 |

#### 3. 手势-指令映射

| 手势 | 指令 | 功能 |
|------|------|------|
| 张开手掌 | TAKEOFF | 集群起飞 (垂直上升至1.5m) |
| 握拳 | LAND | 集群降落 (垂直下降) |
| 食指指向 | HOVER | 集群悬停 |
| ILY手势 | FORMATION | 进入编队 (当前编队类型) |
| OK手势 | CONFIRM | 指令确定 |
| 竖起大拇指 | ALTITUDE_UP | 高度上升 0.5m |
| 向下大拇指 | ALTITUDE_DOWN | 高度下降 0.5m |
| V形手势 (剪刀手) | MOVE_FORWARD | 编队向前飞行 5m |

#### 4. 编队类型 (手绘形状触发)

| 手绘形状 | 编队类型 | 队形描述 |
|----------|----------|----------|
| 三角形 | TRIANGLE | 三角形编队 |
| 正方形 | SQUARE | 正方形编队 |
| 圆形 | CIRCLE | 圆形编队 |
| 五角星 | STAR | 五角星编队 |

---

### 技术改进

#### 1. APF避障算法优化

**问题**: 无人机起飞后高频上下震荡
**原因**: 吸引力在目标点附近过强导致超调

**解决方案**:
```python
# 距离衰减吸引力
if dist < 0.5:
    scale = dist / 0.5  # 近距离时衰减
else:
    scale = 1.0
magnitude = min(self.k_att * dist * scale, self.k_att)
```

**EMA速度平滑**:
```python
self._velocity_history[i] = self.ema_alpha * velocity + (1 - self.ema_alpha) * self._velocity_history[i]
```

**参数调整**:
| 参数 | 原值 | 新值 |
|------|------|------|
| k_att (吸引系数) | 1.2 | 0.8 |
| k_rep (排斥系数) | 1.5 | 2.0 |
| d0 (障碍影响距离) | 2.0 | 1.5 |
| step_size (步长) | 0.02 | 0.015 |
| ema_alpha (平滑系数) | - | 0.3 |

#### 2. 无人机行为逻辑修正

**问题**: 无人机初始位置即为编队队形
**修正**:
1. 初始化时随机分布在地面安全区内
2. 起飞指令仅垂直上升，保持x,y位置
3. 编队指令单独触发队形变换

**状态机**:
```
地面(随机位置) → [起飞] → 悬停(保持位置) → [编队] → 编队飞行
                                          ↓
                                      [降落] → 垂直下降 → 地面
```

#### 3. 新增V形手势识别

在 `detectors/gesture_detector.py` 中添加:
```python
class GestureType(Enum):
    VICTORY = "向前飞行"  # V形手势（剪刀手）
```

识别条件: 仅食指和中指伸直，其余弯曲

---

### UI 重构

#### 1. 布局变更

**原布局**:
```
┌─────────┬─────────────────┬─────────┐
│  控制   │    可视化       │  命令   │
│  面板   ├─────────────────┤  历史   │
│         │    视频流       │         │
└─────────┴─────────────────┴─────────┘
```

**新布局**:
```
┌─────────┬─────────────────┬─────────┐
│  控制   │    可视化       │  命令   │
│  面板   ├─────────────────┤  历史   │
│         │    视频流       ├─────────┤
│         │                 │  无人机 │
│         │                 │  状态   │
└─────────┴─────────────────┴─────────┘
```

#### 2. 可折叠无人机状态卡片

新增 `CollapsibleDroneCard` 类:

**折叠状态**: 显示 UAV ID + 状态
**展开状态**: 显示完整信息
- 位置 (x, y, z)
- 高度
- 速度
- 电量 (颜色指示)
- 信号强度 (颜色指示)

**颜色指示**:
- 电量/信号 > 50%: 绿色
- 20% ~ 50%: 橙色
- < 20%: 红色

#### 3. 删除的组件

- `VideoWidget` 右侧识别状态卡片
- `SwarmView3D` 底部无人机状态面板
- 雷达射线可视化 (enable_lidar=False)

---

### 修改的文件

| 文件 | 修改内容 |
|------|----------|
| `ui/swarm_view_3d.py` | 场景扩大、APF优化、行为逻辑修正、添加MOVE_FORWARD |
| `ui/main_window.py` | 布局重构、CollapsibleDroneCard、手势映射更新 |
| `ui/video_widget.py` | 简化，移除识别状态卡片 |
| `detectors/gesture_detector.py` | 新增VICTORY手势类型 |
| `ros_bridge/ros_bridge.py` | 新增MOVE_FORWARD指令 |

---

### 文件清单更新

```
multimodal_detector/
├── main.py
├── environment.yml
├── requirements.txt
├── README.md
├── PROGRESS.md
├── yolov8n.pt
├── ui/
│   ├── __init__.py
│   ├── styles.py
│   ├── control_panel.py
│   ├── video_widget.py
│   ├── history_table.py
│   ├── main_window.py
│   ├── progress_dialog.py
│   ├── swarm_view_3d.py        # 新增 - 3D可视化
│   └── rviz_widget.py          # 新增 - RViz集成(可选)
├── workers/
│   ├── __init__.py
│   └── camera_worker.py
├── detectors/
│   ├── __init__.py
│   ├── base_detector.py
│   ├── voice_detector.py
│   ├── gesture_detector.py     # 更新 - 新增VICTORY手势
│   ├── image_detector.py
│   └── touch_detector.py
├── ros_bridge/                  # 新增目录
│   ├── __init__.py
│   └── ros_bridge.py           # ROS通信模块
└── progress_log/                # 新增目录
    └── TECHNICAL_DOCUMENT_v2.3.md
```

---

### 运行说明

#### 无ROS环境
```bash
conda activate multimodal
python main.py
# ROS功能将自动禁用，仅本地3D模拟
```

#### 有ROS环境
```bash
# 终端1: 启动ROS Master
roscore

# 终端2: 启动应用
source ~/catkin_ws/devel/setup.bash
conda activate multimodal
python main.py
```

---

### 已完成功能 (v2.3)

- [x] 3D无人机集群可视化 (PyQtGraph OpenGL)
- [x] APF人工势场避障算法
- [x] ROS Bridge 通信模块
- [x] 手势→集群指令映射
- [x] 手绘形状→编队类型映射
- [x] 可折叠无人机状态卡片
- [x] V形手势编队前进指令
- [x] 无人机行为逻辑修正 (随机初始化→垂直起飞→编队)

---

### 待办事项 (v2.3)

- [ ] RViz 实时同步显示
- [ ] 多机协同路径规划
- [ ] 障碍物动态生成
- [ ] 编队切换动画平滑过渡
- [ ] 手势置信度自适应阈值

---

## [2025-12-18 T18] 场景扩大与边界围栏 (v2.4)

### 问题背景

v2.3 版本中场景为 16x16 米，当无人机编队使用 V 形手势向前飞行 5 米时，多次执行后会越过边界，导致无人机集群漂移到无穷远处。

### 解决方案

#### 1. 场景尺寸扩大

| 参数 | v2.3 | v2.4 |
|------|------|------|
| 地图尺寸 | 16×16 m | **32×32 m** |
| 网格间距 | 1 m | **2 m** |
| 安全区半径 | 4.0 m | **6.0 m** |
| 相机距离 | 18 | **35** |
| 相机仰角 | 35° | **40°** |
| 随机障碍物 | 10 个 | **15 个** |
| 障碍物间距 | 0.8 m | **1.2 m** |

#### 2. 边界围栏系统

新增 `_create_boundary_fence()` 方法，创建可视化边界围栏：

| 组件 | 颜色 | 说明 |
|------|------|------|
| 立柱 | 橙黄色 | 每 2m 一根，高度 5m |
| 横杆 | 亮橙色 | 顶部和中部各一根 |
| 角落标记 | 橙红色 | 四角高亮立柱 |
| 地面边界线 | 橙色 | 地面警示线 |

```python
def _create_boundary_fence(self):
    fence_height = 5.0      # 围栏高度
    post_spacing = 2.0      # 立柱间距
    limit = self.boundary_limit  # ±15m
```

#### 3. 边界约束机制

**实时位置约束** (`_update_animation`):
```python
# 限制无人机在围栏内
drone.position[0] = np.clip(drone.position[0], -self.boundary_limit, self.boundary_limit)
drone.position[1] = np.clip(drone.position[1], -self.boundary_limit, self.boundary_limit)
drone.position[2] = max(0.0, drone.position[2])  # 不能低于地面
```

**编队中心约束** (`_execute_move_forward`):
```python
# 确保编队中心不会超出安全范围
max_center = self.boundary_limit - self.formation_radius - 1.0
new_y = np.clip(new_y, -max_center, max_center)
```

**目标位置约束**:
```python
# 目标位置自动钳位到边界内
clamped_pos = (
    np.clip(pos[0], -self.boundary_limit, self.boundary_limit),
    np.clip(pos[1], -self.boundary_limit, self.boundary_limit),
    pos[2]
)
drone.set_target(clamped_pos)
```

### 修改的文件

| 文件 | 修改内容 |
|------|----------|
| `ui/swarm_view_3d.py` | 场景扩大、围栏创建、边界约束 |

### 已完成功能 (v2.4)

- [x] 场景尺寸扩大到 32×32 米
- [x] 边界围栏可视化 (橙黄色警示围栏)
- [x] 无人机实时位置边界约束
- [x] 编队中心位置边界约束
- [x] 目标位置自动钳位
- [x] 固定障碍物位置重新分布
- [x] 动态障碍物路径调整

### 待办事项 (v2.4)

- [ ] 边界接近警告 (视觉/声音提示)
- [ ] 多方向移动指令 (左/右/后退)
- [ ] 编队旋转指令
- [x] 轨迹历史显示

---

## [2025-12-18 T19] 轨迹历史显示 (v2.5)

### 功能描述

为每架无人机添加飞行轨迹历史显示功能，实时记录并可视化无人机的飞行路径。

### 核心特性

#### 1. 轨迹可视化

| 参数 | 值 | 说明 |
|------|-----|------|
| 最大轨迹点数 | 200 | 超过后自动丢弃最旧的点 |
| 采样间隔 | 每3帧 | 避免过于密集的采样 |
| 轨迹线宽 | 2.5px | 清晰可见但不干扰视野 |
| 颜色渐变 | 旧→新 | 透明度 0.2→1.0，亮度 60%→100% |

#### 2. 渐变效果

```python
# 颜色渐变算法
for i in range(n):
    alpha = 0.2 + 0.8 * (i / (n - 1))      # 透明度渐变
    brightness = 0.6 + 0.4 * (i / (n - 1))  # 亮度渐变
    colors[i] = [r * brightness, g * brightness, b * brightness, alpha]
```

- 旧轨迹点：透明度低、颜色暗淡
- 新轨迹点：透明度高、颜色鲜明
- 每架无人机使用各自的主题色

#### 3. 交互控制

| 操作 | 功能 |
|------|------|
| 点击状态栏"轨迹"标签 | 切换轨迹显示开关 |
| 紧急停止 | 自动清除所有轨迹 |

#### 4. 显示逻辑

- 只在飞行状态下记录轨迹 (z > 0.1m)
- 地面状态不记录轨迹
- 降落后保留轨迹显示
- 紧急停止后清除轨迹

### 技术实现

**DroneModel 新增属性**:
```python
trajectory_max_len = 200       # 最大轨迹点数
trajectory_sample_interval = 3  # 采样间隔
_trajectory_points: deque      # 轨迹点存储
_trajectory_visual: GLLinePlotItem  # 轨迹可视化
_trajectory_enabled: bool      # 启用状态
```

**DroneModel 新增方法**:
```python
_create_trajectory_visual()    # 创建轨迹可视化元素
update_trajectory()            # 更新轨迹（每帧调用）
_update_trajectory_visual()    # 更新轨迹线渲染
clear_trajectory()             # 清除轨迹
set_trajectory_enabled()       # 启用/禁用轨迹
```

**SwarmView3D 新增**:
```python
trajectory_enabled: bool       # 全局轨迹开关
_trajectory_label: QLabel      # 状态栏轨迹标签
toggle_trajectory()            # 切换轨迹显示
clear_all_trajectories()       # 清除所有轨迹
set_trajectory_enabled()       # 设置轨迹显示状态
```

### 修改的文件

| 文件 | 修改内容 |
|------|----------|
| `ui/swarm_view_3d.py` | DroneModel 轨迹存储与可视化、SwarmView3D 轨迹控制 |

### 已完成功能 (v2.5)

- [x] 每架无人机独立轨迹存储
- [x] 轨迹颜色渐变可视化
- [x] 状态栏轨迹开关控制
- [x] 点击切换轨迹显示
- [x] 紧急停止清除轨迹
- [x] 采样间隔优化

### 待办事项 (v2.5)

- [ ] 轨迹导出功能 (JSON/CSV)
- [ ] 轨迹回放功能
- [ ] 轨迹长度自定义设置

---

## [2025-12-18 T20] L1 基础动力学模型 (v2.6)

### 功能描述

为 Sim2Real 添加 L1 基础四旋翼动力学模型，使无人机运动更加真实。

### 动力学模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| mass | 1.5 kg | 无人机质量 |
| max_thrust | 30.0 N | 最大推力 (~2g) |
| max_velocity | 3.0 m/s | 最大速度 |
| drag_coeff | 0.5 | 线性阻力系数 |
| gravity | 9.81 m/s² | 重力加速度 |

### 控制器参数

| 参数 | 值 | 说明 |
|------|-----|------|
| kp | 2.0 | 位置比例增益 |
| kd | 1.5 | 速度阻尼增益 |

### 动力学特性

```python
class QuadrotorDynamics:
    """
    L1 基础动力学模型

    状态: [x, y, z, vx, vy, vz]
    控制: [ax_cmd, ay_cmd, az_cmd]

    特性:
    - 质量和重力
    - 推力/加速度限制
    - 速度限制
    - 线性阻力
    - 半隐式欧拉积分
    """
```

### 积分方法

```python
# 半隐式欧拉积分 (Semi-implicit Euler)
velocity = velocity + accel * dt    # 先更新速度
position = position + velocity * dt  # 再更新位置
```

### 与避障算法集成

```python
# 避障速度 -> 期望加速度
avoidance_vel = drone.compute_avoidance_velocity(obstacles)
external_accel = 2.0 * (avoidance_vel - current_vel)  # P控制
drone.step_dynamics(dt, external_accel)
```

### 修改的文件

| 文件 | 修改内容 |
|------|----------|
| `ui/swarm_view_3d.py` | 新增 QuadrotorDynamics 类，DroneModel 集成动力学 |

### 已完成功能 (v2.6)

- [x] QuadrotorDynamics 基础动力学类
- [x] 质量、推力、速度限制
- [x] 线性阻力模型
- [x] PD 位置控制器
- [x] 与避障算法集成
- [x] 边界约束同步

### 后续计划 (L2/L3)

| 层次 | 内容 | 状态 |
|------|------|------|
| L2 | 姿态动力学 (Roll/Pitch/Yaw) | 待实现 |
| L2 | 姿态 PID 控制器 | 待实现 |
| L3 | 电机动力学模型 | 待实现 |
| L3 | 气动阻力 (二次项) | 待实现 |
| L3 | 传感器噪声模型 | 待实现 |

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2025-12-16 | 基础框架搭建，四模态检测实现 |
| v1.1 | 2025-12-17 | 手势优化，无人机指令映射，手绘形状识别 |
| v1.2 | 2025-12-17 | UI交互优化，详见上方 |
| v2.3 | 2025-12-18 | 3D可视化与ROS集成，详见上方 |
| v2.4 | 2025-12-18 | 场景扩大与边界围栏，详见上方 |
| v2.5 | 2025-12-18 | 轨迹历史显示，详见上方 |
| v2.6 | 2025-12-18 | **L1 基础动力学模型 (Sim2Real)**，详见上方 |
