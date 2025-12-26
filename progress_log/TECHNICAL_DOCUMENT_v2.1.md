# 多模态无人机集群控制系统 - 技术文档

> **版本**: v2.2
> **更新日期**: 2025-12-17
> **作者**: 开发团队
> **状态**: 激光雷达传感模拟、APF避障算法、动态障碍物、状态监控面板已实现

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [技术栈](#3-技术栈)
4. [功能模块](#4-功能模块)
5. [ROS 集成](#5-ros-集成)
6. [界面设计](#6-界面设计)
7. [文件结构](#7-文件结构)
8. [API 参考](#8-api-参考)
9. [安装部署](#9-安装部署)
10. [使用指南](#10-使用指南)
11. [障碍物与避障系统](#11-障碍物与避障系统)
12. [开发日志](#12-开发日志)
13. [待办事项](#13-待办事项)

---

## 1. 项目概述

### 1.1 项目简介

**多模态无人机集群控制系统** 是一个基于 PySide6 的桌面应用程序，通过融合多种人机交互方式（语音、手势、图像、触屏）实现对无人机集群的直观控制。系统与 ROS (Robot Operating System) 深度集成，支持在 RViz 中实时可视化无人机集群状态。

### 1.2 核心特性

| 特性 | 描述 |
|------|------|
| **四模态输入** | 语音识别 (Whisper)、手势识别 (MediaPipe)、图像识别 (YOLOv8)、触屏手绘 |
| **无人机指令映射** | 手势→起飞/降落/悬停/编队/高度控制 |
| **手绘编队控制** | 绘制三角形/正方形/圆形/五角星切换编队 |
| **ROS 集成** | 通过 rospy 发布控制指令到 ROS 话题 |
| **PyQtGraph 3D 可视化** | 原生嵌入的无人机集群 3D 场景渲染 |
| **障碍物场景** | 静态柱形点云障碍物 + 动态移动障碍物 |
| **激光雷达模拟** | 2D 激光扫描传感器模拟与可视化 |
| **APF 避障算法** | 人工势场法实现自主避障飞行 |
| **状态监控面板** | 实时显示每架无人机位置、状态、障碍物距离 |
| **实时视频流** | 摄像头画面叠加检测结果 |

### 1.3 应用场景

- 无人机集群演示系统
- 人机交互研究平台
- 多模态融合算法验证
- ROS 机器人控制原型开发

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PySide6 桌面应用程序                              │
├──────────────┬────────────────────────────────────┬─────────────────────┤
│   控制面板    │           中央显示区域              │     历史记录表       │
│              │  ┌─────────────────────────────┐   │                     │
│  - 模态开关   │  │     RViz 3D 可视化 (嵌入)    │   │  - 检测结果列表      │
│  - 阈值设置   │  │     无人机集群实时状态        │   │  - 关键词搜索        │
│  - 摄像头选择 │  └─────────────────────────────┘   │  - 模态筛选          │
│  - 录音按钮   │  ┌─────────────────────────────┐   │  - JSON 导出        │
│  - 重置/导出  │  │     视频流 + 检测叠加         │   │                     │
│              │  │     手势/目标检测/手绘轨迹     │   │                     │
│              │  └─────────────────────────────┘   │                     │
└──────────────┴────────────────────────────────────┴─────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │         ROS Bridge            │
                    │    (rospy 通信桥接模块)        │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            /swarm/command   /swarm/formation  /swarm/visualization
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │    swarm_visualizer_node      │
                    │    (ROS 可视化节点)            │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │         RViz 显示              │
                    │    (嵌入到 PySide6 界面)       │
                    └───────────────────────────────┘
```

### 2.2 数据流图

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   摄像头      │────▶│  帧处理器     │────▶│  视频显示     │
│  (OpenCV)    │     │  (30 FPS)    │     │ (VideoWidget)│
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  手势检测     │     │  图像检测     │     │  触屏检测     │
│ (MediaPipe)  │     │  (YOLOv8)    │     │ (手绘形状)    │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            ▼
                    ┌──────────────┐
                    │  检测结果     │
                    │ (Detection   │
                    │   Result)    │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐    ┌──────────────┐   ┌──────────────┐
│  历史记录     │    │  ROS 发布     │   │  UI 状态更新  │
│  (Table)     │    │  (Bridge)    │   │  (OSD/Card)  │
└──────────────┘    └──────────────┘   └──────────────┘
```

### 2.3 模块依赖关系

```
main.py
    └── ui/
        ├── MainWindow
        │   ├── ControlPanel      (控制面板)
        │   ├── RVizWidget        (RViz 嵌入)  [NEW]
        │   ├── VideoWidget       (视频显示)
        │   ├── HistoryTable      (历史记录)
        │   └── ProgressDialog    (进度对话框)
        │
        └── workers/
            └── CameraWorker      (摄像头管理)

        └── detectors/
            ├── VoiceDetector     (语音识别)
            ├── GestureDetector   (手势识别)
            ├── ImageDetector     (图像识别)
            └── TouchDetector     (触屏/手绘)

        └── ros_bridge/           [NEW]
            └── ROSBridge         (ROS 通信)
```

---

## 3. 技术栈

### 3.1 核心依赖

| 组件 | 版本 | 用途 |
|------|------|------|
| **Python** | 3.10 | 主开发语言 |
| **PySide6** | 6.x | GUI 框架 |
| **OpenCV** | 4.x | 图像处理/视频采集 |
| **PyTorch** | 2.x | 深度学习框架 |
| **CUDA** | 11.8 | GPU 加速 |

### 3.2 AI/ML 模型

| 模型 | 库 | 用途 |
|------|------|------|
| **Whisper** | openai-whisper | 语音识别 |
| **MediaPipe Hands** | mediapipe | 手势识别 |
| **YOLOv8n** | ultralytics | 目标检测 |

### 3.3 ROS 生态

| 组件 | 版本 | 用途 |
|------|------|------|
| **ROS** | Noetic (ROS1) | 机器人中间件 |
| **rospy** | - | Python ROS 客户端 |
| **RViz** | - | 3D 可视化工具 |
| **visualization_msgs** | - | 可视化消息类型 |

### 3.4 运行环境

| 项目 | 要求 |
|------|------|
| **操作系统** | Ubuntu 20.04 LTS |
| **显卡** | NVIDIA (支持 CUDA 11.8) |
| **内存** | ≥ 8GB |
| **摄像头** | USB 或内置摄像头 |
| **麦克风** | 用于语音识别 |

---

## 4. 功能模块

### 4.1 语音识别模块 (VoiceDetector)

**功能**: 通过 Whisper 模型将语音转换为文本，解析无人机控制指令。

**支持的语音指令**:

| 关键词 (中文) | 关键词 (英文) | 对应指令 |
|--------------|--------------|----------|
| 起飞、升空 | takeoff, take off | TAKEOFF |
| 降落、着陆 | land | LAND |
| 悬停、停止 | hover, stop | HOVER |
| 上升、升高 | go up, higher | ALTITUDE_UP |
| 下降、降低 | go down, lower | ALTITUDE_DOWN |
| 紧急停止、急停 | emergency, stop | EMERGENCY_STOP |
| 三角形、三角 | triangle | 三角形编队 |
| 正方形、方形 | square | 正方形编队 |
| 圆形、圆 | circle | 圆形编队 |
| 五角星、星形 | star | 五角星编队 |
| 直线、一字 | line | 直线编队 |

**技术参数**:
- 模型: Whisper base
- 采样率: 16000 Hz
- 录音方式: 点击录音

### 4.2 手势识别模块 (GestureDetector)

**功能**: 使用 MediaPipe Hands 检测手部关键点，识别预定义手势并映射到无人机指令。

**手势映射表**:

| 手势 | 检测规则 | 无人机指令 |
|------|----------|-----------|
| 张开手掌 (Open_Palm) | 所有手指伸直 | 集群起飞 |
| 握拳 (Closed_Fist) | 所有手指弯曲 | 集群降落 |
| 食指向上 (Pointing_Up) | 仅食指伸直 | 集群悬停 |
| ILY 手势 (ILoveYou) | 大拇指+食指+小指伸直 | 编队飞行 |
| OK 手势 (Victory) | 比耶手势 | 指令确认 |
| 竖起大拇指 (Thumb_Up) | 大拇指向上 | 高度上升 |
| 向下大拇指 (Thumb_Down) | 大拇指向下 | 高度下降 |

**技术参数**:
- 检测置信度: ≥ 0.7
- 稳定阈值: 连续 5 帧
- 轨迹长度: 5 帧
- 支持双手独立追踪

### 4.3 图像识别模块 (ImageDetector)

**功能**: 使用 YOLOv8 检测视频帧中的目标物体。

**技术参数**:
- 模型: YOLOv8n (轻量版)
- 支持 COCO 80 类目标
- GPU 加速推理
- 检测框实时绘制

### 4.4 触屏手绘模块 (TouchDetector)

**功能**:
1. 检测鼠标点击/拖动事件
2. 识别手绘闭合形状
3. 将形状映射到编队指令

**支持的形状**:

| 手绘形状 | 识别条件 | 编队类型 |
|----------|----------|----------|
| 三角形 | 3 顶点 + 闭合 | 三角形编队 |
| 正方形 | 4 顶点 + 闭合 | 正方形编队 |
| 圆形 | 圆度 > 0.7 + 闭合 | 圆形编队 |
| 五角星 | ≥4 凹陷点 + 闭合 | 五角星编队 |

**识别算法**:
```python
# 1. 闭合检测
closed = distance(start, end) < 0.15 * max_dimension

# 2. 多边形近似
approx = cv2.approxPolyDP(contour, epsilon, closed=True)
vertices = len(approx)

# 3. 圆度计算
circularity = 4 * pi * area / (perimeter ** 2)

# 4. 凹陷检测
defects = cv2.convexityDefects(contour, hull)
```

---

## 5. ROS 集成

### 5.1 架构设计

系统采用 **PySide6 + ROS Bridge** 架构:

```
┌─────────────────────────────────────────────────────────────┐
│                    PySide6 应用层                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ 手势检测    │  │ 语音识别     │  │ 触屏手绘             │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                    │              │
│         └────────────────┼────────────────────┘              │
│                          ▼                                   │
│              ┌─────────────────────┐                        │
│              │     ROS Bridge      │                        │
│              │  publish_command()  │                        │
│              │  publish_formation()│                        │
│              └──────────┬──────────┘                        │
└─────────────────────────┼───────────────────────────────────┘
                          │ rospy
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      ROS 层                                  │
│                                                             │
│  ┌─────────────────┐                                        │
│  │ /swarm/command  │◄────── 控制指令 (JSON)                  │
│  │ /swarm/formation│◄────── 编队变换 (JSON)                  │
│  │ /swarm/status   │──────► 状态反馈 (JSON)                  │
│  └────────┬────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────┐               │
│  │        swarm_visualizer_node            │               │
│  │  - FormationGenerator (编队位置生成)     │               │
│  │  - 动画插值系统                          │               │
│  │  - Marker 发布                          │               │
│  └────────────────────┬────────────────────┘               │
│                       │                                     │
│                       ▼                                     │
│  ┌─────────────────────────────────────────┐               │
│  │    /swarm/visualization (MarkerArray)   │               │
│  └────────────────────┬────────────────────┘               │
│                       │                                     │
│                       ▼                                     │
│  ┌─────────────────────────────────────────┐               │
│  │              RViz 显示                   │               │
│  │   - 无人机模型 (机身/机臂/旋翼)           │               │
│  │   - 编队连线                             │               │
│  │   - 地面网格                             │               │
│  │   - 文字标签                             │               │
│  └─────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 ROS 话题定义

| 话题名称 | 消息类型 | 方向 | 描述 |
|----------|----------|------|------|
| `/swarm/command` | `std_msgs/String` | 发布 | 集群控制指令 (JSON) |
| `/swarm/formation` | `std_msgs/String` | 发布 | 编队变换指令 (JSON) |
| `/swarm/status` | `std_msgs/String` | 订阅 | 集群状态反馈 (JSON) |
| `/swarm/visualization` | `visualization_msgs/MarkerArray` | 发布 | RViz 可视化数据 |

### 5.3 消息格式

**控制指令** (`/swarm/command`):
```json
{
    "command": "takeoff",
    "timestamp": "2025-12-17T15:30:00.000000",
    "params": {}
}
```

**编队变换** (`/swarm/formation`):
```json
{
    "formation": "triangle",
    "drone_count": 6,
    "timestamp": "2025-12-17T15:30:00.000000"
}
```

**状态反馈** (`/swarm/status`):
```json
{
    "is_flying": true,
    "formation": "circle",
    "altitude": 1.5,
    "drone_count": 6,
    "positions": [[0, 2, 1.5], [1.73, 1, 1.5], ...]
}
```

### 5.4 编队类型

| 编队名称 | 描述 | 参数 |
|----------|------|------|
| `triangle` | 三角形编队 | 最多 3 架 |
| `square` | 正方形编队 | 最多 4 架 |
| `circle` | 圆形编队 | 支持任意数量 |
| `star` | 五角星编队 | 最多 5 架 |
| `line` | 直线编队 | 支持任意数量 |

### 5.5 控制指令

| 指令 | 值 | 描述 |
|------|------|------|
| `takeoff` | TAKEOFF | 集群起飞到预设高度 |
| `land` | LAND | 集群降落到地面 |
| `hover` | HOVER | 原地悬停 |
| `altitude_up` | ALTITUDE_UP | 高度上升 0.5m |
| `altitude_down` | ALTITUDE_DOWN | 高度下降 0.5m |
| `emergency` | EMERGENCY_STOP | 紧急停止，立即着陆 |
| `formation` | FORMATION | 触发编队切换 |
| `confirm` | CONFIRM | 确认当前指令 |

### 5.6 可视化节点 (swarm_visualizer_node)

**文件位置**: `catkin_ws/src/swarm_visualizer/scripts/swarm_visualizer_node.py`

**功能**:
1. 订阅控制指令和编队变换话题
2. 生成编队位置坐标
3. 发布可视化 Marker 到 RViz
4. 动画过渡效果 (平滑插值)

**无人机模型组成**:
- 机身: 圆柱体 (彩色，根据 ID 分配)
- 机臂: 4 个圆柱体 (X 形布局)
- 旋翼: 4 个扁平圆柱体 (前红后黑)
- 标签: 文字 (UAV1, UAV2, ...)

**编队位置生成器** (FormationGenerator):
```python
# 三角形编队
def triangle(center, radius, altitude, count=3):
    for i in range(count):
        angle = 2 * pi * i / count - pi / 2
        x = center[0] + radius * cos(angle)
        y = center[1] + radius * sin(angle)
        positions.append((x, y, altitude))

# 圆形编队
def circle(center, radius, altitude, count=6):
    for i in range(count):
        angle = 2 * pi * i / count
        x = center[0] + radius * cos(angle)
        y = center[1] + radius * sin(angle)
        positions.append((x, y, altitude))
```

---

## 6. 界面设计

### 6.1 整体布局 (v2.0)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Multimodal Detector                          │
├───────────┬───────────────────────────────────────┬─────────────┤
│           │          RViz 3D 可视化 (上部)         │             │
│  控制面板  │  ┌───────────────────────────────────┐ │  历史记录表  │
│           │  │                                   │ │             │
│ ┌───────┐ │  │   ● RViz 运行中          [停止]   │ │ ┌─────────┐ │
│ │模态开关│ │  │  ┌─────────────────────────────┐ │ │ │搜索框   │ │
│ │☑ 语音 │ │  │  │                             │ │ │ │筛选下拉 │ │
│ │☑ 手势 │ │  │  │    无人机集群 3D 显示        │ │ │ └─────────┘ │
│ │☑ 图像 │ │  │  │    (嵌入的 RViz 窗口)        │ │ │ ┌─────────┐ │
│ │☑ 触屏 │ │  │  │                             │ │ │ │时间│模态│ │
│ └───────┘ │  │  └─────────────────────────────┘ │ │ │指令│置信│ │
│ ┌───────┐ │  └───────────────────────────────────┘ │ │度│参数│ │
│ │阈值设置│ │──────────────────────────────────────│ │...│...│ │
│ │融合:0.6│ │          视频流显示 (下部)           │ │ └─────────┘ │
│ │召回:0.5│ │  ┌───────────────────────────────┐   │             │
│ └───────┘ │  │  ┌──────────────────┐          │   │ [清除] [导出]│
│ ┌───────┐ │  │  │FPS: 30.0         │  摄像头  │   │             │
│ │摄像头  │ │  │  │置信度: 85%       │  实时画面│   │             │
│ │[下拉框]│ │  │  │命令: 集群起飞    │  +检测叠加│   │             │
│ │[刷新]  │ │  │  └──────────────────┘          │   │             │
│ └───────┘ │  │                                 │   │             │
│ ┌───────┐ │  │    ← 手势关键点 / 检测框 →      │   │             │
│ │[🎤录音]│ │  │    ← 手绘轨迹 / 标准形状 →      │   │             │
│ │状态文字│ │  │                                 │   │             │
│ └───────┘ │  └───────────────────────────────────┘   │             │
│           │  [启动摄像头]        [停止摄像头]        │             │
│[重置][导出]│                                        │             │
└───────────┴───────────────────────────────────────┴─────────────┘
```

### 6.2 中央区域布局

中央区域采用 **垂直分割** (`QSplitter(Qt.Vertical)`):

| 区域 | 组件 | 功能 |
|------|------|------|
| **上部** | RVizWidget | 嵌入 RViz 3D 可视化 |
| **下部** | VideoWidget | 摄像头视频流 + 检测叠加 |

**分割比例**: 默认 1:1，用户可拖动调整

### 6.3 RViz 嵌入组件 (RVizWidget)

**文件位置**: `ui/rviz_widget.py`

**功能**:
1. 启动 RViz 进程 (`roslaunch`)
2. 通过 `xdotool` 查找 RViz 窗口 ID
3. 使用 `QWindow.fromWinId()` 嵌入窗口
4. 提供启动/停止控制

**状态显示**:
- 未启动: 显示占位符 + "启动 RViz" 按钮
- 运行中: 显示状态栏 (绿色指示灯) + "停止" 按钮
- 独立模式: 如无法嵌入，显示提示信息

### 6.4 快捷键

| 快捷键 | 功能 |
|--------|------|
| `空格` | 开始/停止录音 |
| `S` | 启动/停止摄像头 |
| `R` | 重置统计 |
| `E` | 导出历史 |
| `1` | 切换语音识别 |
| `2` | 切换手势识别 |
| `3` | 切换图像识别 |
| `4` | 切换触屏指令 |
| `F1` | 显示快捷键帮助 |

---

## 7. 文件结构

```
multimodal_detector/
├── main.py                          # 应用入口
├── environment.yml                  # Conda 环境配置
├── requirements.txt                 # pip 依赖
├── README.md                        # 项目说明
├── PROGRESS.md                      # 开发进展 (旧)
├── yolov8n.pt                       # YOLOv8 模型权重
├── start_ros_demo.sh                # ROS 演示启动脚本 [NEW]
│
├── ui/                              # 界面模块
│   ├── __init__.py
│   ├── styles.py                    # QSS 样式定义
│   ├── main_window.py               # 主窗口 (含 ROS 集成)
│   ├── control_panel.py             # 左栏控制面板
│   ├── video_widget.py              # 视频显示组件
│   ├── history_table.py             # 历史记录表格
│   ├── progress_dialog.py           # 进度对话框
│   └── rviz_widget.py               # RViz 嵌入组件 [NEW]
│
├── workers/                         # 工作线程模块
│   ├── __init__.py
│   └── camera_worker.py             # 摄像头管理
│
├── detectors/                       # 检测器模块
│   ├── __init__.py
│   ├── base_detector.py             # 检测器基类
│   ├── voice_detector.py            # 语音识别 (Whisper)
│   ├── gesture_detector.py          # 手势识别 (MediaPipe)
│   ├── image_detector.py            # 图像识别 (YOLOv8)
│   └── touch_detector.py            # 触屏/手绘检测
│
├── ros_bridge/                      # ROS 桥接模块 [NEW]
│   ├── __init__.py
│   └── ros_bridge.py                # ROS 通信实现
│
├── catkin_ws/                       # ROS 工作空间 [NEW]
│   ├── src/
│   │   └── swarm_visualizer/        # 可视化包
│   │       ├── CMakeLists.txt
│   │       ├── package.xml
│   │       ├── scripts/
│   │       │   └── swarm_visualizer_node.py  # 可视化节点
│   │       ├── launch/
│   │       │   └── swarm_visualizer.launch   # 启动文件
│   │       └── rviz/
│   │           └── swarm.rviz       # RViz 配置
│   ├── build/                       # 构建目录
│   └── devel/                       # 开发空间
│
├── progress_log/                    # 进展日志目录 [NEW]
│   └── TECHNICAL_DOCUMENT_v2.0.md   # 本技术文档
│
└── github_reference_project/        # 参考项目
    ├── ego-planner-swarm/           # ZJU FAST Lab
    └── RACER/                       # Robotics STAR Lab
```

---

## 8. API 参考

### 8.1 ROSBridge 类

**文件**: `ros_bridge/ros_bridge.py`

```python
class ROSBridge:
    """ROS 通信桥接类"""

    def __init__(self, node_name: str = "multimodal_detector"):
        """初始化 ROS Bridge"""

    @property
    def is_available(self) -> bool:
        """检查 ROS 是否可用"""

    @property
    def is_initialized(self) -> bool:
        """检查是否已初始化"""

    def initialize(self) -> bool:
        """
        初始化 ROS 节点和话题
        Returns: 是否初始化成功
        """

    def shutdown(self):
        """关闭 ROS 连接"""

    def publish_command(self, command: SwarmCommand, params: dict = None):
        """
        发布集群控制指令
        Args:
            command: SwarmCommand 枚举值
            params: 附加参数字典
        """

    def publish_formation(self, formation: FormationType, drone_count: int = 6):
        """
        发布编队变换指令
        Args:
            formation: FormationType 枚举值
            drone_count: 无人机数量
        """

    def publish_drone_markers(self, positions: List[tuple], colors: List[tuple] = None):
        """
        发布无人机可视化 Marker
        Args:
            positions: 位置列表 [(x, y, z), ...]
            colors: 颜色列表 [(r, g, b, a), ...]
        """
```

### 8.2 SwarmCommand 枚举

```python
class SwarmCommand(Enum):
    TAKEOFF = "takeoff"              # 集群起飞
    LAND = "land"                    # 集群降落
    HOVER = "hover"                  # 集群悬停
    FORMATION = "formation"          # 编队飞行
    CONFIRM = "confirm"              # 指令确认
    ALTITUDE_UP = "altitude_up"      # 高度上升
    ALTITUDE_DOWN = "altitude_down"  # 高度下降
    EMERGENCY_STOP = "emergency"     # 紧急停止
```

### 8.3 FormationType 枚举

```python
class FormationType(Enum):
    TRIANGLE = "triangle"    # 三角形编队
    SQUARE = "square"        # 正方形编队
    CIRCLE = "circle"        # 圆形编队
    STAR = "star"            # 五角星编队
    LINE = "line"            # 直线编队
    CUSTOM = "custom"        # 自定义编队
```

### 8.4 RVizWidget 类

**文件**: `ui/rviz_widget.py`

```python
class RVizWidget(QWidget):
    """RViz 嵌入组件"""

    # 信号
    rviz_started = Signal()           # RViz 启动完成
    rviz_stopped = Signal()           # RViz 已停止
    rviz_error = Signal(str)          # RViz 错误

    def start_rviz(self):
        """启动 RViz"""

    def stop_rviz(self):
        """停止 RViz"""

    def is_running(self) -> bool:
        """检查 RViz 是否运行中"""
```

---

## 9. 安装部署

### 9.1 系统要求

- Ubuntu 20.04 LTS
- NVIDIA 显卡 (支持 CUDA 11.8)
- Python 3.10
- ROS Noetic

### 9.2 安装步骤

**1. 安装 ROS Noetic**
```bash
# 添加 ROS 源
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update

# 安装 ROS Noetic
sudo apt install ros-noetic-desktop-full

# 初始化 rosdep
sudo rosdep init
rosdep update
```

**2. 创建 Conda 环境**
```bash
cd /home/ubuntu/NGW/intern/multimodal_detector
conda env create -f environment.yml
conda activate multimodal
```

**3. 安装额外依赖**
```bash
# 窗口嵌入工具 (可选，用于嵌入 RViz)
sudo apt install xdotool

# 中文字体
sudo apt install fonts-wqy-zenhei fonts-wqy-microhei
```

**4. 构建 ROS 工作空间**
```bash
cd /home/ubuntu/NGW/intern/multimodal_detector/catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make
```

**5. 验证安装**
```bash
# 验证 ROS 环境
source devel/setup.bash
rospack find swarm_visualizer

# 验证 Python 环境
python -c "import rospy; print('rospy OK')"
python -c "from PySide6.QtWidgets import QApplication; print('PySide6 OK')"
```

### 9.3 环境配置

将以下内容添加到 `~/.bashrc`:
```bash
# ROS 环境
source /opt/ros/noetic/setup.bash
source /home/ubuntu/NGW/intern/multimodal_detector/catkin_ws/devel/setup.bash

# Conda 环境
conda activate multimodal
```

---

## 10. 使用指南

### 10.1 快速启动

**方式一: 使用启动脚本**
```bash
cd /home/ubuntu/NGW/intern/multimodal_detector
./start_ros_demo.sh
```

脚本提供三种启动模式:
1. 仅启动 RViz 可视化
2. 仅启动 PySide6 检测器
3. 完整启动 (ROS + PySide6)

**方式二: 手动启动**

终端 1 - ROS 可视化:
```bash
source /opt/ros/noetic/setup.bash
source catkin_ws/devel/setup.bash
roslaunch swarm_visualizer swarm_visualizer.launch
```

终端 2 - PySide6 应用:
```bash
conda activate multimodal
python main.py
```

### 10.2 界面操作

**启动 RViz 可视化**:
1. 启动应用后，中央上部显示 RViz 占位区域
2. 点击 "启动 RViz" 按钮
3. 等待 RViz 启动并嵌入 (约 3-5 秒)
4. 看到绿色状态指示灯表示成功

**启动摄像头**:
1. 选择摄像头设备 (下拉框)
2. 点击 "启动摄像头" 按钮
3. 首次启动会初始化 AI 模型 (显示进度条)

**手势控制无人机**:
1. 勾选 "手势识别"
2. 对摄像头做手势
3. 观察右上角 OSD 显示识别结果
4. 在 RViz 中查看无人机响应

**手绘编队控制**:
1. 勾选 "触屏指令"
2. 在视频区域用鼠标绘制闭合形状
3. 松开鼠标后系统识别形状
4. 视频中央显示标准形状
5. RViz 中无人机切换编队

**语音控制**:
1. 勾选 "语音识别"
2. 点击录音按钮 (或按空格键)
3. 说出指令 (如 "起飞"、"三角形编队")
4. 再次点击停止录音
5. 系统识别并执行指令

### 10.3 常见问题

**Q: RViz 无法嵌入，显示独立窗口**
A: 安装 xdotool: `sudo apt install xdotool`

**Q: 中文显示乱码**
A: 安装中文字体: `sudo apt install fonts-wqy-zenhei`

**Q: 摄像头无法打开**
A: 检查权限: `ls -la /dev/video*`，确保用户在 video 组

**Q: ROS 连接失败**
A: 确保 roscore 运行中: `roscore`

**Q: 手势识别不稳定**
A: 调整光线条件，确保手部完整进入画面

---

## 11. 开发日志

### v1.0 (2025-12-16)
- 项目初始化，基础框架搭建
- 四模态检测器实现
- 三栏布局 UI

### v1.1 (2025-12-17 上午)
- 手势识别优化 (稳定性、轨迹分离)
- 手势→无人机指令映射
- 手绘形状识别功能
- 视频 OSD 信息显示

### v1.2 (2025-12-17 下午)
- UI 交互优化
- 模型加载进度对话框
- 快捷键支持
- 摄像头切换功能
- 历史记录筛选

### v2.0 (2025-12-17 晚间)
- **ROS 集成** (ros_bridge 模块)
- **swarm_visualizer 包** (ROS 可视化节点)
- **RViz 嵌入界面** (中央上下分割布局)
- 检测结果→ROS 指令发布
- 语音指令解析
- 启动脚本

### v2.1 (2025-12-17 深夜) - 3D 可视化重构
- **PyQtGraph 原生 3D 渲染** (替代 RViz 窗口嵌入)
- 无人机模型优化 (机身+机臂+旋翼)
- 编队动画平滑过渡
- 完全嵌入 PySide6，无外部窗口依赖

---

## 12. 3D 可视化方案对比

在实现无人机集群 3D 可视化时，我们评估了三种技术方案：

### 12.1 方案 A：RViz 窗口嵌入 (已废弃)

**技术路线**:
- 使用 `roslaunch` 启动 RViz
- 通过 `xdotool` 查找 RViz 窗口 ID
- 使用 `QWindow.fromWinId()` 嵌入到 PySide6

**优点**:
- 保留 RViz 完整功能
- 可直接使用 ROS 生态工具

**缺点**:
- 依赖窗口管理器，Linux 下不稳定
- 需要安装 xdotool
- 窗口嵌入可能失败，回退到独立窗口

**状态**: ❌ 已废弃 (稳定性问题)

---

### 12.2 方案 B：PyQtGraph 原生 3D 渲染 (已采用) ✅

**技术路线**:
- 使用 `pyqtgraph.opengl` 模块
- 在 PySide6 内部原生渲染 3D 场景
- 自定义无人机模型 (球体+圆柱体+线段)

**优点**:
- 完全嵌入，无外部窗口
- 稳定可靠，跨平台
- 渲染效果可完全自定义
- 性能优秀，60+ FPS

**缺点**:
- 需要自行实现可视化效果
- 功能不如 RViz 丰富

**依赖**:
```bash
pip install pyqtgraph PyOpenGL PyOpenGL_accelerate
```

**状态**: ✅ **已采用** (v2.1)

**核心代码** (`ui/swarm_view_3d.py`):
```python
class SwarmView3D(QWidget):
    """无人机集群 3D 可视化组件"""

    def execute_command(self, command: str):
        """执行控制指令 (起飞/降落/编队等)"""

    def change_formation(self, formation: str, drone_count: int):
        """切换编队类型"""

class DroneModel:
    """单个无人机 3D 模型"""
    # 机身: 球体
    # 机臂: 4条线段 (X形)
    # 旋翼: 4个圆环 (前红后黑)
```

---

### 12.3 方案 C：Web 可视化 (ros3djs) (备选)

**技术路线**:
- 使用 `rosbridge_server` 建立 WebSocket 连接
- 在 `QWebEngineView` 中嵌入 ros3djs 页面
- 通过 JavaScript 渲染 3D 场景

**优点**:
- 跨平台，美观
- 可复用 Web 前端技术
- 支持远程访问

**缺点**:
- 需要 rosbridge_server 依赖
- WebSocket 通信有延迟
- 调试复杂

**状态**: 📋 备选方案 (未实现)

---

### 12.4 方案对比总结

| 特性 | 方案 A (RViz) | 方案 B (PyQtGraph) | 方案 C (ros3djs) |
|------|--------------|-------------------|-----------------|
| **稳定性** | ⚠️ 中等 | ✅ 高 | ✅ 高 |
| **集成度** | ⚠️ 外部窗口 | ✅ 完全嵌入 | ✅ 完全嵌入 |
| **性能** | ✅ 高 | ✅ 高 | ⚠️ 中等 |
| **依赖** | ROS + xdotool | PyQtGraph | rosbridge |
| **开发难度** | 低 | 中 | 高 |
| **可定制性** | ⚠️ 有限 | ✅ 完全 | ✅ 完全 |

**最终选择**: **方案 B (PyQtGraph)** - 稳定性最佳，完全嵌入，性能优秀

---

## 13. 待办事项

### 高优先级
- [ ] 添加无人机轨迹历史显示
- [ ] 实现编队平滑过渡动画优化
- [ ] 添加多机协同任务规划

### 中优先级
- [ ] 性能优化 (检测器多线程)
- [ ] 配置文件支持 (阈值持久化)
- [ ] 更多手势支持
- [ ] Gazebo 仿真集成

### 低优先级
- [ ] 多语言 UI
- [ ] 更多形状支持 (箭头、菱形等)
- [ ] 手势组合指令
- [ ] 移动端控制接口

---

## 附录

### A. 参考项目

1. **ego-planner-swarm** (ZJU FAST Lab)
   - GitHub: https://github.com/ZJU-FAST-Lab/ego-planner-swarm
   - 参考: 无人机可视化方案、odom_visualization

2. **RACER** (Robotics STAR Lab)
   - GitHub: https://github.com/Robotics-STAR-Lab/RACER
   - 参考: 集群规划算法

### B. 相关文档

- [ROS Noetic 官方文档](http://wiki.ros.org/noetic)
- [PySide6 官方文档](https://doc.qt.io/qtforpython/)
- [MediaPipe Hands 文档](https://google.github.io/mediapipe/solutions/hands.html)
- [YOLOv8 文档](https://docs.ultralytics.com/)
- [Whisper 文档](https://github.com/openai/whisper)

### C. 联系方式

如有问题或建议，请通过以下方式联系:
- 项目仓库 Issues
- 开发团队邮箱

---

*文档更新日期: 2025-12-17*
*版本: v2.1*
