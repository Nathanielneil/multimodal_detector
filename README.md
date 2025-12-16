# 多模态检测器 (Multimodal Detector)

基于 PySide6 + OpenCV 的桌面应用，集成语音识别、手势识别、图像识别和触屏指令检测四种模态。

## 功能特性

- **语音识别**: 使用 OpenAI Whisper 进行语音转文字
- **手势识别**: 使用 MediaPipe Hands 检测手部关键点并识别手势
- **图像识别**: 使用 YOLOv8m 进行实时物体检测
- **触屏指令**: 检测视频区域的鼠标点击事件

## 系统要求

- Ubuntu 20.04+
- Python 3.10
- NVIDIA GPU (可选，用于 CUDA 加速)
- 摄像头设备

## 安装步骤

### 1. 创建 Conda 环境

```bash
cd multimodal_detector
conda env create -f environment.yml
conda activate multimodal
```

### 2. 运行应用

```bash
python main.py
```

## 项目结构

```
multimodal_detector/
├── main.py                 # 应用入口
├── environment.yml         # Conda 环境配置
├── README.md
├── ui/                     # UI 组件
│   ├── __init__.py
│   ├── main_window.py      # 主窗口
│   ├── control_panel.py    # 左栏控制面板
│   ├── video_widget.py     # 中栏视频显示
│   ├── history_table.py    # 右栏历史表格
│   └── styles.py           # QSS 样式
├── workers/                # 后台工作类
│   ├── __init__.py
│   └── camera_worker.py    # 摄像头管理
└── detectors/              # 检测器模块
    ├── __init__.py
    ├── base_detector.py    # 检测器基类
    ├── voice_detector.py   # 语音识别 (Whisper)
    ├── gesture_detector.py # 手势识别 (MediaPipe)
    ├── image_detector.py   # 图像识别 (YOLOv8)
    └── touch_detector.py   # 触屏指令
```

## 使用说明

### 界面布局

- **左栏**: 控制面板 - 启用/禁用各模态、设置阈值、重置统计、导出历史
- **中栏**: 视频显示 - 实时摄像头画面，右侧显示四模态状态
- **右栏**: 历史表格 - 所有检测结果的时间线记录

### 操作流程

1. 点击「启动摄像头」按钮开始视频采集
2. 使用左侧复选框启用/禁用各模态检测
3. 点击「🎤 点击录音」按钮进行语音识别
4. 在视频区域点击鼠标触发触屏指令
5. 点击「导出历史」将检测记录保存为 JSON 文件

### 支持的手势

- 握拳
- 张开手掌
- 指向
- 竖起大拇指
- 向下大拇指
- 比V/和平
- OK手势
- 摇滚手势

## 配置选项

### 跟踪算法
- highest: 最高精度
- medium: 平衡模式
- low: 低延迟模式

### 阈值设置
- **融合阈值**: 多模态融合的置信度阈值 (默认 0.50)
- **召回阈值**: 检测结果筛选的置信度阈值 (默认 0.50)

## 依赖说明

| 依赖 | 用途 |
|------|------|
| PySide6 | GUI 框架 |
| opencv-python | 图像处理 |
| openai-whisper | 语音识别 |
| mediapipe | 手势识别 |
| ultralytics | YOLOv8 物体检测 |
| sounddevice | 音频录制 |
| pytorch + CUDA | GPU 加速 |

## 注意事项

1. 首次运行会自动下载 Whisper 和 YOLOv8 模型，请确保网络通畅
2. 语音识别需要麦克风权限
3. 摄像头需要正确连接并授权
4. GPU 加速需要正确安装 CUDA 驱动

## License

MIT License
