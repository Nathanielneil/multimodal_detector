# Ubuntu 22.04 部署指南

本分支名为 `deploy/ubuntu22.04`，目标是 Ubuntu 22.04 LTS + Python 3.10/3.11。程序默认使用 SenseVoice-Small 语音识别，首次启动时会自动下载语音模型和 YOLOv8n 权重。

## 1. 获取代码

在 Gitee 仓库创建完成后，将下面的地址替换为你的仓库地址：

```bash
git clone -b deploy/ubuntu22.04 \
  git@gitee.com:<你的用户名>/multimodal_detector.git
cd multimodal_detector
```

如果使用 HTTPS：

```bash
git clone -b deploy/ubuntu22.04 \
  https://gitee.com/<你的用户名>/multimodal_detector.git
cd multimodal_detector
```

## 2. 安装系统依赖

```bash
sudo apt update
sudo apt install -y \
  git ffmpeg \
  libgl1 libglib2.0-0 \
  libportaudio2 portaudio19-dev \
  libegl1 libxkbcommon-x11-0 libxcb-cursor0
```

这些库分别用于 Git/模型音频处理、OpenCV、PortAudio 麦克风输入和 Qt/OpenGL 图形界面。

## 3. 创建 Python 环境

### CPU 版本（最简单）

Ubuntu 22.04 的系统 Python 通常是 3.10，可直接使用虚拟环境：

```bash
sudo apt install -y python3.10-venv python3-dev
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt
```

### NVIDIA GPU 版本（可选）

先确认 NVIDIA 驱动可用：

```bash
nvidia-smi
```

然后在同一个虚拟环境中安装与显卡架构匹配的 PyTorch。RTX 50 系列（Blackwell）建议使用 CUDA 12.8 wheel；较新的 NVIDIA 驱动可以向下兼容该运行时：

```bash
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

安装后检查：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Conda 版本

有 NVIDIA GPU 时，先创建 Python 3.10 环境，再安装对应的 PyTorch wheel 和项目依赖。下面的命令适用于 RTX 50 系列：

```bash
conda env create -f environment.yml
conda activate multimodal
python -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

CPU-only 主机可使用：

```bash
conda env create -f environment-ubuntu22-cpu.yml
conda activate multimodal-cpu
```

## 4. 安装检查

```bash
python scripts/verify_install.py
```

如果要使用可选的 FunASR 引擎：

```bash
python -m pip install -r requirements-funasr.txt
python scripts/verify_install.py --with-funasr
```

## 5. 首次运行

```bash
source .venv/bin/activate       # 如果使用 venv
python main.py
```

首次初始化可能需要联网下载：

- SenseVoice-Small ONNX 模型（约 250 MB，缓存于 `~/.cache/sensevoice_small`）
- YOLOv8n 权重（缓存于 Ultralytics 默认目录）

如果所在网络需要代理，可复制配置并编辑：

```bash
cp config/default.yaml config/config.yaml
```

然后设置 `proxy.enabled`、`proxy.http` 和 `proxy.https`。`config/config.yaml` 已被 `.gitignore` 忽略，不会被提交。

默认音频设备是系统默认输入设备（`voice.audio_device: null`）。查看设备：

```bash
python -m sounddevice
```

只有在需要固定设备时，才把 `audio_device` 改成对应的整数 ID。摄像头可在左侧面板刷新并选择。

## 6. ASR 引擎选择

默认配置：

```yaml
voice:
  asr_engine: "sensevoice"
```

如果改为 `funasr`，必须先安装 `requirements-funasr.txt` 中的依赖；FunASR 模型更大、首次下载时间更长，但可使用 Paraformer 流式识别。

## 7. ROS 说明

默认 `ros.enabled: false`，不安装 ROS 也可以使用本地 3D 仿真、四种检测模态和多平台演示。

当前桥接代码使用 ROS1 `rospy`。Ubuntu 22.04 官方更适合 ROS2 Humble；如果确实需要 ROS 发布/订阅，建议在 Ubuntu 20.04 + ROS Noetic 环境或容器中运行，或者后续将 `ros_bridge` 迁移到 ROS2。

## 8. 常见问题

### Qt 启动时报 XCB/OpenGL 错误

确认第 2 步中的 Qt/XCB 和 OpenGL 系统库已安装。若日志明确提示
`libxcb-cursor0`，补装下面这组库后重新启动：

```bash
sudo apt update
sudo apt install -y \
  libxcb-cursor0 libxcb-xinerama0 libxkbcommon-x11-0 \
  libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
  libxcb-render-util0 libxcb-randr0 libxcb-shape0 \
  libxcb-xkb1 libxcb-sync1 libegl1
```

程序启动时会优先使用 PySide6 的 Qt 插件，避免 OpenCV 自带插件与 Qt 冲突。远程无桌面环境只适合做导入检查，完整界面需要可用的 X11/Wayland 显示会话。

### RTX 50 系列显示 `no kernel image is available`

通常是安装了过旧的 CUDA/PyTorch wheel。卸载当前 PyTorch，然后按本文的 `cu128` 命令重新安装；不需要为了该 wheel 单独安装完整 CUDA Toolkit，但 NVIDIA 驱动必须正常。

### 没有摄像头或麦克风

应用仍可启动并显示 3D 仿真；在左侧关闭对应模态，或连接设备后点击“刷新设备列表”。

### 模型下载失败

确认网络、代理和磁盘空间。模型权重不在 Git 仓库中，需要每台新电脑单独下载。
