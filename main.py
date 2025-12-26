#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态检测器 - 主入口文件

一个基于 PySide6 + OpenCV 的桌面应用，集成了:
- 语音识别 (OpenAI Whisper)
- 手势识别 (MediaPipe Hands)
- 图像识别 (YOLOv8)
- 触屏指令检测

运行环境:
- Ubuntu 20.04+
- Python 3.10
- CUDA 11.8 (可选，用于 GPU 加速)

依赖安装:
    conda env create -f environment.yml
    conda activate multimodal

运行:
    python main.py

作者: Claude Code
"""

import sys
import os

# 设置代理 (用于下载模型，如 YOLO/Whisper)
# 如果使用 clash-verge 或其他代理，取消下面两行的注释
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from ui.main_window import MainWindow


def main():
    """应用入口函数"""
    # 创建应用实例
    app = QApplication(sys.argv)

    # 设置应用属性
    app.setApplicationName("Multimodal Detector")
    app.setApplicationVersion("1.2.0")
    app.setOrganizationName("Multimodal")

    # 设置默认字体
    font = QFont("Microsoft YaHei", 10)
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)

    # 高 DPI 支持 (Qt6 默认启用，无需手动设置)

    # 创建主窗口
    window = MainWindow()
    window.show()

    # 在状态栏显示欢迎消息
    window.statusBar().showMessage("欢迎使用多模态检测器！点击「启动摄像头」开始。", 5000)

    # 运行事件循环
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
