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

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入配置和日志模块
from config import config
from utils.logger import setup_logging_from_config, get_logger

# 设置日志
setup_logging_from_config(config)
logger = get_logger(__name__)

# 设置代理 (从配置文件读取)
if config.get("proxy.enabled", False):
    http_proxy = config.get("proxy.http", "")
    https_proxy = config.get("proxy.https", "")
    if http_proxy:
        os.environ['HTTP_PROXY'] = http_proxy
        logger.debug(f"HTTP proxy set to: {http_proxy}")
    if https_proxy:
        os.environ['HTTPS_PROXY'] = https_proxy
        logger.debug(f"HTTPS proxy set to: {https_proxy}")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from ui.main_window import MainWindow


def main():
    """应用入口函数"""
    logger.info("Starting Multimodal Detector application")

    # 验证配置
    config_errors = config.validate()
    if config_errors:
        logger.warning(f"Configuration validation warnings: {config_errors}")

    # 创建应用实例
    app = QApplication(sys.argv)

    # 从配置读取应用属性
    app_name = config.get("app.name", "Multimodal Detector")
    app_version = config.get("app.version", "2.13.0")

    app.setApplicationName(app_name)
    app.setApplicationVersion(app_version)
    app.setOrganizationName("Multimodal")

    logger.info(f"Application: {app_name} v{app_version}")

    # 设置默认字体
    font = QFont("Microsoft YaHei", 10)
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)

    # 高 DPI 支持 (Qt6 默认启用，无需手动设置)

    # 创建主窗口
    try:
        window = MainWindow()
        window.show()
        logger.info("Main window created successfully")
    except Exception as e:
        logger.critical(f"Failed to create main window: {e}", exc_info=True)
        sys.exit(1)

    # 在状态栏显示欢迎消息
    window.statusBar().showMessage("欢迎使用多模态检测器！点击「启动摄像头」开始。", 5000)

    # 运行事件循环
    logger.info("Entering main event loop")
    exit_code = app.exec()

    logger.info(f"Application exited with code: {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
