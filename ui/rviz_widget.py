# -*- coding: utf-8 -*-
"""
RViz 嵌入组件 - 将 RViz 窗口嵌入到 PySide6 应用中
"""

import os
import subprocess
import time
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, QProcess, Signal, Slot
from PySide6.QtGui import QWindow


class RVizWidget(QWidget):
    """
    RViz 嵌入组件

    将 RViz 窗口嵌入到 Qt 容器中，实现无人机集群可视化
    """

    # 信号
    rviz_started = Signal()
    rviz_stopped = Signal()
    rviz_error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._rviz_process: Optional[QProcess] = None
        self._rviz_window: Optional[QWindow] = None
        self._rviz_container: Optional[QWidget] = None
        self._window_id: Optional[int] = None
        self._embed_timer: Optional[QTimer] = None
        self._embed_attempts = 0
        self._max_embed_attempts = 30  # 最多尝试30次 (约6秒)

        self._setup_ui()

    def _setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 占位显示 (RViz 未启动时显示)
        self._placeholder = QFrame()
        self._placeholder.setStyleSheet("""
            QFrame {
                background-color: #1a1a2e;
                border: 2px dashed #4a4a6a;
                border-radius: 8px;
            }
        """)
        placeholder_layout = QVBoxLayout(self._placeholder)

        # 标题
        title_label = QLabel("RViz 3D 可视化")
        title_label.setStyleSheet("""
            QLabel {
                color: #8888aa;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)

        # 说明文字
        info_label = QLabel("无人机集群实时状态显示")
        info_label.setStyleSheet("color: #6666888; font-size: 12px;")
        info_label.setAlignment(Qt.AlignCenter)

        # 启动按钮
        self._btn_start = QPushButton("启动 RViz")
        self._btn_start.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 30px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self._btn_start.clicked.connect(self.start_rviz)

        # 状态标签
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #888888; font-size: 11px;")
        self._status_label.setAlignment(Qt.AlignCenter)

        placeholder_layout.addStretch()
        placeholder_layout.addWidget(title_label)
        placeholder_layout.addWidget(info_label)
        placeholder_layout.addSpacing(20)
        placeholder_layout.addWidget(self._btn_start, alignment=Qt.AlignCenter)
        placeholder_layout.addWidget(self._status_label)
        placeholder_layout.addStretch()

        layout.addWidget(self._placeholder)

        # RViz 容器 (嵌入窗口用)
        self._rviz_frame = QFrame()
        self._rviz_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
            }
        """)
        self._rviz_frame.hide()

        rviz_layout = QVBoxLayout(self._rviz_frame)
        rviz_layout.setContentsMargins(0, 0, 0, 0)
        rviz_layout.setSpacing(0)

        # 顶部工具栏
        toolbar = QFrame()
        toolbar.setFixedHeight(32)
        toolbar.setStyleSheet("""
            QFrame {
                background-color: #363636;
                border-bottom: 1px solid #4a4a4a;
            }
        """)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 0, 8, 0)

        # 状态指示
        self._rviz_status = QLabel("● RViz 运行中")
        self._rviz_status.setStyleSheet("color: #4CAF50; font-size: 12px;")

        # 停止按钮
        btn_stop = QPushButton("停止")
        btn_stop.setFixedSize(60, 24)
        btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        btn_stop.clicked.connect(self.stop_rviz)

        toolbar_layout.addWidget(self._rviz_status)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(btn_stop)

        rviz_layout.addWidget(toolbar)

        # RViz 窗口容器
        self._window_container = QWidget()
        self._window_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rviz_layout.addWidget(self._window_container)

        layout.addWidget(self._rviz_frame)

    def start_rviz(self):
        """启动 RViz"""
        if self._rviz_process is not None:
            return

        self._status_label.setText("正在启动 RViz...")
        self._btn_start.setEnabled(False)

        # 检查 ROS 环境
        ros_distro = os.environ.get('ROS_DISTRO')
        if not ros_distro:
            # 尝试设置 ROS 环境
            self._setup_ros_env()

        # 获取 catkin 工作空间路径
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        catkin_ws = os.path.join(script_dir, 'catkin_ws')

        # 构建启动命令
        launch_cmd = f"""
        source /opt/ros/noetic/setup.bash
        if [ -f "{catkin_ws}/devel/setup.bash" ]; then
            source "{catkin_ws}/devel/setup.bash"
        fi
        roslaunch swarm_visualizer swarm_visualizer.launch rviz:=true
        """

        # 使用 QProcess 启动
        self._rviz_process = QProcess(self)
        self._rviz_process.finished.connect(self._on_rviz_finished)
        self._rviz_process.errorOccurred.connect(self._on_rviz_error)

        self._rviz_process.start('bash', ['-c', launch_cmd])

        # 启动定时器尝试捕获窗口
        self._embed_attempts = 0
        self._embed_timer = QTimer(self)
        self._embed_timer.timeout.connect(self._try_embed_window)
        self._embed_timer.start(200)  # 每200ms尝试一次

    def _setup_ros_env(self):
        """设置 ROS 环境变量"""
        ros_setup = '/opt/ros/noetic/setup.bash'
        if os.path.exists(ros_setup):
            # 读取环境变量
            result = subprocess.run(
                ['bash', '-c', f'source {ros_setup} && env'],
                capture_output=True, text=True
            )
            for line in result.stdout.split('\n'):
                if '=' in line:
                    key, _, value = line.partition('=')
                    if key.startswith('ROS') or key in ['PYTHONPATH', 'LD_LIBRARY_PATH', 'CMAKE_PREFIX_PATH']:
                        os.environ[key] = value

    def _try_embed_window(self):
        """尝试捕获并嵌入 RViz 窗口"""
        self._embed_attempts += 1

        if self._embed_attempts > self._max_embed_attempts:
            self._embed_timer.stop()
            self._status_label.setText("无法嵌入 RViz 窗口，请检查 ROS 环境")
            self._btn_start.setEnabled(True)
            return

        # 使用 xdotool 查找 RViz 窗口
        try:
            result = subprocess.run(
                ['xdotool', 'search', '--name', 'RViz'],
                capture_output=True, text=True, timeout=1
            )

            window_ids = result.stdout.strip().split('\n')
            if window_ids and window_ids[0]:
                # 找到窗口，尝试嵌入
                self._window_id = int(window_ids[0])
                self._embed_timer.stop()
                self._embed_rviz_window()

        except subprocess.TimeoutExpired:
            pass
        except FileNotFoundError:
            # xdotool 未安装，使用备用方案
            self._embed_timer.stop()
            self._use_standalone_mode()
        except Exception as e:
            self._status_label.setText(f"查找窗口失败: {str(e)[:30]}")

    def _embed_rviz_window(self):
        """嵌入 RViz 窗口"""
        if self._window_id is None:
            return

        try:
            # 创建 QWindow 从窗口 ID
            self._rviz_window = QWindow.fromWinId(self._window_id)

            if self._rviz_window:
                # 创建容器 widget
                self._rviz_container = QWidget.createWindowContainer(
                    self._rviz_window, self._window_container
                )
                self._rviz_container.setSizePolicy(
                    QSizePolicy.Expanding, QSizePolicy.Expanding
                )

                # 添加到布局
                container_layout = QVBoxLayout(self._window_container)
                container_layout.setContentsMargins(0, 0, 0, 0)
                container_layout.addWidget(self._rviz_container)

                # 切换显示
                self._placeholder.hide()
                self._rviz_frame.show()

                self.rviz_started.emit()

        except Exception as e:
            self._status_label.setText(f"嵌入窗口失败: {str(e)[:30]}")
            self._use_standalone_mode()

    def _use_standalone_mode(self):
        """使用独立模式 (RViz 在单独窗口运行)"""
        self._status_label.setText("RViz 在独立窗口运行中")
        self._status_label.setStyleSheet("color: #4CAF50; font-size: 11px;")

        # 显示简化的状态面板
        self._placeholder.hide()
        self._rviz_frame.show()

        # 在容器中显示提示信息
        info_layout = QVBoxLayout(self._window_container)
        info_label = QLabel(
            "RViz 在独立窗口运行\n\n"
            "提示: 安装 xdotool 可实现窗口嵌入\n"
            "sudo apt install xdotool"
        )
        info_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 14px;
            }
        """)
        info_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(info_label)

        self.rviz_started.emit()

    def stop_rviz(self):
        """停止 RViz"""
        if self._embed_timer:
            self._embed_timer.stop()

        if self._rviz_process:
            self._rviz_process.terminate()
            self._rviz_process.waitForFinished(3000)
            if self._rviz_process.state() != QProcess.NotRunning:
                self._rviz_process.kill()
            self._rviz_process = None

        # 同时停止 roslaunch 启动的节点
        try:
            subprocess.run(['pkill', '-f', 'swarm_visualizer'], timeout=2)
            subprocess.run(['pkill', '-f', 'rviz'], timeout=2)
        except:
            pass

        self._cleanup_ui()
        self.rviz_stopped.emit()

    def _cleanup_ui(self):
        """清理 UI"""
        # 清理容器
        if self._rviz_container:
            self._rviz_container.setParent(None)
            self._rviz_container.deleteLater()
            self._rviz_container = None

        # 清理 window_container 的布局
        layout = self._window_container.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            QWidget().setLayout(layout)

        self._rviz_window = None
        self._window_id = None

        # 切换显示
        self._rviz_frame.hide()
        self._placeholder.show()
        self._btn_start.setEnabled(True)
        self._status_label.setText("")
        self._status_label.setStyleSheet("color: #888888; font-size: 11px;")

    @Slot(int, QProcess.ExitStatus)
    def _on_rviz_finished(self, exit_code, exit_status):
        """RViz 进程结束"""
        self._rviz_process = None
        self._cleanup_ui()

    @Slot(QProcess.ProcessError)
    def _on_rviz_error(self, error):
        """RViz 进程错误"""
        error_msgs = {
            QProcess.FailedToStart: "启动失败",
            QProcess.Crashed: "进程崩溃",
            QProcess.Timedout: "超时",
            QProcess.WriteError: "写入错误",
            QProcess.ReadError: "读取错误",
            QProcess.UnknownError: "未知错误",
        }
        msg = error_msgs.get(error, "错误")
        self._status_label.setText(f"RViz {msg}")
        self._btn_start.setEnabled(True)
        self.rviz_error.emit(msg)

    def is_running(self) -> bool:
        """检查 RViz 是否运行中"""
        return self._rviz_process is not None and self._rviz_process.state() == QProcess.Running

    def closeEvent(self, event):
        """关闭事件"""
        self.stop_rviz()
        super().closeEvent(event)
