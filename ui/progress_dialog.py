"""
进度对话框 - 用于显示模型加载等耗时操作的进度
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar,
    QApplication
)
from PySide6.QtCore import Qt, Signal, QTimer


class ProgressDialog(QDialog):
    """
    进度对话框

    用于显示模型加载、初始化等耗时操作的进度。
    支持确定进度和不确定进度两种模式。
    """

    # 取消信号
    cancelled = Signal()

    def __init__(self, title: str = "请稍候", parent=None):
        super().__init__(parent)
        self._setup_ui(title)

    def _setup_ui(self, title: str):
        """初始化UI"""
        self.setWindowTitle(title)
        self.setFixedSize(400, 120)
        self.setModal(True)
        self.setWindowFlags(
            Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # 状态标签
        self._status_label = QLabel("正在初始化...")
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self._status_label)

        # 进度条
        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background-color: #f5f5f5;
                height: 20px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #1e88e5;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self._progress_bar)

        # 详情标签
        self._detail_label = QLabel("")
        self._detail_label.setAlignment(Qt.AlignCenter)
        self._detail_label.setStyleSheet("color: #757575; font-size: 12px;")
        layout.addWidget(self._detail_label)

    def set_status(self, text: str):
        """设置状态文本"""
        self._status_label.setText(text)
        QApplication.processEvents()

    def set_detail(self, text: str):
        """设置详情文本"""
        self._detail_label.setText(text)
        QApplication.processEvents()

    def set_progress(self, value: int):
        """设置进度值 (0-100)"""
        self._progress_bar.setValue(value)
        QApplication.processEvents()

    def set_indeterminate(self, indeterminate: bool = True):
        """设置为不确定进度模式（动画滚动条）"""
        if indeterminate:
            self._progress_bar.setMinimum(0)
            self._progress_bar.setMaximum(0)  # 设为0表示不确定模式
        else:
            self._progress_bar.setMinimum(0)
            self._progress_bar.setMaximum(100)
        QApplication.processEvents()

    def update_for_detector(self, detector_name: str, step: int, total: int):
        """
        更新检测器加载进度

        Args:
            detector_name: 检测器名称
            step: 当前步骤
            total: 总步骤数
        """
        self.set_status(f"正在加载: {detector_name}")
        self.set_progress(int(step / total * 100))
        self.set_detail(f"步骤 {step}/{total}")


class LoadingOverlay(QDialog):
    """
    加载遮罩层

    用于在主窗口上显示半透明的加载提示
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._animate)
        self._dots = 0
        self._base_text = "加载中"

    def _setup_ui(self):
        """初始化UI"""
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        # 加载容器
        container = QLabel()
        container.setFixedSize(200, 80)
        container.setAlignment(Qt.AlignCenter)
        container.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.7);
            border-radius: 10px;
            color: white;
            font-size: 16px;
        """)
        self._container = container
        layout.addWidget(container)

    def _animate(self):
        """动画更新"""
        self._dots = (self._dots + 1) % 4
        dots = "." * self._dots
        self._container.setText(f"{self._base_text}{dots}")

    def show_loading(self, text: str = "加载中"):
        """显示加载提示"""
        self._base_text = text
        self._container.setText(text)
        self._animation_timer.start(500)
        self.show()
        QApplication.processEvents()

    def hide_loading(self):
        """隐藏加载提示"""
        self._animation_timer.stop()
        self.hide()
