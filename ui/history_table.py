"""
右栏命令历史表 - 显示所有模态的识别历史记录
"""

import json
from datetime import datetime
from typing import List, Dict, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QLabel, QFileDialog,
    QMessageBox, QLineEdit, QComboBox, QPushButton
)
from PySide6.QtCore import Qt, Signal

from detectors.base_detector import DetectionResult


class HistoryTable(QWidget):
    """
    命令历史表组件

    显示所有检测结果的历史记录，包含:
    - 时间
    - 指令
    - 检测类型
    - 置信度
    - 参数（识别详情）
    - 其他

    支持 JSON 格式导出
    """

    # 列定义
    COLUMNS = ["时间", "指令", "检测类型", "置信度", "参数", "其他"]
    COLUMN_KEYS = ["time", "command", "modal_type", "confidence", "details", "extra"]

    # 模态类型显示名称
    MODAL_DISPLAY_NAMES = {
        "voice": "语音识别",
        "gesture": "手势识别",
        "image": "图像识别",
        "touch": "触屏指令",
        "simulation": "仿真指令",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history: List[Dict[str, Any]] = []
        self._setup_ui()

    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)

        # 标题
        title = QLabel("命令历史")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(title)

        # ===== 筛选区域 =====
        filter_layout = QHBoxLayout()
        filter_layout.setSpacing(5)

        # 搜索框
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("搜索指令...")
        self._search_input.setToolTip("输入关键词筛选指令")
        self._search_input.textChanged.connect(self._apply_filter)
        filter_layout.addWidget(self._search_input, 2)

        # 模态筛选下拉框
        self._modal_filter = QComboBox()
        self._modal_filter.addItem("全部", "all")
        self._modal_filter.addItem("语音", "voice")
        self._modal_filter.addItem("手势", "gesture")
        self._modal_filter.addItem("图像", "image")
        self._modal_filter.addItem("触屏", "touch")
        self._modal_filter.addItem("仿真", "simulation")
        self._modal_filter.setToolTip("按模态类型筛选")
        self._modal_filter.currentIndexChanged.connect(self._apply_filter)
        filter_layout.addWidget(self._modal_filter, 1)

        # 清除筛选按钮
        self._btn_clear_filter = QPushButton("清除")
        self._btn_clear_filter.setProperty("class", "secondary")
        self._btn_clear_filter.setMinimumWidth(60)
        self._btn_clear_filter.setToolTip("清除筛选条件")
        self._btn_clear_filter.clicked.connect(self._clear_filter)
        filter_layout.addWidget(self._btn_clear_filter)

        layout.addLayout(filter_layout)

        # 表格
        self._table = QTableWidget()
        self._table.setColumnCount(len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(self.COLUMNS)

        # 表格设置
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 不可编辑
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)  # 整行选择
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)  # 单选
        self._table.verticalHeader().setVisible(False)  # 隐藏行号
        self._table.setAlternatingRowColors(True)  # 交替行颜色
        self._table.setSortingEnabled(False)  # 禁用排序（保持时间顺序）

        # 表头设置
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)  # 自动拉伸
        header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # 设置最小列宽
        self._table.setColumnWidth(0, 80)   # 时间
        self._table.setColumnWidth(1, 120)  # 指令
        self._table.setColumnWidth(2, 80)   # 检测类型
        self._table.setColumnWidth(3, 60)   # 置信度
        self._table.setColumnWidth(4, 150)  # 参数
        self._table.setColumnWidth(5, 80)   # 其他

        layout.addWidget(self._table)

        # 记录数统计
        self._count_label = QLabel("共 0 条记录")
        self._count_label.setStyleSheet("color: #757575; font-size: 11px;")
        layout.addWidget(self._count_label)

    def add_result(self, result: DetectionResult):
        """
        添加检测结果到历史表

        Args:
            result: 检测结果对象
        """
        # 构建历史记录
        record = {
            "time": result.timestamp.strftime("%H:%M:%S"),
            "command": result.command,
            "modal_type": self.MODAL_DISPLAY_NAMES.get(
                result.modal_type, result.modal_type
            ),
            "confidence": f"{result.confidence:.2f}",
            "details": self._format_details(result.details),
            "extra": "",
            "raw": result.to_dict(),  # 保存原始数据用于导出
        }
        self._history.append(record)

        # 添加行到表格
        row = self._table.rowCount()
        self._table.insertRow(row)

        for col, key in enumerate(self.COLUMN_KEYS):
            item = QTableWidgetItem(str(record.get(key, "")))
            item.setToolTip(str(record.get(key, "")))  # 悬停显示完整内容
            self._table.setItem(row, col, item)

        # 滚动到最新行
        self._table.scrollToBottom()

        # 更新计数
        self._update_count()

    def _format_details(self, details: Dict[str, Any]) -> str:
        """格式化详情字典为字符串"""
        if not details:
            return ""

        # 提取关键信息
        key_fields = [
            "primary_object", "gesture_type", "event_type",
            "language", "handedness", "position",
            "platform_label", "command_label", "mode"
        ]

        parts = []
        for key in key_fields:
            if key in details:
                value = details[key]
                if isinstance(value, (list, tuple)):
                    value = ",".join(map(str, value))
                parts.append(f"{value}")

        return " | ".join(parts) if parts else str(details)[:50]

    def clear(self):
        """清空历史记录"""
        self._table.setRowCount(0)
        self._history.clear()
        self._update_count()

    def _update_count(self):
        """更新记录计数"""
        count = len(self._history)
        self._count_label.setText(f"共 {count} 条记录")

    def get_history(self) -> List[Dict[str, Any]]:
        """获取所有历史记录"""
        return [record["raw"] for record in self._history]

    def export_to_json(self, file_path: str = None) -> bool:
        """
        导出历史记录为 JSON 文件

        Args:
            file_path: 文件路径，None 则弹出保存对话框

        Returns:
            bool: 是否导出成功
        """
        if not self._history:
            QMessageBox.information(self, "提示", "没有历史记录可导出")
            return False

        # 如果没有指定路径，弹出保存对话框
        if file_path is None:
            default_name = f"detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "导出历史记录",
                default_name,
                "JSON 文件 (*.json)"
            )

        if not file_path:
            return False

        try:
            export_data = {
                "export_time": datetime.now().isoformat(),
                "total_records": len(self._history),
                "records": [record["raw"] for record in self._history],
            }

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            QMessageBox.information(
                self, "导出成功",
                f"已导出 {len(self._history)} 条记录到:\n{file_path}"
            )
            return True

        except Exception as e:
            QMessageBox.critical(
                self, "导出失败",
                f"导出文件时发生错误:\n{str(e)}"
            )
            return False

    def add_sample_data(self):
        """添加示例数据（用于演示）"""
        sample_results = [
            DetectionResult(
                modal_type="voice",
                command="打开文件管理器",
                confidence=0.92,
                details={"language": "zh", "duration": 2.5}
            ),
            DetectionResult(
                modal_type="gesture",
                command="竖起大拇指",
                confidence=0.88,
                details={"handedness": "Right", "gesture_type": "THUMBS_UP"}
            ),
            DetectionResult(
                modal_type="image",
                command="检测到: person",
                confidence=0.95,
                details={"primary_object": "person", "total_count": 3}
            ),
            DetectionResult(
                modal_type="touch",
                command="单击 (320, 180)",
                confidence=1.0,
                details={"event_type": "CLICK", "position": (320, 180)}
            ),
        ]

        for result in sample_results:
            self.add_result(result)

    def _apply_filter(self):
        """应用筛选条件"""
        search_text = self._search_input.text().lower().strip()
        modal_filter = self._modal_filter.currentData()

        # 模态类型映射（用于匹配）
        modal_display_to_type = {
            "语音识别": "voice",
            "手势识别": "gesture",
            "图像识别": "image",
            "触屏指令": "touch",
            "仿真指令": "simulation",
        }

        visible_count = 0
        for row in range(self._table.rowCount()):
            show_row = True

            # 获取该行的指令和模态类型
            command_item = self._table.item(row, 1)  # 指令列
            modal_item = self._table.item(row, 2)    # 检测类型列

            command_text = command_item.text().lower() if command_item else ""
            modal_text = modal_item.text() if modal_item else ""
            modal_type = modal_display_to_type.get(modal_text, "")

            # 搜索筛选
            if search_text and search_text not in command_text:
                show_row = False

            # 模态筛选
            if modal_filter != "all" and modal_type != modal_filter:
                show_row = False

            self._table.setRowHidden(row, not show_row)
            if show_row:
                visible_count += 1

        # 更新计数显示
        total_count = len(self._history)
        if search_text or modal_filter != "all":
            self._count_label.setText(f"显示 {visible_count}/{total_count} 条记录")
        else:
            self._count_label.setText(f"共 {total_count} 条记录")

    def _clear_filter(self):
        """清除筛选条件"""
        self._search_input.clear()
        self._modal_filter.setCurrentIndex(0)  # 选择"全部"
        self._apply_filter()
