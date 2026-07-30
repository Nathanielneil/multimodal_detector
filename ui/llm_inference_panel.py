# -*- coding: utf-8 -*-
"""
大模型推理演示面板 - 固定剧本回放（浅色主题，带淡入动效与推理指示灯）

这是纯前端演示效果，不接入真实大模型推理。剧本文本由产品/演示需求提供，
分两阶段触发：用户输入"开始命令"后回放第一段剧本（环境扫描 + 空中集群分
配），再输入"区域信息"后回放第二段剧本（地面单位调度 + 收尾返航）。
"""

from typing import List, Tuple

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QLabel
from PySide6.QtCore import Qt, QTimer, QVariantAnimation, QAbstractAnimation, QEasingCurve
from PySide6.QtGui import QTextCursor, QTextCharFormat, QColor

_MONO_FONT = "font-family: 'Consolas', 'Menlo', 'Monaco', monospace;"

_PROMPT_STAGE0 = "[终端系统] 请输入开始命令："
_PROMPT_STAGE1 = "请在无人机到达并发现目标后，输入区域信息:（区域信息）"

_SCRIPT_STAGE0: List[str] = [
    "[系统提示] 收到指令。我将按照[环境检测与区域划分] -> [空中集群分配] -> [目标锁定与地面单位适配] -> [区域收尾与返航]的顺序列控当前任务。",
    "",
    ">>> 执行全局环境扫描与拓扑划分...",
    "  -> [思考] 接收全局地图点云，开始进行3D语义重建...",
    "  -> [思考] 提取场景边界特征，识别到大面积平坦路面与局部堆叠障碍物，综合判定当前场景为标准车库及局部复杂废墟。",
    "  -> [思考] 基于有效探索面积与无人机视野范围，通过 Voronoi 图形聚类算法，将待探索环境均匀划分为 5 个最优子区域。",
    "[执行结果] 由无人机编队起始，结合初始地图尺寸与当前可视场景的三维特征进行区域分割。当前场景评估为标准车库场景，已按均匀覆盖策略将其划分为 5 个互不重叠的探索子区域。",
    "",
    ">>> 执行空中集群最优路径分配...",
    "  -> [思考] 获取在线可用空中单位：无人机1至5及当前三维坐标。",
    "  -> [思考] 评估当前各区域先验信息较少，目标存在概率呈均匀分布。为实现探索收益最大化，建立距离代价矩阵。",
    "  -> [思考] 采用 Dijkstra 最短路径算法生成匹配解，优化集群飞行轨迹以避免航线交叉。",
    "[执行结果] 读取当前在线节点：无人机1至5及其位置状态。因初期场景先验信息不足，5个区域均具备同等目标存在概率。系统采用最短路径全局寻优原则，指派 无人机1->区域5，无人机2->区域3，无人机3->区域1，无人机4->区域2，无人机5->区域4。",
    "",
    "===========================================================================",
    "",
    "[系统状态] 无人机编队已出发，正在前往各自探索区域...",
    "[系统状态] 等待前线感知数据回传。",
]

_SCRIPT_STAGE1: List[str] = [
    "【前线信号接入】 无人机1 传回高优数据流：“区域5发现疑似目标！”",
    "【前线信号接入】 无人机2 传回高优数据流：“区域3发现疑似目标！”",
    "",
    ">>> 执行场景语义适配与地面单位调度...",
    "  -> [思考] 并发解析前线回传视觉图像...",
    "  -> [思考] 分析无人机1区域：目标周边存在大量散落碎石与断层台阶，地面崎岖度(Roughness)>0.7。轮式底盘无法通行，匹配地形技能 ->唤醒机械狗。",
    "  -> [思考] 分析无人机2区域：目标周围为平坦水泥路面，无明显障碍物，地形崎岖度<0.1。为追求最高抵近速度，匹配地形技能 ->唤醒无人车。",
    "  -> [思考] 生成地面单位行进拓扑航线，下发指令。",
    "[执行结果] 前线目标已锁定，开始分配抵近任务：",
    "- 无人机1区域评估为崎岖路面，已调度跨障能力强的机械狗前往目标执行任务；",
    "- 无人机2区域评估为平坦路面，已调度高速机动的无人车快速前往目标执行任务。",
    "",
    ">>> 执行待探索区域检查与集群收尾...",
    "  -> [思考] 持续监控全局栅格地图，实时更新占据概率图...",
    "  -> [思考] 校验区域覆盖率：无人机1负责的区域5覆盖率已达 100%，其余区域均无遗漏。",
    "  -> [思考] 遍历全局待搜索任务队列，当前高优目标数为 0，探索阶段收敛。",
    "  -> [思考] 触发任务闭环机制，为全机队(空+地)生成防碰撞返航航线。",
    "[执行结果] 区域探索校验完毕：无人机1已完整覆盖区域5且无剩余待搜索目标，该子任务结束，下达返航指令。其余无人机及地面单位状态同步核查完毕，全域探索任务顺利闭环，全编队安全返航。",
]

_PLAYBACK_INTERVAL_MS = 400
_FADE_DURATION_MS = 260

# 行样式规则：(前缀匹配, 文字颜色, 是否斜体, 是否加粗)
_LINE_STYLE_RULES: List[Tuple[str, str, bool, bool]] = [
    (">", "#0d47a1", False, True),            # 用户输入回显 "> xxx"
    ("[终端系统]", "#1565c0", False, True),
    ("[系统提示]", "#1565c0", False, True),
    ("[系统状态]", "#1565c0", False, False),
    ("请在无人机到达", "#1565c0", False, True),
    ("  -> [思考]", "#757575", True, False),
    ("[执行结果]", "#2e7d32", False, True),
    ("- ", "#2e7d32", False, False),
    ("【前线信号接入】", "#ef6c00", False, True),
    (">>> ", "#1e88e5", False, True),
    ("===", "#bdbdbd", False, False),
]
_DEFAULT_LINE_COLOR = "#424242"


def _line_style(text: str) -> Tuple[str, bool, bool]:
    for prefix, color, italic, bold in _LINE_STYLE_RULES:
        if text.startswith(prefix):
            return color, italic, bold
    return _DEFAULT_LINE_COLOR, False, False


def _blend(start: QColor, end: QColor, t: float) -> QColor:
    r = start.red() + (end.red() - start.red()) * t
    g = start.green() + (end.green() - start.green()) * t
    b = start.blue() + (end.blue() - start.blue()) * t
    return QColor(int(r), int(g), int(b))


class LLMInferencePanel(QWidget):
    """
    大模型推理演示面板：固定剧本回放（演示用，非真实推理）。

    Stage 0: 等待"开始命令" -> 回放 _SCRIPT_STAGE0 -> 提示输入区域信息
    Stage 1: 等待"区域信息" -> 回放 _SCRIPT_STAGE1 -> 结束
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stage = 0
        self._playback_lines: List[str] = []
        self._playback_index = 0

        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(_PLAYBACK_INTERVAL_MS)
        self._playback_timer.timeout.connect(self._play_next_line)

        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(500)
        self._pulse_timer.timeout.connect(self._toggle_pulse)
        self._pulse_on = False

        self._setup_ui()
        self._append_line(_PROMPT_STAGE0)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setFixedHeight(240)
        self._output.setStyleSheet(f"""
            QTextEdit {{
                background-color: #fafafa;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 6px;
                font-size: 11px;
                {_MONO_FONT}
            }}
        """)
        layout.addWidget(self._output)

        # 推理状态指示行：待命/推理中，配合圆点呼吸动效
        status_row = QHBoxLayout()
        status_row.setContentsMargins(2, 0, 2, 0)
        status_row.setSpacing(6)

        self._status_dot = QLabel("●")
        self._status_dot.setFixedWidth(14)
        status_row.addWidget(self._status_dot)

        self._status_label = QLabel("待命")
        self._status_label.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        status_row.addWidget(self._status_label)
        status_row.addStretch()
        layout.addLayout(status_row)
        self._set_idle_status()

        self._input = QLineEdit()
        self._input.setPlaceholderText("（命令）")
        self._input.setStyleSheet(f"""
            QLineEdit {{
                background-color: #ffffff;
                color: #212121;
                border: 1px solid #d0d7de;
                border-radius: 4px;
                padding: 6px 8px;
                {_MONO_FONT}
            }}
            QLineEdit:focus {{
                border-color: #1e88e5;
            }}
        """)
        self._input.returnPressed.connect(self._on_input_submitted)
        layout.addWidget(self._input)

    def _set_idle_status(self):
        self._pulse_timer.stop()
        self._status_dot.setStyleSheet("color: #bdbdbd; font-size: 12px;")
        self._status_label.setText("待命")
        self._status_label.setStyleSheet("color: #9e9e9e; font-size: 11px;")

    def _set_busy_status(self):
        self._pulse_on = True
        self._status_label.setText("推理中...")
        self._status_label.setStyleSheet("color: #1e88e5; font-size: 11px; font-weight: bold;")
        self._pulse_timer.start()

    def _toggle_pulse(self):
        self._pulse_on = not self._pulse_on
        color = "#1e88e5" if self._pulse_on else "#bbdefb"
        self._status_dot.setStyleSheet(f"color: {color}; font-size: 12px;")

    def _append_line(self, text: str):
        """插入一行文本，并对其做颜色淡入动效。"""
        doc_cursor = QTextCursor(self._output.document())
        doc_cursor.movePosition(QTextCursor.End)
        if not self._output.toPlainText() == "" or doc_cursor.position() > 0:
            doc_cursor.insertBlock()

        if not text:
            self._scroll_to_bottom()
            return

        color, italic, bold = _line_style(text)
        target = QColor(color)

        fmt = QTextCharFormat()
        fmt.setFontItalic(italic)
        fmt.setFontWeight(700 if bold else 400)
        fmt.setForeground(_blend(QColor("#fafafa"), target, 0.0))

        start = doc_cursor.position()
        doc_cursor.setCharFormat(fmt)
        doc_cursor.insertText(text)
        end = doc_cursor.position()

        self._scroll_to_bottom()
        self._animate_fade_in(start, end, target, italic, bold)

    def _animate_fade_in(self, start: int, end: int, target: QColor, italic: bool, bold: bool):
        anim = QVariantAnimation(self)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setDuration(_FADE_DURATION_MS)
        anim.setEasingCurve(QEasingCurve.OutCubic)

        def on_value_changed(t):
            cursor = QTextCursor(self._output.document())
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.KeepAnchor)
            fmt = QTextCharFormat()
            fmt.setFontItalic(italic)
            fmt.setFontWeight(700 if bold else 400)
            fmt.setForeground(_blend(QColor("#fafafa"), target, t))
            cursor.mergeCharFormat(fmt)

        anim.valueChanged.connect(on_value_changed)
        anim.start(QAbstractAnimation.DeleteWhenStopped)

    def _scroll_to_bottom(self):
        self._output.verticalScrollBar().setValue(
            self._output.verticalScrollBar().maximum()
        )

    def _on_input_submitted(self):
        text = self._input.text().strip()
        if not text or self._playback_timer.isActive():
            return

        self._append_line(f"> {text}")
        self._input.clear()
        self._input.setEnabled(False)
        self._set_busy_status()

        if self._stage == 0:
            self._start_playback(_SCRIPT_STAGE0)
        elif self._stage == 1:
            self._start_playback(_SCRIPT_STAGE1)

    def _start_playback(self, lines: List[str]):
        self._playback_lines = lines
        self._playback_index = 0
        self._playback_timer.start()

    def _play_next_line(self):
        if self._playback_index >= len(self._playback_lines):
            self._playback_timer.stop()
            self._on_playback_finished()
            return

        self._append_line(self._playback_lines[self._playback_index])
        self._playback_index += 1

    def _on_playback_finished(self):
        if self._stage == 0:
            self._stage = 1
            self._append_line("")
            self._append_line(_PROMPT_STAGE1)
            self._input.setEnabled(True)
            self._input.setFocus()
            self._set_idle_status()
        else:
            self._input.setEnabled(False)
            self._set_idle_status()
