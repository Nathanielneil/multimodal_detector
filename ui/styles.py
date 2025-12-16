"""
UI 样式模块 - 浅色主题 QSS 样式表
主色: #1e88e5 (蓝色)
悬停色: #1565c0 (深蓝)
"""

# 主题颜色常量
PRIMARY_COLOR = "#1e88e5"
PRIMARY_HOVER = "#1565c0"
PRIMARY_DISABLED = "#9e9e9e"
BACKGROUND_COLOR = "#f5f5f5"
CARD_BACKGROUND = "#ffffff"
BORDER_COLOR = "#e0e0e0"
TEXT_COLOR = "#212121"
TEXT_SECONDARY = "#757575"
SELECTED_ROW = "#e3f2fd"
TABLE_HEADER_BG = "#eeeeee"

# 模态颜色 - 用于四模态可视化
MODAL_COLORS = {
    "voice": "#4caf50",      # 绿色 - 语音
    "gesture": "#ff9800",    # 橙色 - 手势
    "image": "#2196f3",      # 蓝色 - 图像
    "touch": "#9c27b0",      # 紫色 - 触屏
}

# 全局 QSS 样式表
MAIN_STYLESHEET = f"""
/* ===== 全局样式 ===== */
QMainWindow, QWidget {{
    background-color: {BACKGROUND_COLOR};
    color: {TEXT_COLOR};
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 13px;
}}

/* ===== 分割器 ===== */
QSplitter::handle {{
    background-color: {BORDER_COLOR};
    width: 2px;
}}
QSplitter::handle:hover {{
    background-color: {PRIMARY_COLOR};
}}

/* ===== 控制面板容器 ===== */
QGroupBox {{
    background-color: {CARD_BACKGROUND};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    margin-top: 12px;
    padding: 10px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: {TEXT_COLOR};
}}

/* ===== 复选框 ===== */
QCheckBox {{
    spacing: 8px;
    padding: 6px;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 2px solid {BORDER_COLOR};
    background-color: {CARD_BACKGROUND};
}}
QCheckBox::indicator:checked {{
    background-color: {PRIMARY_COLOR};
    border-color: {PRIMARY_COLOR};
}}
QCheckBox::indicator:hover {{
    border-color: {PRIMARY_COLOR};
}}

/* ===== 下拉框 ===== */
QComboBox {{
    padding: 6px 12px;
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    background-color: {CARD_BACKGROUND};
    min-height: 28px;
}}
QComboBox:hover {{
    border-color: {PRIMARY_COLOR};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {TEXT_SECONDARY};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {CARD_BACKGROUND};
    border: 1px solid {BORDER_COLOR};
    selection-background-color: {SELECTED_ROW};
}}

/* ===== 数值输入框 ===== */
QDoubleSpinBox, QSpinBox {{
    padding: 6px 12px;
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    background-color: {CARD_BACKGROUND};
    min-height: 28px;
}}
QDoubleSpinBox:hover, QSpinBox:hover {{
    border-color: {PRIMARY_COLOR};
}}
QDoubleSpinBox:focus, QSpinBox:focus {{
    border-color: {PRIMARY_COLOR};
    border-width: 2px;
}}

/* ===== 标签 ===== */
QLabel {{
    color: {TEXT_COLOR};
    padding: 2px;
}}
QLabel[class="secondary"] {{
    color: {TEXT_SECONDARY};
    font-size: 12px;
}}

/* ===== 按钮 - 主要按钮(蓝色) ===== */
QPushButton {{
    background-color: {PRIMARY_COLOR};
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
    min-height: 32px;
}}
QPushButton:hover {{
    background-color: {PRIMARY_HOVER};
}}
QPushButton:pressed {{
    background-color: #0d47a1;
}}
QPushButton:disabled {{
    background-color: {PRIMARY_DISABLED};
    color: #e0e0e0;
}}

/* ===== 按钮 - 次要按钮(白底蓝边) ===== */
QPushButton[class="secondary"] {{
    background-color: {CARD_BACKGROUND};
    color: {PRIMARY_COLOR};
    border: 1px solid {PRIMARY_COLOR};
}}
QPushButton[class="secondary"]:hover {{
    background-color: {SELECTED_ROW};
}}

/* ===== 按钮 - 危险按钮(红色) ===== */
QPushButton[class="danger"] {{
    background-color: #f44336;
}}
QPushButton[class="danger"]:hover {{
    background-color: #d32f2f;
}}

/* ===== 表格 ===== */
QTableWidget {{
    background-color: {CARD_BACKGROUND};
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    gridline-color: {BORDER_COLOR};
    selection-background-color: {SELECTED_ROW};
    selection-color: {TEXT_COLOR};
}}
QTableWidget::item {{
    padding: 8px;
    border-bottom: 1px solid {BORDER_COLOR};
}}
QTableWidget::item:selected {{
    background-color: {SELECTED_ROW};
    color: {TEXT_COLOR};
}}
QHeaderView::section {{
    background-color: {TABLE_HEADER_BG};
    color: {TEXT_COLOR};
    padding: 10px;
    border: none;
    border-bottom: 2px solid {BORDER_COLOR};
    font-weight: bold;
}}

/* ===== 滚动条 ===== */
QScrollBar:vertical {{
    background-color: {BACKGROUND_COLOR};
    width: 10px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical {{
    background-color: {BORDER_COLOR};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: {TEXT_SECONDARY};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* ===== 视频显示区域 ===== */
QLabel[class="video-display"] {{
    background-color: #000000;
    border: 2px solid {BORDER_COLOR};
    border-radius: 6px;
}}

/* ===== 模态状态卡片 ===== */
QFrame[class="modal-card"] {{
    background-color: {CARD_BACKGROUND};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 8px;
}}
QFrame[class="modal-card-voice"] {{
    border-left: 4px solid #4caf50;
}}
QFrame[class="modal-card-gesture"] {{
    border-left: 4px solid #ff9800;
}}
QFrame[class="modal-card-image"] {{
    border-left: 4px solid #2196f3;
}}
QFrame[class="modal-card-touch"] {{
    border-left: 4px solid #9c27b0;
}}

/* ===== 工具提示 ===== */
QToolTip {{
    background-color: #424242;
    color: white;
    border: none;
    padding: 6px 10px;
    border-radius: 4px;
}}

/* ===== 录音按钮特殊样式 ===== */
QPushButton[class="record-btn"] {{
    background-color: #f44336;
    border-radius: 20px;
    min-width: 40px;
    min-height: 40px;
    max-width: 40px;
    max-height: 40px;
}}
QPushButton[class="record-btn"]:hover {{
    background-color: #d32f2f;
}}
QPushButton[class="record-btn"]:checked {{
    background-color: #b71c1c;
}}
"""

# 模态状态指示器样式函数
def get_modal_indicator_style(modal_type: str, is_active: bool = False) -> str:
    """获取模态状态指示器的样式"""
    color = MODAL_COLORS.get(modal_type, PRIMARY_COLOR)
    opacity = "1.0" if is_active else "0.5"
    return f"""
        background-color: {color};
        border-radius: 4px;
        opacity: {opacity};
    """
