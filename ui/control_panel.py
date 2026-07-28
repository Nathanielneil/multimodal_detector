"""
左栏控制面板 - 检测器控制组件
包含: 模态开关、跟踪算法选择、阈值设置、操作按钮
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QCheckBox, QLabel, QComboBox, QDoubleSpinBox,
    QPushButton, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import Signal, Qt


class ControlPanel(QWidget):
    """
    左侧控制面板组件

    Signals:
        modal_toggled: 模态开关切换 (modal_name: str, enabled: bool)
        algorithm_changed: 跟踪算法变更 (algorithm: str)
        threshold_changed: 阈值变更 (threshold_type: str, value: float)
        reset_clicked: 重置统计按钮点击
        export_clicked: 导出历史按钮点击
        record_clicked: 录音按钮点击 (is_recording: bool)
        camera_changed: 摄像头切换 (camera_id: int)
        refresh_cameras_clicked: 刷新摄像头列表按钮点击
        load_map_clicked: 加载点云地图按钮点击
    """

    # 信号定义
    modal_toggled = Signal(str, bool)
    algorithm_changed = Signal(str)
    threshold_changed = Signal(str, float)
    reset_clicked = Signal()
    export_clicked = Signal()
    record_clicked = Signal(bool)
    camera_changed = Signal(int)
    refresh_cameras_clicked = Signal()
    load_map_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        """初始化界面布局"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # ===== 模态开关组 =====
        modal_group = QGroupBox("检测模态")
        modal_layout = QVBoxLayout(modal_group)

        # 四个模态复选框
        self.cb_voice = QCheckBox("语音识别")
        self.cb_voice.setChecked(True)
        self.cb_voice.setToolTip("启用/禁用语音识别功能")

        self.cb_gesture = QCheckBox("手势识别")
        self.cb_gesture.setChecked(True)
        self.cb_gesture.setToolTip("启用/禁用手势识别功能")

        self.cb_image = QCheckBox("图像识别")
        self.cb_image.setChecked(True)
        self.cb_image.setToolTip("启用/禁用物体检测功能")

        self.cb_touch = QCheckBox("触屏指令")
        self.cb_touch.setChecked(True)
        self.cb_touch.setToolTip("启用/禁用触屏/鼠标点击检测")

        modal_layout.addWidget(self.cb_voice)
        modal_layout.addWidget(self.cb_gesture)
        modal_layout.addWidget(self.cb_image)
        modal_layout.addWidget(self.cb_touch)

        layout.addWidget(modal_group)

        # ===== 摄像头设置组 =====
        camera_group = QGroupBox("摄像头设置")
        camera_layout = QVBoxLayout(camera_group)

        # 摄像头选择下拉框
        camera_row = QHBoxLayout()
        camera_label = QLabel("摄像头:")
        self.combo_camera = QComboBox()
        self.combo_camera.addItem("摄像头 0", 0)
        self.combo_camera.setToolTip("选择要使用的摄像头设备")
        camera_row.addWidget(camera_label)
        camera_row.addWidget(self.combo_camera, 1)
        camera_layout.addLayout(camera_row)

        # 刷新按钮
        self.btn_refresh_cameras = QPushButton("刷新设备列表")
        self.btn_refresh_cameras.setProperty("class", "secondary")
        self.btn_refresh_cameras.setToolTip("扫描可用的摄像头设备")
        camera_layout.addWidget(self.btn_refresh_cameras)

        layout.addWidget(camera_group)

        # ===== 3D 场景地图组 =====
        map_group = QGroupBox("3D 场景地图")
        map_layout = QVBoxLayout(map_group)

        self.btn_load_map = QPushButton("加载点云地图...")
        self.btn_load_map.setProperty("class", "secondary")
        self.btn_load_map.setToolTip("选择 .pcd 点云文件作为 3D 场景的先验地图")
        map_layout.addWidget(self.btn_load_map)

        layout.addWidget(map_group)

        # ===== 跟踪算法组 =====
        algo_group = QGroupBox("跟踪设置")
        algo_layout = QVBoxLayout(algo_group)

        # 跟踪算法下拉框
        algo_row = QHBoxLayout()
        algo_label = QLabel("跟踪算法:")
        self.combo_algorithm = QComboBox()
        self.combo_algorithm.addItems(["highest", "medium", "low"])
        self.combo_algorithm.setCurrentText("highest")
        self.combo_algorithm.setToolTip("选择目标跟踪的精度级别")
        algo_row.addWidget(algo_label)
        algo_row.addWidget(self.combo_algorithm, 1)
        algo_layout.addLayout(algo_row)

        layout.addWidget(algo_group)

        # ===== 阈值设置组 =====
        threshold_group = QGroupBox("阈值设置")
        threshold_layout = QVBoxLayout(threshold_group)

        # 融合阈值
        fusion_row = QHBoxLayout()
        fusion_label = QLabel("融合阈值:")
        self.spin_fusion = QDoubleSpinBox()
        self.spin_fusion.setRange(0.0, 1.0)
        self.spin_fusion.setSingleStep(0.01)
        self.spin_fusion.setValue(0.50)
        self.spin_fusion.setDecimals(2)
        self.spin_fusion.setToolTip("多模态融合的置信度阈值")
        fusion_row.addWidget(fusion_label)
        fusion_row.addWidget(self.spin_fusion, 1)
        threshold_layout.addLayout(fusion_row)

        # 召回阈值
        recall_row = QHBoxLayout()
        recall_label = QLabel("召回阈值:")
        self.spin_recall = QDoubleSpinBox()
        self.spin_recall.setRange(0.0, 1.0)
        self.spin_recall.setSingleStep(0.01)
        self.spin_recall.setValue(0.50)
        self.spin_recall.setDecimals(2)
        self.spin_recall.setToolTip("检测结果召回的置信度阈值")
        recall_row.addWidget(recall_label)
        recall_row.addWidget(self.spin_recall, 1)
        threshold_layout.addLayout(recall_row)

        layout.addWidget(threshold_group)

        # ===== 语音录制组 =====
        voice_group = QGroupBox("语音录制")
        voice_layout = QVBoxLayout(voice_group)

        # 录音按钮
        self.btn_record = QPushButton("点击录音")
        self.btn_record.setCheckable(True)
        self.btn_record.setToolTip("点击开始录音，再次点击停止并识别")
        voice_layout.addWidget(self.btn_record)

        # 录音状态标签 (居中显示)
        self.label_record_status = QLabel("待命")
        self.label_record_status.setProperty("class", "secondary")
        self.label_record_status.setAlignment(Qt.AlignCenter)
        voice_layout.addWidget(self.label_record_status)

        layout.addWidget(voice_group)

        # ===== 操作按钮组 =====
        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout(action_group)

        # 重置统计按钮
        self.btn_reset = QPushButton("重置统计")
        self.btn_reset.setProperty("class", "secondary")
        self.btn_reset.setToolTip("清空命令历史并重置所有模态状态")
        action_layout.addWidget(self.btn_reset)

        # 导出历史按钮
        self.btn_export = QPushButton("导出历史")
        self.btn_export.setProperty("class", "secondary")
        self.btn_export.setToolTip("将命令历史导出为JSON文件")
        action_layout.addWidget(self.btn_export)

        layout.addWidget(action_group)

        # 弹性空间
        layout.addSpacerItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

    def _connect_signals(self):
        """连接内部信号到外部信号"""
        # 模态开关
        self.cb_voice.toggled.connect(
            lambda checked: self.modal_toggled.emit("voice", checked)
        )
        self.cb_gesture.toggled.connect(
            lambda checked: self.modal_toggled.emit("gesture", checked)
        )
        self.cb_image.toggled.connect(
            lambda checked: self.modal_toggled.emit("image", checked)
        )
        self.cb_touch.toggled.connect(
            lambda checked: self.modal_toggled.emit("touch", checked)
        )

        # 跟踪算法
        self.combo_algorithm.currentTextChanged.connect(
            self.algorithm_changed.emit
        )

        # 阈值变更
        self.spin_fusion.valueChanged.connect(
            lambda value: self.threshold_changed.emit("fusion", value)
        )
        self.spin_recall.valueChanged.connect(
            lambda value: self.threshold_changed.emit("recall", value)
        )

        # 操作按钮
        self.btn_reset.clicked.connect(self.reset_clicked.emit)
        self.btn_export.clicked.connect(self.export_clicked.emit)
        self.btn_record.toggled.connect(self._on_record_toggled)

        # 摄像头选择
        self.combo_camera.currentIndexChanged.connect(self._on_camera_changed)
        self.btn_refresh_cameras.clicked.connect(self.refresh_cameras_clicked.emit)

        # 点云地图加载
        self.btn_load_map.clicked.connect(self.load_map_clicked.emit)

    def _on_record_toggled(self, checked: bool):
        """处理录音按钮切换"""
        if checked:
            self.btn_record.setText("[录音中...]")
            self.label_record_status.setText("正在录音...")
        else:
            self.btn_record.setText("点击录音")
            self.label_record_status.setText("处理中...")
        self.record_clicked.emit(checked)

    def set_record_status(self, status: str):
        """设置录音状态文本 (过滤识别结果，只显示简洁状态)"""
        # 过滤掉识别结果，不在状态栏显示
        if "识别完成" in status or "识别到" in status:
            self.label_record_status.setText("待命")
            return
        # 简化状态显示
        status_map = {
            "正在加载语音模型...": "加载模型...",
            "语音模型加载完成": "模型就绪",
            "正在加载 Whisper 模型...": "加载模型...",  # legacy message
            "Whisper 模型加载完成": "模型就绪",  # legacy message
            "正在录音...": "录音中...",
            "录音为空": "录音为空",
            "正在识别...": "识别中...",
            "未识别到语音": "未识别到",
            "语音检测器未初始化": "未初始化",
            "语音检测器已释放": "待命",
            "语音模态已禁用": "已禁用",
            "语音模型加载失败": "加载失败",
        }
        # 处理重采样消息
        if "重采样" in status:
            self.label_record_status.setText("处理音频...")
            return
        # 使用映射或原状态
        display_status = status_map.get(status, status)
        self.label_record_status.setText(display_status)

    def is_modal_enabled(self, modal_name: str) -> bool:
        """检查指定模态是否启用"""
        modal_map = {
            "voice": self.cb_voice,
            "gesture": self.cb_gesture,
            "image": self.cb_image,
            "touch": self.cb_touch,
        }
        checkbox = modal_map.get(modal_name)
        return checkbox.isChecked() if checkbox else False

    def get_fusion_threshold(self) -> float:
        """获取融合阈值"""
        return self.spin_fusion.value()

    def get_recall_threshold(self) -> float:
        """获取召回阈值"""
        return self.spin_recall.value()

    def get_algorithm(self) -> str:
        """获取当前跟踪算法"""
        return self.combo_algorithm.currentText()

    def _on_camera_changed(self, index: int):
        """处理摄像头切换"""
        camera_id = self.combo_camera.currentData()
        if camera_id is not None:
            self.camera_changed.emit(camera_id)

    def update_camera_list(self, camera_ids: list):
        """
        更新摄像头列表

        Args:
            camera_ids: 可用摄像头ID列表
        """
        current_id = self.combo_camera.currentData()

        # 阻止信号触发
        self.combo_camera.blockSignals(True)
        self.combo_camera.clear()

        for cam_id in camera_ids:
            self.combo_camera.addItem(f"摄像头 {cam_id}", cam_id)

        # 尝试恢复之前的选择
        if current_id is not None:
            index = self.combo_camera.findData(current_id)
            if index >= 0:
                self.combo_camera.setCurrentIndex(index)

        self.combo_camera.blockSignals(False)

    def get_selected_camera(self) -> int:
        """获取当前选择的摄像头ID"""
        return self.combo_camera.currentData() or 0
