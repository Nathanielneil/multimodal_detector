# -*- coding: utf-8 -*-
"""
地面平台 3D 模型 - 机器狗 (四足) 与无人车 (轮式)

与 swarm_view_3d.DroneModel 共用同一个 GLViewWidget 场景。
地面平台运动学简化为: 位置 (x, y) + 朝向 (yaw) + 高度状态，
不使用无人机的飞行动力学/激光雷达，保持轻量。

接口与 DroneModel 对齐:
    set_position(pos) / set_target(pos) / set_yaw(yaw)
    step(dt)  - 每帧平滑趋近目标位置与朝向
    remove()  - 从场景移除所有 GL 元素
"""

import math
from enum import Enum
from pathlib import Path
from typing import List, Tuple

import numpy as np

import pyqtgraph.opengl as gl
from utils.logger import get_logger
from .model_assets import apply_item_transform, create_model_asset

logger = get_logger(__name__)

_MODEL_ROOT = Path(__file__).resolve().parent.parent / "assets" / "models"


class PlatformType(Enum):
    """控制平台类型 - 用于界面选择器与指令路由"""
    DRONE = "drone"          # 无人机集群
    ROBOT_DOG = "robot_dog"  # 机器狗
    UGV = "ugv"              # 无人车


class RobotDogCommand(Enum):
    """机器狗指令集 (独立于无人机)"""
    STAND = "stand"          # 站立
    LIE_DOWN = "lie_down"    # 趴下
    SIT = "sit"              # 坐下
    FORWARD = "forward"      # 前进
    BACKWARD = "backward"    # 后退
    TURN_LEFT = "turn_left"  # 左转
    TURN_RIGHT = "turn_right"  # 右转
    STOP = "stop"            # 停止


class UGVCommand(Enum):
    """无人车指令集 (独立于无人机)"""
    START = "start"          # 启动
    PARK = "park"            # 停车
    SPEED_UP = "speed_up"    # 加速
    SPEED_DOWN = "speed_down"  # 减速
    FORWARD = "forward"      # 前进
    BACKWARD = "backward"    # 后退
    TURN_LEFT = "turn_left"  # 左转
    TURN_RIGHT = "turn_right"  # 右转


# 地面平台共用的运动参数
_MOVE_STEP = 3.0          # 单次"前进/后退"沿朝向移动的距离 (米)
_TURN_STEP = math.pi / 4  # 单次"左转/右转"的角度 (45°)
_POS_LERP = 0.12          # 位置趋近插值系数 (每帧)
_YAW_LERP = 0.15          # 朝向趋近插值系数 (每帧)


def _wrap_angle(a: float) -> float:
    """把角度归一化到 [-pi, pi]"""
    return (a + math.pi) % (2 * math.pi) - math.pi


class RobotDogModel:
    """
    机器狗 3D 模型 - 四足 (躯干 + 头 + 4 条腿)

    姿态 (pose):
        stand    - 站立，躯干抬至 LEG_LEN 高度
        lie_down - 趴下，躯干贴近地面
        sit      - 坐下，后部低、前部抬起
    运动:
        位置 (x, y) + 朝向 yaw；前进/后退沿朝向走 _MOVE_STEP，转向改 yaw
    """

    COLOR = (0.96, 0.97, 0.94, 1.0)   # 机器狗：Go2 风格白色外壳
    PANEL_COLOR = (0.78, 0.80, 0.80, 1.0)
    DARK_COLOR = (0.08, 0.08, 0.075, 1.0)
    LEG_COLOR = (0.16, 0.16, 0.15, 1.0)
    FOOT_COLOR = (0.035, 0.035, 0.032, 1.0)
    BODY_LEN = 0.82   # 躯干长 (沿朝向 x)
    BODY_WID = 0.34   # 躯干宽
    BODY_THK = 0.22   # 躯干厚
    LEG_LEN = 0.44    # 腿长 (站立时躯干离地高度)

    def __init__(self, dog_id: int, view: gl.GLViewWidget):
        self.dog_id = dog_id
        self.view = view
        self.color = self.COLOR

        # 状态: 位置 / 目标位置 / 朝向 / 目标朝向 / 姿态
        self.position = np.array([0.0, 0.0, 0.0])
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0
        self.target_yaw = 0.0
        self.pose = "stand"                 # stand | lie_down | sit
        self._body_h = self.LEG_LEN         # 当前躯干离地高度 (随姿态平滑变化)
        self._target_body_h = self.LEG_LEN

        self.elements: List = []
        # 当前官方 Go2 OBJ 在 PyQtGraph/OpenGL 场景里观感偏碎，演示默认使用实体低模。
        # 源模型仍保留在 assets/models/unitree_go2 作为后续离线重拓扑/贴图优化来源。
        self.asset = None
        self._create_visual()
        self._update_transform()

    def _create_visual(self):
        """创建机器狗的 GL 元素：实体躯干、头部、传感器、四条腿和脚掌。"""
        # 躯干 — 实体白色外壳
        self.body = gl.GLMeshItem(
            meshdata=self._box_mesh(self.BODY_LEN, self.BODY_WID, self.BODY_THK),
            smooth=False, color=self.color, shader='shaded', glOptions='opaque'
        )
        self.view.addItem(self.body)
        self.elements.append(self.body)

        # 背部控制舱和底部暗色结构，增强实体感
        self.top_pack = gl.GLMeshItem(
            meshdata=self._box_mesh(0.52, 0.23, 0.07),
            smooth=False, color=self.PANEL_COLOR, shader='shaded',
            glOptions='opaque'
        )
        self.view.addItem(self.top_pack)
        self.elements.append(self.top_pack)

        self.belly = gl.GLMeshItem(
            meshdata=self._box_mesh(0.72, 0.22, 0.05),
            smooth=False, color=self.DARK_COLOR, shader='shaded',
            glOptions='opaque'
        )
        self.view.addItem(self.belly)
        self.elements.append(self.belly)

        self.side_panels: List[gl.GLMeshItem] = []
        for _ in range(2):
            panel = gl.GLMeshItem(
                meshdata=self._box_mesh(0.54, 0.045, 0.10),
                smooth=False, color=self.PANEL_COLOR, shader='shaded',
                glOptions='opaque'
            )
            self.view.addItem(panel)
            self.elements.append(panel)
            self.side_panels.append(panel)

        # 前端传感器头部
        self.head = gl.GLMeshItem(
            meshdata=self._box_mesh(0.24, 0.22, 0.17),
            smooth=False, color=self.color, shader='shaded', glOptions='opaque'
        )
        self.view.addItem(self.head)
        self.elements.append(self.head)

        self.camera = gl.GLMeshItem(
            meshdata=self._box_mesh(0.045, 0.15, 0.085),
            smooth=False, color=self.DARK_COLOR, shader='shaded',
            glOptions='opaque'
        )
        self.view.addItem(self.camera)
        self.elements.append(self.camera)

        # 4 条实体腿 — 每条腿: 髋部白色护罩、上下腿和脚掌
        self.legs: List[dict] = []
        for _ in range(4):
            hip = gl.GLMeshItem(
                meshdata=self._box_mesh(0.14, 0.10, 0.14),
                smooth=False, color=self.color, shader='shaded',
                glOptions='opaque'
            )
            upper = gl.GLMeshItem(
                meshdata=self._box_mesh(0.075, 0.075, 0.24),
                smooth=False, color=self.LEG_COLOR, shader='shaded',
                glOptions='opaque'
            )
            lower = gl.GLMeshItem(
                meshdata=self._box_mesh(0.065, 0.065, 0.24),
                smooth=False, color=self.LEG_COLOR, shader='shaded',
                glOptions='opaque'
            )
            foot = gl.GLMeshItem(
                meshdata=self._box_mesh(0.20, 0.10, 0.055),
                smooth=False, color=self.FOOT_COLOR, shader='shaded',
                glOptions='opaque'
            )
            for item in (hip, upper, lower, foot):
                self.view.addItem(item)
                self.elements.append(item)
            self.legs.append({
                "hip": hip,
                "upper": upper,
                "lower": lower,
                "foot": foot,
            })

    @staticmethod
    def _box_mesh(lx: float, ly: float, lz: float):
        """生成以原点为中心的长方体 MeshData"""
        x, y, z = lx / 2, ly / 2, lz / 2
        verts = np.array([
            [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
            [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
        ])
        faces = np.array([
            [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
            [1, 2, 6], [1, 6, 5], [0, 3, 7], [0, 7, 4],
        ])
        return gl.MeshData(vertexes=verts, faces=faces)

    def execute_command(self, cmd: str):
        """执行机器狗指令 (cmd 为 RobotDogCommand.value)"""
        if cmd == RobotDogCommand.STAND.value:
            self.pose = "stand"
            self._target_body_h = self.LEG_LEN
        elif cmd == RobotDogCommand.LIE_DOWN.value:
            self.pose = "lie_down"
            self._target_body_h = self.BODY_THK * 0.6
            self.target_position = self.position.copy()
        elif cmd == RobotDogCommand.SIT.value:
            self.pose = "sit"
            self._target_body_h = self.LEG_LEN * 0.55
            self.target_position = self.position.copy()
        elif cmd == RobotDogCommand.FORWARD.value:
            self._step_along_heading(+_MOVE_STEP)
        elif cmd == RobotDogCommand.BACKWARD.value:
            self._step_along_heading(-_MOVE_STEP)
        elif cmd == RobotDogCommand.TURN_LEFT.value:
            self.target_yaw = _wrap_angle(self.target_yaw + _TURN_STEP)
        elif cmd == RobotDogCommand.TURN_RIGHT.value:
            self.target_yaw = _wrap_angle(self.target_yaw - _TURN_STEP)
        elif cmd == RobotDogCommand.STOP.value:
            self.target_position = self.position.copy()

        self.clamp_to_bounds(15.0)

    def _step_along_heading(self, dist: float):
        """沿当前目标朝向移动 dist (趴下/坐下时不移动)"""
        if self.pose != "stand":
            return
        nx = self.target_position[0] + dist * math.cos(self.target_yaw)
        ny = self.target_position[1] + dist * math.sin(self.target_yaw)
        self.target_position = np.array([nx, ny, 0.0])

    def set_position(self, pos: Tuple[float, float, float]):
        self.position = np.array(pos, dtype=float)
        self.target_position = self.position.copy()
        self._update_transform()

    def set_target(self, pos: Tuple[float, float, float]):
        self.target_position = np.array(pos, dtype=float)

    def set_yaw(self, yaw: float):
        self.yaw = yaw
        self.target_yaw = yaw

    def clamp_to_bounds(self, limit: float):
        """把目标位置钳制在 ±limit 围栏内"""
        self.target_position[0] = float(np.clip(self.target_position[0], -limit, limit))
        self.target_position[1] = float(np.clip(self.target_position[1], -limit, limit))

    def step(self, dt: float):
        """每帧平滑趋近目标位置、朝向、躯干高度"""
        # 位置插值
        self.position += (self.target_position - self.position) * _POS_LERP
        # 朝向插值 (走最短弧)
        dyaw = _wrap_angle(self.target_yaw - self.yaw)
        self.yaw = _wrap_angle(self.yaw + dyaw * _YAW_LERP)
        # 躯干高度插值
        self._body_h += (self._target_body_h - self._body_h) * _POS_LERP
        self._update_transform()

    def _update_transform(self):
        """根据 position / yaw / _body_h 更新所有 GL 元素"""
        x, y, _ = self.position
        h = self._body_h
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)

        if self.asset:
            self.asset.apply_pose((x, y, 0.0), self.yaw)
            return

        def to_world(lx, ly, lz):
            """局部坐标 (沿朝向 x 向前) -> 世界坐标"""
            wx = x + lx * cy - ly * sy
            wy = y + lx * sy + ly * cy
            return [wx, wy, lz]

        # 躯干: 平移到 (x, y, h)，绕 z 轴旋转 yaw
        apply_item_transform(self.body, (x, y, h), self.yaw)

        apply_item_transform(
            self.top_pack,
            (x, y, h + self.BODY_THK * 0.58),
            self.yaw,
        )

        apply_item_transform(
            self.belly,
            (x, y, h - self.BODY_THK * 0.50),
            self.yaw,
        )

        for panel, side in zip(self.side_panels, (1, -1)):
            sx, sy_, sz = to_world(0.0, side * self.BODY_WID * 0.56, h + 0.005)
            apply_item_transform(panel, (sx, sy_, sz), self.yaw)

        # 头部: 躯干前端上方
        hx, hy, hz = to_world(self.BODY_LEN * 0.5 + 0.08, 0, h + 0.01)
        apply_item_transform(self.head, (hx, hy, hz), self.yaw)

        cam_x, cam_y, cam_z = to_world(self.BODY_LEN * 0.5 + 0.225, 0, h + 0.015)
        apply_item_transform(self.camera, (cam_x, cam_y, cam_z), self.yaw)

        # 4 条腿: 上腿/下腿/脚掌实体。坐下时后腿更短，趴下时整体贴近地面。
        leg_offsets = [
            (self.BODY_LEN * 0.35, self.BODY_WID * 0.58),    # 左前
            (self.BODY_LEN * 0.35, -self.BODY_WID * 0.58),   # 右前
            (-self.BODY_LEN * 0.35, self.BODY_WID * 0.58),   # 左后
            (-self.BODY_LEN * 0.35, -self.BODY_WID * 0.58),  # 右后
        ]
        for i, (lx, ly) in enumerate(leg_offsets):
            is_rear = lx < 0
            hip_z = h - self.BODY_THK * 0.35
            if self.pose == "lie_down":
                foot_z = max(0.0, h - self.LEG_LEN * 0.35)
            elif self.pose == "sit" and is_rear:
                foot_z = max(0.02, h - self.LEG_LEN * 0.45)
            else:
                foot_z = 0.03

            knee_z = (hip_z + foot_z) * 0.5
            upper_mid_z = (hip_z + knee_z) * 0.5
            lower_mid_z = (knee_z + foot_z) * 0.5
            foot_mid_z = foot_z + 0.025

            # 前后腿略微错开，视觉上更像四足结构
            knee_x_offset = 0.05 if lx > 0 else -0.05
            hip_x, hip_y, _ = to_world(lx, ly, hip_z)
            upper_x, upper_y, _ = to_world(lx + knee_x_offset * 0.5, ly, upper_mid_z)
            lower_x, lower_y, _ = to_world(lx + knee_x_offset, ly, lower_mid_z)
            foot_x, foot_y, _ = to_world(lx + knee_x_offset * 1.2, ly, foot_mid_z)

            leg = self.legs[i]
            for item, px, py, pz in (
                (leg["hip"], hip_x, hip_y, hip_z),
                (leg["upper"], upper_x, upper_y, upper_mid_z),
                (leg["lower"], lower_x, lower_y, lower_mid_z),
                (leg["foot"], foot_x, foot_y, foot_mid_z),
            ):
                apply_item_transform(item, (px, py, pz), self.yaw)

    def remove(self):
        """从场景移除所有元素"""
        for el in self.elements:
            self.view.removeItem(el)
        self.elements.clear()

    def get_status(self) -> dict:
        """返回机器狗仿真状态。"""
        return {
            "platform": PlatformType.ROBOT_DOG.value,
            "name": "机器狗",
            "position": tuple(self.position),
            "yaw": self.yaw,
            "pose": self.pose,
            "status": self._status_text(),
            "speed": float(np.linalg.norm(self.target_position - self.position)),
        }

    def _status_text(self) -> str:
        status_map = {
            "stand": "站立",
            "lie_down": "趴下",
            "sit": "坐下",
        }
        return status_map.get(self.pose, "待命")


class UGVModel:
    """
    无人车 3D 模型 - 轮式平台 (车体 + 驾驶舱 + 4 个轮子)

    运动:
        位置 (x, y) + 朝向 yaw；启动后可前进/后退/转向，停车后锁定位置。
    """

    COLOR = (0.15, 0.75, 0.85, 1.0)  # 无人车：青蓝色
    BODY_LEN = 0.9
    BODY_WID = 0.45
    BODY_THK = 0.18
    CABIN_LEN = 0.34
    CABIN_WID = 0.34
    CABIN_THK = 0.18
    WHEEL_RADIUS = 0.12
    WHEEL_WIDTH = 0.09

    def __init__(self, ugv_id: int, view: gl.GLViewWidget):
        self.ugv_id = ugv_id
        self.view = view
        self.color = self.COLOR

        self.position = np.array([0.0, 0.0, 0.0])
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0
        self.target_yaw = 0.0
        self.active = False
        self.speed_level = 1

        self.elements: List = []
        self.asset = create_model_asset(
            _MODEL_ROOT / "clearpath_husky",
            self.view,
            default_color=self.color,
        )
        if self.asset:
            self.elements.extend(self.asset.items)
            self._create_asset_overlays()
        else:
            self._create_visual()
        self._update_transform()

    def _create_visual(self):
        """创建无人车 GL 元素。"""
        self.body = gl.GLMeshItem(
            meshdata=RobotDogModel._box_mesh(self.BODY_LEN, self.BODY_WID, self.BODY_THK),
            smooth=False, color=self.color, shader='shaded', glOptions='opaque'
        )
        self.view.addItem(self.body)
        self.elements.append(self.body)

        self.cabin = gl.GLMeshItem(
            meshdata=RobotDogModel._box_mesh(self.CABIN_LEN, self.CABIN_WID, self.CABIN_THK),
            smooth=False, color=(0.08, 0.45, 0.55, 1.0), shader='shaded', glOptions='opaque'
        )
        self.view.addItem(self.cabin)
        self.elements.append(self.cabin)

        self.roof_sensor = gl.GLMeshItem(
            meshdata=gl.MeshData.cylinder(
                rows=6, cols=16,
                radius=[0.08, 0.08],
                length=0.10
            ),
            smooth=True, color=(0.02, 0.02, 0.02, 1.0), shader='shaded',
            glOptions='opaque'
        )
        self.view.addItem(self.roof_sensor)
        self.elements.append(self.roof_sensor)

        self.front_bumper = gl.GLMeshItem(
            meshdata=RobotDogModel._box_mesh(0.08, self.BODY_WID * 1.08, 0.08),
            smooth=False, color=(0.02, 0.02, 0.02, 1.0), shader='shaded',
            glOptions='opaque'
        )
        self.view.addItem(self.front_bumper)
        self.elements.append(self.front_bumper)

        self.wheels: List[gl.GLMeshItem] = []
        for _ in range(4):
            wheel = gl.GLMeshItem(
                meshdata=gl.MeshData.cylinder(
                    rows=8, cols=20,
                    radius=[self.WHEEL_RADIUS, self.WHEEL_RADIUS],
                    length=self.WHEEL_WIDTH
                ),
                smooth=True, color=(0.03, 0.03, 0.03, 1.0), shader='shaded',
                glOptions='opaque'
            )
            self.view.addItem(wheel)
            self.wheels.append(wheel)
            self.elements.append(wheel)

        self.heading_line = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), color=(0.9, 1.0, 0.2, 1.0),
            width=3, antialias=True
        )
        self.view.addItem(self.heading_line)
        self.elements.append(self.heading_line)

    def _create_asset_overlays(self):
        """为外部小车模型追加演示用传感器和朝向线。"""
        self.roof_sensor = gl.GLMeshItem(
            meshdata=gl.MeshData.cylinder(
                rows=6, cols=16,
                radius=[0.075, 0.075],
                length=0.10
            ),
            smooth=True, color=(0.02, 0.02, 0.02, 1.0), shader='shaded',
            glOptions='opaque'
        )
        self.view.addItem(self.roof_sensor)
        self.elements.append(self.roof_sensor)

        self.heading_line = gl.GLLinePlotItem(
            pos=np.zeros((2, 3)), color=(0.9, 1.0, 0.2, 1.0),
            width=3, antialias=True
        )
        self.view.addItem(self.heading_line)
        self.elements.append(self.heading_line)

    def execute_command(self, cmd: str):
        """执行无人车指令 (cmd 为 UGVCommand.value)。"""
        if cmd == UGVCommand.START.value:
            self.active = True
        elif cmd == UGVCommand.PARK.value:
            self.active = False
            self.target_position = self.position.copy()
        elif cmd == UGVCommand.SPEED_UP.value:
            self.speed_level = min(3, self.speed_level + 1)
        elif cmd == UGVCommand.SPEED_DOWN.value:
            self.speed_level = max(1, self.speed_level - 1)
        elif cmd == UGVCommand.FORWARD.value:
            self._step_along_heading(+self._move_distance())
        elif cmd == UGVCommand.BACKWARD.value:
            self._step_along_heading(-self._move_distance())
        elif cmd == UGVCommand.TURN_LEFT.value:
            self.target_yaw = _wrap_angle(self.target_yaw + _TURN_STEP)
        elif cmd == UGVCommand.TURN_RIGHT.value:
            self.target_yaw = _wrap_angle(self.target_yaw - _TURN_STEP)

        self.clamp_to_bounds(15.0)

    def _move_distance(self) -> float:
        return _MOVE_STEP * (0.65 + 0.35 * self.speed_level)

    def _step_along_heading(self, dist: float):
        """沿当前目标朝向移动 dist，停车时不移动。"""
        if not self.active:
            return
        nx = self.target_position[0] + dist * math.cos(self.target_yaw)
        ny = self.target_position[1] + dist * math.sin(self.target_yaw)
        self.target_position = np.array([nx, ny, 0.0])

    def set_position(self, pos: Tuple[float, float, float]):
        self.position = np.array(pos, dtype=float)
        self.target_position = self.position.copy()
        self._update_transform()

    def set_target(self, pos: Tuple[float, float, float]):
        self.target_position = np.array(pos, dtype=float)

    def set_yaw(self, yaw: float):
        self.yaw = yaw
        self.target_yaw = yaw

    def clamp_to_bounds(self, limit: float):
        self.target_position[0] = float(np.clip(self.target_position[0], -limit, limit))
        self.target_position[1] = float(np.clip(self.target_position[1], -limit, limit))

    def step(self, dt: float):
        """每帧平滑趋近目标位置和朝向。"""
        self.position += (self.target_position - self.position) * _POS_LERP
        dyaw = _wrap_angle(self.target_yaw - self.yaw)
        self.yaw = _wrap_angle(self.yaw + dyaw * _YAW_LERP)
        self._update_transform()

    def _update_transform(self):
        x, y, _ = self.position
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)

        def to_world(lx, ly, lz):
            wx = x + lx * cy - ly * sy
            wy = y + lx * sy + ly * cy
            return [wx, wy, lz]

        if self.asset:
            self.asset.apply_pose((x, y, 0.0), self.yaw)

            sensor_x, sensor_y, sensor_z = to_world(0.08, 0.0, 0.58)
            apply_item_transform(
                self.roof_sensor,
                (sensor_x, sensor_y, sensor_z),
                self.yaw,
            )

            start = np.array(to_world(0.0, 0.0, 0.68))
            end = np.array(to_world(0.95, 0.0, 0.68))
            self.heading_line.setData(pos=np.array([start, end]))
            return

        body_z = self.BODY_THK * 0.5 + 0.08

        apply_item_transform(self.body, (x, y, body_z), self.yaw)

        cx, cy_, cz = to_world(0.08, 0.0, body_z + self.BODY_THK * 0.5)
        apply_item_transform(self.cabin, (cx, cy_, cz), self.yaw)

        sensor_x, sensor_y, sensor_z = to_world(
            0.03, 0.0, body_z + self.BODY_THK * 0.5 + self.CABIN_THK + 0.05
        )
        apply_item_transform(self.roof_sensor, (sensor_x, sensor_y, sensor_z), self.yaw)

        bumper_x, bumper_y, bumper_z = to_world(self.BODY_LEN * 0.52, 0.0, body_z)
        apply_item_transform(self.front_bumper, (bumper_x, bumper_y, bumper_z), self.yaw)

        wheel_offsets = [
            (self.BODY_LEN * 0.34, self.BODY_WID * 0.55),
            (self.BODY_LEN * 0.34, -self.BODY_WID * 0.55),
            (-self.BODY_LEN * 0.34, self.BODY_WID * 0.55),
            (-self.BODY_LEN * 0.34, -self.BODY_WID * 0.55),
        ]
        for wheel, (lx, ly) in zip(self.wheels, wheel_offsets):
            wx, wy, wz = to_world(lx, ly, self.WHEEL_RADIUS)
            apply_item_transform(
                wheel,
                (wx, wy, wz),
                self.yaw,
                rotation_deg=(90, 0, 0),
            )

        start = np.array(to_world(0.0, 0.0, body_z + 0.2))
        end = np.array(to_world(0.75, 0.0, body_z + 0.2))
        self.heading_line.setData(pos=np.array([start, end]))

    def remove(self):
        """从场景移除所有元素。"""
        for el in self.elements:
            self.view.removeItem(el)
        self.elements.clear()

    def get_status(self) -> dict:
        """返回无人车仿真状态。"""
        return {
            "platform": PlatformType.UGV.value,
            "name": "无人车",
            "position": tuple(self.position),
            "yaw": self.yaw,
            "active": self.active,
            "speed_level": self.speed_level,
            "status": self._status_text(),
            "speed": float(np.linalg.norm(self.target_position - self.position)),
        }

    def _status_text(self) -> str:
        if not self.active:
            return "停车"
        return f"运行 S{self.speed_level}"
