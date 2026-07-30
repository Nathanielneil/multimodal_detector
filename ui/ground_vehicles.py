# -*- coding: utf-8 -*-
"""
地面平台状态模型 - 机器狗 (四足) 与无人车 (轮式)

与 swarm_view_3d.DroneModel 共用同一个 GLViewWidget 场景。
地面平台运动学简化为: 位置 (x, y) + 朝向 (yaw) + 高度状态，
不使用无人机的飞行动力学/激光雷达，保持轻量。

真实点云场景尺度下几何机身模型不可见，本模块不再持有 3D 网格 GL 元素，
渲染改为 SwarmView3D 统一维护的一层固定像素大小 2D 标记点 (颜色+编号标签)。

接口与 DroneModel 对齐:
    set_position(pos) / set_target(pos) / set_yaw(yaw)
    step(dt)  - 每帧平滑趋近目标位置与朝向
    remove()  - 从场景移除所有 GL 元素 (当前为空实现，保留接口兼容)
"""

import math
from enum import Enum
from typing import List, Tuple

import numpy as np

import pyqtgraph.opengl as gl
from utils.logger import get_logger

logger = get_logger(__name__)


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
    机器狗状态模型 - 位置/姿态/朝向，不再持有 3D 网格

    真实点云场景尺度下几何模型不可见，渲染改为 SwarmView3D 统一维护的
    一层固定像素大小 2D 标记点 (颜色+编号标签)，本类只负责位置状态。

    姿态 (pose):
        stand    - 站立
        lie_down - 趴下
        sit      - 坐下
    运动:
        位置 (x, y) + 朝向 yaw；前进/后退沿朝向走 _MOVE_STEP，转向改 yaw
    """

    MARKER_COLOR = (0.95, 0.55, 0.15, 1.0)   # 机器狗：2D 标记层颜色 (橙色)
    LEG_LEN = 0.44    # 腿长 (站立时躯干离地高度，仅用于姿态高度状态)

    def __init__(self, dog_id: int, view: gl.GLViewWidget):
        self.dog_id = dog_id
        self.view = view
        self.color = self.MARKER_COLOR

        # 状态: 位置 / 目标位置 / 朝向 / 目标朝向 / 姿态
        self.position = np.array([0.0, 0.0, 0.0])
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0
        self.target_yaw = 0.0
        self.pose = "stand"                 # stand | lie_down | sit
        self._body_h = self.LEG_LEN         # 当前躯干离地高度 (随姿态平滑变化)
        self._target_body_h = self.LEG_LEN

        self.elements: List = []

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
        """
        位置更新后的钩子 (原用于同步躯干/头部/四腿网格变换)。

        机身几何已移除，渲染改由 SwarmView3D 的统一 2D 标记层每帧读取
        self.position 完成，这里不再需要做任何事，仅保留方法签名以兼容
        set_position/step 的调用。
        """
        pass

    def remove(self):
        """从场景移除所有元素 (当前无持有的 GL item，保留接口兼容)"""
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
    无人车状态模型 - 位置/朝向，不再持有 3D 网格

    真实点云场景尺度下几何模型不可见，渲染改为 SwarmView3D 统一维护的
    一层固定像素大小 2D 标记点 (颜色+编号标签)，本类只负责位置状态。

    运动:
        位置 (x, y) + 朝向 yaw；启动后可前进/后退/转向，停车后锁定位置。
    """

    MARKER_COLOR = (0.15, 0.75, 0.85, 1.0)  # 无人车：2D 标记层颜色 (青蓝色)

    def __init__(self, ugv_id: int, view: gl.GLViewWidget):
        self.ugv_id = ugv_id
        self.view = view
        self.color = self.MARKER_COLOR

        self.position = np.array([0.0, 0.0, 0.0])
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.yaw = 0.0
        self.target_yaw = 0.0
        self.active = False
        self.speed_level = 1

        self.elements: List = []

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
        """
        位置更新后的钩子 (原用于同步车体/驾驶舱/轮子网格变换)。

        机身几何已移除，渲染改由 SwarmView3D 的统一 2D 标记层每帧读取
        self.position 完成，这里不再需要做任何事，仅保留方法签名以兼容
        set_position/step 的调用。
        """
        pass

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
