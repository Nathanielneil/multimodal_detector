# -*- coding: utf-8 -*-
"""
无人机集群 3D 可视化组件 - 使用 PyQtGraph 原生渲染

特点:
- 完全嵌入 PySide6，无外部窗口
- 实时响应控制指令
- 平滑动画过渡
"""

import math
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from enum import Enum
from collections import deque

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFont

import pyqtgraph as pg
import pyqtgraph.opengl as gl


class FormationType(Enum):
    """编队类型"""
    TRIANGLE = "triangle"
    SQUARE = "square"
    CIRCLE = "circle"
    STAR = "star"
    LINE = "line"


class ObstacleGenerator:
    """
    障碍物生成器 - 生成柱形点云障碍物

    参考 ego-planner-swarm 项目的实现方式
    """

    # 障碍物颜色
    OBSTACLE_COLOR = (0.6, 0.3, 0.1, 0.8)  # 棕色

    @staticmethod
    def generate_cylinder_points(
        center: Tuple[float, float],
        radius: float,
        height: float,
        resolution: float = 0.05
    ) -> np.ndarray:
        """
        生成单个圆柱体的表面点云

        Args:
            center: 圆柱中心 (x, y)
            radius: 圆柱半径
            height: 圆柱高度
            resolution: 点云分辨率

        Returns:
            点云数组 (N, 3)
        """
        points = []

        # 圆周上的点数
        n_theta = max(8, int(2 * np.pi * radius / resolution))
        # 高度方向的点数
        n_z = max(2, int(height / resolution))

        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        z_vals = np.linspace(0, height, n_z)

        # 生成圆柱侧面点云
        for z in z_vals:
            for t in theta:
                x = center[0] + radius * np.cos(t)
                y = center[1] + radius * np.sin(t)
                points.append([x, y, z])

        return np.array(points)

    @staticmethod
    def generate_box_points(
        center: Tuple[float, float],
        width: float,
        depth: float,
        height: float,
        resolution: float = 0.05
    ) -> np.ndarray:
        """
        生成矩形柱体的表面点云 (空心)

        Args:
            center: 中心 (x, y)
            width: x方向宽度
            depth: y方向深度
            height: 高度
            resolution: 点云分辨率

        Returns:
            点云数组 (N, 3)
        """
        points = []

        nx = max(2, int(width / resolution))
        ny = max(2, int(depth / resolution))
        nz = max(2, int(height / resolution))

        x_vals = np.linspace(center[0] - width/2, center[0] + width/2, nx)
        y_vals = np.linspace(center[1] - depth/2, center[1] + depth/2, ny)
        z_vals = np.linspace(0, height, nz)

        # 只生成表面点 (空心柱体)
        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                for k, z in enumerate(z_vals):
                    # 判断是否在边界上
                    on_x_boundary = (i == 0 or i == nx - 1)
                    on_y_boundary = (j == 0 or j == ny - 1)
                    on_z_boundary = (k == 0 or k == nz - 1)

                    if on_x_boundary or on_y_boundary or on_z_boundary:
                        points.append([x, y, z])

        return np.array(points)

    @staticmethod
    def generate_random_obstacles(
        map_size: Tuple[float, float] = (8.0, 8.0),
        num_obstacles: int = 8,
        min_radius: float = 0.15,
        max_radius: float = 0.4,
        min_height: float = 1.0,
        max_height: float = 3.0,
        safe_zone_radius: float = 2.5,
        min_spacing: float = 0.3,
        seed: int = 42
    ) -> List[Dict]:
        """
        随机生成障碍物配置 (带碰撞检测)

        Args:
            map_size: 地图尺寸 (x, y)
            num_obstacles: 障碍物数量
            min_radius: 最小半径
            max_radius: 最大半径
            min_height: 最小高度
            max_height: 最大高度
            safe_zone_radius: 中心安全区半径 (无人机起飞区)
            min_spacing: 障碍物之间最小间距
            seed: 随机种子

        Returns:
            障碍物配置列表
        """
        np.random.seed(seed)
        obstacles = []
        max_attempts = 100  # 每个障碍物最多尝试次数

        for _ in range(num_obstacles):
            # 先随机生成尺寸
            radius = np.random.uniform(min_radius, max_radius)
            height = np.random.uniform(min_height, max_height)

            # 尝试找到不重叠的位置
            placed = False
            for attempt in range(max_attempts):
                x = np.random.uniform(-map_size[0]/2 + radius, map_size[0]/2 - radius)
                y = np.random.uniform(-map_size[1]/2 + radius, map_size[1]/2 - radius)

                # 检查是否在安全区外
                dist_to_center = np.sqrt(x**2 + y**2)
                if dist_to_center < safe_zone_radius + radius:
                    continue

                # 检查与已有障碍物是否重叠
                overlap = False
                for obs in obstacles:
                    ox, oy = obs['center']
                    or_ = obs['radius']
                    dist = np.sqrt((x - ox)**2 + (y - oy)**2)
                    # 两圆心距离需要大于两半径之和加上最小间距
                    if dist < radius + or_ + min_spacing:
                        overlap = True
                        break

                if not overlap:
                    obstacles.append({
                        'center': (x, y),
                        'radius': radius,
                        'height': height,
                        'type': 'cylinder'
                    })
                    placed = True
                    break

            if not placed:
                # 无法放置更多障碍物，空间不足
                break

        return obstacles


class LidarSensor:
    """
    激光雷达传感器模拟

    模拟2D激光雷达扫描，检测周围障碍物
    """

    def __init__(
        self,
        max_range: float = 2.0,
        num_rays: int = 36,
        fov: float = 360.0
    ):
        """
        Args:
            max_range: 最大探测距离 (米)
            num_rays: 射线数量
            fov: 视场角 (度)
        """
        self.max_range = max_range
        self.num_rays = num_rays
        self.fov = math.radians(fov)

        # 预计算射线角度
        self.ray_angles = np.linspace(
            -self.fov / 2, self.fov / 2, num_rays, endpoint=False
        )

        # 检测结果
        self.hit_points: List[np.ndarray] = []
        self.distances: List[float] = []

    def scan(
        self,
        position: np.ndarray,
        obstacles: List[Dict],
        dynamic_obstacles: List[Dict] = None
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        执行激光雷达扫描

        Args:
            position: 传感器位置 (x, y, z)
            obstacles: 静态障碍物列表 [{'center': (x,y), 'radius': r}, ...]
            dynamic_obstacles: 动态障碍物列表

        Returns:
            (hit_points, distances): 击中点列表和距离列表
        """
        self.hit_points = []
        self.distances = []

        all_obstacles = list(obstacles)
        if dynamic_obstacles:
            all_obstacles.extend(dynamic_obstacles)

        for angle in self.ray_angles:
            # 射线方向
            dx = math.cos(angle)
            dy = math.sin(angle)

            min_dist = self.max_range
            hit_point = None

            # 检测与所有障碍物的交点
            for obs in all_obstacles:
                ox, oy = obs['center']
                r = obs.get('radius', 0.3)

                # 射线-圆交点检测
                dist = self._ray_circle_intersection(
                    position[0], position[1], dx, dy, ox, oy, r
                )

                if dist is not None and dist < min_dist:
                    min_dist = dist
                    hit_point = np.array([
                        position[0] + dx * dist,
                        position[1] + dy * dist,
                        position[2]
                    ])

            self.distances.append(min_dist)
            if hit_point is not None:
                self.hit_points.append(hit_point)
            else:
                # 未击中，记录最大距离点
                self.hit_points.append(np.array([
                    position[0] + dx * self.max_range,
                    position[1] + dy * self.max_range,
                    position[2]
                ]))

        return self.hit_points, self.distances

    def _ray_circle_intersection(
        self, rx, ry, dx, dy, cx, cy, r
    ) -> Optional[float]:
        """
        射线与圆的交点检测

        Args:
            rx, ry: 射线起点
            dx, dy: 射线方向 (单位向量)
            cx, cy: 圆心
            r: 圆半径

        Returns:
            距离，如果没有交点则返回 None
        """
        # 射线起点到圆心的向量
        fx = rx - cx
        fy = ry - cy

        a = dx * dx + dy * dy  # = 1 (单位向量)
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - r * r

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        # 返回最近的正交点
        if t1 > 0.01:
            return t1
        elif t2 > 0.01:
            return t2
        return None


class ObstacleAvoidance:
    """
    避障算法 - 人工势场法 (Artificial Potential Field)

    - 目标点产生引力
    - 障碍物产生斥力
    - 计算合力方向调整速度
    """

    def __init__(
        self,
        attractive_gain: float = 1.0,
        repulsive_gain: float = 0.8,
        influence_distance: float = 1.5,
        max_speed: float = 0.5
    ):
        """
        Args:
            attractive_gain: 引力增益
            repulsive_gain: 斥力增益
            influence_distance: 障碍物影响距离
            max_speed: 最大速度
        """
        self.k_att = attractive_gain
        self.k_rep = repulsive_gain
        self.d0 = influence_distance
        self.max_speed = max_speed

    def compute_velocity(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        obstacles: List[Dict],
        dynamic_obstacles: List[Dict] = None
    ) -> np.ndarray:
        """
        计算避障后的速度向量

        Args:
            current_pos: 当前位置 (x, y, z)
            target_pos: 目标位置 (x, y, z)
            obstacles: 静态障碍物列表
            dynamic_obstacles: 动态障碍物列表

        Returns:
            速度向量 (vx, vy, vz)
        """
        # 引力 (指向目标)
        f_att = self._attractive_force(current_pos, target_pos)

        # 斥力 (远离障碍物)
        f_rep = np.zeros(3)
        all_obstacles = list(obstacles)
        if dynamic_obstacles:
            all_obstacles.extend(dynamic_obstacles)

        for obs in all_obstacles:
            f_rep += self._repulsive_force(current_pos, obs)

        # 合力
        f_total = f_att + f_rep

        # 限制最大速度
        speed = np.linalg.norm(f_total)
        if speed > self.max_speed:
            f_total = f_total / speed * self.max_speed

        return f_total

    def _attractive_force(
        self, current: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        """计算引力 - 快速归位"""
        diff = target - current
        dist = np.linalg.norm(diff)

        if dist < 0.05:
            # 非常接近目标时，返回零力避免震荡
            return np.zeros(3)

        # 距离衰减：仅在非常接近时衰减 (0.2m内)
        if dist < 0.2:
            scale = dist / 0.2
        else:
            scale = 1.0

        # 引力与距离成比例，快速归位
        magnitude = min(self.k_att * dist * scale, self.k_att * 1.5)
        return magnitude * diff / dist

    def _repulsive_force(
        self, current: np.ndarray, obstacle: Dict
    ) -> np.ndarray:
        """计算斥力"""
        ox, oy = obstacle['center']
        r = obstacle.get('radius', 0.3)
        obs_pos = np.array([ox, oy, current[2]])

        diff = current - obs_pos
        dist = np.linalg.norm(diff[:2])  # 只考虑水平距离
        dist_to_surface = dist - r

        if dist_to_surface >= self.d0 or dist_to_surface < 0.01:
            return np.zeros(3)

        # 斥力公式: k_rep * (1/d - 1/d0) * (1/d^2) * direction
        magnitude = self.k_rep * (1.0 / dist_to_surface - 1.0 / self.d0) * \
                    (1.0 / (dist_to_surface ** 2))

        direction = diff / (dist + 0.001)
        direction[2] = 0  # 水平斥力

        return magnitude * direction


class DynamicObstacle:
    """
    动态障碍物 - 沿路径移动的障碍物
    """

    def __init__(
        self,
        waypoints: List[Tuple[float, float]],
        radius: float = 0.25,
        height: float = 2.0,
        speed: float = 0.3,
        color: Tuple[float, float, float, float] = (0.9, 0.2, 0.2, 0.9)
    ):
        """
        Args:
            waypoints: 路径点列表 [(x1,y1), (x2,y2), ...]
            radius: 障碍物半径
            height: 障碍物高度
            speed: 移动速度 (m/s)
            color: 颜色
        """
        self.waypoints = waypoints
        self.radius = radius
        self.height = height
        self.speed = speed
        self.color = color

        # 当前状态
        self.current_waypoint_idx = 0
        self.position = np.array([waypoints[0][0], waypoints[0][1], 0.0])
        self.direction = 1  # 1: 正向, -1: 反向

        # 可视化元素
        self.scatter: Optional[gl.GLScatterPlotItem] = None

    def update(self, dt: float) -> np.ndarray:
        """
        更新位置

        Args:
            dt: 时间步长 (秒)

        Returns:
            新位置
        """
        if len(self.waypoints) < 2:
            return self.position

        # 目标路径点
        target_idx = self.current_waypoint_idx + self.direction
        if target_idx >= len(self.waypoints):
            target_idx = len(self.waypoints) - 2
            self.direction = -1
        elif target_idx < 0:
            target_idx = 1
            self.direction = 1

        target = np.array([
            self.waypoints[target_idx][0],
            self.waypoints[target_idx][1],
            0.0
        ])

        # 移动
        diff = target - self.position
        dist = np.linalg.norm(diff[:2])

        if dist < 0.1:
            # 到达路径点
            self.current_waypoint_idx = target_idx
        else:
            # 向目标移动
            move = diff / dist * self.speed * dt
            self.position += move

        return self.position

    def get_config(self) -> Dict:
        """获取障碍物配置 (用于碰撞检测)"""
        return {
            'center': (self.position[0], self.position[1]),
            'radius': self.radius,
            'height': self.height,
            'type': 'dynamic'
        }

    def create_visual(self, view: gl.GLViewWidget) -> gl.GLScatterPlotItem:
        """创建可视化元素"""
        points = ObstacleGenerator.generate_cylinder_points(
            center=(0, 0),  # 相对位置
            radius=self.radius,
            height=self.height,
            resolution=0.1
        )

        # 设置颜色
        colors = np.zeros((len(points), 4))
        for i, pt in enumerate(points):
            height_ratio = pt[2] / self.height
            colors[i] = [
                self.color[0] * (0.7 + 0.3 * height_ratio),
                self.color[1] * (0.7 + 0.3 * height_ratio),
                self.color[2] * (0.7 + 0.3 * height_ratio),
                self.color[3]
            ]

        self.scatter = gl.GLScatterPlotItem(
            pos=points + self.position,
            color=colors,
            size=4,
            pxMode=True
        )
        view.addItem(self.scatter)
        return self.scatter

    def update_visual(self):
        """更新可视化位置"""
        if self.scatter is None:
            return

        # 重新生成点云位置
        points = ObstacleGenerator.generate_cylinder_points(
            center=(0, 0),
            radius=self.radius,
            height=self.height,
            resolution=0.1
        )
        self.scatter.setData(pos=points + self.position)


class SwarmCommand(Enum):
    """集群指令"""
    TAKEOFF = "takeoff"
    LAND = "land"
    HOVER = "hover"
    FORMATION = "formation"
    CONFIRM = "confirm"
    ALTITUDE_UP = "altitude_up"
    ALTITUDE_DOWN = "altitude_down"
    EMERGENCY_STOP = "emergency"
    MOVE_FORWARD = "move_forward"  # 编队向前飞行


class FormationGenerator:
    """编队位置生成器"""

    @staticmethod
    def triangle(center: Tuple[float, float], radius: float, altitude: float, count: int = 3) -> List[Tuple]:
        """
        三角形编队 - 支持任意数量无人机

        布局：多层三角形，每层递减
        """
        positions = []
        remaining = count
        layer = 0
        layer_radius = radius

        while remaining > 0:
            # 每层的三角形顶点数
            layer_count = min(3, remaining)
            layer_alt = altitude + layer * 0.5

            for i in range(layer_count):
                angle = 2 * math.pi * i / 3 - math.pi / 2
                x = center[0] + layer_radius * math.cos(angle)
                y = center[1] + layer_radius * math.sin(angle)
                positions.append((x, y, layer_alt))

            remaining -= layer_count
            layer += 1
            layer_radius *= 0.6  # 内层缩小

            # 如果只剩1个，放中心
            if remaining == 1:
                positions.append((center[0], center[1], altitude + layer * 0.5))
                break

        return positions

    @staticmethod
    def square(center: Tuple[float, float], size: float, altitude: float, count: int = 4) -> List[Tuple]:
        """
        正方形编队 - 支持任意数量无人机

        布局：多层正方形，每层递减
        """
        positions = []
        remaining = count
        layer = 0
        layer_size = size

        while remaining > 0:
            half = layer_size / 2
            layer_alt = altitude + layer * 0.5

            # 4个角
            corners = [
                (center[0] - half, center[1] - half, layer_alt),
                (center[0] + half, center[1] - half, layer_alt),
                (center[0] + half, center[1] + half, layer_alt),
                (center[0] - half, center[1] + half, layer_alt),
            ]

            for corner in corners[:min(4, remaining)]:
                positions.append(corner)

            remaining -= min(4, remaining)
            layer += 1
            layer_size *= 0.5  # 内层缩小

            # 如果只剩1个，放中心
            if remaining == 1:
                positions.append((center[0], center[1], altitude + layer * 0.5))
                break

        return positions

    @staticmethod
    def circle(center: Tuple[float, float], radius: float, altitude: float, count: int = 6) -> List[Tuple]:
        """圆形编队"""
        positions = []
        for i in range(count):
            angle = 2 * math.pi * i / count
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            positions.append((x, y, altitude))
        return positions

    @staticmethod
    def star(center: Tuple[float, float], radius: float, altitude: float, count: int = 5) -> List[Tuple]:
        """
        五角星编队 - 支持任意数量无人机

        布局：
        - 1-5架：外圈5个顶点
        - 6-10架：内圈5个点（五角星的内凹点）
        - 11架：中心点
        - 12+架：中心垂直堆叠
        """
        positions = []
        inner_radius = radius * 0.382  # 黄金比例内圈

        # 外圈5个顶点
        for i in range(min(count, 5)):
            angle = 2 * math.pi * i / 5 - math.pi / 2
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            positions.append((x, y, altitude))

        # 内圈5个点（旋转36度，形成五角星内凹）
        for i in range(5, min(count, 10)):
            angle = 2 * math.pi * (i - 5) / 5 - math.pi / 2 + math.pi / 5
            x = center[0] + inner_radius * math.cos(angle)
            y = center[1] + inner_radius * math.sin(angle)
            positions.append((x, y, altitude))

        # 中心点
        if count > 10:
            positions.append((center[0], center[1], altitude))

        # 超过11架：中心垂直堆叠
        for i in range(11, count):
            positions.append((center[0], center[1], altitude + 0.5 * (i - 10)))

        return positions

    @staticmethod
    def line(center: Tuple[float, float], spacing: float, altitude: float, count: int = 6) -> List[Tuple]:
        """直线编队"""
        positions = []
        start_x = center[0] - (count - 1) * spacing / 2
        for i in range(count):
            x = start_x + i * spacing
            positions.append((x, center[1], altitude))
        return positions


class QuadrotorDynamics:
    """
    L1 四旋翼动力学模型 (Sim2Real) - 基础版本

    状态:
    - 位置: [x, y, z] (世界坐标系)
    - 速度: [vx, vy, vz] (世界坐标系)

    控制:
    - PD位置控制器
    - 直接加速度输出

    特性:
    - 质量、推力限制
    - 线性阻力
    - 半隐式欧拉积分
    """

    def __init__(
        self,
        mass: float = 1.5,           # 质量 (kg)
        max_thrust: float = 30.0,    # 最大推力 (N), ~2g
        max_velocity: float = 5.0,   # 最大速度 (m/s)
        drag_coeff: float = 0.3,     # 线性阻力系数
        gravity: float = 9.81,       # 重力加速度
    ):
        self.mass = mass
        self.max_thrust = max_thrust
        self.max_velocity = max_velocity
        self.drag_coeff = drag_coeff
        self.gravity = gravity

        # 计算最大加速度
        self.max_accel = max_thrust / mass - gravity
        self.max_accel_down = max_thrust / mass + gravity

        # ========== 状态变量 ==========
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])

        # 控制输入
        self._target_position = np.array([0.0, 0.0, 0.0])

        # ========== 控制器参数 ==========
        self.kp = 2.0      # 位置比例增益
        self.kd = 1.5      # 位置微分增益

    def set_state(self, position: np.ndarray, velocity: np.ndarray = None, **kwargs):
        """设置状态"""
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float) if velocity is not None else np.zeros(3)

    def set_target(self, target: np.ndarray, **kwargs):
        """设置目标位置"""
        self._target_position = np.array(target, dtype=float)

    def step(self, dt: float, external_accel: np.ndarray = None) -> np.ndarray:
        """
        动力学积分一步

        Args:
            dt: 时间步长 (秒)
            external_accel: 外部加速度 (如避障力), 可选

        Returns:
            新位置
        """
        if external_accel is not None:
            # 使用外部加速度 (来自避障等)
            accel = external_accel.copy()
        else:
            # PD控制
            pos_error = self._target_position - self.position
            accel = self.kp * pos_error - self.kd * self.velocity

        # 加速度限幅 (考虑推力限制)
        accel_xy = accel[:2]
        accel_xy_norm = np.linalg.norm(accel_xy)
        if accel_xy_norm > self.max_accel:
            accel[:2] = accel_xy / accel_xy_norm * self.max_accel

        accel[2] = np.clip(accel[2], -self.max_accel_down, self.max_accel)

        # 阻力
        drag = -self.drag_coeff * self.velocity

        # 总加速度
        total_accel = accel + drag

        # 半隐式欧拉积分
        self.velocity = self.velocity + total_accel * dt

        # 速度限幅
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_velocity:
            self.velocity = self.velocity / speed * self.max_velocity

        # 更新位置
        self.position = self.position + self.velocity * dt

        return self.position

    def get_state(self) -> dict:
        """获取完整状态"""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'speed': np.linalg.norm(self.velocity),
            'distance_to_target': np.linalg.norm(self._target_position - self.position),
        }

    def is_at_target(self, tolerance: float = 0.1) -> bool:
        """检查是否到达目标"""
        dist = np.linalg.norm(self._target_position - self.position)
        speed = np.linalg.norm(self.velocity)
        return dist < tolerance and speed < 0.1


class DroneModel:
    """
    单个无人机 3D 模型 - 简洁清晰的四旋翼样式

    结构:
    - 中央机身 (扁平圆柱)
    - 4个机臂 (线条)
    - 4个电机座 (小圆柱)
    - 4个旋翼 (圆环 + 桨叶)
    - 起落架 (线条)
    """

    # 无人机颜色列表
    COLORS = [
        (0.2, 0.6, 1.0, 1.0),  # 蓝
        (1.0, 0.4, 0.2, 1.0),  # 橙红
        (0.3, 0.9, 0.3, 1.0),  # 绿
        (1.0, 0.9, 0.2, 1.0),  # 黄
        (0.9, 0.3, 0.9, 1.0),  # 紫
        (0.2, 0.9, 0.9, 1.0),  # 青
        (1.0, 0.5, 0.7, 1.0),  # 粉
        (0.7, 0.5, 1.0, 1.0),  # 淡紫
    ]

    # 模型参数
    ARM_LENGTH = 0.15       # 机臂长度
    ARM_ANGLES = [45, 135, 225, 315]  # X形布局
    ROTOR_RADIUS = 0.06     # 旋翼半径
    BODY_RADIUS = 0.035     # 机身半径 (更小)
    BODY_HEIGHT = 0.015     # 机身高度

    def __init__(self, drone_id: int, view: gl.GLViewWidget, enable_lidar: bool = True, use_dynamics: bool = True):
        self.drone_id = drone_id
        self.view = view
        self.position = np.array([0.0, 0.0, 0.0])
        self.target_position = np.array([0.0, 0.0, 0.0])

        # 获取颜色
        self.color = self.COLORS[drone_id % len(self.COLORS)]

        # 所有3D元素
        self.elements = []

        # L1 动力学模型 (Sim2Real)
        self.use_dynamics = use_dynamics
        if use_dynamics:
            self.dynamics = QuadrotorDynamics(
                mass=1.5,            # 1.5kg 典型小型无人机
                max_thrust=30.0,     # ~2g 推力
                max_velocity=3.0,    # 3 m/s 最大速度
                drag_coeff=0.5,      # 阻力系数
            )
        else:
            self.dynamics = None

        # 轨迹历史 (时间衰减模式)
        self.trajectory_max_len = 60        # 最大轨迹点数 (~6秒)
        self.trajectory_sample_interval = 3  # 每N帧采样一次
        self._trajectory_frame_count = 0
        self._trajectory_points: deque = deque(maxlen=self.trajectory_max_len)
        self._trajectory_visual: Optional[gl.GLLinePlotItem] = None
        self._trajectory_enabled = True

        # 激光雷达传感器
        self.enable_lidar = enable_lidar
        self.lidar = LidarSensor(max_range=1.8, num_rays=24, fov=360.0) if enable_lidar else None
        self.lidar_visual: Optional[gl.GLLinePlotItem] = None
        self.lidar_hits_visual: Optional[gl.GLScatterPlotItem] = None

        # 避障算法 (针对扩大场景优化参数)
        self.obstacle_avoidance = ObstacleAvoidance(
            attractive_gain=4.0,      # 增强引力 (2.0→4.0 加快编队恢复)
            repulsive_gain=0.5,       # 斥力
            influence_distance=0.6,   # 缩小影响距离 (0.8→0.6 更早恢复)
            max_speed=4.0             # 提高最大速度 (3.0→4.0)
        )

        # 避障模式开关
        self.avoidance_enabled = True

        # 创建 3D 元素
        self._create_body()
        self._create_arms()
        self._create_motors()
        self._create_rotors()
        self._create_landing_gear()
        self._create_trajectory_visual()
        if enable_lidar:
            self._create_lidar_visual()

    def _create_body(self):
        """创建机身 (扁平圆柱)"""
        mesh_data = gl.MeshData.cylinder(
            rows=6, cols=12,
            radius=[self.BODY_RADIUS, self.BODY_RADIUS],
            length=self.BODY_HEIGHT
        )
        self.body = gl.GLMeshItem(
            meshdata=mesh_data,
            smooth=True,
            color=self.color,
            shader='shaded',
            glOptions='opaque'
        )
        self.view.addItem(self.body)
        self.elements.append(self.body)

    def _create_arms(self):
        """创建机臂 (4根线条)"""
        self.arms = []
        for angle in self.ARM_ANGLES:
            rad = math.radians(angle)
            end_x = self.ARM_LENGTH * math.cos(rad)
            end_y = self.ARM_LENGTH * math.sin(rad)

            # 机臂线段
            arm_pts = np.array([
                [0, 0, 0],
                [end_x, end_y, 0]
            ])
            arm = gl.GLLinePlotItem(
                pos=arm_pts,
                color=(0.3, 0.3, 0.3, 1.0),
                width=4,
                antialias=True
            )
            self.arms.append(arm)
            self.view.addItem(arm)
            self.elements.append(arm)

    def _create_motors(self):
        """创建电机座 (4个小圆柱)"""
        self.motors = []
        motor_radius = 0.012
        motor_height = 0.012

        for angle in self.ARM_ANGLES:
            rad = math.radians(angle)
            pos_x = self.ARM_LENGTH * math.cos(rad)
            pos_y = self.ARM_LENGTH * math.sin(rad)

            mesh_data = gl.MeshData.cylinder(
                rows=4, cols=8,
                radius=[motor_radius, motor_radius],
                length=motor_height
            )
            motor = gl.GLMeshItem(
                meshdata=mesh_data,
                smooth=True,
                color=(0.25, 0.25, 0.25, 1.0),
                shader='shaded',
                glOptions='opaque'
            )
            motor.translate(pos_x, pos_y, 0)
            self.motors.append(motor)
            self.view.addItem(motor)
            self.elements.append(motor)

    def _create_rotors(self):
        """创建旋翼 (4个圆盘 + 桨叶)"""
        self.rotors = []

        for i, angle in enumerate(self.ARM_ANGLES):
            rad = math.radians(angle)
            pos_x = self.ARM_LENGTH * math.cos(rad)
            pos_y = self.ARM_LENGTH * math.sin(rad)

            # 旋翼圆环
            theta = np.linspace(0, 2 * np.pi, 32)
            rotor_pts = np.zeros((32, 3))
            rotor_pts[:, 0] = pos_x + self.ROTOR_RADIUS * np.cos(theta)
            rotor_pts[:, 1] = pos_y + self.ROTOR_RADIUS * np.sin(theta)
            rotor_pts[:, 2] = 0.015

            # 前方旋翼红色，后方旋翼深灰色
            is_front = angle in [45, 315]
            rotor_color = (0.9, 0.2, 0.2, 0.9) if is_front else (0.4, 0.4, 0.4, 0.9)

            rotor = gl.GLLinePlotItem(
                pos=rotor_pts,
                color=rotor_color,
                width=2.5,
                antialias=True
            )
            self.rotors.append(rotor)
            self.view.addItem(rotor)
            self.elements.append(rotor)

            # 旋翼桨叶 (两片)
            blade_len = self.ROTOR_RADIUS * 0.85
            blade_pts = np.array([
                [pos_x - blade_len, pos_y, 0.015],
                [pos_x + blade_len, pos_y, 0.015],
            ])
            blade = gl.GLLinePlotItem(
                pos=blade_pts,
                color=rotor_color,
                width=3,
                antialias=True
            )
            self.rotors.append(blade)
            self.view.addItem(blade)
            self.elements.append(blade)

    def _create_landing_gear(self):
        """创建起落架 (4条腿)"""
        self.legs = []
        leg_angles = [0, 90, 180, 270]
        leg_radius = self.BODY_RADIUS * 0.8
        leg_height = 0.035

        for angle in leg_angles:
            rad = math.radians(angle)
            base_x = leg_radius * math.cos(rad)
            base_y = leg_radius * math.sin(rad)

            leg_pts = np.array([
                [base_x, base_y, -self.BODY_HEIGHT/2],
                [base_x * 1.3, base_y * 1.3, -self.BODY_HEIGHT/2 - leg_height]
            ])
            leg = gl.GLLinePlotItem(
                pos=leg_pts,
                color=(0.35, 0.35, 0.35, 1.0),
                width=2.5,
                antialias=True
            )
            self.legs.append(leg)
            self.view.addItem(leg)
            self.elements.append(leg)

    def _create_trajectory_visual(self):
        """创建轨迹可视化元素"""
        # 初始化空轨迹线
        self._trajectory_visual = gl.GLLinePlotItem(
            pos=np.zeros((1, 3)),
            color=self.color,
            width=2.5,
            antialias=True,
            mode='line_strip'
        )
        self.view.addItem(self._trajectory_visual)
        self.elements.append(self._trajectory_visual)

    def update_trajectory(self):
        """更新轨迹 - 记录当前位置并更新可视化"""
        if not self._trajectory_enabled:
            return

        # 采样控制
        self._trajectory_frame_count += 1
        if self._trajectory_frame_count < self.trajectory_sample_interval:
            return
        self._trajectory_frame_count = 0

        # 只在有高度时记录轨迹 (z > 0.05)
        if self.position[2] < 0.05:
            return

        # 添加当前位置到轨迹
        self._trajectory_points.append(self.position.copy())

        # 更新可视化
        self._update_trajectory_visual()

    def _update_trajectory_visual(self):
        """更新轨迹线的可视化"""
        if not self._trajectory_visual or len(self._trajectory_points) < 2:
            return

        # 转换为数组
        pts = np.array(list(self._trajectory_points))

        # 创建颜色渐变 (旧点透明，新点不透明)
        n = len(pts)
        colors = np.zeros((n, 4))
        for i in range(n):
            # 从旧到新: alpha 从 0.2 渐变到 1.0
            alpha = 0.2 + 0.8 * (i / max(1, n - 1))
            # 亮度也渐变
            brightness = 0.6 + 0.4 * (i / max(1, n - 1))
            colors[i] = [
                self.color[0] * brightness,
                self.color[1] * brightness,
                self.color[2] * brightness,
                alpha
            ]

        self._trajectory_visual.setData(pos=pts, color=colors)

    def clear_trajectory(self):
        """清除轨迹历史"""
        self._trajectory_points.clear()
        self._trajectory_frame_count = 0
        if self._trajectory_visual:
            self._trajectory_visual.setData(pos=np.zeros((1, 3)))

    def set_trajectory_enabled(self, enabled: bool):
        """启用/禁用轨迹显示"""
        self._trajectory_enabled = enabled
        if self._trajectory_visual:
            if enabled:
                self._update_trajectory_visual()
            else:
                self._trajectory_visual.setData(pos=np.zeros((1, 3)))

    def _create_lidar_visual(self):
        """创建激光雷达可视化元素"""
        # 激光射线 (从无人机中心向外)
        num_rays = self.lidar.num_rays if self.lidar else 24
        ray_pts = np.zeros((num_rays * 2, 3))  # 每条射线2个点
        self.lidar_visual = gl.GLLinePlotItem(
            pos=ray_pts,
            color=(0.0, 1.0, 0.5, 0.3),
            width=1,
            antialias=True,
            mode='lines'
        )
        self.view.addItem(self.lidar_visual)
        self.elements.append(self.lidar_visual)

        # 击中点 (检测到障碍物的点)
        self.lidar_hits_visual = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            color=(1.0, 0.3, 0.0, 0.9),
            size=6,
            pxMode=True
        )
        self.view.addItem(self.lidar_hits_visual)
        self.elements.append(self.lidar_hits_visual)

    def update_lidar(self, obstacles: List[Dict], dynamic_obstacles: List[Dict] = None):
        """
        更新激光雷达扫描和可视化

        Args:
            obstacles: 静态障碍物列表
            dynamic_obstacles: 动态障碍物列表
        """
        if not self.enable_lidar or not self.lidar:
            return

        # 执行扫描
        hit_points, distances = self.lidar.scan(
            self.position, obstacles, dynamic_obstacles
        )

        # 更新射线可视化
        ray_pts = []
        for hit_pt in hit_points:
            ray_pts.append(self.position)
            ray_pts.append(hit_pt)

        if ray_pts:
            self.lidar_visual.setData(pos=np.array(ray_pts))

        # 更新击中点可视化 (只显示检测到障碍物的点)
        actual_hits = []
        for i, dist in enumerate(distances):
            if dist < self.lidar.max_range - 0.01:
                actual_hits.append(hit_points[i])

        if actual_hits:
            self.lidar_hits_visual.setData(pos=np.array(actual_hits))
        else:
            self.lidar_hits_visual.setData(pos=np.zeros((1, 3)))

    def compute_avoidance_velocity(
        self,
        obstacles: List[Dict],
        dynamic_obstacles: List[Dict] = None
    ) -> np.ndarray:
        """
        计算避障速度

        Args:
            obstacles: 静态障碍物列表
            dynamic_obstacles: 动态障碍物列表

        Returns:
            速度向量
        """
        if not self.avoidance_enabled:
            # 直接返回指向目标的速度
            diff = self.target_position - self.position
            dist = np.linalg.norm(diff)
            if dist < 0.01:
                return np.zeros(3)
            return diff / dist * 0.3

        return self.obstacle_avoidance.compute_velocity(
            self.position, self.target_position, obstacles, dynamic_obstacles
        )

    def set_position(self, pos: Tuple[float, float, float]):
        """设置位置"""
        self.position = np.array(pos)
        if self.dynamics:
            self.dynamics.set_state(self.position)
        self._update_transform()

    def set_target(self, pos: Tuple[float, float, float]):
        """设置目标位置"""
        self.target_position = np.array(pos)
        if self.dynamics:
            self.dynamics.set_target(self.target_position)

    def step_dynamics(self, dt: float, external_accel: np.ndarray = None):
        """
        使用动力学模型更新位置

        Args:
            dt: 时间步长
            external_accel: 外部加速度 (避障力等)
        """
        if self.dynamics:
            self.position = self.dynamics.step(dt, external_accel)
            self._update_transform()

    def get_velocity(self) -> np.ndarray:
        """获取当前速度"""
        if self.dynamics:
            return self.dynamics.velocity.copy()
        return np.zeros(3)

    def get_attitude(self) -> np.ndarray:
        """获取当前姿态 [roll, pitch, yaw] (弧度) - L1无姿态"""
        return np.zeros(3)

    def get_attitude_deg(self) -> tuple:
        """获取当前姿态 (度) - L1无姿态"""
        return (0.0, 0.0, 0.0)

    def update_animation(self, t: float):
        """更新动画 (t: 0-1 插值因子) - 旧方法，保留兼容"""
        # 平滑插值
        t_smooth = t * t * (3 - 2 * t)  # smoothstep
        self.position = self.position + (self.target_position - self.position) * t_smooth * 0.1
        self._update_transform()

    def _rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """计算旋转矩阵 (ZYX欧拉角)"""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ])
        return R

    def _rotate_point(self, local_pos: np.ndarray, R: np.ndarray, world_pos: np.ndarray) -> np.ndarray:
        """将局部坐标旋转并平移到世界坐标"""
        return R @ local_pos + world_pos

    def _update_transform(self):
        """更新所有元素的位置 (L1: 仅位置，无姿态倾斜)"""
        x, y, z = self.position

        # 更新机身
        self.body.resetTransform()
        self.body.translate(x, y, z)

        # 更新机臂
        for i, arm in enumerate(self.arms):
            angle = self.ARM_ANGLES[i]
            rad = math.radians(angle)
            arm_pts = np.array([
                [x, y, z],
                [x + self.ARM_LENGTH * math.cos(rad),
                 y + self.ARM_LENGTH * math.sin(rad), z]
            ])
            arm.setData(pos=arm_pts)

        # 更新电机座
        for i, motor in enumerate(self.motors):
            angle = self.ARM_ANGLES[i]
            rad = math.radians(angle)
            motor_x = x + self.ARM_LENGTH * math.cos(rad)
            motor_y = y + self.ARM_LENGTH * math.sin(rad)
            motor.resetTransform()
            motor.translate(motor_x, motor_y, z)

        # 更新旋翼
        rotor_idx = 0
        for i, angle in enumerate(self.ARM_ANGLES):
            rad = math.radians(angle)
            motor_x = x + self.ARM_LENGTH * math.cos(rad)
            motor_y = y + self.ARM_LENGTH * math.sin(rad)
            motor_z = z + 0.015

            # 旋翼圆环
            theta = np.linspace(0, 2 * np.pi, 32)
            rotor_pts = np.zeros((32, 3))
            for j in range(32):
                rotor_pts[j] = [
                    motor_x + self.ROTOR_RADIUS * np.cos(theta[j]),
                    motor_y + self.ROTOR_RADIUS * np.sin(theta[j]),
                    motor_z
                ]
            self.rotors[rotor_idx].setData(pos=rotor_pts)
            rotor_idx += 1

            # 桨叶
            blade_len = self.ROTOR_RADIUS * 0.85
            blade_pts = np.array([
                [motor_x - blade_len, motor_y, motor_z],
                [motor_x + blade_len, motor_y, motor_z]
            ])
            self.rotors[rotor_idx].setData(pos=blade_pts)
            rotor_idx += 1

        # 更新起落架
        leg_angles = [0, 90, 180, 270]
        leg_radius = self.BODY_RADIUS * 0.8
        leg_height = 0.035

        for i, leg in enumerate(self.legs):
            angle = leg_angles[i]
            rad = math.radians(angle)
            base_x = x + leg_radius * math.cos(rad)
            base_y = y + leg_radius * math.sin(rad)
            base_z = z - self.BODY_HEIGHT / 2

            foot_x = x + leg_radius * 1.3 * math.cos(rad)
            foot_y = y + leg_radius * 1.3 * math.sin(rad)
            foot_z = base_z - leg_height

            leg_pts = np.array([
                [base_x, base_y, base_z],
                [foot_x, foot_y, foot_z]
            ])
            leg.setData(pos=leg_pts)

    def remove(self):
        """从场景中移除"""
        for element in self.elements:
            self.view.removeItem(element)
        self.elements.clear()


class SwarmView3D(QWidget):
    """
    无人机集群 3D 可视化组件

    使用 PyQtGraph OpenGL 原生渲染，完全嵌入 PySide6
    """

    # 信号
    command_executed = Signal(str)  # 指令执行完成

    def __init__(self, parent=None):
        super().__init__(parent)

        # 状态
        self.drone_count = 6
        self.formation_radius = 2.0
        self.base_altitude = 1.5
        self.center = (0.0, 0.0)

        self.current_formation = FormationType.CIRCLE
        self.is_flying = False
        self.in_formation = False  # 是否已进入编队
        self.current_altitude = 0.0

        # 边界限制 (32x32 场景，留 1m 缓冲)
        self.boundary_limit = 15.0  # ±15m 水平边界
        self.max_altitude = 8.0     # 最大飞行高度 8m

        # 无人机模型
        self.drones: List[DroneModel] = []

        # 编队连线
        self.formation_lines: Optional[gl.GLLinePlotItem] = None

        # 动画
        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._update_animation)
        self._animation_timer.start(33)  # 30 FPS
        self._last_animation_time = None  # 用于计算实际 dt

        # 静态障碍物
        self.obstacles: List[gl.GLScatterPlotItem] = []
        self.obstacle_configs: List[Dict] = []  # 障碍物配置 (用于碰撞检测)

        # 动态障碍物
        self.dynamic_obstacles: List[DynamicObstacle] = []

        # 避障模式
        self.avoidance_enabled = True

        # 轨迹显示
        self.trajectory_enabled = True

        self._setup_ui()
        self._setup_3d_scene()
        self._create_boundary_fence()  # 先创建边界围栏
        self._create_obstacles()
        self._create_dynamic_obstacles()
        self._create_drones()
        self._init_ground_positions()

    def _setup_ui(self):
        """设置界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 顶部状态栏
        status_bar = QFrame()
        status_bar.setFixedHeight(36)
        status_bar.setStyleSheet("""
            QFrame {
                background-color: #2a2a3a;
                border-bottom: 1px solid #3a3a4a;
            }
        """)
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(10, 0, 10, 0)

        # 状态指示
        self._status_indicator = QLabel("● 地面待命")
        self._status_indicator.setStyleSheet("color: #888888; font-size: 12px;")

        # 编队类型
        self._formation_label = QLabel("编队: 圆形")
        self._formation_label.setStyleSheet("color: #4FC3F7; font-size: 12px;")

        # 高度显示
        self._altitude_label = QLabel("高度: 0.0m")
        self._altitude_label.setStyleSheet("color: #81C784; font-size: 12px;")

        # 无人机数量
        self._count_label = QLabel(f"无人机: {self.drone_count}架")
        self._count_label.setStyleSheet("color: #FFB74D; font-size: 12px;")

        # 避障状态
        self._avoidance_label = QLabel("避障: 开启")
        self._avoidance_label.setStyleSheet("color: #4CAF50; font-size: 12px;")

        # 激光雷达状态
        self._lidar_label = QLabel("雷达: 待机")
        self._lidar_label.setStyleSheet("color: #888888; font-size: 12px;")

        # 轨迹显示状态 (可点击切换)
        self._trajectory_label = QLabel("轨迹: 开启")
        self._trajectory_label.setStyleSheet("color: #CE93D8; font-size: 12px;")
        self._trajectory_label.setCursor(Qt.PointingHandCursor)
        self._trajectory_label.mousePressEvent = self._on_trajectory_label_clicked

        status_layout.addWidget(self._status_indicator)
        status_layout.addStretch()
        status_layout.addWidget(self._formation_label)
        status_layout.addSpacing(15)
        status_layout.addWidget(self._altitude_label)
        status_layout.addSpacing(15)
        status_layout.addWidget(self._count_label)
        status_layout.addSpacing(15)
        status_layout.addWidget(self._avoidance_label)
        status_layout.addSpacing(15)
        status_layout.addWidget(self._lidar_label)
        status_layout.addSpacing(15)
        status_layout.addWidget(self._trajectory_label)

        layout.addWidget(status_bar)

        # 3D 视图容器
        self._view_container = QWidget()
        self._view_layout = QVBoxLayout(self._view_container)
        self._view_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view_container, 1)  # 占据全部空间

    def _setup_3d_scene(self):
        """设置 3D 场景"""
        # 创建 OpenGL 视图 - 调整相机距离适应 32x32 场景
        self._gl_widget = gl.GLViewWidget()
        self._gl_widget.setCameraPosition(distance=35, elevation=40, azimuth=45)
        self._gl_widget.setBackgroundColor(pg.mkColor(30, 30, 40))

        # 添加地面网格 - 32x32 米场景
        grid = gl.GLGridItem()
        grid.setSize(32, 32, 1)
        grid.setSpacing(2, 2, 1)  # 2米间隔
        grid.setColor((100, 100, 100, 80))
        self._gl_widget.addItem(grid)

        # 添加坐标轴
        axis = gl.GLAxisItem()
        axis.setSize(2, 2, 2)
        self._gl_widget.addItem(axis)

        # 添加到布局
        self._view_layout.addWidget(self._gl_widget)

    def _create_boundary_fence(self):
        """创建边界围栏 - 四周 + 顶部天花板"""
        fence_height = self.max_altitude  # 围栏高度 = 最大飞行高度
        post_spacing = 2.0  # 立柱间距
        limit = self.boundary_limit

        # 围栏颜色 (橙黄色警示色)
        fence_color = (1.0, 0.6, 0.1, 0.8)
        post_color = (0.8, 0.5, 0.1, 1.0)
        ceiling_color = (0.8, 0.2, 0.2, 0.4)  # 天花板：半透明红色

        # 创建四面围栏立柱和横杆
        all_post_points = []
        all_rail_points = []

        # 四条边
        edges = [
            ((-limit, -limit), (-limit, limit)),   # 左边 (x = -limit)
            ((limit, -limit), (limit, limit)),     # 右边 (x = +limit)
            ((-limit, -limit), (limit, -limit)),   # 下边 (y = -limit)
            ((-limit, limit), (limit, limit)),     # 上边 (y = +limit)
        ]

        for (x1, y1), (x2, y2) in edges:
            # 计算该边的长度和立柱数量
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            num_posts = int(length / post_spacing) + 1

            for i in range(num_posts):
                t = i / max(1, num_posts - 1)
                px = x1 + t * (x2 - x1)
                py = y1 + t * (y2 - y1)

                # 立柱点 (从地面到围栏顶部)
                for h in np.linspace(0, fence_height, 15):
                    all_post_points.append([px, py, h])

            # 横杆 (顶部、中部、底部)
            for h in [fence_height, fence_height * 0.5, 0.1]:
                all_rail_points.append([x1, y1, h])
                all_rail_points.append([x2, y2, h])

        # 创建立柱散点
        if all_post_points:
            post_scatter = gl.GLScatterPlotItem(
                pos=np.array(all_post_points),
                color=post_color,
                size=4,
                pxMode=True
            )
            self._gl_widget.addItem(post_scatter)

        # 创建横杆线条
        if all_rail_points:
            rail_lines = gl.GLLinePlotItem(
                pos=np.array(all_rail_points),
                color=fence_color,
                width=2,
                antialias=True,
                mode='lines'
            )
            self._gl_widget.addItem(rail_lines)

        # 添加四个角落的高亮标记 (从地面到天花板)
        corner_positions = [
            (-limit, -limit), (-limit, limit),
            (limit, -limit), (limit, limit)
        ]
        corner_points = []
        for cx, cy in corner_positions:
            for h in np.linspace(0, fence_height, 20):
                corner_points.append([cx, cy, h])

        if corner_points:
            corner_scatter = gl.GLScatterPlotItem(
                pos=np.array(corner_points),
                color=(1.0, 0.3, 0.0, 1.0),  # 更亮的橙红色
                size=6,
                pxMode=True
            )
            self._gl_widget.addItem(corner_scatter)

        # 底部边界线 (更明显的地面标记)
        ground_line_points = []
        for (x1, y1), (x2, y2) in edges:
            ground_line_points.append([x1, y1, 0.05])
            ground_line_points.append([x2, y2, 0.05])

        if ground_line_points:
            ground_lines = gl.GLLinePlotItem(
                pos=np.array(ground_line_points),
                color=(1.0, 0.4, 0.0, 0.9),
                width=3,
                antialias=True,
                mode='lines'
            )
            self._gl_widget.addItem(ground_lines)

        # ========== 顶部天花板 (Z轴围栏) ==========
        ceiling_points = []
        ceiling_lines = []
        ceiling_spacing = 4.0  # 天花板网格间距

        # 天花板网格线 (X方向)
        for x in np.arange(-limit, limit + 0.1, ceiling_spacing):
            ceiling_lines.append([x, -limit, fence_height])
            ceiling_lines.append([x, limit, fence_height])

        # 天花板网格线 (Y方向)
        for y in np.arange(-limit, limit + 0.1, ceiling_spacing):
            ceiling_lines.append([-limit, y, fence_height])
            ceiling_lines.append([limit, y, fence_height])

        # 天花板边框 (更明显)
        ceiling_border = [
            [-limit, -limit, fence_height], [limit, -limit, fence_height],
            [limit, -limit, fence_height], [limit, limit, fence_height],
            [limit, limit, fence_height], [-limit, limit, fence_height],
            [-limit, limit, fence_height], [-limit, -limit, fence_height],
        ]

        if ceiling_lines:
            ceiling_grid = gl.GLLinePlotItem(
                pos=np.array(ceiling_lines),
                color=ceiling_color,
                width=1,
                antialias=True,
                mode='lines'
            )
            self._gl_widget.addItem(ceiling_grid)

        if ceiling_border:
            ceiling_border_lines = gl.GLLinePlotItem(
                pos=np.array(ceiling_border),
                color=(1.0, 0.3, 0.3, 0.8),  # 更明显的红色边框
                width=3,
                antialias=True,
                mode='lines'
            )
            self._gl_widget.addItem(ceiling_border_lines)

    def _create_obstacles(self):
        """创建柱形点云障碍物 (带碰撞检测)"""
        # 固定障碍物配置 (建筑物风格) - 分布在 32x32 场景中
        fixed_obstacles = [
            {'center': (-10.0, 10.0), 'width': 1.2, 'depth': 1.2, 'height': 5.0},
            {'center': (11.0, -9.0), 'width': 1.0, 'depth': 1.4, 'height': 4.5},
            {'center': (10.0, 9.0), 'width': 0.9, 'depth': 0.9, 'height': 4.0},
            {'center': (-9.0, -10.0), 'width': 1.1, 'depth': 1.1, 'height': 4.2},
            {'center': (-12.0, 0.0), 'width': 1.0, 'depth': 1.0, 'height': 4.8},
            {'center': (12.0, 0.0), 'width': 0.8, 'depth': 1.2, 'height': 4.3},
        ]

        # 计算固定障碍物的等效圆形半径 (用于碰撞检测)
        fixed_circles = []
        for obs in fixed_obstacles:
            # 使用对角线的一半作为等效半径
            equiv_radius = np.sqrt(obs['width']**2 + obs['depth']**2) / 2
            fixed_circles.append({
                'center': obs['center'],
                'radius': equiv_radius
            })

        # 生成随机障碍物配置 (避开固定障碍物)
        # 32x32 场景，扩大安全区和障碍物间距
        obstacle_configs = ObstacleGenerator.generate_random_obstacles(
            map_size=(32.0, 32.0),       # 扩大地图: 16x16 -> 32x32
            num_obstacles=15,            # 适当增加障碍物
            min_radius=0.2,
            max_radius=0.5,
            min_height=2.0,
            max_height=4.5,
            safe_zone_radius=6.0,        # 扩大安全区: 4.0 -> 6.0
            min_spacing=1.2,             # 增大间距
            seed=123
        )

        # 过滤掉与固定障碍物重叠的随机障碍物
        filtered_configs = []
        for config in obstacle_configs:
            cx, cy = config['center']
            cr = config['radius']
            overlap = False
            for fc in fixed_circles:
                fx, fy = fc['center']
                fr = fc['radius']
                dist = np.sqrt((cx - fx)**2 + (cy - fy)**2)
                if dist < cr + fr + 0.3:  # 0.3m 间距
                    overlap = True
                    break
            if not overlap:
                filtered_configs.append(config)

        # 为每个随机障碍物生成点云
        for config in filtered_configs:
            points = ObstacleGenerator.generate_cylinder_points(
                center=config['center'],
                radius=config['radius'],
                height=config['height'],
                resolution=0.08
            )

            if len(points) == 0:
                continue

            # 根据高度设置颜色渐变 (底部深色，顶部浅色)
            colors = np.zeros((len(points), 4))
            for i, pt in enumerate(points):
                height_ratio = pt[2] / config['height']
                # 棕色到橙色渐变
                colors[i] = [
                    0.5 + 0.3 * height_ratio,  # R
                    0.25 + 0.2 * height_ratio,  # G
                    0.1,                         # B
                    0.9                          # A
                ]

            scatter = gl.GLScatterPlotItem(
                pos=points,
                color=colors,
                size=3,
                pxMode=True
            )
            self._gl_widget.addItem(scatter)
            self.obstacles.append(scatter)
            # 保存配置用于碰撞检测
            self.obstacle_configs.append(config)

        # 生成固定障碍物点云
        for config in fixed_obstacles:
            points = ObstacleGenerator.generate_box_points(
                center=config['center'],
                width=config['width'],
                depth=config['depth'],
                height=config['height'],
                resolution=0.06
            )

            if len(points) == 0:
                continue

            # 灰色建筑物颜色
            colors = np.zeros((len(points), 4))
            for i, pt in enumerate(points):
                height_ratio = pt[2] / config['height']
                colors[i] = [
                    0.4 + 0.15 * height_ratio,
                    0.4 + 0.15 * height_ratio,
                    0.45 + 0.15 * height_ratio,
                    0.9
                ]

            scatter = gl.GLScatterPlotItem(
                pos=points,
                color=colors,
                size=3,
                pxMode=True
            )
            self._gl_widget.addItem(scatter)
            self.obstacles.append(scatter)
            # 保存固定障碍物配置 (使用等效圆形)
            equiv_radius = np.sqrt(config['width']**2 + config['depth']**2) / 2
            self.obstacle_configs.append({
                'center': config['center'],
                'radius': equiv_radius,
                'height': config['height'],
                'type': 'box'
            })

    def _create_dynamic_obstacles(self):
        """创建动态障碍物"""
        # 定义动态障碍物路径 - 在外围活动，不干扰中心编队区 (适应 32x32 场景)
        # 动态障碍物路径 - 避开所有静态障碍物
        # 静态障碍物位置: (-10,10), (11,-9), (10,9), (-9,-10), (-12,0), (12,0)
        # 随机障碍物在 6m 安全区外分布
        dynamic_configs = [
            {
                # 红色: 在左侧安全走廊移动 (避开 -12,0 和 -10,10)
                'waypoints': [(-6.0, -5.0), (-6.0, 5.0), (-4.0, 5.0), (-4.0, -5.0)],
                'radius': 0.3,
                'height': 3.0,
                'speed': 1.8,
                'color': (0.9, 0.15, 0.15, 0.9)  # 红色
            },
            {
                # 绿色: 在右侧安全走廊移动 (避开 12,0 和 10,9)
                'waypoints': [(6.0, -5.0), (6.0, 5.0), (4.0, 5.0), (4.0, -5.0)],
                'radius': 0.35,
                'height': 3.5,
                'speed': 1.5,
                'color': (0.15, 0.9, 0.15, 0.9)  # 绿色
            },
            {
                # 蓝色: 在前方水平移动 (安全区边缘)
                'waypoints': [(-5.0, 7.0), (5.0, 7.0), (5.0, 5.0), (-5.0, 5.0)],
                'radius': 0.28,
                'height': 2.8,
                'speed': 2.0,
                'color': (0.15, 0.15, 0.9, 0.9)  # 蓝色
            },
        ]

        for config in dynamic_configs:
            obs = DynamicObstacle(
                waypoints=config['waypoints'],
                radius=config['radius'],
                height=config['height'],
                speed=config['speed'],
                color=config['color']
            )
            obs.create_visual(self._gl_widget)
            self.dynamic_obstacles.append(obs)

    def _create_drones(self):
        """创建无人机模型"""
        for i in range(self.drone_count):
            # 禁用激光雷达可视化，保持场景简洁
            drone = DroneModel(i, self._gl_widget, enable_lidar=False)
            self.drones.append(drone)

    def _init_ground_positions(self):
        """初始化地面位置 - 随机分布在安全区内"""
        import random
        positions = []
        for i in range(self.drone_count):
            # 在中心安全区内随机分布 (半径 3m 内)
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(0.5, 2.5)
            x = self.center[0] + radius * np.cos(angle)
            y = self.center[1] + radius * np.sin(angle)
            positions.append((x, y, 0.0))

        for drone, pos in zip(self.drones, positions):
            drone.set_position(pos)
            drone.set_target(pos)
            # 保存地面位置，用于垂直起降
            drone._ground_position = np.array(pos)

    def _update_formation_lines(self):
        """更新编队连线 - 只在编队状态下显示"""
        if not self.is_flying or not self.in_formation or len(self.drones) < 2:
            if self.formation_lines:
                self._gl_widget.removeItem(self.formation_lines)
                self.formation_lines = None
            return

        # 收集所有位置
        positions = [drone.position for drone in self.drones]
        n = len(positions)

        # 根据编队类型创建连线
        pts = []

        if self.current_formation == FormationType.STAR and n >= 5:
            # 五角星特殊连线
            # 外圈五边形 (0-1-2-3-4-0)
            for i in range(min(5, n)):
                pts.append(positions[i])
                pts.append(positions[(i + 1) % 5])

            # 内圈五边形 (5-6-7-8-9-5)
            if n >= 10:
                for i in range(5, 10):
                    pts.append(positions[i])
                    next_idx = 5 + ((i - 5 + 1) % 5)
                    pts.append(positions[next_idx])

                # 外圈到内圈的连接线（星形辐射）
                for i in range(5):
                    if i < n and (i + 5) < n:
                        pts.append(positions[i])
                        pts.append(positions[i + 5])
        else:
            # 其他编队：普通闭合多边形
            for i in range(n):
                pts.append(positions[i])
                pts.append(positions[(i + 1) % n])

        pts_array = np.array(pts)

        if self.formation_lines:
            self.formation_lines.setData(pos=pts_array)
        else:
            self.formation_lines = gl.GLLinePlotItem(
                pos=pts_array,
                color=(1.0, 0.8, 0.0, 0.6),
                width=2,
                antialias=True,
                mode='lines'
            )
            self._gl_widget.addItem(self.formation_lines)

    def _update_animation(self):
        """动画更新 - 使用 L1 动力学模型"""
        current_time = time.time()
        if self._last_animation_time is None:
            dt = 0.033  # 首次调用使用默认值
        else:
            dt = current_time - self._last_animation_time
            # 限制 dt 范围，防止极端情况
            dt = max(0.01, min(dt, 0.1))  # 10-100ms
        self._last_animation_time = current_time

        # 更新动态障碍物位置
        dynamic_configs = []
        for obs in self.dynamic_obstacles:
            obs.update(dt)
            obs.update_visual()
            dynamic_configs.append(obs.get_config())

        # 更新每架无人机
        for drone in self.drones:
            if self.is_flying:
                # 使用动力学模型
                if drone.use_dynamics and drone.dynamics:
                    # 计算避障加速度
                    if self.avoidance_enabled:
                        # 获取避障速度方向，转换为加速度
                        avoidance_vel = drone.compute_avoidance_velocity(
                            self.obstacle_configs, dynamic_configs
                        )
                        # 将避障速度转换为期望加速度 (P控制)
                        current_vel = drone.dynamics.velocity
                        external_accel = 2.0 * (avoidance_vel - current_vel)
                    else:
                        external_accel = None

                    # 动力学积分
                    drone.step_dynamics(dt, external_accel)

                    # 同步速度到旧属性 (兼容)
                    drone._smoothed_velocity = drone.dynamics.velocity.copy()
                else:
                    # 旧方法 (无动力学模型)
                    to_target = drone.target_position - drone.position
                    dist_to_target = np.linalg.norm(to_target)

                    if dist_to_target < 0.08:
                        drone.position = drone.target_position.copy()
                        drone._update_transform()
                        continue

                    if self.avoidance_enabled:
                        velocity = drone.compute_avoidance_velocity(
                            self.obstacle_configs, dynamic_configs
                        )
                    else:
                        if dist_to_target > 0.1:
                            velocity = to_target / dist_to_target * 0.5
                        else:
                            velocity = np.zeros(3)

                    if not hasattr(drone, '_smoothed_velocity'):
                        drone._smoothed_velocity = velocity.copy()
                    else:
                        alpha = 0.3
                        drone._smoothed_velocity = alpha * velocity + (1 - alpha) * drone._smoothed_velocity

                    move = drone._smoothed_velocity * dt * 3
                    drone.position = drone.position + move
                    drone._update_transform()

                # 边界约束：限制无人机在围栏内 (X/Y/Z)
                drone.position[0] = np.clip(drone.position[0], -self.boundary_limit, self.boundary_limit)
                drone.position[1] = np.clip(drone.position[1], -self.boundary_limit, self.boundary_limit)
                drone.position[2] = np.clip(drone.position[2], 0.0, self.max_altitude)

                # 同步边界约束到动力学模型
                if drone.dynamics:
                    drone.dynamics.position = drone.position.copy()

                drone._update_transform()

                # 更新轨迹
                if self.trajectory_enabled:
                    drone.update_trajectory()
            else:
                # 地面状态 - 普通动画
                drone.update_animation(0.5)
                # 降落过程中也要更新轨迹 (z > 0.05 时继续记录)
                if self.trajectory_enabled and drone.position[2] > 0.05:
                    drone.update_trajectory()

        self._update_formation_lines()

    def _update_status_display(self):
        """更新状态显示"""
        # 状态指示
        if self.is_flying:
            if self.in_formation:
                self._status_indicator.setText("● 编队飞行")
                self._status_indicator.setStyleSheet("color: #2196F3; font-size: 12px;")
            else:
                self._status_indicator.setText("● 悬停中")
                self._status_indicator.setStyleSheet("color: #4CAF50; font-size: 12px;")
        else:
            self._status_indicator.setText("● 地面待命")
            self._status_indicator.setStyleSheet("color: #888888; font-size: 12px;")

        # 编队类型
        formation_names = {
            FormationType.TRIANGLE: "三角形",
            FormationType.SQUARE: "正方形",
            FormationType.CIRCLE: "圆形",
            FormationType.STAR: "五角星",
            FormationType.LINE: "直线",
        }
        if self.in_formation:
            self._formation_label.setText(f"编队: {formation_names.get(self.current_formation, '未知')}")
            self._formation_label.setStyleSheet("color: #4FC3F7; font-size: 12px;")
        else:
            self._formation_label.setText("编队: 未编队")
            self._formation_label.setStyleSheet("color: #888888; font-size: 12px;")

        # 高度
        self._altitude_label.setText(f"高度: {self.current_altitude:.1f}m")

        # 数量
        self._count_label.setText(f"无人机: {self.drone_count}架")

    def _generate_formation_positions(self, formation: FormationType, altitude: float) -> List[Tuple]:
        """生成编队位置"""
        generators = {
            FormationType.TRIANGLE: lambda: FormationGenerator.triangle(
                self.center, self.formation_radius, altitude, self.drone_count
            ),
            FormationType.SQUARE: lambda: FormationGenerator.square(
                self.center, self.formation_radius * 1.5, altitude, self.drone_count
            ),
            FormationType.CIRCLE: lambda: FormationGenerator.circle(
                self.center, self.formation_radius, altitude, self.drone_count
            ),
            FormationType.STAR: lambda: FormationGenerator.star(
                self.center, self.formation_radius, altitude, self.drone_count
            ),
            FormationType.LINE: lambda: FormationGenerator.line(
                self.center, 0.8, altitude, self.drone_count
            ),
        }
        return generators.get(formation, generators[FormationType.CIRCLE])()

    # ========== 公开接口 ==========

    @Slot(str)
    def execute_command(self, command: str):
        """
        执行控制指令

        Args:
            command: 指令字符串 (SwarmCommand.value)
        """
        print(f"[3D视图] execute_command: {command}, is_flying={self.is_flying}")
        if command == SwarmCommand.TAKEOFF.value or command == "takeoff":
            self._execute_takeoff()
            print(f"[3D视图] 起飞后 is_flying={self.is_flying}")
        elif command == SwarmCommand.LAND.value or command == "land":
            self._execute_land()
        elif command == SwarmCommand.HOVER.value or command == "hover":
            pass  # 悬停不需要特殊处理
        elif command == SwarmCommand.FORMATION.value or command == "formation":
            # 编队飞行 - 使用当前编队类型进入编队
            self._execute_formation()
        elif command == SwarmCommand.ALTITUDE_UP.value or command == "altitude_up":
            self._execute_altitude_change(0.5)
        elif command == SwarmCommand.ALTITUDE_DOWN.value or command == "altitude_down":
            self._execute_altitude_change(-0.5)
        elif command == SwarmCommand.EMERGENCY_STOP.value or command == "emergency":
            self._execute_emergency()
        elif command == SwarmCommand.MOVE_FORWARD.value or command == "move_forward":
            self._execute_move_forward(5.0)  # 向前飞行5米

        self._update_status_display()
        self.command_executed.emit(command)

    @Slot(str, int)
    def change_formation(self, formation: str, drone_count: int = None):
        """
        切换编队 - 只有在飞行状态下才会进入编队

        Args:
            formation: 编队类型字符串
            drone_count: 无人机数量 (可选)
        """
        print(f"[编队] change_formation 调用: formation={formation}, is_flying={self.is_flying}")
        # 必须在飞行状态下才能编队
        if not self.is_flying:
            print(f"[编队] 忽略：无人机未起飞")
            return

        # 解析编队类型
        formation_map = {
            "triangle": FormationType.TRIANGLE,
            "square": FormationType.SQUARE,
            "circle": FormationType.CIRCLE,
            "star": FormationType.STAR,
            "line": FormationType.LINE,
        }
        new_formation = formation_map.get(formation.lower(), FormationType.CIRCLE)

        if drone_count and drone_count != self.drone_count:
            self._update_drone_count(drone_count)

        self.current_formation = new_formation
        self.in_formation = True  # 标记已进入编队
        positions = self._generate_formation_positions(new_formation, self.current_altitude)
        print(f"[编队] 生成 {len(positions)} 个位置，编队类型={new_formation.value}")

        for i, (drone, pos) in enumerate(zip(self.drones, positions)):
            drone.set_target(pos)
            print(f"[编队] 无人机 {i} 目标位置: {pos}")

        self._update_status_display()
        print(f"[编队] 编队变换完成")

    def _execute_formation(self):
        """执行编队 - 使用当前编队类型进入编队"""
        if not self.is_flying:
            return  # 必须在飞行状态下才能编队

        self.in_formation = True
        positions = self._generate_formation_positions(self.current_formation, self.current_altitude)

        for drone, pos in zip(self.drones, positions):
            drone.set_target(pos)

    def _execute_move_forward(self, distance: float = 5.0):
        """
        执行编队向前飞行

        Args:
            distance: 向前飞行的距离（米）
        """
        if not self.is_flying or not self.in_formation:
            return  # 必须在编队飞行状态下才能移动

        # 更新编队中心位置（Y轴正方向为前方）
        new_y = self.center[1] + distance

        # 边界约束：确保编队中心不会超出安全范围（留出编队半径的余量）
        max_center = self.boundary_limit - self.formation_radius - 1.0
        new_y = np.clip(new_y, -max_center, max_center)

        new_center = (self.center[0], new_y)
        self.center = new_center

        # 生成新的编队位置
        positions = self._generate_formation_positions(self.current_formation, self.current_altitude)

        # 设置每架无人机的新目标位置（同时约束目标在边界内）
        for drone, pos in zip(self.drones, positions):
            clamped_pos = (
                np.clip(pos[0], -self.boundary_limit, self.boundary_limit),
                np.clip(pos[1], -self.boundary_limit, self.boundary_limit),
                pos[2]
            )
            drone.set_target(clamped_pos)

    def _execute_takeoff(self):
        """执行起飞 - 垂直上升到指定高度，保持 x,y 位置"""
        if self.is_flying:
            return

        self.is_flying = True
        self.in_formation = False  # 起飞后还未进入编队
        self.current_altitude = self.base_altitude

        # 垂直上升：保持当前 x,y，只改变 z
        for drone in self.drones:
            current_pos = drone.position.copy()
            target = (current_pos[0], current_pos[1], self.current_altitude)
            drone.set_target(target)
            # 保存起飞位置
            if not hasattr(drone, '_ground_position'):
                drone._ground_position = current_pos.copy()
                drone._ground_position[2] = 0.0

    def _execute_land(self):
        """执行降落 - 垂直下降到地面"""
        if not self.is_flying:
            return

        self.is_flying = False
        self.in_formation = False
        self.current_altitude = 0.0

        # 垂直下降：保持当前 x,y，z 降为 0
        for drone in self.drones:
            current_pos = drone.position.copy()
            target = (current_pos[0], current_pos[1], 0.0)
            drone.set_target(target)

    def _execute_altitude_change(self, delta: float):
        """执行高度变化 - 保持 x,y 位置"""
        if not self.is_flying:
            return

        # 限制高度在 0.5 ~ max_altitude 之间
        self.current_altitude = np.clip(self.current_altitude + delta, 0.5, self.max_altitude)

        # 如果已编队，按编队位置调整高度；否则保持当前 x,y
        if self.in_formation:
            positions = self._generate_formation_positions(self.current_formation, self.current_altitude)
            for drone, pos in zip(self.drones, positions):
                drone.set_target(pos)
        else:
            for drone in self.drones:
                current_pos = drone.position.copy()
                target = (current_pos[0], current_pos[1], self.current_altitude)
                drone.set_target(target)

    def _execute_emergency(self):
        """紧急停止"""
        self.is_flying = False
        self.in_formation = False
        self.current_altitude = 0.0
        self._init_ground_positions()
        # 清除所有轨迹
        self.clear_all_trajectories()

    def _update_drone_count(self, count: int):
        """更新无人机数量"""
        # 移除多余的无人机
        while len(self.drones) > count:
            drone = self.drones.pop()
            drone.remove()

        # 添加新的无人机
        while len(self.drones) < count:
            drone = DroneModel(len(self.drones), self._gl_widget, enable_lidar=False)
            self.drones.append(drone)

        self.drone_count = count

    def set_drone_count(self, count: int):
        """设置无人机数量 - 保持现有无人机位置不变"""
        old_count = self.drone_count

        if count == old_count:
            return

        if count > old_count:
            # 增加无人机: 现有的保持原位，新增的放到随机位置
            import random
            while len(self.drones) < count:
                new_id = len(self.drones)
                drone = DroneModel(new_id, self._gl_widget, enable_lidar=False)

                if self.is_flying:
                    # 飞行中: 新无人机加入当前编队
                    altitude = self.current_altitude
                    positions = self._generate_formation_positions(self.current_formation, altitude)
                    if new_id < len(positions):
                        pos = positions[new_id]
                    else:
                        # 超出编队位置，放到中心附近
                        pos = (self.center[0], self.center[1], altitude)
                    drone.set_position(pos)
                    drone.set_target(pos)
                else:
                    # 地面待命: 新无人机放到随机位置
                    angle = random.uniform(0, 2 * np.pi)
                    radius = random.uniform(0.5, 2.5)
                    x = self.center[0] + radius * np.cos(angle)
                    y = self.center[1] + radius * np.sin(angle)
                    pos = (x, y, 0.0)
                    drone.set_position(pos)
                    drone.set_target(pos)
                    drone._ground_position = pos

                self.drones.append(drone)

            self.drone_count = count

        else:
            # 减少无人机: 只移除多余的，现有的保持原位
            while len(self.drones) > count:
                drone = self.drones.pop()
                drone.remove()
            self.drone_count = count

        self._update_status_display()

    # ========== 轨迹控制 ==========

    def _on_trajectory_label_clicked(self, event):
        """轨迹标签点击事件 - 切换轨迹显示"""
        self.toggle_trajectory()

    def toggle_trajectory(self):
        """切换轨迹显示开关"""
        self.trajectory_enabled = not self.trajectory_enabled

        # 更新所有无人机的轨迹显示状态
        for drone in self.drones:
            drone.set_trajectory_enabled(self.trajectory_enabled)

        # 更新状态标签
        if self.trajectory_enabled:
            self._trajectory_label.setText("轨迹: 开启")
            self._trajectory_label.setStyleSheet("color: #CE93D8; font-size: 12px;")
        else:
            self._trajectory_label.setText("轨迹: 关闭")
            self._trajectory_label.setStyleSheet("color: #888888; font-size: 12px;")

    def clear_all_trajectories(self):
        """清除所有无人机的轨迹"""
        for drone in self.drones:
            drone.clear_trajectory()

    def set_trajectory_enabled(self, enabled: bool):
        """设置轨迹显示状态"""
        self.trajectory_enabled = enabled
        for drone in self.drones:
            drone.set_trajectory_enabled(enabled)

        if enabled:
            self._trajectory_label.setText("轨迹: 开启")
            self._trajectory_label.setStyleSheet("color: #CE93D8; font-size: 12px;")
        else:
            self._trajectory_label.setText("轨迹: 关闭")
            self._trajectory_label.setStyleSheet("color: #888888; font-size: 12px;")

    def get_drone_status_list(self) -> list:
        """
        获取所有无人机的状态信息（含模拟数据）

        Returns:
            状态列表 [{'id', 'position', 'status', 'color', 'battery', 'signal', 'speed'}, ...]
        """
        import random

        status_list = []
        for i, drone in enumerate(self.drones):
            pos = drone.position
            status = "飞行中" if self.is_flying else "待机"
            color = DroneModel.COLORS[i % len(DroneModel.COLORS)]

            # 计算速度（基于平滑速度）
            speed = 0.0
            if hasattr(drone, '_smoothed_velocity'):
                speed = np.linalg.norm(drone._smoothed_velocity)

            # 模拟电量（飞行时缓慢消耗）
            if not hasattr(drone, '_sim_battery'):
                drone._sim_battery = 95 + random.randint(0, 5)
            if self.is_flying and random.random() < 0.02:  # 2%概率消耗
                drone._sim_battery = max(10, drone._sim_battery - 1)

            # 模拟信号强度（随距离变化）
            dist_from_center = np.linalg.norm(pos[:2])
            signal = max(30, min(100, 100 - int(dist_from_center * 5) + random.randint(-5, 5)))

            status_list.append({
                'id': i,
                'position': (pos[0], pos[1], pos[2]),
                'status': status,
                'color': color,
                'battery': drone._sim_battery,
                'signal': signal,
                'speed': speed
            })
        return status_list
