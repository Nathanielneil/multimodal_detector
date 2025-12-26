# -*- coding: utf-8 -*-
"""
ROS Bridge - PySide6 与 ROS 的桥接模块

使用 rospy 实现与 ROS 的通信
"""

import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, List
from datetime import datetime

# ROS imports (延迟导入，避免非ROS环境报错)
_ros_available = False
try:
    import rospy
    from std_msgs.msg import String, Header
    from geometry_msgs.msg import Point, Pose, PoseArray, Twist
    from visualization_msgs.msg import Marker, MarkerArray
    _ros_available = True
except ImportError:
    pass


class FormationType(Enum):
    """编队类型 - 对应手绘形状"""
    TRIANGLE = "triangle"      # 三角形编队
    SQUARE = "square"          # 正方形编队
    CIRCLE = "circle"          # 圆形编队
    STAR = "star"              # 五角星编队
    LINE = "line"              # 直线编队
    CUSTOM = "custom"          # 自定义编队


class SwarmCommand(Enum):
    """集群指令 - 对应手势识别"""
    TAKEOFF = "takeoff"            # 集群起飞 (张开手掌)
    LAND = "land"                  # 集群降落 (握拳)
    HOVER = "hover"                # 集群悬停 (食指指向)
    FORMATION = "formation"        # 编队飞行 (ILY手势)
    CONFIRM = "confirm"            # 指令确定 (OK手势)
    ALTITUDE_UP = "altitude_up"    # 高度上升 (竖起大拇指)
    ALTITUDE_DOWN = "altitude_down"  # 高度下降 (向下大拇指)
    EMERGENCY_STOP = "emergency"   # 紧急停止
    MOVE_FORWARD = "move_forward"  # 向前飞行 (V形手势)


@dataclass
class DroneState:
    """单个无人机状态"""
    drone_id: int
    position: tuple = (0.0, 0.0, 0.0)  # x, y, z
    orientation: tuple = (0.0, 0.0, 0.0, 1.0)  # quaternion
    velocity: tuple = (0.0, 0.0, 0.0)
    battery: float = 100.0
    status: str = "idle"  # idle, flying, hovering, landing


@dataclass
class SwarmState:
    """集群状态"""
    drones: List[DroneState] = field(default_factory=list)
    formation: FormationType = FormationType.TRIANGLE
    command_executing: Optional[SwarmCommand] = None
    timestamp: datetime = field(default_factory=datetime.now)


class ROSBridge:
    """
    ROS Bridge 主类

    负责:
    - 初始化 ROS 节点
    - 发布集群控制指令
    - 订阅集群状态
    - 发布可视化 Marker
    """

    def __init__(self, node_name: str = "multimodal_detector"):
        """
        初始化 ROS Bridge

        Args:
            node_name: ROS 节点名称
        """
        self._node_name = node_name
        self._initialized = False
        self._publishers = {}
        self._subscribers = {}
        self._swarm_state = SwarmState()
        self._state_callbacks = []
        self._ros_thread = None

        # Topic 名称
        self.TOPIC_COMMAND = "/swarm/command"
        self.TOPIC_FORMATION = "/swarm/formation"
        self.TOPIC_STATUS = "/swarm/status"
        self.TOPIC_MARKER = "/swarm/visualization"

    @property
    def is_available(self) -> bool:
        """检查 ROS 是否可用"""
        return _ros_available

    @property
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized

    def initialize(self) -> bool:
        """
        初始化 ROS 节点和话题

        Returns:
            bool: 是否初始化成功
        """
        if not _ros_available:
            print("[ROSBridge] ROS not available. Install rospy to enable ROS features.")
            return False

        if self._initialized:
            return True

        try:
            # 初始化 ROS 节点 (anonymous=True 允许多实例)
            rospy.init_node(self._node_name, anonymous=True, disable_signals=True)

            # 创建发布者
            self._publishers["command"] = rospy.Publisher(
                self.TOPIC_COMMAND, String, queue_size=10
            )
            self._publishers["formation"] = rospy.Publisher(
                self.TOPIC_FORMATION, String, queue_size=10
            )
            self._publishers["marker"] = rospy.Publisher(
                self.TOPIC_MARKER, MarkerArray, queue_size=10
            )

            # 创建订阅者
            self._subscribers["status"] = rospy.Subscriber(
                self.TOPIC_STATUS, String, self._status_callback
            )

            self._initialized = True
            print(f"[ROSBridge] ROS node '{self._node_name}' initialized successfully")
            return True

        except Exception as e:
            print(f"[ROSBridge] Failed to initialize ROS: {e}")
            return False

    def shutdown(self):
        """关闭 ROS 连接"""
        if self._initialized:
            try:
                rospy.signal_shutdown("Application closing")
            except:
                pass
            self._initialized = False

    def publish_command(self, command: SwarmCommand, params: dict = None):
        """
        发布集群控制指令

        Args:
            command: 控制指令
            params: 附加参数
        """
        if not self._initialized:
            print(f"[ROSBridge] Not initialized, command '{command.value}' not sent")
            return

        import json
        msg_data = {
            "command": command.value,
            "timestamp": datetime.now().isoformat(),
            "params": params or {}
        }

        msg = String()
        msg.data = json.dumps(msg_data)
        self._publishers["command"].publish(msg)
        print(f"[ROSBridge] Published command: {command.value}")

    def publish_formation(self, formation: FormationType, drone_count: int = 6):
        """
        发布编队指令

        Args:
            formation: 编队类型
            drone_count: 无人机数量
        """
        if not self._initialized:
            print(f"[ROSBridge] Not initialized, formation '{formation.value}' not sent")
            return

        import json
        msg_data = {
            "formation": formation.value,
            "drone_count": drone_count,
            "timestamp": datetime.now().isoformat()
        }

        msg = String()
        msg.data = json.dumps(msg_data)
        self._publishers["formation"].publish(msg)
        print(f"[ROSBridge] Published formation: {formation.value}")

    def publish_drone_markers(self, positions: List[tuple], colors: List[tuple] = None):
        """
        发布无人机可视化 Marker 到 RViz

        Args:
            positions: 无人机位置列表 [(x, y, z), ...]
            colors: 颜色列表 [(r, g, b, a), ...], 默认蓝色
        """
        if not self._initialized:
            return

        marker_array = MarkerArray()

        for i, pos in enumerate(positions):
            # 无人机主体 Marker (球体)
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "drones"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = pos[2]
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.15

            if colors and i < len(colors):
                marker.color.r = colors[i][0]
                marker.color.g = colors[i][1]
                marker.color.b = colors[i][2]
                marker.color.a = colors[i][3] if len(colors[i]) > 3 else 1.0
            else:
                marker.color.r = 0.2
                marker.color.g = 0.6
                marker.color.b = 1.0
                marker.color.a = 1.0

            marker_array.markers.append(marker)

            # 无人机编号文字
            text_marker = Marker()
            text_marker.header.frame_id = "world"
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "drone_labels"
            text_marker.id = i + 100
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = pos[0]
            text_marker.pose.position.y = pos[1]
            text_marker.pose.position.z = pos[2] + 0.4

            text_marker.text = f"UAV{i+1}"
            text_marker.scale.z = 0.2
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0

            marker_array.markers.append(text_marker)

        self._publishers["marker"].publish(marker_array)

    def publish_formation_lines(self, positions: List[tuple]):
        """
        发布编队连线 Marker

        Args:
            positions: 无人机位置列表
        """
        if not self._initialized or len(positions) < 2:
            return

        marker_array = MarkerArray()

        # 编队连线
        line_marker = Marker()
        line_marker.header.frame_id = "world"
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = "formation_lines"
        line_marker.id = 0
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD

        line_marker.scale.x = 0.02  # 线宽
        line_marker.color.r = 1.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0
        line_marker.color.a = 0.6

        # 连接相邻无人机
        for i in range(len(positions)):
            j = (i + 1) % len(positions)

            p1 = Point()
            p1.x, p1.y, p1.z = positions[i]

            p2 = Point()
            p2.x, p2.y, p2.z = positions[j]

            line_marker.points.append(p1)
            line_marker.points.append(p2)

        marker_array.markers.append(line_marker)
        self._publishers["marker"].publish(marker_array)

    def _status_callback(self, msg):
        """状态回调函数"""
        import json
        try:
            data = json.loads(msg.data)
            # 更新内部状态
            # 触发回调
            for callback in self._state_callbacks:
                callback(data)
        except Exception as e:
            print(f"[ROSBridge] Status parse error: {e}")

    def register_status_callback(self, callback: Callable):
        """注册状态更新回调"""
        self._state_callbacks.append(callback)

    def get_swarm_state(self) -> SwarmState:
        """获取当前集群状态"""
        return self._swarm_state


# 单例模式
_bridge_instance: Optional[ROSBridge] = None

def get_ros_bridge() -> ROSBridge:
    """获取 ROSBridge 单例"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = ROSBridge()
    return _bridge_instance
