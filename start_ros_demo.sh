#!/bin/bash
# ========================================
# 无人机集群多模态控制系统启动脚本
# ========================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  无人机集群多模态控制系统${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATKIN_WS="${SCRIPT_DIR}/catkin_ws"

# 检查 ROS 环境
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${YELLOW}[INFO]${NC} 正在设置 ROS 环境..."
    source /opt/ros/noetic/setup.bash
fi

# 检查 catkin 工作空间
if [ -f "${CATKIN_WS}/devel/setup.bash" ]; then
    echo -e "${YELLOW}[INFO]${NC} 正在加载 catkin 工作空间..."
    source "${CATKIN_WS}/devel/setup.bash"
else
    echo -e "${RED}[ERROR]${NC} catkin 工作空间未构建。请先运行:"
    echo "  cd ${CATKIN_WS} && catkin_make"
    exit 1
fi

# 启动模式选择
echo ""
echo -e "${YELLOW}请选择启动模式:${NC}"
echo "  1. 仅启动 RViz 可视化 (ROS 节点)"
echo "  2. 仅启动 PySide6 多模态检测器"
echo "  3. 完整启动 (ROS + PySide6)"
echo ""
read -p "请输入选项 [1-3]: " choice

case $choice in
    1)
        echo -e "${GREEN}[启动]${NC} ROS 可视化节点..."
        roslaunch swarm_visualizer swarm_visualizer.launch
        ;;
    2)
        echo -e "${GREEN}[启动]${NC} PySide6 多模态检测器..."
        cd "${SCRIPT_DIR}"
        python3 main.py
        ;;
    3)
        echo -e "${GREEN}[启动]${NC} 完整系统..."
        echo -e "${YELLOW}[INFO]${NC} 在新终端启动 ROS 可视化..."

        # 在新终端启动 ROS
        gnome-terminal --title="ROS Visualizer" -- bash -c "
            source /opt/ros/noetic/setup.bash
            source ${CATKIN_WS}/devel/setup.bash
            echo '正在启动 ROS 可视化节点...'
            roslaunch swarm_visualizer swarm_visualizer.launch
        " 2>/dev/null || xterm -title "ROS Visualizer" -e "
            source /opt/ros/noetic/setup.bash
            source ${CATKIN_WS}/devel/setup.bash
            roslaunch swarm_visualizer swarm_visualizer.launch
        " &

        # 等待 ROS 启动
        sleep 3

        echo -e "${GREEN}[启动]${NC} PySide6 多模态检测器..."
        cd "${SCRIPT_DIR}"
        python3 main.py
        ;;
    *)
        echo -e "${RED}[ERROR]${NC} 无效选项"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}[完成]${NC} 系统已退出"
