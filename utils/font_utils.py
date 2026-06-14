"""
跨平台中文字体查找工具
"""

import sys
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)

# 按平台优先级排列的候选字体路径
_FONT_PATHS = {
    "darwin": [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode MS.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ],
    "linux": [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ],
    "win32": [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ],
}


def find_cjk_font(size: int = 24):
    """
    按平台查找可用的中文字体，返回 PIL ImageFont 对象。
    找不到合适字体时退回到 PIL 默认字体。
    """
    from PIL import ImageFont

    platform = sys.platform
    # darwin → darwin, linux* → linux, win32 → win32
    key = "darwin" if platform == "darwin" else ("win32" if platform == "win32" else "linux")
    candidates = _FONT_PATHS.get(key, []) + _FONT_PATHS.get("linux", [])

    for path in candidates:
        try:
            font = ImageFont.truetype(path, size)
            logger.debug(f"已加载字体: {path}")
            return font
        except OSError:
            continue

    logger.warning("未找到中文字体，使用 PIL 默认字体")
    return ImageFont.load_default()


def find_qt_font_family() -> str:
    """
    返回当前平台上可用的中文 Qt 字体名。
    """
    from PySide6.QtGui import QFontDatabase
    candidates = [
        "PingFang SC",       # macOS
        "Heiti SC",          # macOS fallback
        "Microsoft YaHei",   # Windows
        "WenQuanYi Zen Hei", # Linux
        "Noto Sans CJK SC",  # Linux/cross-platform
        "Arial Unicode MS",
    ]
    available = set(QFontDatabase.families())
    for name in candidates:
        if name in available:
            return name
    return "sans-serif"
