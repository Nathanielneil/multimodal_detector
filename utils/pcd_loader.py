# -*- coding: utf-8 -*-
"""
轻量 PCD (Point Cloud Data, v0.7) 加载器

仅支持提取 x/y/z 三个字段，支持 ASCII 和 binary 两种 DATA 编码。
不引入 open3d/pypcd 等第三方点云库，保持项目依赖最小化。
"""

from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


_NUMPY_TYPE_MAP = {
    ("F", 4): np.float32,
    ("F", 8): np.float64,
    ("U", 1): np.uint8,
    ("U", 2): np.uint16,
    ("U", 4): np.uint32,
    ("U", 8): np.uint64,
    ("I", 1): np.int8,
    ("I", 2): np.int16,
    ("I", 4): np.int32,
    ("I", 8): np.int64,
}


def _parse_header(fh) -> Dict[str, str]:
    """逐行读取 PCD 头部，直到 DATA 行（含）为止，返回字段字典。"""
    header: Dict[str, str] = {}
    while True:
        raw = fh.readline()
        if not raw:
            raise ValueError("PCD 文件在读到 DATA 行之前已结束")
        line = raw.decode("ascii", errors="replace").strip()
        if not line or line.startswith("#"):
            continue
        key, _, rest = line.partition(" ")
        header[key.upper()] = rest.strip()
        if key.upper() == "DATA":
            break
    return header


def load_pcd_xyz(path: Union[str, Path]) -> np.ndarray:
    """
    解析 PCD v0.7 文件，返回形状为 (N, 3) 的 float32 数组 (x, y, z)。

    Args:
        path: PCD 文件路径

    Returns:
        np.ndarray: (N, 3) 点坐标

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 头部缺少必要字段，或 FIELDS 中没有 x/y/z，或 DATA 编码不支持
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PCD 文件不存在: {path}")

    with path.open("rb") as fh:
        header = _parse_header(fh)
        data_start = fh.tell()

        fields = header.get("FIELDS", "").split()
        sizes = [int(v) for v in header.get("SIZE", "").split()]
        types = header.get("TYPE", "").split()
        counts_raw = header.get("COUNT", "")
        counts = [int(v) for v in counts_raw.split()] if counts_raw else [1] * len(fields)
        data_mode = header.get("DATA", "").strip().lower()

        if not fields or not sizes or not types:
            raise ValueError(f"PCD 头部缺少 FIELDS/SIZE/TYPE: {path}")
        if not (len(fields) == len(sizes) == len(types) == len(counts)):
            raise ValueError(f"PCD 头部 FIELDS/SIZE/TYPE/COUNT 长度不一致: {path}")

        try:
            num_points = int(header["POINTS"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"PCD 头部缺少合法的 POINTS 字段: {path}") from exc

        try:
            xyz_indices = [fields.index(axis) for axis in ("x", "y", "z")]
        except ValueError as exc:
            raise ValueError(f"PCD 文件缺少 x/y/z 字段，FIELDS={fields}: {path}") from exc

        if num_points == 0:
            return np.empty((0, 3), dtype=np.float32)

        if data_mode == "ascii":
            fh.seek(data_start)
            points = np.empty((num_points, 3), dtype=np.float32)
            for i in range(num_points):
                raw_line = fh.readline()
                if not raw_line:
                    raise ValueError(f"PCD ASCII 数据行数不足 POINTS={num_points}: {path}")
                tokens = raw_line.decode("ascii", errors="replace").split()
                points[i] = [float(tokens[idx]) for idx in xyz_indices]
            return points

        if data_mode == "binary":
            dtype_fields = []
            for name, size, type_char, count in zip(fields, sizes, types, counts):
                np_type = _NUMPY_TYPE_MAP.get((type_char.upper(), size))
                if np_type is None:
                    raise ValueError(
                        f"不支持的 PCD 字段类型 {name}: TYPE={type_char} SIZE={size}: {path}"
                    )
                if count == 1:
                    dtype_fields.append((name, np_type))
                else:
                    dtype_fields.append((name, np_type, (count,)))

            struct_dtype = np.dtype(dtype_fields)
            raw = fh.read(num_points * struct_dtype.itemsize)
            if len(raw) < num_points * struct_dtype.itemsize:
                raise ValueError(f"PCD binary 数据长度不足 POINTS={num_points}: {path}")

            records = np.frombuffer(raw, dtype=struct_dtype, count=num_points)
            points = np.stack(
                [records["x"].astype(np.float32),
                 records["y"].astype(np.float32),
                 records["z"].astype(np.float32)],
                axis=1,
            )
            return points

        raise ValueError(f"不支持的 PCD DATA 编码 '{data_mode}' (仅支持 ascii/binary): {path}")


def downsample_points(
    points: np.ndarray,
    max_points: Optional[int],
    seed: int = 42,
) -> np.ndarray:
    """
    均匀随机降采样。max_points 为 None 或 <=0，或点数已经不超过上限时原样返回。

    Args:
        points: (N, 3) 点坐标
        max_points: 降采样后的最大点数
        seed: 随机种子，保证同一文件每次加载采样结果一致

    Returns:
        np.ndarray: 降采样后的点坐标
    """
    if max_points is None or max_points <= 0 or len(points) <= max_points:
        return points

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices]


def estimate_floor_height(points: np.ndarray, low_percentile: float = 40.0, bins: int = 60) -> float:
    """
    估算点云的地面高度 (Z 值)。

    室内点云 (如地下车库) 的地面通常是扫描/重建中密度最高、最平坦的一层，
    在 Z 轴直方图上表现为一个明显的峰值。只在最低 low_percentile% 的点里
    找密度峰值，避免天花板/墙体等其他密集结构干扰。

    Args:
        points: (N, 3) 点坐标
        low_percentile: 只在 Z 值最低的这个百分比范围内寻找地面峰值
        bins: 直方图分箱数量

    Returns:
        float: 估算的地面高度。点数不足时返回 Z 最小值。
    """
    if len(points) == 0:
        return 0.0

    z = points[:, 2]
    if len(z) < bins:
        return float(z.min())

    threshold = np.percentile(z, low_percentile)
    low_z = z[z <= threshold]
    if len(low_z) < bins:
        low_z = z

    hist, edges = np.histogram(low_z, bins=bins)
    peak_idx = int(np.argmax(hist))
    return float((edges[peak_idx] + edges[peak_idx + 1]) / 2.0)
