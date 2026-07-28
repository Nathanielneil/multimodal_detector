# -*- coding: utf-8 -*-
"""Tests for utils.pcd_loader"""

import struct

import numpy as np
import pytest

from utils.pcd_loader import load_pcd_xyz, downsample_points, estimate_floor_height


def _write_binary_pcd(path, points: np.ndarray):
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {len(points)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\n"
        "DATA binary\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(points.astype(np.float32).tobytes())


def _write_ascii_pcd(path, points: np.ndarray):
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {len(points)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\n"
        "DATA ascii\n"
    )
    with open(path, "w", encoding="ascii") as fh:
        fh.write(header)
        for x, y, z in points:
            fh.write(f"{x} {y} {z}\n")


class TestLoadPcdXyz:
    def test_binary_roundtrip(self, tmp_path):
        points = np.array(
            [[1.0, 2.0, 3.0], [-1.5, 0.5, 4.25], [10.0, -10.0, 0.0]],
            dtype=np.float32,
        )
        path = tmp_path / "cloud.pcd"
        _write_binary_pcd(path, points)

        loaded = load_pcd_xyz(path)
        assert loaded.shape == (3, 3)
        np.testing.assert_allclose(loaded, points, atol=1e-5)

    def test_ascii_roundtrip(self, tmp_path):
        points = np.array(
            [[1.0, 2.0, 3.0], [-1.5, 0.5, 4.25]],
            dtype=np.float32,
        )
        path = tmp_path / "cloud_ascii.pcd"
        _write_ascii_pcd(path, points)

        loaded = load_pcd_xyz(path)
        assert loaded.shape == (2, 3)
        np.testing.assert_allclose(loaded, points, atol=1e-5)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pcd_xyz(tmp_path / "does_not_exist.pcd")

    def test_missing_xyz_field_raises(self, tmp_path):
        path = tmp_path / "no_xyz.pcd"
        header = (
            "# .PCD v0.7\nVERSION 0.7\nFIELDS intensity\nSIZE 4\nTYPE F\n"
            "COUNT 1\nWIDTH 1\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
            "POINTS 1\nDATA binary\n"
        )
        with open(path, "wb") as fh:
            fh.write(header.encode("ascii"))
            fh.write(struct.pack("<f", 1.0))

        with pytest.raises(ValueError):
            load_pcd_xyz(path)

    def test_empty_points(self, tmp_path):
        path = tmp_path / "empty.pcd"
        _write_binary_pcd(path, np.empty((0, 3), dtype=np.float32))
        loaded = load_pcd_xyz(path)
        assert loaded.shape == (0, 3)

    def test_real_project_pcd_if_present(self):
        """若项目根目录存在 global_map_ref_uav1.pcd，验证解析结果与已知范围一致。"""
        from pathlib import Path

        project_root = Path(__file__).parent.parent
        pcd_path = project_root / "global_map_ref_uav1.pcd"
        if not pcd_path.exists():
            pytest.skip("global_map_ref_uav1.pcd 不存在，跳过真实文件校验")

        points = load_pcd_xyz(pcd_path)
        assert points.shape == (789437, 3)
        assert -14.0 < points[:, 0].min() < -13.0
        assert 40.0 < points[:, 0].max() < 41.0
        assert -21.0 < points[:, 1].min() < -20.0
        assert 59.0 < points[:, 1].max() < 60.0


class TestDownsamplePoints:
    def test_no_downsample_when_under_limit(self):
        points = np.random.rand(10, 3).astype(np.float32)
        result = downsample_points(points, max_points=100)
        assert result is points

    def test_downsample_reduces_count(self):
        points = np.random.rand(1000, 3).astype(np.float32)
        result = downsample_points(points, max_points=100, seed=1)
        assert result.shape == (100, 3)

    def test_downsample_deterministic_with_seed(self):
        points = np.random.rand(1000, 3).astype(np.float32)
        result_a = downsample_points(points, max_points=50, seed=7)
        result_b = downsample_points(points, max_points=50, seed=7)
        np.testing.assert_array_equal(result_a, result_b)

    def test_none_max_points_returns_original(self):
        points = np.random.rand(5, 3).astype(np.float32)
        result = downsample_points(points, max_points=None)
        assert result is points


class TestEstimateFloorHeight:
    def test_finds_dense_low_cluster_as_floor(self):
        rng = np.random.default_rng(0)
        # 地面：密集的一层点在 z=0 附近
        floor_pts = np.stack([
            rng.uniform(-10, 10, 5000),
            rng.uniform(-10, 10, 5000),
            rng.normal(0.0, 0.02, 5000),
        ], axis=1)
        # 墙体/天花板：稀疏一些，分布在更高的位置
        wall_pts = np.stack([
            rng.uniform(-10, 10, 500),
            rng.uniform(-10, 10, 500),
            rng.uniform(1.0, 3.0, 500),
        ], axis=1)
        points = np.vstack([floor_pts, wall_pts]).astype(np.float32)

        floor_z = estimate_floor_height(points)
        assert abs(floor_z - 0.0) < 0.2

    def test_empty_points_returns_zero(self):
        points = np.empty((0, 3), dtype=np.float32)
        assert estimate_floor_height(points) == 0.0

    def test_real_project_pcd_floor_height_if_present(self):
        """已知该文件的地面峰值约在 z=-0.15 (最低40%区间内密度最高的直方图桶)。"""
        from pathlib import Path

        project_root = Path(__file__).parent.parent
        pcd_path = project_root / "global_map_ref_uav1.pcd"
        if not pcd_path.exists():
            pytest.skip("global_map_ref_uav1.pcd 不存在，跳过真实文件校验")

        points = load_pcd_xyz(pcd_path)
        floor_z = estimate_floor_height(points)
        assert -0.4 < floor_z < 0.1
