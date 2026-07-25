# -*- coding: utf-8 -*-
"""
Lightweight OBJ asset loading for the embedded PyQtGraph 3D scene.

The runtime format is intentionally small:
    assets/models/<name>/model.json
    assets/models/<name>/*.obj

URDF/MJCF/xacro files are kept as source formats only.  The UI loads
pre-aligned OBJ meshes so the demo does not depend on ROS or model converters.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph.opengl as gl

from utils.logger import get_logger

logger = get_logger(__name__)


_MESH_CACHE: Dict[Path, gl.MeshData] = {}


def _as_color(value: Sequence[float], fallback: Tuple[float, float, float, float]):
    if not value:
        return fallback
    vals = tuple(float(v) for v in value)
    if len(vals) == 3:
        return vals + (1.0,)
    if len(vals) >= 4:
        return vals[:4]
    return fallback


def _parse_obj_face_token(token: str, vertex_count: int) -> int:
    """Return a zero-based vertex index from OBJ face token forms."""
    head = token.split("/")[0]
    if not head:
        raise ValueError(f"Invalid OBJ face token: {token!r}")
    idx = int(head)
    if idx < 0:
        return vertex_count + idx
    return idx - 1


def load_obj_mesh(path: Path) -> gl.MeshData:
    """
    Load a simple triangulated MeshData from OBJ.

    Supported:
    - v x y z [optional vertex colors ignored]
    - f a b c ...
    - face tokens with v/vt/vn syntax

    Materials/textures are intentionally ignored; colors are set per part from
    model.json.  This keeps the renderer dependency-free.
    """
    path = path.resolve()
    cached = _MESH_CACHE.get(path)
    if cached is not None:
        return cached

    vertices: List[List[float]] = []
    faces: List[Tuple[int, int, int]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                idxs = [
                    _parse_obj_face_token(token, len(vertices))
                    for token in parts[1:]
                ]
                for i in range(1, len(idxs) - 1):
                    faces.append((idxs[0], idxs[i], idxs[i + 1]))

    if not vertices or not faces:
        raise ValueError(f"OBJ mesh has no renderable triangles: {path}")

    mesh = gl.MeshData(
        vertexes=np.array(vertices, dtype=float),
        faces=np.array(faces, dtype=np.uint32),
    )
    _MESH_CACHE[path] = mesh
    return mesh


def apply_item_transform(
    item,
    position: Sequence[float],
    yaw: float,
    scale: float = 1.0,
    rotation_deg: Optional[Sequence[float]] = None,
):
    """
    Apply local rotations/scale first, then yaw, then world translation.

    PyQtGraph composes global transforms in a way that makes translate-then-
    rotate easy to misuse.  Keeping this order in one helper prevents model
    parts from orbiting around the world origin.
    """
    item.resetTransform()

    if rotation_deg:
        rx, ry, rz = (list(rotation_deg) + [0.0, 0.0, 0.0])[:3]
        if rx:
            item.rotate(float(rx), 1, 0, 0)
        if ry:
            item.rotate(float(ry), 0, 1, 0)
        if rz:
            item.rotate(float(rz), 0, 0, 1)

    if scale != 1.0:
        item.scale(float(scale), float(scale), float(scale))

    if yaw:
        item.rotate(math.degrees(float(yaw)), 0, 0, 1)

    x, y, z = position
    item.translate(float(x), float(y), float(z))


@dataclass
class ModelAssetPart:
    """One GL item plus its local transform inside a model asset."""

    item: gl.GLMeshItem
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0


@dataclass
class LoadedModelAsset:
    """GL items created from one model manifest."""

    name: str
    parts: List[ModelAssetPart]
    scale: float = 1.0
    rotation_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def items(self) -> List[gl.GLMeshItem]:
        return [part.item for part in self.parts]

    def apply_pose(self, position: Sequence[float], yaw: float):
        x, y, z = position
        ox, oy, oz = self.offset
        cy, sy = math.cos(yaw), math.sin(yaw)

        for part in self.parts:
            px = (ox + part.offset[0]) * self.scale
            py = (oy + part.offset[1]) * self.scale
            pz = (oz + part.offset[2]) * self.scale
            pose = (
                x + px * cy - py * sy,
                y + px * sy + py * cy,
                z + pz,
            )
            rotation = tuple(
                self.rotation_deg[i] + part.rotation_deg[i]
                for i in range(3)
            )
            apply_item_transform(
                part.item,
                pose,
                yaw,
                scale=self.scale * part.scale,
                rotation_deg=rotation,
            )

    def remove_from(self, view: gl.GLViewWidget):
        for part in self.parts:
            view.removeItem(part.item)
        self.parts.clear()


def _iter_part_specs(manifest: dict) -> Iterable[dict]:
    parts = manifest.get("parts")
    if parts:
        return parts
    mesh = manifest.get("mesh")
    if not mesh:
        return []
    return [{"mesh": mesh, "color": manifest.get("color")}]


def create_model_asset(
    asset_dir: Path,
    view: gl.GLViewWidget,
    default_color: Tuple[float, float, float, float],
) -> Optional[LoadedModelAsset]:
    """Create GL mesh items from assets/models/<name>/model.json."""
    manifest_path = asset_dir / "model.json"
    if not manifest_path.exists():
        return None

    created: List[gl.GLMeshItem] = []
    parts: List[ModelAssetPart] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        name = str(manifest.get("name") or asset_dir.name)
        scale = float(manifest.get("scale", 1.0))
        rotation_deg = tuple(float(v) for v in manifest.get("rotation_deg", (0, 0, 0)))
        rotation_deg = (rotation_deg + (0.0, 0.0, 0.0))[:3]
        offset = tuple(float(v) for v in manifest.get("offset", (0, 0, 0)))
        offset = (offset + (0.0, 0.0, 0.0))[:3]
        smooth = bool(manifest.get("smooth", False))
        shader = str(manifest.get("shader", "shaded"))
        gl_options = str(manifest.get("gl_options", "opaque"))

        for spec in _iter_part_specs(manifest):
            mesh_path = asset_dir / str(spec["mesh"])
            mesh = load_obj_mesh(mesh_path)
            color = _as_color(spec.get("color"), default_color)
            item = gl.GLMeshItem(
                meshdata=mesh,
                smooth=bool(spec.get("smooth", smooth)),
                color=color,
                shader=str(spec.get("shader", shader)),
                glOptions=str(spec.get("gl_options", gl_options)),
            )
            view.addItem(item)
            created.append(item)
            part_offset = tuple(float(v) for v in spec.get("offset", (0, 0, 0)))
            part_offset = (part_offset + (0.0, 0.0, 0.0))[:3]
            part_rotation = tuple(
                float(v) for v in spec.get("rotation_deg", (0, 0, 0))
            )
            part_rotation = (part_rotation + (0.0, 0.0, 0.0))[:3]
            parts.append(
                ModelAssetPart(
                    item=item,
                    offset=part_offset,
                    rotation_deg=part_rotation,
                    scale=float(spec.get("scale", 1.0)),
                )
            )

        if not parts:
            return None

        logger.info("Loaded 3D model asset: %s (%d part(s))", name, len(parts))
        return LoadedModelAsset(
            name=name,
            parts=parts,
            scale=scale,
            rotation_deg=rotation_deg,
            offset=offset,
        )
    except Exception as exc:
        for item in created:
            view.removeItem(item)
        logger.warning("Failed to load model asset %s: %s", asset_dir, exc)
        return None
