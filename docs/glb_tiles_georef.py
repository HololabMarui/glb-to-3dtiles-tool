#!/usr/bin/env python3
"""
Lightweight GLB -> chunked GLB -> selectable GLB/B3DM 3D Tiles set.

Purpose:
- Split a GLB into a few chunks based on mesh units
- Optionally convert each chunk GLB into B3DM with 3d-tiles-tools
- Write a georeferenced tileset.json with root.transform
- Allow content.uri to point to either aligned B3DM or source GLB

Notes:
- This is intentionally simple. It splits by mesh groups, not by faces.
- If the source GLB effectively contains only one mesh, it may still create
  only one chunk.
- Requires: numpy, trimesh. For B3DM output, also requires 3d-tiles-tools via npx.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import trimesh

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


@dataclass
class ChunkInfo:
    index: int
    glb_path: Path
    b3dm_path: Path | None
    aligned_b3dm_path: Path | None
    bounds: np.ndarray
    mesh_count: int


def run(cmd: Sequence[str]) -> None:
    print("[RUN]", " ".join(str(x) for x in cmd))
    subprocess.run(list(map(str, cmd)), check=True)


def ecef_from_lon_lat_height(lon_deg: float, lat_deg: float, height: float) -> np.ndarray:
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + height) * cos_lat * cos_lon
    y = (n + height) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + height) * sin_lat
    return np.array([x, y, z], dtype=float)


def enu_axes(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    east = np.array([-sin_lon, cos_lon, 0.0], dtype=float)
    north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=float)
    up = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dtype=float)
    return np.column_stack([east, north, up])


def rot_x(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def hpr_rotation_matrix(heading_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    h = math.radians(heading_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    return rot_z(h) @ rot_y(p) @ rot_x(r)


def make_transform(lon_deg: float, lat_deg: float, height: float, heading_deg: float, pitch_deg: float, roll_deg: float) -> List[float]:
    translation = ecef_from_lon_lat_height(lon_deg, lat_deg, height)
    r_ecef_from_enu = enu_axes(lon_deg, lat_deg)
    r_local = hpr_rotation_matrix(heading_deg, pitch_deg, roll_deg)
    r = r_ecef_from_enu @ r_local
    m = np.eye(4, dtype=float)
    m[:3, :3] = r
    m[:3, 3] = translation
    return m.T.reshape(-1).tolist()


def box_from_bounds(bounds: np.ndarray) -> List[float]:
    bmin = bounds[0]
    bmax = bounds[1]
    center = (bmin + bmax) / 2.0
    half = (bmax - bmin) / 2.0
    return [
        float(center[0]), float(center[1]), float(center[2]),
        float(half[0]), 0.0, 0.0,
        0.0, float(half[1]), 0.0,
        0.0, 0.0, float(half[2]),
    ]


def load_scene_meshes(glb_path: Path) -> List[trimesh.Trimesh]:
    scene_or_mesh = trimesh.load(glb_path, force="scene")
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        meshes = [scene_or_mesh]
    else:
        meshes = []
        for _, geom in scene_or_mesh.geometry.items():
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0 and len(geom.faces) > 0:
                meshes.append(geom.copy())
    if not meshes:
        raise RuntimeError("No meshes found in input GLB.")
    return meshes


def split_evenly(items: Sequence[trimesh.Trimesh], chunks: int) -> List[List[trimesh.Trimesh]]:
    chunks = max(1, min(chunks, len(items)))
    groups: List[List[trimesh.Trimesh]] = [[] for _ in range(chunks)]
    for idx, item in enumerate(items):
        groups[idx % chunks].append(item)
    return [g for g in groups if g]


def scene_bounds(meshes: Iterable[trimesh.Trimesh]) -> np.ndarray:
    bounds_list = [m.bounds for m in meshes]
    mins = np.vstack([b[0] for b in bounds_list])
    maxs = np.vstack([b[1] for b in bounds_list])
    return np.array([mins.min(axis=0), maxs.max(axis=0)], dtype=float)


def export_chunk_glb(meshes: List[trimesh.Trimesh], out_path: Path) -> np.ndarray:
    scene = trimesh.Scene()
    for i, mesh in enumerate(meshes):
        scene.add_geometry(mesh, node_name=f"mesh_{i}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = scene.export(file_type="glb")
    out_path.write_bytes(data)
    return scene_bounds(meshes)


def create_b3dm(glb_path: Path, out_tiles_dir: Path, prefix: str) -> tuple[Path, Path]:
    raw = out_tiles_dir / f"{prefix}.b3dm"
    aligned = out_tiles_dir / f"{prefix}_aligned.b3dm"
    run(["npx", "3d-tiles-tools", "glbToB3dm", "-i", glb_path, "-o", raw])
    run(["npx", "3d-tiles-tools", "updateAlignment", "-i", raw, "-o", aligned])
    return raw, aligned


def write_tileset_json(out_path: Path, chunks: List[ChunkInfo], geometric_error: float, refine: str,
                       lon: float, lat: float, height: float, heading: float, pitch: float, roll: float,
                       content_format: str) -> None:
    mins = np.vstack([c.bounds[0] for c in chunks])
    maxs = np.vstack([c.bounds[1] for c in chunks])
    root_bounds = np.array([mins.min(axis=0), maxs.max(axis=0)], dtype=float)

    children = []
    for chunk in chunks:
        if content_format == "glb":
            uri = chunk.glb_path.relative_to(out_path.parent).as_posix()
        else:
            if chunk.aligned_b3dm_path is None:
                raise RuntimeError("content-format=b3dm requires B3DM conversion output.")
            uri = chunk.aligned_b3dm_path.relative_to(out_path.parent).as_posix()
        children.append({
            "boundingVolume": {"box": box_from_bounds(chunk.bounds)},
            "geometricError": 0,
            "content": {"uri": uri},
        })

    tileset = {
        "asset": {"version": "1.1"},
        "geometricError": float(geometric_error),
        "root": {
            "boundingVolume": {"box": box_from_bounds(root_bounds)},
            "geometricError": float(max(1.0, geometric_error / 2.0)),
            "refine": refine,
            "transform": make_transform(lon, lat, height, heading, pitch, roll),
            "children": children,
        },
    }
    out_path.write_text(json.dumps(tileset, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(out_path: Path, input_glb: Path, chunks: List[ChunkInfo], args: argparse.Namespace) -> None:
    report = {
        "input": str(input_glb),
        "chunk_count": len(chunks),
        "parameters": {
            "chunks": args.chunks,
            "content_format": args.content_format,
            "lon": args.lon,
            "lat": args.lat,
            "height": args.height,
            "heading": args.heading,
            "pitch": args.pitch,
            "roll": args.roll,
            "geometric_error": args.geometric_error,
            "refine": args.refine,
        },
        "chunks": [
            {
                "index": c.index,
                "mesh_count": c.mesh_count,
                "glb": str(c.glb_path),
                "b3dm": str(c.b3dm_path) if c.b3dm_path else None,
                "aligned_b3dm": str(c.aligned_b3dm_path) if c.aligned_b3dm_path else None,
                "bounds": c.bounds.tolist(),
            }
            for c in chunks
        ],
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split GLB into a few mesh-based chunks and write a georeferenced tileset.")
    p.add_argument("--input", required=True, help="Input GLB path")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--chunks", type=int, default=3, help="Requested number of chunks (default: 3)")
    p.add_argument("--tileset-name", default="tileset.json", help="Tileset file name")
    p.add_argument("--content-format", default="b3dm", choices=["b3dm", "glb"], help="Write tileset content URIs as b3dm or glb")
    p.add_argument("--skip-b3dm", action="store_true", help="Only export chunk GLBs; skip b3dm conversion")
    p.add_argument("--refine", default="ADD", choices=["ADD", "REPLACE"], help="Root refine mode")
    p.add_argument("--geometric-error", type=float, default=200.0, help="Root geometric error")
    p.add_argument("--lon", type=float, required=True, help="Longitude in degrees")
    p.add_argument("--lat", type=float, required=True, help="Latitude in degrees")
    p.add_argument("--height", type=float, default=0.0, help="Height in meters")
    p.add_argument("--heading", type=float, default=0.0, help="Heading in degrees")
    p.add_argument("--pitch", type=float, default=0.0, help="Pitch in degrees")
    p.add_argument("--roll", type=float, default=0.0, help="Roll in degrees")
    return p.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    input_glb = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    tiles_src_dir = output_dir / "tiles_src"
    tiles_dir = output_dir / "tiles"
    tileset_path = output_dir / args.tileset_name
    report_path = output_dir / "build_report.json"

    if not input_glb.exists():
        print(f"Input GLB not found: {input_glb}", file=sys.stderr)
        return 1
    if args.content_format == "b3dm" and args.skip_b3dm:
        print("content-format=b3dm cannot be used with --skip-b3dm.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    tiles_src_dir.mkdir(parents=True, exist_ok=True)
    need_b3dm = (args.content_format == "b3dm") and (not args.skip_b3dm)
    if need_b3dm:
        tiles_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which("npx") is None:
            print("npx not found. Install Node.js/npm or use --content-format glb.", file=sys.stderr)
            return 3

    meshes = load_scene_meshes(input_glb)
    groups = split_evenly(meshes, args.chunks)
    print(f"[INFO] source meshes: {len(meshes)}")
    print(f"[INFO] actual chunks : {len(groups)}")
    print(f"[INFO] content format: {args.content_format}")

    chunk_infos: List[ChunkInfo] = []
    for index, group in enumerate(groups):
        chunk_glb = tiles_src_dir / f"chunk_{index:03d}.glb"
        bounds = export_chunk_glb(group, chunk_glb)
        raw_b3dm = None
        aligned_b3dm = None
        if need_b3dm:
            raw_b3dm, aligned_b3dm = create_b3dm(chunk_glb, tiles_dir, f"chunk_{index:03d}")
        chunk_infos.append(ChunkInfo(index=index, glb_path=chunk_glb, b3dm_path=raw_b3dm,
                                     aligned_b3dm_path=aligned_b3dm, bounds=bounds, mesh_count=len(group)))

    write_tileset_json(tileset_path, chunk_infos, geometric_error=args.geometric_error, refine=args.refine,
                       lon=args.lon, lat=args.lat, height=args.height, heading=args.heading,
                       pitch=args.pitch, roll=args.roll, content_format=args.content_format)
    write_report(report_path, input_glb, chunk_infos, args)

    print(f"[OK] wrote {tileset_path}")
    print(f"[OK] wrote {report_path}")
    for c in chunk_infos:
        target = c.glb_path if args.content_format == "glb" else c.aligned_b3dm_path
        print(f"[OK] chunk {c.index:03d}: meshes={c.mesh_count}, target={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
