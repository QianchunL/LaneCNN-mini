#!/usr/bin/env python3
"""Prepare lane_in/lane_out ROI samples from nuScenes mini.

The script uses nuScenes ego poses and map layers to derive binary labels:

- lane_in: shifted ego footprint polygon mostly overlaps drivable/lane map layers.
- lane_out: shifted ego footprint polygon mostly falls outside those layers.
- ignore: ambiguous footprint overlap or optional front-corridor check fails.

It writes train/val CSV files consumed by the later image classification
pipeline. Images are not copied; CSV paths are relative to the nuScenes
dataroot by default.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.map_expansion.map_api import NuScenesMap
    from pyquaternion import Quaternion
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
except ImportError as exc:  # pragma: no cover - handled at runtime.
    raise SystemExit(
        "Missing dependencies. Install them with:\n"
        "  pip install nuscenes-devkit pyquaternion shapely\n"
    ) from exc


LABEL_OUT = 0
LABEL_IN = 1


@dataclass(frozen=True)
class SampleRow:
    image_path: str
    roi_x1: int
    roi_y1: int
    roi_x2: int
    roi_y2: int
    label: int
    lateral_offset_m: float
    location: str
    scene_token: str
    sample_token: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate lane_in/lane_out ROI CSVs from nuScenes mini."
    )
    parser.add_argument(
        "--dataroot",
        type=Path,
        required=True,
        help="Path to nuScenes dataroot containing v1.0-mini and samples/.",
    )
    parser.add_argument(
        "--version",
        default="v1.0-mini",
        help="nuScenes version, defaults to v1.0-mini.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for train.csv, val.csv, and summary.txt.",
    )
    parser.add_argument(
        "--camera",
        default="CAM_FRONT",
        help="Camera channel to use. Defaults to CAM_FRONT.",
    )
    parser.add_argument(
        "--offsets-m",
        default="0,-0.8,0.8,-1.6,1.6,-2.4,2.4",
        help="Comma-separated ego lateral offsets in meters. Positive is left.",
    )
    parser.add_argument(
        "--vehicle-length-m",
        type=float,
        default=4.6,
        help="Approximate ego footprint length.",
    )
    parser.add_argument(
        "--vehicle-width-m",
        type=float,
        default=1.9,
        help="Approximate ego footprint width.",
    )
    parser.add_argument(
        "--map-layers",
        default="lane,lane_connector,drivable_area",
        help="Comma-separated nuScenes map polygon layers used for overlap.",
    )
    parser.add_argument(
        "--in-threshold",
        type=float,
        default=0.8,
        help="Minimum footprint-map area overlap ratio for lane_in.",
    )
    parser.add_argument(
        "--out-threshold",
        type=float,
        default=0.5,
        help="Maximum footprint-map area overlap ratio for lane_out.",
    )
    parser.add_argument(
        "--check-front-corridor",
        action="store_true",
        help="Filter lane_in samples whose front corridor is not mostly drivable.",
    )
    parser.add_argument(
        "--front-distances-m",
        default="5,10,15",
        help="Comma-separated forward distances used by front-corridor check.",
    )
    parser.add_argument(
        "--front-laterals-m",
        default="-0.6,0,0.6",
        help="Comma-separated lateral points used by front-corridor check.",
    )
    parser.add_argument(
        "--front-in-threshold",
        type=float,
        default=0.8,
        help="Minimum front-corridor inside ratio to keep a lane_in sample.",
    )
    parser.add_argument(
        "--roi-width-frac",
        type=float,
        default=0.42,
        help="ROI width as a fraction of image width.",
    )
    parser.add_argument(
        "--roi-y1-frac",
        type=float,
        default=0.52,
        help="Top ROI coordinate as a fraction of image height.",
    )
    parser.add_argument(
        "--roi-y2-frac",
        type=float,
        default=0.95,
        help="Bottom ROI coordinate as a fraction of image height.",
    )
    parser.add_argument(
        "--roi-lateral-span-m",
        type=float,
        default=10.0,
        help="Approximate meters represented by full image width for ROI shift.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Scene-level validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for scene split.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute image paths instead of paths relative to dataroot.",
    )
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_str_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_maps(dataroot: Path) -> dict[str, NuScenesMap]:
    locations = [
        "boston-seaport",
        "singapore-hollandvillage",
        "singapore-onenorth",
        "singapore-queenstown",
    ]
    return {
        location: NuScenesMap(dataroot=str(dataroot), map_name=location)
        for location in locations
    }


def scene_location(nusc: NuScenes, scene: dict) -> str:
    log = nusc.get("log", scene["log_token"])
    return log["location"]


def ego_to_global_xy(ego_pose: dict, forward_m: float, lateral_m: float) -> tuple[float, float]:
    rotation = Quaternion(ego_pose["rotation"]).rotation_matrix
    translation = ego_pose["translation"]
    local = [forward_m, lateral_m, 0.0]
    global_point = rotation.dot(local) + translation
    return float(global_point[0]), float(global_point[1])


def ego_box_polygon(
    ego_pose: dict,
    length_m: float,
    width_m: float,
    lateral_offset_m: float,
) -> Polygon:
    """Build the shifted ego footprint as a global-coordinate polygon."""

    half_length = length_m / 2.0
    half_width = width_m / 2.0
    corners_ego = [
        (-half_length, lateral_offset_m - half_width),
        (half_length, lateral_offset_m - half_width),
        (half_length, lateral_offset_m + half_width),
        (-half_length, lateral_offset_m + half_width),
    ]
    corners_global = [
        ego_to_global_xy(ego_pose, forward_m, lateral_m)
        for forward_m, lateral_m in corners_ego
    ]
    polygon = Polygon(corners_global)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def polygon_tokens_for_record(record: dict, layer_name: str) -> list[str]:
    if layer_name == "drivable_area":
        return list(record.get("polygon_tokens", []))

    polygon_token = record.get("polygon_token")
    return [polygon_token] if polygon_token else []


def build_map_geometry(nusc_map: NuScenesMap, layer_names: Sequence[str]):
    polygons = []
    for layer_name in layer_names:
        records = getattr(nusc_map, layer_name, None)
        if records is None:
            raise ValueError(f"Unsupported nuScenes map layer: {layer_name}")

        for record in records:
            for polygon_token in polygon_tokens_for_record(record, layer_name):
                try:
                    polygon = nusc_map.extract_polygon(polygon_token)
                except Exception:
                    continue
                if polygon.is_empty:
                    continue
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                if not polygon.is_empty:
                    polygons.append(polygon)

    if not polygons:
        raise ValueError(f"No polygons loaded for map {nusc_map.map_name}")

    geometry = unary_union(polygons)
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    return geometry


def overlap_ratio(footprint_polygon: Polygon, map_geometry) -> float:
    """Compute area(footprint intersect map) / area(footprint)."""

    if footprint_polygon.is_empty or footprint_polygon.area <= 0:
        return 0.0

    try:
        intersection = footprint_polygon.intersection(map_geometry)
    except Exception:
        footprint_polygon = footprint_polygon.buffer(0)
        intersection = footprint_polygon.intersection(map_geometry)

    return float(intersection.area / footprint_polygon.area)


def point_inside_map(nusc_map: NuScenesMap, x: float, y: float) -> bool:
    """Return whether a point is inside a usable driving layer.

    drivable_area is the broadest and most stable layer for this binary task.
    lane/lane_connector add a stricter lane-level signal when available.
    """

    for layer_name in ("lane", "lane_connector", "drivable_area"):
        try:
            token = nusc_map.record_on_point(x, y, layer_name)
        except Exception:
            token = ""
        if token:
            return True
    return False


def inside_ratio(
    nusc_map: NuScenesMap,
    ego_pose: dict,
    points_ego: Sequence[tuple[float, float]],
) -> float:
    if not points_ego:
        return 0.0

    inside = 0
    for forward_m, lateral_m in points_ego:
        x, y = ego_to_global_xy(ego_pose, forward_m, lateral_m)
        if point_inside_map(nusc_map, x, y):
            inside += 1
    return inside / len(points_ego)


def label_from_ratio(ratio: float, in_threshold: float, out_threshold: float) -> int | None:
    if ratio >= in_threshold:
        return LABEL_IN
    if ratio <= out_threshold:
        return LABEL_OUT
    return None


def image_size(sample_data: dict) -> tuple[int, int]:
    width = sample_data.get("width")
    height = sample_data.get("height")
    if width and height:
        return int(width), int(height)

    # nuScenes camera samples are 1600x900. Keep this fallback to avoid opening
    # every image while still supporting older metadata dumps.
    return 1600, 900


def roi_for_offset(
    image_width: int,
    image_height: int,
    lateral_offset_m: float,
    roi_width_frac: float,
    roi_y1_frac: float,
    roi_y2_frac: float,
    roi_lateral_span_m: float,
) -> tuple[int, int, int, int] | None:
    roi_width = image_width * roi_width_frac
    # Positive ego lateral is left, while image x grows to the right.
    center_x = image_width * (0.5 - lateral_offset_m / roi_lateral_span_m)

    x1 = round(center_x - roi_width / 2.0)
    x2 = round(center_x + roi_width / 2.0)
    y1 = round(image_height * roi_y1_frac)
    y2 = round(image_height * roi_y2_frac)

    if x1 < 0 or x2 > image_width or y1 < 0 or y2 > image_height or x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def iter_scene_samples(nusc: NuScenes, scene: dict) -> Iterable[dict]:
    token = scene["first_sample_token"]
    while token:
        sample = nusc.get("sample", token)
        yield sample
        token = sample["next"]


def build_rows(args: argparse.Namespace) -> tuple[list[SampleRow], dict[str, int]]:
    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=True)
    maps = load_maps(args.dataroot)
    offsets_m = parse_float_list(args.offsets_m)
    map_layers = parse_str_list(args.map_layers)
    front_distances_m = parse_float_list(args.front_distances_m)
    front_laterals_m = parse_float_list(args.front_laterals_m)
    map_geometries = {
        location: build_map_geometry(nusc_map, map_layers)
        for location, nusc_map in maps.items()
    }

    rows: list[SampleRow] = []
    stats = {
        "samples_seen": 0,
        "candidate_rois": 0,
        "kept": 0,
        "ignored_ambiguous": 0,
        "ignored_roi_outside": 0,
        "ignored_front_corridor": 0,
        "lane_in": 0,
        "lane_out": 0,
    }

    for scene in nusc.scene:
        location = scene_location(nusc, scene)
        nusc_map = maps[location]
        map_geometry = map_geometries[location]

        for sample in iter_scene_samples(nusc, scene):
            stats["samples_seen"] += 1
            if args.camera not in sample["data"]:
                continue

            cam_data = nusc.get("sample_data", sample["data"][args.camera])
            ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])
            width, height = image_size(cam_data)
            image_path = cam_data["filename"]
            if args.absolute_paths:
                image_path = str(args.dataroot / image_path)

            for lateral_offset_m in offsets_m:
                stats["candidate_rois"] += 1
                roi = roi_for_offset(
                    width,
                    height,
                    lateral_offset_m,
                    args.roi_width_frac,
                    args.roi_y1_frac,
                    args.roi_y2_frac,
                    args.roi_lateral_span_m,
                )
                if roi is None:
                    stats["ignored_roi_outside"] += 1
                    continue

                footprint_polygon = ego_box_polygon(
                    ego_pose,
                    args.vehicle_length_m,
                    args.vehicle_width_m,
                    lateral_offset_m,
                )
                footprint_ratio = overlap_ratio(footprint_polygon, map_geometry)
                label = label_from_ratio(
                    footprint_ratio,
                    args.in_threshold,
                    args.out_threshold,
                )
                if label is None:
                    stats["ignored_ambiguous"] += 1
                    continue

                if args.check_front_corridor and label == LABEL_IN:
                    corridor_points = [
                        (forward_m, lateral_offset_m + lateral_m)
                        for forward_m in front_distances_m
                        for lateral_m in front_laterals_m
                    ]
                    corridor_ratio = inside_ratio(nusc_map, ego_pose, corridor_points)
                    if corridor_ratio < args.front_in_threshold:
                        stats["ignored_front_corridor"] += 1
                        continue

                x1, y1, x2, y2 = roi
                rows.append(
                    SampleRow(
                        image_path=image_path,
                        roi_x1=x1,
                        roi_y1=y1,
                        roi_x2=x2,
                        roi_y2=y2,
                        label=label,
                        lateral_offset_m=lateral_offset_m,
                        location=location,
                        scene_token=scene["token"],
                        sample_token=sample["token"],
                    )
                )
                stats["kept"] += 1
                if label == LABEL_IN:
                    stats["lane_in"] += 1
                else:
                    stats["lane_out"] += 1

    return rows, stats


def split_by_scene(
    rows: Sequence[SampleRow],
    val_ratio: float,
    seed: int,
) -> tuple[list[SampleRow], list[SampleRow]]:
    scene_tokens = sorted({row.scene_token for row in rows})
    rng = random.Random(seed)
    rng.shuffle(scene_tokens)

    val_count = max(1, math.ceil(len(scene_tokens) * val_ratio)) if scene_tokens else 0
    val_scenes = set(scene_tokens[:val_count])

    train_rows = [row for row in rows if row.scene_token not in val_scenes]
    val_rows = [row for row in rows if row.scene_token in val_scenes]
    return train_rows, val_rows


def write_csv(path: Path, rows: Sequence[SampleRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "roi_x1",
        "roi_y1",
        "roi_x2",
        "roi_y2",
        "label",
        "lateral_offset_m",
        "location",
        "scene_token",
        "sample_token",
    ]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_path": row.image_path,
                    "roi_x1": row.roi_x1,
                    "roi_y1": row.roi_y1,
                    "roi_x2": row.roi_x2,
                    "roi_y2": row.roi_y2,
                    "label": row.label,
                    "lateral_offset_m": f"{row.lateral_offset_m:.3f}",
                    "location": row.location,
                    "scene_token": row.scene_token,
                    "sample_token": row.sample_token,
                }
            )


def write_summary(path: Path, stats: dict[str, int], train_rows: Sequence[SampleRow], val_rows: Sequence[SampleRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["nuScenes binary lane ROI preparation summary", ""]
    for key in sorted(stats):
        lines.append(f"{key}: {stats[key]}")
    lines.extend(
        [
            "",
            f"train_rows: {len(train_rows)}",
            f"val_rows: {len(val_rows)}",
            "",
            "labels:",
            "  0 = lane_out",
            "  1 = lane_in",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows, stats = build_rows(args)
    train_rows, val_rows = split_by_scene(rows, args.val_ratio, args.seed)

    write_csv(args.output_dir / "train.csv", train_rows)
    write_csv(args.output_dir / "val.csv", val_rows)
    write_summary(args.output_dir / "summary.txt", stats, train_rows, val_rows)

    print(f"Wrote {len(train_rows)} train rows to {args.output_dir / 'train.csv'}")
    print(f"Wrote {len(val_rows)} val rows to {args.output_dir / 'val.csv'}")
    print(f"Wrote summary to {args.output_dir / 'summary.txt'}")
    print(f"lane_in={stats['lane_in']} lane_out={stats['lane_out']}")


if __name__ == "__main__":
    main()
