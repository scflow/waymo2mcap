#!/usr/bin/env python3
"""
Import Foxglove-compatible MCAP files (Waymo Open Dataset conversions) into
LanceDB, following the episode-centric Physical AI data architecture.

Tables
------
episodes       one row per MCAP file: media location, time range, stats
camera_frames  one row per (episode, frame, camera): JPEG bytes + 2D labels
lidar_frames   one row per (episode, frame): raw point buffer + layout metadata
ego_poses      one row per (episode, frame): pose + twist
objects_3d     one row per (episode, frame, object): label / position / size

Rows are keyed by (episode_id, frame_index, timestamp_ns) so raw media in the
source MCAP can still be located for full-fidelity random access.

Usage
-----
python mcap_to_lancedb.py                          # import data/mcap/**/*.mcap
python mcap_to_lancedb.py --input <file.mcap>      # single file
python mcap_to_lancedb.py --max-frames 10          # smoke test
python mcap_to_lancedb.py --rebuild                # drop and recreate tables
"""

import argparse
import json
import re
import time
from pathlib import Path

import lancedb
from lancedb.index import BTree
import pyarrow as pa
from foxglove_schemas_protobuf import (
    CompressedImage_pb2,
    ImageAnnotations_pb2,
    LinePrimitive_pb2,
    Odometry_pb2,
    PointCloud_pb2,
    PoseInFrame_pb2,
    SceneUpdate_pb2,
)
from mcap.reader import make_reader

DECODERS = {
    "foxglove.CompressedImage": CompressedImage_pb2.CompressedImage,
    "foxglove.PointCloud": PointCloud_pb2.PointCloud,
    "foxglove.SceneUpdate": SceneUpdate_pb2.SceneUpdate,
    "foxglove.PoseInFrame": PoseInFrame_pb2.PoseInFrame,
    "foxglove.ImageAnnotations": ImageAnnotations_pb2.ImageAnnotations,
    "foxglove.Odometry": Odometry_pb2.Odometry,
}

CAMERA_IMG_RE = re.compile(r"^/camera/(\w+)/compressed$")
CAMERA_ANN_RE = re.compile(r"^/camera/(\w+)/annotations$")
MAP_RE = re.compile(r"^/map/(\w+)$")

EPISODES_SCHEMA = pa.schema([
    ("episode_id", pa.string()),
    ("source_type", pa.string()),
    ("mcap_path", pa.string()),
    ("file_size_bytes", pa.int64()),
    ("start_time_ns", pa.int64()),
    ("end_time_ns", pa.int64()),
    ("duration_s", pa.float64()),
    ("frame_count", pa.int32()),
    ("num_messages", pa.int64()),
    ("num_images", pa.int64()),
    ("num_lidar_points", pa.int64()),
    ("num_objects_3d", pa.int64()),
    ("num_map_features", pa.int64()),
    ("cameras", pa.list_(pa.string())),
    ("topic_counts", pa.string()),  # JSON {topic: message_count}
    ("imported_at", pa.string()),
])

CAMERA_FRAMES_SCHEMA = pa.schema([
    ("episode_id", pa.string()),
    ("frame_index", pa.int32()),
    ("camera", pa.string()),
    ("timestamp_ns", pa.int64()),
    ("format", pa.string()),
    ("image", pa.binary()),
    ("num_labels", pa.int32()),
    ("labels", pa.list_(pa.struct([
        ("text", pa.string()),
        ("x", pa.float64()),
        ("y", pa.float64()),
    ]))),
])

LIDAR_FRAMES_SCHEMA = pa.schema([
    ("episode_id", pa.string()),
    ("frame_index", pa.int32()),
    ("timestamp_ns", pa.int64()),
    ("num_points", pa.int32()),
    ("point_stride", pa.int32()),
    ("field_names", pa.list_(pa.string())),
    ("points", pa.binary()),
])

EGO_POSES_SCHEMA = pa.schema([
    ("episode_id", pa.string()),
    ("frame_index", pa.int32()),
    ("timestamp_ns", pa.int64()),
    ("px", pa.float64()), ("py", pa.float64()), ("pz", pa.float64()),
    ("qx", pa.float64()), ("qy", pa.float64()), ("qz", pa.float64()), ("qw", pa.float64()),
    ("vx", pa.float64()), ("vy", pa.float64()), ("vz", pa.float64()),
    ("speed_mps", pa.float64()),
])

OBJECTS_3D_SCHEMA = pa.schema([
    ("episode_id", pa.string()),
    ("frame_index", pa.int32()),
    ("timestamp_ns", pa.int64()),
    ("object_id", pa.string()),
    ("label", pa.string()),
    ("pos_x", pa.float64()), ("pos_y", pa.float64()), ("pos_z", pa.float64()),
    ("size_x", pa.float64()), ("size_y", pa.float64()), ("size_z", pa.float64()),
    ("quat_x", pa.float64()), ("quat_y", pa.float64()), ("quat_z", pa.float64()), ("quat_w", pa.float64()),
    ("color_r", pa.float32()), ("color_g", pa.float32()), ("color_b", pa.float32()), ("color_a", pa.float32()),
])

MAP_FEATURES_SCHEMA = pa.schema([
    ("episode_id", pa.string()),
    ("layer", pa.string()),          # lanes / road_lines / road_edges / ...
    ("geometry_type", pa.string()),  # line_strip / line_loop / point
    ("timestamp_ns", pa.int64()),
    ("num_points", pa.int32()),
    ("points", pa.list_(pa.struct([
        ("x", pa.float64()), ("y", pa.float64()), ("z", pa.float64()),
    ]))),
    # world-frame bounding box, enables spatial filtering near ego poses
    ("min_x", pa.float64()), ("min_y", pa.float64()),
    ("max_x", pa.float64()), ("max_y", pa.float64()),
])

# Flush thresholds: bound memory while keeping Lance writes large enough
# to be efficient on the WSL /mnt/d filesystem.
FLUSH_EVERY = {
    "camera_frames": 128,
    "lidar_frames": 32,
    "ego_poses": 2048,
    "objects_3d": 8192,
    "map_features": 512,
}


class TableBuffer:
    """Accumulates rows column-wise and flushes them into a LanceDB table."""

    def __init__(self, db, name, schema, rebuild):
        if rebuild and name in db.table_names():
            db.drop_table(name)
        if name in db.table_names():
            self.table = db.open_table(name)
        else:
            self.table = db.create_table(name, schema=schema, mode="create")
        self.schema = schema
        self.rows = []  # list of dicts
        self.total = 0

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= FLUSH_EVERY.get(self.table.name, 1):
            self.flush()

    def flush(self):
        if not self.rows:
            return
        columns = {f.name: [row.get(f.name) for row in self.rows] for f in self.schema}
        batch = pa.Table.from_pydict(columns, schema=self.schema)
        self.table.add(batch)
        self.total += len(self.rows)
        self.rows = []

    def count(self):
        return self.total


def episode_id_for(path: Path) -> str:
    stem = path.stem
    if stem.startswith("individual_files_training_"):
        return stem[len("individual_files_training_"):]
    return stem


def source_type_for(path: Path) -> str:
    return "scenario" if path.stem.startswith("scenario") else "perception"


class FrameAccumulator:
    """Holds decoded messages for the frame currently being read. All messages
    of one frame share the same log_time in the converted MCAPs, so a change
    of log_time finalizes the previous frame."""

    def __init__(self):
        self.timestamp_ns = None
        self.images = {}    # camera -> CompressedImage
        self.labels = {}    # camera -> [(text, x, y)]
        self.lidar = None   # PointCloud
        self.pose = None    # PoseInFrame
        self.odom = None    # Odometry
        self.boxes = []     # [(entity, cube, label)]
        self.map_rows = []  # map feature row dicts (written once per file)

    def reset(self, timestamp_ns):
        self.timestamp_ns = timestamp_ns
        self.images.clear()
        self.labels.clear()
        self.lidar = None
        self.pose = None
        self.odom = None
        self.boxes = []
        self.map_rows = []


def finalize_frame(acc, buffers, episode_id, frame_index, stats, store_lidar=True):
    ts = acc.timestamp_ns

    for camera, img in acc.images.items():
        labels = acc.labels.get(camera, [])
        buffers["camera_frames"].add({
            "episode_id": episode_id,
            "frame_index": frame_index,
            "camera": camera,
            "timestamp_ns": ts,
            "format": img.format or "jpeg",
            "image": img.data,
            "num_labels": len(labels),
            "labels": [{"text": t, "x": x, "y": y} for t, x, y in labels],
        })
        stats["num_images"] += 1

    if acc.lidar is not None and store_lidar:
        pc = acc.lidar
        buffers["lidar_frames"].add({
            "episode_id": episode_id,
            "frame_index": frame_index,
            "timestamp_ns": ts,
            "num_points": len(pc.data) // pc.point_stride if pc.point_stride else 0,
            "point_stride": pc.point_stride,
            "field_names": [f.name for f in pc.fields],
            "points": pc.data,
        })
    if acc.lidar is not None:
        stats["num_lidar_points"] += len(acc.lidar.data) // acc.lidar.point_stride

    if acc.pose is not None:
        p = acc.pose.pose.position
        q = acc.pose.pose.orientation
        row = {
            "episode_id": episode_id,
            "frame_index": frame_index,
            "timestamp_ns": ts,
            "px": p.x, "py": p.y, "pz": p.z,
            "qx": q.x, "qy": q.y, "qz": q.z, "qw": q.w,
            "vx": None, "vy": None, "vz": None, "speed_mps": None,
        }
        if acc.odom is not None:
            lin = acc.odom.linear_velocity
            row["vx"], row["vy"], row["vz"] = lin.x, lin.y, lin.z
            row["speed_mps"] = (lin.x**2 + lin.y**2 + lin.z**2) ** 0.5
        buffers["ego_poses"].add(row)

    for entity, cube, label in acc.boxes:
        pos, size, quat, color = cube.pose.position, cube.size, cube.pose.orientation, cube.color
        buffers["objects_3d"].add({
            "episode_id": episode_id,
            "frame_index": frame_index,
            "timestamp_ns": ts,
            "object_id": entity.id,
            "label": label,
            "pos_x": pos.x, "pos_y": pos.y, "pos_z": pos.z,
            "size_x": size.x, "size_y": size.y, "size_z": size.z,
            "quat_x": quat.x, "quat_y": quat.y, "quat_z": quat.z, "quat_w": quat.w,
            "color_r": color.r, "color_g": color.g, "color_b": color.b, "color_a": color.a,
        })
        stats["num_objects_3d"] += 1

    for row in acc.map_rows:
        buffers["map_features"].add(row)
        stats["num_map_features"] += 1


def decode_map_update(msg, layer, episode_id, ts):
    """Convert a foxglove.SceneUpdate on a /map/<layer> topic into
    map_features rows. One entity per layer; each line/cylinder primitive is
    one feature in world coordinates."""
    rows = []
    for entity in msg.entities:
        for line in entity.lines:
            pts = [{"x": p.x, "y": p.y, "z": p.z} for p in line.points]
            if not pts:
                continue
            rows.append({
                "episode_id": episode_id,
                "layer": layer,
                "geometry_type": LinePrimitive_pb2.LinePrimitive.Type.Name(
                    line.type).lower(),
                "timestamp_ns": ts,
                "num_points": len(pts),
                "points": pts,
                "min_x": min(p["x"] for p in pts),
                "min_y": min(p["y"] for p in pts),
                "max_x": max(p["x"] for p in pts),
                "max_y": max(p["y"] for p in pts),
            })
        for cyl in entity.cylinders:
            p = cyl.pose.position
            rows.append({
                "episode_id": episode_id,
                "layer": layer,
                "geometry_type": "point",
                "timestamp_ns": ts,
                "num_points": 1,
                "points": [{"x": p.x, "y": p.y, "z": p.z}],
                "min_x": p.x, "min_y": p.y, "max_x": p.x, "max_y": p.y,
            })
    return rows


def import_mcap(path: Path, buffers, max_frames=None, store_lidar=True, force=False):
    episode_id = episode_id_for(path)
    episodes_tbl = buffers["episodes"].table

    if not force and episodes_tbl.count_rows(f"episode_id = '{episode_id}'") > 0:
        print(f"[skip] {episode_id}: already imported (use --force to reimport)")
        return

    print(f"[import] {path.name}")
    t0 = time.time()
    stats = {"num_images": 0, "num_lidar_points": 0, "num_objects_3d": 0,
             "num_map_features": 0}

    with open(path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        schemas = summary.schemas
        topic_counts = {
            summary.channels[cid].topic: cnt
            for cid, cnt in (summary.statistics.channel_message_counts or {}).items()
        }
        cameras = sorted({
            m.group(1) for m in (CAMERA_IMG_RE.match(t) for t in topic_counts) if m
        })

        acc = FrameAccumulator()
        frame_index = -1
        skipped_schemas = set()

        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            decoder = DECODERS.get(schema.name)
            if decoder is None:
                if schema.name not in skipped_schemas:
                    skipped_schemas.add(schema.name)
                continue

            ts = message.log_time
            if acc.timestamp_ns is not None and ts != acc.timestamp_ns:
                # frame boundary: finalize the accumulated frame, unless we
                # have already reached the max-frames limit
                if max_frames is not None and frame_index + 1 >= max_frames:
                    break
                frame_index += 1
                finalize_frame(acc, buffers, episode_id, frame_index, stats, store_lidar)
                acc.reset(ts)
            if acc.timestamp_ns is None:
                acc.reset(ts)

            msg = decoder.FromString(message.data)
            if (m := CAMERA_IMG_RE.match(topic)) is not None:
                acc.images[m.group(1)] = msg
            elif (m := CAMERA_ANN_RE.match(topic)) is not None:
                acc.labels[m.group(1)] = [
                    (t.text, t.position.x, t.position.y) for t in msg.texts
                ]
            elif topic == "/lidar/points":
                acc.lidar = msg
            elif topic == "/perception/boxes_3d":
                for entity in msg.entities:
                    label = entity.texts[0].text if entity.texts else ""
                    for cube in entity.cubes:
                        acc.boxes.append((entity, cube, label))
            elif topic == "/ego_pose":
                acc.pose = msg
            elif topic == "/odometry":
                acc.odom = msg
            elif (m := MAP_RE.match(topic)) is not None:
                acc.map_rows.extend(decode_map_update(msg, m.group(1), episode_id, ts))
        else:
            # finalize the last frame when the loop wasn't broken early
            if acc.timestamp_ns is not None:
                frame_index += 1
                finalize_frame(acc, buffers, episode_id, frame_index, stats, store_lidar)

        s = summary.statistics
        start_ns, end_ns = s.message_start_time, s.message_end_time

        for b in buffers.values():
            b.flush()

        buffers["episodes"].add({
            "episode_id": episode_id,
            "source_type": source_type_for(path),
            "mcap_path": str(path.resolve()),
            "file_size_bytes": path.stat().st_size,
            "start_time_ns": start_ns,
            "end_time_ns": end_ns,
            "duration_s": (end_ns - start_ns) / 1e9,
            "frame_count": frame_index + 1,
            "num_messages": s.message_count,
            "num_images": stats["num_images"],
            "num_lidar_points": stats["num_lidar_points"],
            "num_objects_3d": stats["num_objects_3d"],
            "num_map_features": stats["num_map_features"],
            "cameras": cameras,
            "topic_counts": json.dumps(topic_counts, ensure_ascii=False),
            "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        buffers["episodes"].flush()

    elapsed = time.time() - t0
    print(f"  frames={frame_index + 1} images={stats['num_images']} "
          f"points={stats['num_lidar_points']:,} boxes={stats['num_objects_3d']:,} "
          f"({elapsed:.1f}s)")


def create_scalar_indexes(db):
    for name in ["camera_frames", "lidar_frames", "ego_poses", "objects_3d"]:
        if name in db.table_names():
            try:
                db.open_table(name).create_index("episode_id", config=BTree())
            except Exception as e:  # index may already exist
                print(f"  index {name}.episode_id: {e}")


def main():
    parser = argparse.ArgumentParser(description="Import MCAP files into LanceDB")
    parser.add_argument("--db", default="data/lancedb", help="LanceDB directory")
    parser.add_argument("--input", default="data/mcap",
                        help="MCAP file or directory containing .mcap files")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Import only the first N frames per file (smoke test)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Drop existing tables before import")
    parser.add_argument("--force", action="store_true",
                        help="Reimport episodes that already exist")
    parser.add_argument("--no-lidar", action="store_true",
                        help="Skip storing point cloud buffers (counts still recorded)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_file():
        mcap_files = [input_path]
    else:
        mcap_files = sorted(input_path.rglob("*.mcap"))
    if not mcap_files:
        print(f"No .mcap files found under {input_path}")
        return

    db = lancedb.connect(args.db)
    buffers = {
        "episodes": TableBuffer(db, "episodes", EPISODES_SCHEMA, args.rebuild),
        "camera_frames": TableBuffer(db, "camera_frames", CAMERA_FRAMES_SCHEMA, args.rebuild),
        "lidar_frames": TableBuffer(db, "lidar_frames", LIDAR_FRAMES_SCHEMA, args.rebuild),
        "ego_poses": TableBuffer(db, "ego_poses", EGO_POSES_SCHEMA, args.rebuild),
        "objects_3d": TableBuffer(db, "objects_3d", OBJECTS_3D_SCHEMA, args.rebuild),
        "map_features": TableBuffer(db, "map_features", MAP_FEATURES_SCHEMA, args.rebuild),
    }

    print(f"LanceDB: {args.db} | files: {len(mcap_files)}")
    for path in mcap_files:
        import_mcap(path, buffers, max_frames=args.max_frames,
                    store_lidar=not args.no_lidar, force=args.force)

    for name, b in buffers.items():
        print(f"  table {name:15s} total rows added this run: {b.count():,}")

    create_scalar_indexes(db)
    print("Done.")


if __name__ == "__main__":
    main()
