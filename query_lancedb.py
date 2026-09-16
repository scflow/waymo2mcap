#!/usr/bin/env python3
"""
Query examples against the LanceDB database built by mcap_to_lancedb.py.

Demonstrates the episode-centric analysis patterns:
  1. table overview / episode registry
  2. structured filtering (episode, camera, time range)
  3. aggregation (object label distribution per episode)
  4. cross-table join (3D boxes -> camera frame evidence)
  5. media retrieval (extract JPEG / point cloud from Lance blobs)

Usage: python query_lancedb.py [--db data/lancedb] [--demo all|overview|...]
"""

import argparse
from collections import Counter

import lancedb
import numpy as np


def table_names(db):
    """Normalize list_tables() output across lancedb versions (plain list vs
    namespace ListTablesResponse)."""
    names = db.list_tables()
    if hasattr(names, "tables"):
        names = names.tables
    return [n if isinstance(n, str) else n[0] for n in names]


def overview(db):
    print("=" * 70)
    print("1. TABLE OVERVIEW")
    print("=" * 70)
    for name in table_names(db):
        tbl = db.open_table(name)
        print(f"  {name:15s} rows={tbl.count_rows():,}")

    print("\n" + "=" * 70)
    print("EPISODE REGISTRY")
    print("=" * 70)
    rows = db.open_table("episodes").to_arrow().to_pylist()
    for r in rows:
        print(f"  [{r['source_type']:9s}] {r['episode_id']}")
        print(f"    duration={r['duration_s']:.1f}s frames={r['frame_count']} "
              f"images={r['num_images']:,} points={r['num_lidar_points']:,} "
              f"boxes={r['num_objects_3d']:,}")
        print(f"    media: {r['mcap_path']}")


def filter_demo(db):
    print("\n" + "=" * 70)
    print("2. STRUCTURED FILTER: FRONT camera, frames 10-12 of first episode")
    print("=" * 70)
    ep_id = db.open_table("episodes").to_arrow().column("episode_id")[0].as_py()
    cf = db.open_table("camera_frames")
    rows = (cf.search()
            .where(f"episode_id = '{ep_id}' AND camera = 'FRONT' "
                   f"AND frame_index >= 10 AND frame_index <= 12")
            .select(["frame_index", "camera", "timestamp_ns", "num_labels"])
            .limit(10).to_list())
    for r in rows:
        print(f"  frame={r['frame_index']:3d} ts={r['timestamp_ns']} labels={r['num_labels']}")

    print("\n" + "=" * 70)
    print("   TIME RANGE: ego poses with speed > 15 m/s (fast driving)")
    print("=" * 70)
    ep = db.open_table("ego_poses")
    rows = (ep.search()
            .where("speed_mps > 15")
            .select(["episode_id", "frame_index", "speed_mps"])
            .limit(8).to_list())
    for r in rows:
        print(f"  {r['episode_id'][:40]:40s} frame={r['frame_index']:3d} "
              f"speed={r['speed_mps']:.1f} m/s")
    print(f"  total: {ep.count_rows('speed_mps > 15'):,} frames above 15 m/s")


def aggregate_demo(db):
    print("\n" + "=" * 70)
    print("3. AGGREGATION: 3D object label distribution per episode")
    print("=" * 70)
    objs = db.open_table("objects_3d")
    arrow = objs.search().select(["episode_id", "label"]).limit(None).to_arrow()
    data = arrow.to_pydict()
    per_ep = {}
    for eid, label in zip(data["episode_id"], data["label"]):
        per_ep.setdefault(eid, Counter())[label] += 1
    for eid, counts in per_ep.items():
        top = ", ".join(f"{k}:{v}" for k, v in counts.most_common(5))
        print(f"  {eid[:50]:50s} {top}")


def join_demo(db):
    print("\n" + "=" * 70)
    print("4. CROSS-TABLE: large objects ahead of ego -> evidence image")
    print("=" * 70)
    objs = db.open_table("objects_3d")
    rows = (objs.search()
            .where("pos_x > 20 AND pos_x < 60 AND size_z > 2.5")
            .select(["episode_id", "frame_index", "label", "pos_x", "pos_y", "size_z"])
            .limit(5).to_list())
    if not rows:
        print("  (no matching objects)")
        return
    cf = db.open_table("camera_frames")
    for r in rows:
        print(f"  {r['label']} at x={r['pos_x']:.1f}m (size_z={r['size_z']:.1f}m) "
              f"frame={r['frame_index']}")
        img_row = (cf.search()
                   .where(f"episode_id = '{r['episode_id']}' "
                          f"AND frame_index = {r['frame_index']} AND camera = 'FRONT'")
                   .select(["image"]).limit(1).to_list())
        if img_row:
            print(f"    -> FRONT evidence image: {len(img_row[0]['image']):,} bytes")


def media_demo(db, out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("5. MEDIA RETRIEVAL: extract JPEG image and point cloud")
    print("=" * 70)
    cf = db.open_table("camera_frames")
    row = cf.search().where("camera = 'FRONT'").select(["episode_id", "frame_index", "image"]).limit(1).to_list()[0]
    img_path = f"{out_dir}/lance_front_f{row['frame_index']}.jpg"
    with open(img_path, "wb") as f:
        f.write(row["image"])
    print(f"  image -> {img_path} ({len(row['image']):,} bytes, "
          f"episode={row['episode_id'][:40]}, frame={row['frame_index']})")

    lf = db.open_table("lidar_frames")
    lrow = (lf.search()
            .where(f"episode_id = '{row['episode_id']}' AND frame_index = {row['frame_index']}")
            .limit(1).to_list())
    if lrow:
        lrow = lrow[0]
        pts = np.frombuffer(lrow["points"], dtype=np.float32).reshape(-1, 5)
        pts_path = f"{out_dir}/lance_points_f{row['frame_index']}.npy"
        np.save(pts_path, pts)
        print(f"  points -> {pts_path} shape={pts.shape} "
              f"(x,y,z,intensity,elongation; range x=[{pts[:,0].min():.0f}, {pts[:,0].max():.0f}]m)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/lancedb")
    parser.add_argument("--out", default="data/lancedb_exports")
    parser.add_argument("--demo", default="all",
                        choices=["all", "overview", "filter", "aggregate", "join", "media"])
    args = parser.parse_args()

    db = lancedb.connect(args.db)
    demos = {
        "overview": overview,
        "filter": filter_demo,
        "aggregate": aggregate_demo,
        "join": join_demo,
        "media": lambda db: media_demo(db, args.out),
    }
    if args.demo == "all":
        for fn in demos.values():
            fn(db)
    else:
        demos[args.demo](db)


if __name__ == "__main__":
    main()
