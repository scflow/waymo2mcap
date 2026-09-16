#!/usr/bin/env python3
"""Inventory: how many episodes, frames per episode, lidar/3d coverage."""
import lancedb
from collections import Counter, defaultdict

db = lancedb.connect("/mnt/d/src/waymo2mcap/data/lancedb")

eps = db.open_table("episodes").to_arrow().to_pylist()
print("=== episodes registry ===")
for r in eps:
    print(f"  {r['source_type']:9s} {r['episode_id'][:60]}")
    print(f"    frames={r['frame_count']} images={r['num_images']} "
          f"points={r['num_lidar_points']:,} boxes={r['num_objects_3d']}")
    print(f"    {r['mcap_path']}")

cf = db.open_table("camera_frames")
rows = (cf.search()
        .select(["episode_id", "frame_index", "camera"])
        .to_arrow().to_pylist())
by_ep = defaultdict(set)
by_ep_cam = Counter()
for r in rows:
    by_ep[r["episode_id"]].add(r["frame_index"])
    by_ep_cam[(r["episode_id"], r["camera"])] += 1

print("\n=== camera_frames ===")
print("total", len(rows))
for ep, frames in sorted(by_ep.items()):
    cams = sorted({c for (e, c) in by_ep_cam if e == ep})
    print(f"  {ep[:56]}")
    print(f"    unique frames={len(frames)}  range={min(frames)}-{max(frames)}  cams={cams}")

lf = db.open_table("lidar_frames")
lid = (lf.search().select(["episode_id", "frame_index"]).to_arrow().to_pylist())
lid_by_ep = defaultdict(set)
for r in lid:
    lid_by_ep[r["episode_id"]].add(r["frame_index"])
print("\n=== lidar_frames ===")
print("total", len(lid))
for ep, frames in sorted(lid_by_ep.items()):
    print(f"  {ep[:56]}  frames={len(frames)} range={min(frames)}-{max(frames)}")

o3 = db.open_table("objects_3d")
o3rows = (o3.search().select(["episode_id", "frame_index"]).to_arrow().to_pylist())
o3_by_ep = defaultdict(set)
for r in o3rows:
    o3_by_ep[r["episode_id"]].add(r["frame_index"])
print("\n=== objects_3d ===")
print("total", len(o3rows))
for ep, frames in sorted(o3_by_ep.items()):
    print(f"  {ep[:56]}  frames={len(frames)} boxes={sum(1 for x in o3rows if x['episode_id']==ep)}")

import os
print("\n=== mcap files ===")
for fn in sorted(os.listdir("/mnt/d/src/waymo2mcap/data/mcap")):
    if fn.endswith(".mcap"):
        p = os.path.join("/mnt/d/src/waymo2mcap/data/mcap", fn)
        print(f"  {os.path.getsize(p)/1e6:.0f} MB  {fn}")
