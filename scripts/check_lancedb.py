#!/usr/bin/env python3
"""Quick health check of the LanceDB tables."""
import lancedb

DB = "/mnt/d/src/waymo2mcap/data/lancedb"
db = lancedb.connect(DB)

tables = db.list_tables()
if hasattr(tables, "tables"):
    names = [t.name if hasattr(t, "name") else str(t) for t in tables.tables]
else:
    names = [str(t) for t in tables]
print("tables:", names)

for name in ["episodes", "camera_frames", "lidar_frames",
             "ego_poses", "objects_3d", "map_features"]:
    try:
        t = db.open_table(name)
        print(f"  {name}: {t.count_rows():,} rows  fields={len(t.schema.names)}")
    except Exception as e:
        print(f"  {name}: MISSING ({e})")

cam = db.open_table("camera_frames")
print("\ncamera_frames schema:", cam.schema.names)
print("has image_embedding:", "image_embedding" in cam.schema.names)

# map features layer breakdown
mf = db.open_table("map_features")
tbl = mf.search().select(["layer", "feature_id", "episode_id"]).limit(5).to_arrow()
print("\nmap_features sample:", tbl.to_pydict())

# count by layer
full = mf.search().select(["layer"]).to_arrow()
print("map_features layer counts:")
import collections
print(collections.Counter(full.column("layer").to_pylist()))
