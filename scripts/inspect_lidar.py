import lancedb
import numpy as np

db = lancedb.connect("/mnt/d/src/waymo2mcap/data/lancedb")
lf = db.open_table("lidar_frames")
print("schema:", lf.schema)
row = lf.search().limit(1).to_arrow().to_pylist()[0]
print("keys:", row.keys())
print("episode_id:", row["episode_id"], "frame_index:", row["frame_index"], "timestamp_ns:", row["timestamp_ns"])
pts_bytes = row["points"]
print("points length bytes:", len(pts_bytes))
# 20 bytes per point: x, y, z, intensity, elongation (float32)
pts = np.frombuffer(pts_bytes, dtype=np.float32).reshape(-1, 5)
print("points shape:", pts.shape)
print("sample point [x, y, z, intensity, elongation]:", pts[0])
print("min:", pts[:, :3].min(axis=0), "max:", pts[:, :3].max(axis=0))
