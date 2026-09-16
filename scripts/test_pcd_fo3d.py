import os
import sys
import numpy as np
import fiftyone as fo
import fiftyone.core.threed as fo3d

pts = np.random.randn(1000, 4).astype(np.float32)
os.makedirs("/mnt/d/src/waymo2mcap/data/fiftyone_media/test", exist_ok=True)
pcd_path = "/mnt/d/src/waymo2mcap/data/fiftyone_media/test/test.pcd"

n_pts = len(pts)
header = (
    "# .PCD v0.7 - Point Cloud Data file format\n"
    "VERSION 0.7\n"
    "FIELDS x y z intensity\n"
    "SIZE 4 4 4 4\n"
    "TYPE F F F F\n"
    "COUNT 1 1 1 1\n"
    "WIDTH {}\n"
    "HEIGHT 1\n"
    "VIEWPOINT 0 0 0 1 0 0 0\n"
    "POINTS {}\n"
    "DATA binary\n"
).format(n_pts, n_pts).encode("ascii")

with open(pcd_path, "wb") as f:
    f.write(header)
    f.write(pts.tobytes())

print("wrote pcd:", pcd_path, "bytes:", os.path.getsize(pcd_path))

fo3d_path = "/mnt/d/src/waymo2mcap/data/fiftyone_media/test/test.fo3d"
scene = fo3d.Scene()
pcd_node = fo3d.PointCloud("lidar", pcd_path)
scene.add(pcd_node)
scene.write(fo3d_path)
print("wrote fo3d scene:", fo3d_path)
