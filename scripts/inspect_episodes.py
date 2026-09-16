import os, sys
sys.path.insert(0, "/mnt/d/src/fiftyone")
import fiftyone as fo

os.environ["FIFTYONE_DATASET_STORAGE"] = "lance"
os.environ["FIFTYONE_DATASET_STORAGE_URI"] = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"
fo.config.dataset_storage = "lance"
fo.config.dataset_storage_uri = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"

ds = fo.load_dataset("waymo-episodes-video")
print("Dataset:", ds.name)
print("Samples count:", ds.count())

for s in ds:
    print(f"Sample: {s.id} | Episode: {s.episode_id[:35]} | Duration: {s.duration_s}s | Frames: {len(s.frames)}")
    f1 = s.frames[1]
    det2d = f1.get_field("detections_2d")
    kp = f1.get_field("keypoints")
    det3d = f1.get_field("detections_3d")
    print(f"   Frame 1: 2D Boxes={len(det2d.detections) if det2d else 0}, Keypoints={len(kp.keypoints[0].points) if kp and kp.keypoints else 0}, 3D Boxes={len(det3d.detections) if det3d else 0}")
