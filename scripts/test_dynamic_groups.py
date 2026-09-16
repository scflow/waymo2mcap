import os, sys
sys.path.insert(0, "/mnt/d/src/fiftyone")
import fiftyone as fo

os.environ["FIFTYONE_DATASET_STORAGE"] = "lance"
os.environ["FIFTYONE_DATASET_STORAGE_URI"] = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"
fo.config.dataset_storage = "lance"
fo.config.dataset_storage_uri = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"

ds = fo.load_dataset("waymo-fused-frames")
print("Dataset:", ds.name)
print("Count:", ds.count())

view = ds.group_by("episode_id", order_by="frame_index")
groups = list(view.iter_dynamic_groups())
print("Number of dynamic groups (episodes):", len(groups))
for g in groups:
    first_s = g.first()
    print(f"Episode: {first_s.episode_id[:35]} | frames count: {len(g)} | first speed: {first_s.speed_mps:.2f} m/s | 3D objects: {first_s.num_objects}")
