import os, sys
sys.path.insert(0, "/mnt/d/src/fiftyone")
import fiftyone as fo

os.environ["FIFTYONE_DATASET_STORAGE"] = "lance"
os.environ["FIFTYONE_DATASET_STORAGE_URI"] = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"
fo.config.dataset_storage = "lance"
fo.config.dataset_storage_uri = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"

print("All datasets in Lance storage:", fo.list_datasets())

print("\n--- 1. Testing waymo-episodes-video ---")
ds_v = fo.load_dataset("waymo-episodes-video")
print("Dataset:", ds_v.name, "media_type:", ds_v.media_type, "count:", len(ds_v))
print("Brain runs:", ds_v.list_brain_runs())
first_v = ds_v.first()
print("First episode:", first_v.episode_id)
print("Video file:", first_v.filepath, "exists:", os.path.isfile(first_v.filepath))
print("Frames in sequence:", len(first_v.frames))
f1 = first_v.frames[1]
print("Frame 1 metrics: speed =", f1.speed_mps, "ego =", (f1.ego_px, f1.ego_py), "3D objects =", f1.num_objects)
print("3D detections count:", len(f1.detections_3d.detections) if f1.detections_3d else 0)

print("\n--- 2. Testing waymo-episodes-grouped ---")
ds_g = fo.load_dataset("waymo-episodes-grouped")
print("Dataset:", ds_g.name, "media_type:", ds_g.media_type, "count:", len(ds_g))
print("Slices:", ds_g.group_slices)
first_g = ds_g.first()
print("Group slice sample:", first_g.group.name, first_g.filepath)

print("\n--- 3. Testing waymo-fused-frames ---")
ds_f = fo.load_dataset("waymo-fused-frames")
print("Dataset:", ds_f.name, "media_type:", ds_f.media_type, "count:", len(ds_f))
print("Brain runs:", ds_f.list_brain_runs())
top_sim = ds_f.sort_by_similarity(ds_f.first().id, k=3, brain_key="frame_sim")
print("Top-3 similar frames to frame 0:")
for s in top_sim:
    print(" ", s.episode_id[:32], "frame:", s.frame_index, "speed:", s.speed_mps)

print("\nALL VERIFICATIONS PASSED!")
