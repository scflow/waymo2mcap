import os, sys
sys.path.insert(0, "/mnt/d/src/fiftyone")
import fiftyone as fo

os.environ["FIFTYONE_DATASET_STORAGE"] = "lance"
os.environ["FIFTYONE_DATASET_STORAGE_URI"] = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"
fo.config.dataset_storage = "lance"
fo.config.dataset_storage_uri = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"

ds = fo.load_dataset("waymo-episodes-video")
print("dataset:", ds)
print("sample count:", len(ds))

s = ds.first()
print("sample frames:", len(s.frames))
f1 = s.frames[1]
print("f1 keys:", list(f1.field_names))
print("f1 filepath:", f1.filepath)

# Test to_frames
try:
    v1 = ds.to_frames(sample_frames=False)
    print("to_frames(sample_frames=False):", len(v1))
except Exception as e:
    print("v1 error:", e)

try:
    v2 = ds.to_frames(sample_frames=True)
    print("to_frames(sample_frames=True):", len(v2))
except Exception as e:
    print("v2 error:", e)
