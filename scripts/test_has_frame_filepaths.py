import os, sys
sys.path.insert(0, "/mnt/d/src/fiftyone")
import fiftyone as fo

os.environ["FIFTYONE_DATASET_STORAGE"] = "lance"
os.environ["FIFTYONE_DATASET_STORAGE_URI"] = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"
fo.config.dataset_storage = "lance"
fo.config.dataset_storage_uri = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"

ds = fo.load_dataset("waymo-episodes-video")
print("has_frame_filepaths before:", ds._doc.has_frame_filepaths)

ds._doc.has_frame_filepaths = True
ds.save()
print("has_frame_filepaths after save:", ds._doc.has_frame_filepaths)

try:
    frames = ds.to_frames(sample_frames=False)
    print("to_frames count:", len(frames))
    f_first = frames.first()
    print("first frame:", f_first)
except Exception as e:
    import traceback
    traceback.print_exc()
