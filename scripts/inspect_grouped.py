import os, sys
sys.path.insert(0, "/mnt/d/src/fiftyone")
import fiftyone as fo

os.environ["FIFTYONE_DATASET_STORAGE"] = "lance"
os.environ["FIFTYONE_DATASET_STORAGE_URI"] = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"
fo.config.dataset_storage = "lance"
fo.config.dataset_storage_uri = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"

ds = fo.load_dataset("waymo-episodes-grouped")
print("sample_collection_name:", ds._sample_collection_name)
print("frame_collection_name:", ds._frame_collection_name)
print("media_type:", ds.media_type)
print("group_slices:", ds.group_slices)
print("group_field:", ds.group_field)
