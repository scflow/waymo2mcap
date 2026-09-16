import os
import sys

sys.path.insert(0, "/mnt/d/src/fiftyone")
import fiftyone as fo

os.environ["FIFTYONE_DATASET_STORAGE"] = "lance"
os.environ["FIFTYONE_DATASET_STORAGE_URI"] = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"
fo.config.dataset_storage = "lance"
fo.config.dataset_storage_uri = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"

print("dataset_storage:", fo.config.dataset_storage)

test_name = "_test_video_lance"
if fo.dataset_exists(test_name):
    fo.delete_dataset(test_name)

ds = fo.Dataset(test_name)
ds.media_type = "video"
print("created dataset:", ds, "media_type:", ds.media_type)

# Try adding a dummy video sample with frames
s = fo.Sample(filepath="/mnt/d/src/waymo2mcap/data/test_dummy.mp4")
s["episode_id"] = "test_ep_1"
ds.add_sample(s)
print("added sample:", s.id)

# Try accessing frames
try:
    print("sample frames:", s.frames)
    s.frames[1]["test_field"] = "val_1"
    s.frames[2]["test_field"] = "val_2"
    s.save()
    print("saved sample with frames successfully!")
    
    # Reload and verify
    ds_reloaded = fo.load_dataset(test_name)
    s_reloaded = ds_reloaded.first()
    print("reloaded sample frames count:", len(s_reloaded.frames))
    print("frame 1:", s_reloaded.frames[1])
except Exception as e:
    import traceback
    print("Frames error:")
    traceback.print_exc()

fo.delete_dataset(test_name)
print("cleaned up")
