import os
import sys

sys.path.insert(0, "/mnt/d/src/fiftyone")
import fiftyone as fo

os.environ["FIFTYONE_DATASET_STORAGE"] = "lance"
os.environ["FIFTYONE_DATASET_STORAGE_URI"] = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"
fo.config.dataset_storage = "lance"
fo.config.dataset_storage_uri = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"

test_name = "_test_group_lance"
if fo.dataset_exists(test_name):
    fo.delete_dataset(test_name)

ds = fo.Dataset(test_name)
ds.add_group_field("group", default="video")
print("created grouped dataset:", ds, "group_field:", ds.group_field, "group_slices:", ds.group_slices)

# Add group samples
group = fo.Group()
s_video = fo.Sample(filepath="/mnt/d/src/waymo2mcap/data/test_dummy.mp4", group=group.element("video"))
s_3d = fo.Sample(filepath="/mnt/d/src/waymo2mcap/data/test_dummy.pcd", group=group.element("point_cloud"))
ds.add_samples([s_video, s_3d])

print("added group samples!")
ds_reloaded = fo.load_dataset(test_name)
print("reloaded group slices:", ds_reloaded.group_slices)
print("sample count:", len(ds_reloaded))

fo.delete_dataset(test_name)
print("cleaned up group test")
