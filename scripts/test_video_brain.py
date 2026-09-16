import os
import sys

sys.path.insert(0, "/mnt/d/src/fiftyone")
import fiftyone as fo
import fiftyone.brain as fob
import numpy as np

os.environ["FIFTYONE_DATASET_STORAGE"] = "lance"
os.environ["FIFTYONE_DATASET_STORAGE_URI"] = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"
fo.config.dataset_storage = "lance"
fo.config.dataset_storage_uri = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"

test_name = "_test_video_brain"
if fo.dataset_exists(test_name):
    fo.delete_dataset(test_name)

ds = fo.Dataset(test_name)
ds.media_type = "video"

# 2 sample videos
s1 = fo.Sample(filepath="/mnt/d/src/waymo2mcap/data/v1.mp4", episode_id="ep1")
s2 = fo.Sample(filepath="/mnt/d/src/waymo2mcap/data/v2.mp4", episode_id="ep2")
ds.add_samples([s1, s2])

# Add frame data
for s in ds:
    for f in range(1, 4):
        s.frames[f]["speed"] = f * 2.0
    s.save()

print("dataset prepared:", ds)

# Test Episode-level embedding
emb_samples = np.random.randn(2, 512).astype(np.float32)
emb_samples /= np.linalg.norm(emb_samples, axis=1, keepdims=True)

fob.compute_similarity(
    ds,
    embeddings=emb_samples,
    backend="lancedb",
    brain_key="ep_sim",
    table_name="_test_ep_sim",
    uri="/mnt/d/src/waymo2mcap/data/fo_lance_demo",
    metric="cosine"
)
print("sample similarity attached!")

# Test Frame-level view
frames_view = ds.to_frames()
print("frames view:", len(frames_view), "frames")

fo.delete_dataset(test_name)
print("cleaned up successfully")
