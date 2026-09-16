import imageio.v3 as iio
import numpy as np
from PIL import Image

frame = Image.open("/mnt/d/src/waymo2mcap/data/fiftyone_media/test/surround_frame_test.jpg")
frame_arr = np.array(frame)

video_path = "/mnt/d/src/waymo2mcap/data/fiftyone_media/test/test_video.mp4"
writer = iio.imiter(video_path, plugin="pyav" if False else "ffmpeg", fps=10, codec="libx264")
# Or using imageio get_writer:
import imageio
w = imageio.get_writer(video_path, fps=10, codec="libx264", quality=8, pixelformat="yuv420p")
for _ in range(20):
    w.append_data(frame_arr)
w.close()
import os
print("Wrote test video:", video_path, "size:", os.path.getsize(video_path))
