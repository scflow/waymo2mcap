import io
import time
import lancedb
import numpy as np
from PIL import Image, ImageDraw, ImageFont

db = lancedb.connect("/mnt/d/src/waymo2mcap/data/lancedb")
cf = db.open_table("camera_frames")

ep = "segment-10017090168044687777_6380_000_6400_000_with_camera_labels"
rows = (cf.search()
        .where(f"episode_id = '{ep}' AND frame_index = 0")
        .select(["camera", "image", "num_labels"])
        .to_arrow().to_pylist())

cams = {r["camera"]: r for r in rows}
print("found cameras:", list(cams.keys()))

t0 = time.time()
W, H = 480, 320
canvas = Image.new("RGB", (W * 3, H * 2), (18, 18, 22))
draw = ImageDraw.Draw(canvas)

# Cam placements:
# (0, 0): FRONT_LEFT, (1, 0): FRONT, (2, 0): FRONT_RIGHT
# (0, 1): SIDE_LEFT,  (1, 1): HUD,   (2, 1): SIDE_RIGHT
layout = {
    "FRONT_LEFT": (0, 0),
    "FRONT": (W, 0),
    "FRONT_RIGHT": (W * 2, 0),
    "SIDE_LEFT": (0, H),
    "SIDE_RIGHT": (W * 2, H),
}

for cam_name, (x, y) in layout.items():
    if cam_name in cams and cams[cam_name]["image"]:
        raw_img = Image.open(io.BytesIO(cams[cam_name]["image"]))
        resized = raw_img.resize((W, H), Image.BILINEAR)
        canvas.paste(resized, (x, y))
        # label
        draw.rectangle([x + 8, y + 8, x + 110, y + 26], fill=(0, 0, 0, 180))
        draw.text((x + 12, y + 10), cam_name.replace("_", " "), fill=(240, 240, 240))

# Render HUD in center-bottom (W, H)
hx, hy = W, H
draw.rectangle([hx + 4, hy + 4, hx + W - 4, hy + H - 4], fill=(24, 26, 32), outline=(50, 54, 66), width=2)
draw.text((hx + 20, hy + 20), "WAYMO EPISODE TELEMETRY", fill=(70, 130, 245))
draw.text((hx + 20, hy + 50), f"Episode: {ep[:32]}...", fill=(200, 200, 210))
draw.text((hx + 20, hy + 85), f"Frame: 0 / 198 (0.00s)", fill=(255, 255, 255))
draw.text((hx + 20, hy + 120), "Speed: 12.4 m/s (44.6 km/h)", fill=(100, 220, 150))
draw.text((hx + 20, hy + 155), "3D Objects: 28 detected", fill=(255, 180, 80))
draw.text((hx + 20, hy + 190), "LiDAR: 183,680 points active", fill=(180, 140, 255))
draw.text((hx + 20, hy + 235), "Foxglove Synchronized Surround View", fill=(120, 125, 140))

out_path = "/mnt/d/src/waymo2mcap/data/fiftyone_media/test/surround_frame_test.jpg"
canvas.save(out_path, "JPEG", quality=85)
print(f"Generated test frame in {time.time() - t0:.3f}s: {out_path}")
