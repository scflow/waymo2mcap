import lancedb, io
from PIL import Image

db = lancedb.connect("/mnt/d/src/waymo2mcap/data/lancedb")
cf = db.open_table("camera_frames")
cams = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
for c in cams:
    row = cf.search().where(f"camera = '{c}'").limit(1).to_arrow().to_pylist()[0]
    img = Image.open(io.BytesIO(row["image"]))
    print(f"Camera {c}: size = {img.size}")
