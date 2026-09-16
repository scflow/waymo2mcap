import io
import lancedb
from PIL import Image

db = lancedb.connect("/mnt/d/src/waymo2mcap/data/lancedb")
cf = db.open_table("camera_frames")
row = cf.search().limit(1).to_arrow().to_pylist()[0]
img_bytes = row["image"]
img = Image.open(io.BytesIO(img_bytes))
print("image size:", img.size, "format:", img.format, "mode:", img.mode)
