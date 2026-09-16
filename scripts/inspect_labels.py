import lancedb
db = lancedb.connect("/mnt/d/src/waymo2mcap/data/lancedb")
cf = db.open_table("camera_frames")
rows = cf.search().where("num_labels > 0").limit(3).to_arrow().to_pylist()
for r in rows:
    print(r["camera"], r["num_labels"], r["labels"][:2])
