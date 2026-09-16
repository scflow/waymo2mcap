import lancedb

db = lancedb.connect("/mnt/d/src/waymo2mcap/data/lancedb")
cf = db.open_table("camera_frames")
total = cf.count_rows()
print("total camera_frames:", total)

null_count = (cf.search().where("image_embedding IS NULL").select(["episode_id"]).to_arrow().num_rows)
print("null embeddings count:", null_count)

row = cf.search().where("image_embedding IS NOT NULL").limit(1).select(["camera", "image_embedding"]).to_arrow().to_pylist()[0]
emb = row["image_embedding"]
print("sample emb type:", type(emb), "len:", len(emb))
