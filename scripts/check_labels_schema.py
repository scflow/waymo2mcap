import lancedb

db = lancedb.connect("/mnt/d/src/waymo2mcap/data/lancedb")
cf = db.open_table("camera_frames")
row = (cf.search()
       .where("camera = 'FRONT' AND num_labels > 0")
       .limit(1)
       .select(["episode_id", "frame_index", "labels", "num_labels"])
       .to_arrow().to_pylist())[0]

print("labels count:", row["num_labels"])
print("sample label entry:", row["labels"][0] if row["labels"] else None)
print("all labels:", row["labels"])
