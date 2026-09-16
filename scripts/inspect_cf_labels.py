import lancedb

db = lancedb.connect("/mnt/d/src/waymo2mcap/data/lancedb")
cf = db.open_table("camera_frames")
print("Total rows:", cf.count_rows())

rows = cf.search().where("num_labels > 5 AND camera = 'FRONT'").limit(2).to_arrow().to_pylist()
for r in rows:
    print(f"Episode: {r['episode_id'][:30]} | Frame: {r['frame_index']} | Camera: {r['camera']} | Labels: {len(r['labels'])}")
    for lb in r["labels"][:3]:
        print("   ", lb)
