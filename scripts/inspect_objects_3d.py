import lancedb

db = lancedb.connect("/mnt/d/src/waymo2mcap/data/lancedb")
o3 = db.open_table("objects_3d")
print("Total 3D objects in LanceDB:", o3.count_rows())

rows = o3.search().limit(3).to_arrow().to_pylist()
for r in rows:
    print(f"Episode: {r['episode_id'][:30]} | Frame: {r['frame_index']} | Label: {r['label']}")
    print(f"   pos: [{r['pos_x']:.2f}, {r['pos_y']:.2f}, {r['pos_z']:.2f}] | size: [{r['size_x']:.2f}, {r['size_y']:.2f}, {r['size_z']:.2f}] | quat: [{r['quat_x']:.2f}, {r['quat_y']:.2f}, {r['quat_z']:.2f}, {r['quat_w']:.2f}]")
