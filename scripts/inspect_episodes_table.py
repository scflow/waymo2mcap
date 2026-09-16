import lancedb

db = lancedb.connect("/mnt/d/src/waymo2mcap/data/lancedb")
ep_tbl = db.open_table("episodes")
print("Total episodes in LanceDB:", ep_tbl.count_rows())

rows = ep_tbl.to_arrow().to_pylist()
for r in rows:
    dur = r.get('duration_s') or 0.0
    pts = r.get('num_lidar_points') or 0
    tc = str(r.get('topic_counts') or '')[:60]
    print(f"Episode: {r['episode_id']} | dur: {dur:.1f}s | frames: {r.get('frame_count')} | 3D objs: {r.get('num_objects_3d')} | pts: {pts:,} | topics: {tc}")
