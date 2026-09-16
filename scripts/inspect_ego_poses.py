import lancedb

db = lancedb.connect("/mnt/d/src/waymo2mcap/data/lancedb")
ep_tbl = db.open_table("ego_poses")
print("Total ego poses in LanceDB:", ep_tbl.count_rows())

rows = ep_tbl.search().limit(3).to_arrow().to_pylist()
for r in rows:
    print(f"Episode: {r['episode_id'][:30]} | Frame: {r['frame_index']} | speed: {r['speed_mps']:.2f} m/s ({r['speed_mps']*3.6:.1f} km/h)")
    print(f"   pos: [{r['px']:.2f}, {r['py']:.2f}, {r['pz']:.2f}] | vel: [{r['vx']:.2f}, {r['vy']:.2f}, {r['vz']:.2f}]")
