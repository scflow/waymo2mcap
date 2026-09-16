import lancedb
import numpy as np

db = lancedb.connect("/mnt/d/src/waymo2mcap/data/lancedb")
tbl = db.open_table("episodes")
print("schema:", tbl.schema)

vec = np.random.randn(512).astype(np.float32)
print("vec shape:", vec.shape)

res = tbl.search(vec, vector_column_name="episode_embedding").limit(3).to_arrow().to_pylist()
print("result count:", len(res))
for r in res:
    print(r["episode_id"], r["_distance"])
