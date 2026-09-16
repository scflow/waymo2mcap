import lancedb, json

db = lancedb.connect("/mnt/d/src/waymo2mcap/data/fo_lance_demo")
reg = db.open_table("_fo_registry")
rows = reg.to_arrow().to_pylist()
print("Registry rows count:", len(rows))
for r in rows:
    d = json.loads(r["doc_json"])
    print("Dataset:", d.get("name"), "sample_collection_name:", d.get("sample_collection_name"))
