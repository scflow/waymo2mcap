import os, sys
sys.path.insert(0, "/mnt/d/src/fiftyone")
import fiftyone as fo

os.environ["FIFTYONE_DATASET_STORAGE"] = "lance"
os.environ["FIFTYONE_DATASET_STORAGE_URI"] = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"
fo.config.dataset_storage = "lance"
fo.config.dataset_storage_uri = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"

for name in fo.list_datasets():
    print(f"\nTesting dataset: {name}")
    try:
        ds = fo.load_dataset(name)
        print("  Loaded:", ds.name, "| Media:", ds.media_type, "| Count:", ds.count())
        st = ds.stats()
        print("  Stats OK:", st.get("samples_count"))
    except Exception as e:
        print("  ERROR:", type(e).__name__, e)
