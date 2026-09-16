import os, sys
sys.path.insert(0, "/mnt/d/src/fiftyone")

os.environ["FIFTYONE_DATASET_STORAGE"] = "lance"
os.environ["FIFTYONE_DATASET_STORAGE_URI"] = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"

import fiftyone as fo
from fiftyone.server.query import Dataset
from fiftyone.server.paginator import _get_items_lance
from fiftyone.server.utils import from_dict

def from_db(doc: dict):
    doc = Dataset.modifier(doc)
    return from_dict(Dataset, doc)

try:
    res = _get_items_lance(from_db, "name", "", 20)
    print("res edges count:", len(res.edges))
    for e in res.edges:
        print("  dataset:", e.node.name, e.node.media_type)
except Exception as e:
    import traceback
    traceback.print_exc()
