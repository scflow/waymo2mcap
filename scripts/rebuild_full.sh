#!/usr/bin/env bash
set -euo pipefail
source /home/zr/anaconda3/etc/profile.d/conda.sh
conda activate viz
export FIFTYONE_DATASET_STORAGE=lance
export FIFTYONE_DATASET_STORAGE_URI=/mnt/d/src/waymo2mcap/data/fo_lance_demo
export PYTHONPATH=/mnt/d/src/fiftyone
cd /mnt/d/src/waymo2mcap

python - <<'PY'
import sys
sys.path.insert(0, "/mnt/d/src/fiftyone")
import fiftyone as fo
fo.config.dataset_storage = "lance"
fo.config.dataset_storage_uri = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"
# drop the MCAP-backed dataset — LanceDB is the store
if fo.dataset_exists("waymo-episodes"):
    fo.delete_dataset("waymo-episodes")
    print("dropped waymo-episodes")
print("datasets:", fo.list_datasets())
PY

echo "=== full rebuild from LanceDB ==="
python fo/rebuild_fo_full.py --frames --brain
