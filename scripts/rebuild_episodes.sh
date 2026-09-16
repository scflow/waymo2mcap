#!/usr/bin/env bash
set -euo pipefail

source /home/zr/anaconda3/etc/profile.d/conda.sh
conda activate viz

export FIFTYONE_DATASET_STORAGE=lance
export FIFTYONE_DATASET_STORAGE_URI=/mnt/d/src/waymo2mcap/data/fo_lance_demo
export PYTHONPATH=/mnt/d/src/fiftyone

cd /mnt/d/src/waymo2mcap

echo "=========================================================="
echo " STEP 1: Compute Dual-Level Embeddings in LanceDB"
echo "=========================================================="
python embed_episodes.py --compute

echo "=========================================================="
echo " STEP 2: Rebuild FiftyOne Episode Sequence Dataset"
echo "=========================================================="
python fo/rebuild_fo_episodes.py --video --grouped --brain

echo "=========================================================="
echo " STEP 3: Verify Datasets in Lance Storage"
echo "=========================================================="
python - <<'PY'
import os, sys
sys.path.insert(0, "/mnt/d/src/fiftyone")
import fiftyone as fo

os.environ["FIFTYONE_DATASET_STORAGE"] = "lance"
os.environ["FIFTYONE_DATASET_STORAGE_URI"] = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"
fo.config.dataset_storage = "lance"
fo.config.dataset_storage_uri = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"

print("Datasets available on Lance backend:", fo.list_datasets())

ds_vid = fo.load_dataset("waymo-episodes-video")
print("\n--- waymo-episodes-video ---")
print("Media type:", ds_vid.media_type)
print("Samples:", len(ds_vid))
print("Brain runs:", ds_vid.list_brain_runs())
first_s = ds_vid.first()
print("First sample ID:", first_s.id, "Episode:", first_s.episode_id)
print("Frames count:", len(first_s.frames))
f1 = first_s.frames[1]
print("Frame 1 fields:", list(f1.field_names))

ds_grp = fo.load_dataset("waymo-episodes-grouped")
print("\n--- waymo-episodes-grouped ---")
print("Media type:", ds_grp.media_type)
print("Group slices:", ds_grp.group_slices)
print("Samples:", len(ds_grp))
PY

echo "=========================================================="
echo " ALL EPISODE PIPELINE STEPS COMPLETED SUCCESSFULLY!"
echo "=========================================================="
