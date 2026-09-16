#!/usr/bin/env bash
set -euo pipefail
source /home/zr/anaconda3/etc/profile.d/conda.sh
conda activate viz
pip install 'fiftyone==1.22.0' lancedb pillow -q
python - <<'PY'
import fiftyone as fo
import lancedb
print("fo", fo.__version__)
print("lancedb", lancedb.__version__)
print("datasets", fo.list_datasets())
PY
