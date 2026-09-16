#!/bin/bash
set -e
source /home/zr/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-drive-dreams
cd /mnt/d/src/Cosmos-Drive-Dreams/cosmos-drive-dreams-toolkits

INPUT=/mnt/d/src/waymo2mcap/data/waymo_rds_hq
OUTPUT=/mnt/d/src/Cosmos-Drive-Dreams/outputs/waymo_mv_hdmap
CLIP=10017090168044687777_6380_000_6400_000

export USE_RAY=False
export TF_CPP_MIN_LOG_LEVEL=2

echo "=== single clip test: $CLIP ==="
python render_from_rds_hq.py \
  -i "$INPUT" \
  -o "$OUTPUT" \
  -d waymo_mv \
  -c pinhole \
  -s lidar \
  -s world_scenario \
  -cj "$CLIP"

echo "=== output tree ==="
find "$OUTPUT" -type f | head -50
