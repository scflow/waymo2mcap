#!/bin/bash
set -e
source /home/zr/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-drive-dreams
cd /mnt/d/src/Cosmos-Drive-Dreams/cosmos-drive-dreams-toolkits

INPUT=/mnt/d/src/waymo2mcap/data/waymo_rds_hq
OUTPUT=/mnt/d/src/Cosmos-Drive-Dreams/outputs/waymo_mv_hdmap

# Ray for multi-clip parallelism
export USE_RAY=True

python render_from_rds_hq.py \
  -i "$INPUT" \
  -o "$OUTPUT" \
  -d waymo_mv \
  -c pinhole \
  -s lidar \
  -s world_scenario

echo "RENDER_DONE"
echo "=== counts ==="
for d in "$OUTPUT"/hdmap/*/; do
  n=$(ls "$d" 2>/dev/null | wc -l)
  echo "$d : $n"
done
