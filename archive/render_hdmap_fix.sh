#!/bin/bash
set -e
source /home/zr/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-drive-dreams
cd /mnt/d/src/Cosmos-Drive-Dreams/cosmos-drive-dreams-toolkits

INPUT=/mnt/d/src/waymo2mcap/data/waymo_rds_hq
OUTPUT=/mnt/d/src/Cosmos-Drive-Dreams/outputs/waymo_mv_hdmap
export USE_RAY=False

for CLIP in \
  10061305430875486848_1080_000_1100_000 \
  10072140764565668044_4060_000_4080_000
do
  echo "=== rendering $CLIP ==="
  python render_from_rds_hq.py \
    -i "$INPUT" \
    -o "$OUTPUT" \
    -d waymo_mv \
    -c pinhole \
    -s lidar \
    -s world_scenario \
    -cj "$CLIP"
done

echo "REPAIR_DONE"
bash /mnt/d/src/waymo2mcap/audit_hdmap.sh
