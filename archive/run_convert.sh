#!/bin/bash
set -e
cd /mnt/d/src/Cosmos-Drive-Dreams/cosmos-drive-dreams-toolkits
PY=/home/zr/anaconda3/envs/waymo/bin/python
INPUT=/mnt/d/src/waymo2mcap/data/tfrecord_clean
OUTPUT=/mnt/d/src/waymo2mcap/data/waymo_rds_hq
mkdir -p "$OUTPUT"
export TF_CPP_MIN_LOG_LEVEL=2
$PY convert_waymo_to_rds_hq.py \
  -i "$INPUT" \
  -o "$OUTPUT" \
  -n 3
echo "CONVERT_DONE"
ls -la "$OUTPUT"
