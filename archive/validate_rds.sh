#!/bin/bash
set -e
OUT=/mnt/d/src/waymo2mcap/data/waymo_rds_hq
PY=/home/zr/anaconda3/envs/waymo/bin/python
echo "=== disk usage ==="
du -sh "$OUT"
du -sh "$OUT"/*
echo "=== sample clip contents ==="
CLIP=10017090168044687777_6380_000_6400_000
$PY - <<'PY'
from pathlib import Path
from webdataset import WebDataset, non_empty
root = Path('/mnt/d/src/waymo2mcap/data/waymo_rds_hq')
clip = '10017090168044687777_6380_000_6400_000'

def peek(tar):
    url = tar.as_posix()
    ds = WebDataset(url, nodesplitter=non_empty, workersplitter=None, shardshuffle=False)
    s = next(iter(ds))
    keys = list(s.keys())
    print(tar.name, 'keys sample:', keys[:8], '... total', len(keys))
    return s

s = peek(root/'pinhole_intrinsic'/f'{clip}.tar')
print('  intrinsic front:', s.get('pinhole_intrinsic.front.npy'))
s = peek(root/'pose'/f'{clip}.tar')
s2 = peek(root/'lidar_raw'/f'{clip}.tar')
s3 = peek(root/'timestamp'/f'{clip}.tar')
s4 = peek(root/'3d_lanes'/f'{clip}.tar')
print('VALIDATION_OK')
PY
