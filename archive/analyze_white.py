#!/bin/bash
source /home/zr/anaconda3/etc/profile.d/conda.sh
conda activate cosmos-drive-dreams
python - <<'PY'
import numpy as np
import imageio.v2 as imageio
from pathlib import Path
from collections import Counter

# sample frames from rendered hdmap
paths = [
    Path('/mnt/d/src/Cosmos-Drive-Dreams/outputs/waymo_mv_hdmap/hdmap/pinhole_front/10017090168044687777_6380_000_6400_000_0.mp4'),
    Path('/mnt/d/src/Cosmos-Drive-Dreams/outputs/waymo_mv_hdmap/hdmap/pinhole_front/10023947602400723454_1120_000_1140_000_0.mp4'),
]
cfg = {
    'lanelines': [98, 183, 249],
    'lanes': [56, 103, 221],
    'poles': [66, 40, 144],
    'road_boundaries': [200, 36, 35],
    'wait_lines': [185, 63, 34],
    'crosswalks': [206, 131, 63],
    'road_markings': [126, 204, 205],
    'traffic_signs': [131, 175, 155],
    'traffic_lights': [252, 157, 155],
    'Car': [255, 0, 0],
    'Truck': [0, 0, 255],
    'Pedestrian': [0, 255, 0],
    'Cyclist': [255, 255, 0],
    'Others': [255, 255, 255],
}

def nearest_name(rgb):
    rgb = np.array(rgb, dtype=float)
    best, bd = None, 1e9
    for k,v in cfg.items():
        d = np.linalg.norm(rgb - np.array(v, dtype=float))
        if d < bd:
            bd, best = d, k
    return best, bd

for p in paths:
    r = imageio.get_reader(p)
    for fi in [0, 30, 60]:
        f = r.get_data(fi)
        # non-black pixels
        mask = f.sum(axis=2) > 30
        pix = f[mask]
        # near-white
        white_mask = (f[:,:,0]>200)&(f[:,:,1]>200)&(f[:,:,2]>200)
        print(f'=== {p.name} frame {fi} ===')
        print('  nonblack', int(mask.sum()), 'near-white', int(white_mask.sum()))
        if white_mask.sum():
            wp = f[white_mask]
            print('  white mean', wp.mean(axis=0), 'min', wp.min(axis=0), 'max', wp.max(axis=0))
            # sample a few
            ys, xs = np.where(white_mask)
            print('  white sample coords', list(zip(ys[:5].tolist(), xs[:5].tolist())))
            print('  white sample rgb', wp[:5].tolist())
        # top colors
        q = (pix//32*32)
        keys = [tuple(x) for x in q]
        c = Counter(keys).most_common(8)
        print('  top colors (quantized):', c)
        for col,_ in c:
            name, d = nearest_name(col)
            print(f'    {col} -> {name} d={d:.1f}')
    r.close()
PY
