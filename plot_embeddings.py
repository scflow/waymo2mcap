#!/usr/bin/env python3
"""
Render a standalone 2D scatter of the CLIP embeddings.

PCA (no extra deps beyond numpy) → HTML with plotly if available,
else a self-contained SVG. Colored by camera, hover shows frame.

Run (viz env):
  python plot_embeddings.py
"""

import os
import sys

sys.path.insert(0, "/mnt/d/src/fiftyone")

import numpy as np

os.environ["FIFTYONE_DATASET_STORAGE"] = "lance"
os.environ["FIFTYONE_DATASET_STORAGE_URI"] = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"

import fiftyone as fo

fo.config.dataset_storage = "lance"
fo.config.dataset_storage_uri = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"

OUT = "/mnt/d/src/waymo2mcap/data/lancedb_exports/embedding_scatter.html"
OUT_SVG = "/mnt/d/src/waymo2mcap/data/lancedb_exports/embedding_scatter.svg"


def pca_2d(X):
    X = X - X.mean(axis=0)
    # SVD is fine for 400x512
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :2] * S[:2]


def main():
    ds = fo.load_dataset("waymo-curated")
    ids, cams, frames, eps, vecs = [], [], [], [], []
    for s in ds.iter_samples():
        try:
            e = s["embedding"]
        except Exception:
            e = None
        if e is None:
            continue
        ids.append(s.id)
        cams.append(s.camera or "?")
        frames.append(s.frame_index)
        eps.append((s.episode_id or "")[:40])
        vecs.append(e)

    X = np.asarray(vecs, dtype=np.float64)
    print("loaded", X.shape)
    xy = pca_2d(X)
    # variance explained
    Xc = X - X.mean(axis=0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    ev = sv ** 2 / (sv ** 2).sum()
    print("PCA var explained: %.1f%% / %.1f%%" % (ev[0] * 100, ev[1] * 100))

    uniq = sorted(set(cams))
    palette = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    ]
    color_of = {c: palette[i % len(palette)] for i, c in enumerate(uniq)}

    # --- try plotly ---
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        for cam in uniq:
            idx = [i for i, c in enumerate(cams) if c == cam]
            fig.add_trace(go.Scatter(
                x=xy[idx, 0], y=xy[idx, 1],
                mode="markers",
                name=cam,
                marker=dict(size=7, color=color_of[cam], opacity=0.85),
                text=[
                    "cam=%s<br>frame=%d<br>ep=%s<br>id=%s" % (
                        cams[i], frames[i], eps[i], ids[i][:12])
                    for i in idx
                ],
                hoverinfo="text",
            ))
        fig.update_layout(
            title="waymo-curated · CLIP ViT-B/32 embeddings (PCA)<br>"
                  "<sup>%d samples · var explained %.1f%% + %.1f%%</sup>"
                  % (len(ids), ev[0] * 100, ev[1] * 100),
            xaxis_title="PC1", yaxis_title="PC2",
            template="plotly_dark",
            width=960, height=700,
        )
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        fig.write_html(OUT, include_plotlyjs="cdn")
        print("wrote", OUT)
        return
    except ImportError:
        print("plotly not installed — falling back to SVG")

    # --- SVG fallback ---
    os.makedirs(os.path.dirname(OUT_SVG), exist_ok=True)
    W, H, PAD = 960, 700, 50
    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    y0, y1 = xy[:, 1].min(), xy[:, 1].max()
    sx = (W - 2 * PAD) / max(x1 - x0, 1e-9)
    sy = (H - 2 * PAD) / max(y1 - y0, 1e-9)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
        'style="background:#111">' % (W, H),
        '<text x="%d" y="28" fill="#eee" font-size="16" '
        'font-family="sans-serif">waymo-curated · CLIP embeddings (PCA)</text>'
        % PAD,
    ]
    for i in range(len(ids)):
        cx = PAD + (xy[i, 0] - x0) * sx
        cy = H - PAD - (xy[i, 1] - y0) * sy
        parts.append(
            '<circle cx="%.1f" cy="%.1f" r="4" fill="%s" opacity="0.85">'
            '<title>%s frame=%d</title></circle>'
            % (cx, cy, color_of[cams[i]], cams[i], frames[i])
        )
    # legend
    for j, cam in enumerate(uniq):
        parts.append(
            '<circle cx="%d" cy="%d" r="5" fill="%s"/>'
            '<text x="%d" y="%d" fill="#ccc" font-size="12" '
            'font-family="sans-serif">%s</text>'
            % (W - 140, 50 + j * 22, color_of[cam], W - 128, 54 + j * 22, cam)
        )
    parts.append("</svg>")
    with open(OUT_SVG, "w") as f:
        f.write("\n".join(parts))
    print("wrote", OUT_SVG)


if __name__ == "__main__":
    main()
