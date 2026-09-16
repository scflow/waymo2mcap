#!/usr/bin/env python3
"""
Vector search demo on camera_frames.image_embedding (CLIP ViT-B/32, 512-d, cosine).

Demos:
  1. text -> image semantic search (needs transformers in clip env)
  2. image -> image reverse lookup (find near-duplicates of a query frame)
  3. hybrid: vector + structured filter (episode / camera / frame range)

Usage (clip env):
  python vector_search_demo.py [--db data/lancedb] [--export-dir data/lancedb_exports]
"""
import argparse
import io
import os

import lancedb
import numpy as np


def list_tables(db):
    t = db.list_tables()
    if hasattr(t, "tables"):
        return [x.name if hasattr(x, "name") else str(x) for x in t.tables]
    return [str(x) for x in t]


def export_jpeg(image_bytes, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(image_bytes)
    return path


def demo_text_search(table, export_dir, queries, k=5):
    """Text -> image via CLIP text encoder."""
    import torch
    from transformers import CLIPProcessor, CLIPTextModelWithProjection

    MODEL_ID = "openai/clip-vit-base-patch32"
    print(f"\n=== text->image search ({MODEL_ID}) ===")
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    # transformers 5.x: CLIPModel(**text_only) still enters the vision tower
    # and crashes on None pixel_values; use the text-only projection head.
    model = CLIPTextModelWithProjection.from_pretrained(MODEL_ID)
    model.eval()

    for q in queries:
        inputs = processor(text=[q], return_tensors="pt", padding=True)
        with torch.no_grad():
            emb = model(**inputs).text_embeds.numpy().astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        qvec = emb[0].tolist()

        res = (table.search(qvec, vector_column_name="image_embedding")
               .metric("cosine")
               .limit(k)
               .to_arrow())
        print(f"\nquery: {q!r}")
        for i in range(len(res)):
            row = res.slice(i, 1).to_pydict()
            dist = row.get("_distance", [None])[0]
            # cosine distance = 1 - cos_sim; lower is better
            sim = 1.0 - dist if dist is not None else None
            ep = row["episode_id"][0]
            fi = row["frame_index"][0]
            cam = row["camera"][0]
            print(f"  #{i+1} sim={sim:.4f}  {ep} frame={fi} cam={cam}")
            out = os.path.join(export_dir, "vector_text",
                               f"{q.replace(' ', '_')[:30]}_{i}.jpg")
            export_jpeg(row["image"][0], out)
        print(f"  exported to {export_dir}/vector_text/")


def demo_image_search(table, export_dir, k=5):
    """Image -> image: use row 0's embedding as query."""
    print("\n=== image->image reverse lookup ===")
    src = table.search().select(
        ["episode_id", "frame_index", "camera", "image", "image_embedding"]
    ).limit(1).to_arrow()
    qvec = src.column("image_embedding")[0].as_py()
    ep = src.column("episode_id")[0].as_py()
    fi = src.column("frame_index")[0].as_py()
    cam = src.column("camera")[0].as_py()
    print(f"query frame: {ep} frame={fi} cam={cam}")

    res = (table.search(qvec, vector_column_name="image_embedding")
           .metric("cosine")
           .limit(k + 1)
           .to_arrow())
    for i in range(len(res)):
        row = res.slice(i, 1).to_pydict()
        dist = row["_distance"][0]
        sim = 1.0 - dist
        same = (row["episode_id"][0] == ep
                and row["frame_index"][0] == fi
                and row["camera"][0] == cam)
        tag = " [SELF]" if same else ""
        print(f"  #{i+1} sim={sim:.4f}  {row['episode_id'][0]} "
              f"frame={row['frame_index'][0]} cam={row['camera'][0]}{tag}")
        if not same:
            out = os.path.join(export_dir, "vector_image", f"nn_{i}.jpg")
            export_jpeg(row["image"][0], out)
    print(f"  exported to {export_dir}/vector_image/")


def demo_hybrid(table, export_dir, k=5):
    """Vector + structured filter: same camera only."""
    print("\n=== hybrid: vector + camera=FRONT ===")
    src = table.search().where("camera = 'FRONT'").select(
        ["episode_id", "frame_index", "camera", "image", "image_embedding"]
    ).limit(1).to_arrow()
    qvec = src.column("image_embedding")[0].as_py()
    print(f"query: {src.column('episode_id')[0].as_py()} "
          f"frame={src.column('frame_index')[0].as_py()} cam=FRONT")

    res = (table.search(qvec, vector_column_name="image_embedding")
           .metric("cosine")
           .where("camera = 'FRONT'")
           .limit(k + 1)
           .to_arrow())
    for i in range(len(res)):
        row = res.slice(i, 1).to_pydict()
        sim = 1.0 - row["_distance"][0]
        print(f"  #{i+1} sim={sim:.4f}  {row['episode_id'][0]} "
              f"frame={row['frame_index'][0]} cam={row['camera'][0]}")
    print(f"  (hybrid filter applied: camera='FRONT')")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/lancedb")
    ap.add_argument("--export-dir", default="data/lancedb_exports")
    ap.add_argument("--skip-text", action="store_true",
                    help="skip text search (avoids loading CLIP text tower)")
    args = ap.parse_args()

    db = lancedb.connect(args.db)
    names = list_tables(db)
    print("tables:", names)
    table = db.open_table("camera_frames")
    fields = table.schema.names
    print("camera_frames fields:", fields)
    assert "image_embedding" in fields, "run embed_lancedb.py first"

    n = table.count_rows()
    print(f"rows: {n:,}")

    if not args.skip_text:
        demo_text_search(table, args.export_dir, [
            "a car driving on a highway",
            "pedestrian crossing the street",
            "bicycle rider in the bike lane",
            "nighttime urban intersection",
        ], k=5)
    demo_image_search(table, args.export_dir, k=5)
    demo_hybrid(table, args.export_dir, k=5)
    print("\nall vector demos done")


if __name__ == "__main__":
    main()
