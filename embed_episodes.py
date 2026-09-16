#!/usr/bin/env python3
"""
Dual-Level Multimodal Embedding Engine for Waymo Episodes & Frames.

Computes:
1. Frame-level 5-Camera Spatial Fusion Embeddings (360° surround view)
   - Direction-weighted fusion of FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT
   - Normalized 512-d CLIP vector per frame
   - Written to LanceDB table: `frame_fused_embeddings`
2. Episode-level Temporal Event Embeddings
   - Temporal sequence pooling of frame vectors across each full episode
   - Normalized 512-d vector per episode
   - Written to LanceDB `episodes` table (as `episode_embedding`)
3. Natural Language Search CLI
   - Text -> Episode search (retrieves full driving scenarios)
   - Text -> Frame search (retrieves exact 360° driving moments)

Usage:
  # Compute and save embeddings into LanceDB:
  python embed_episodes.py --compute

  # Search episodes with text query (requires clip env):
  python embed_episodes.py --query "busy intersection with pedestrians" --target episode
  python embed_episodes.py --query "high speed driving" --target frame
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import lancedb
import numpy as np
import pyarrow as pa

DB_PATH = "/mnt/d/src/waymo2mcap/data/lancedb"
CAMS_WEIGHTS = {
    "FRONT": 0.36,
    "FRONT_LEFT": 0.16,
    "FRONT_RIGHT": 0.16,
    "SIDE_LEFT": 0.16,
    "SIDE_RIGHT": 0.16,
}


def normalize(v):
    norm = np.linalg.norm(v)
    if norm > 1e-6:
        return v / norm
    return v


def compute_all_embeddings(db_path=DB_PATH):
    print(f"Connecting to LanceDB at {db_path}...")
    db = lancedb.connect(db_path)

    # 1. Fetch camera frames
    cf = db.open_table("camera_frames")
    print("Reading camera_frames embeddings...")
    t0 = time.time()
    rows = (cf.search()
            .where("image_embedding IS NOT NULL")
            .select(["episode_id", "frame_index", "camera", "timestamp_ns", "image_embedding"])
            .to_arrow().to_pylist())
    print(f"Read {len(rows)} camera embeddings in {time.time() - t0:.2f}s")

    # Group by (episode_id, frame_index)
    frames_dict = defaultdict(dict)
    ts_dict = {}
    for r in rows:
        key = (r["episode_id"], r["frame_index"])
        frames_dict[key][r["camera"]] = np.array(r["image_embedding"], dtype=np.float32)
        if key not in ts_dict:
            ts_dict[key] = r["timestamp_ns"]

    # 2. Compute Frame-level fused embeddings
    print("Computing 5-camera spatial fusion embeddings...")
    fused_rows = []
    episodes_frames = defaultdict(list)

    for (ep_id, fi), cam_map in sorted(frames_dict.items()):
        fused_vec = np.zeros(512, dtype=np.float32)
        total_w = 0.0

        for cam, weight in CAMS_WEIGHTS.items():
            if cam in cam_map:
                fused_vec += weight * cam_map[cam]
                total_w += weight

        if total_w > 0:
            fused_vec /= total_w
            fused_vec = normalize(fused_vec)

        fused_rows.append({
            "episode_id": ep_id,
            "frame_index": fi,
            "timestamp_ns": ts_dict[(ep_id, fi)],
            "vector": fused_vec.tolist(),
        })
        episodes_frames[ep_id].append((fi, fused_vec))

    print(f"Fused {len(fused_rows)} multi-camera frame embeddings.")

    # 3. Save to LanceDB table: frame_fused_embeddings
    print("Saving table 'frame_fused_embeddings' to LanceDB...")
    table_names = db.list_tables()
    if hasattr(table_names, "tables"):
        table_names = table_names.tables
    names = [n if isinstance(n, str) else n[0] for n in table_names]

    if "frame_fused_embeddings" in names:
        db.drop_table("frame_fused_embeddings")

    schema = pa.schema([
        pa.field("episode_id", pa.string()),
        pa.field("frame_index", pa.int32()),
        pa.field("timestamp_ns", pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), 512)),
    ])

    pa_table = pa.Table.from_pylist(fused_rows, schema=schema)
    fused_table = db.create_table("frame_fused_embeddings", pa_table)
    print("Created 'frame_fused_embeddings' table with rows:", fused_table.count_rows())

    # Build vector index on frame_fused_embeddings
    try:
        print("Building IVF-FLAT vector index on frame_fused_embeddings...")
        fused_table.create_index(
            metric="cosine",
            vector_column_name="vector",
            index_type="IVF_FLAT"
        )
        print("Vector index built successfully.")
    except Exception as e:
        print("Note on index building:", e)

    # 4. Compute Episode-level Temporal Embeddings
    print("\nComputing Episode-level temporal embeddings...")
    ep_rows = db.open_table("episodes").to_arrow().to_pylist()
    updated_ep_rows = []

    for ep in ep_rows:
        ep_id = ep["episode_id"]
        frame_list = episodes_frames.get(ep_id, [])
        if frame_list:
            frame_vecs = np.array([vec for _, vec in frame_list], dtype=np.float32)
            # Temporal pooling (average over sequence)
            ep_vec = np.mean(frame_vecs, axis=0)
            ep_vec = normalize(ep_vec)
        else:
            ep_vec = np.zeros(512, dtype=np.float32)

        ep_dict = dict(ep)
        ep_dict["episode_embedding"] = ep_vec.tolist()
        updated_ep_rows.append(ep_dict)
        print(f"  Episode: {ep_id[:45]}... frames={len(frame_list)} emb_norm={np.linalg.norm(ep_vec):.4f}")

    # 5. Update LanceDB `episodes` table
    print("Updating 'episodes' table with episode_embedding column...")
    if "episodes" in names:
        db.drop_table("episodes")

    ep_schema = pa.schema([
        pa.field("episode_id", pa.string()),
        pa.field("mcap_path", pa.string()),
        pa.field("frame_count", pa.int32()),
        pa.field("duration_s", pa.float64()),
        pa.field("num_images", pa.int32()),
        pa.field("num_lidar_points", pa.int64()),
        pa.field("num_objects_3d", pa.int32()),
        pa.field("source_type", pa.string()),
        pa.field("episode_embedding", pa.list_(pa.float32(), 512)),
    ])
    ep_table = db.create_table("episodes", pa.Table.from_pylist(updated_ep_rows, schema=ep_schema))
    print(f"Saved {ep_table.count_rows()} episodes with embeddings!")

    print("\nSUCCESS: All dual-level embeddings computed and persisted in LanceDB.")


def get_clip_text_embedding(text):
    """Encode text into 512-d CLIP vector (requires clip env)."""
    import torch
    from transformers import CLIPProcessor, CLIPTextModelWithProjection

    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPTextModelWithProjection.from_pretrained(model_name)
    model.eval()

    inputs = processor(text=[text], padding=True, return_tensors="pt")
    with torch.no_grad():
        emb = model(**inputs).text_embeds.cpu().numpy().astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return emb[0].tolist()


def search(query_text, target="episode", top_k=5, db_path=DB_PATH):
    print(f"\nSearching for '{query_text}' (target: {target})...")
    query_vec = get_clip_text_embedding(query_text)

    db = lancedb.connect(db_path)
    if target == "episode":
        tbl = db.open_table("episodes")
        results = (tbl.search(query_vec, vector_column_name="episode_embedding")
                   .metric("cosine")
                   .where("source_type = 'perception'")
                   .limit(top_k)
                   .to_arrow().to_pylist())

        print(f"\n--- TOP {top_k} EPISODE MATCHES (COSINE SIMILARITY) ---")
        for i, r in enumerate(results, 1):
            dist = r.get("_distance", 0.0)
            sim = 1.0 - dist
            print(f"[{i}] Cosine Sim: {sim:.4f}  |  Episode: {r['episode_id']}")
            print(f"    Frames: {r['frame_count']}  |  3D Boxes: {r['num_objects_3d']}  |  Duration: {r['duration_s']:.1f}s")
    else:
        tbl = db.open_table("frame_fused_embeddings")
        results = (tbl.search(query_vec, vector_column_name="vector")
                   .metric("cosine")
                   .limit(top_k)
                   .to_arrow().to_pylist())

        print(f"\n--- TOP {top_k} FRAME MATCHES (360° SURROUND) ---")
        for i, r in enumerate(results, 1):
            dist = r.get("_distance", 0.0)
            sim = 1.0 - dist
            t_sec = r['frame_index'] * 0.1
            print(f"[{i}] Cosine Sim: {sim:.4f}  |  Frame: {r['frame_index']:03d} ({t_sec:.1f}s)")
            print(f"    Episode: {r['episode_id']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", action="store_true", help="compute and save embeddings")
    parser.add_argument("--query", type=str, default=None, help="text search query")
    parser.add_argument("--target", choices=["episode", "frame"], default="episode", help="search target")
    parser.add_argument("--top-k", type=int, default=5, help="number of results")
    args = parser.parse_args()

    if args.compute or not args.query:
        compute_all_embeddings()

    if args.query:
        search(args.query, target=args.target, top_k=args.top_k)


if __name__ == "__main__":
    main()
