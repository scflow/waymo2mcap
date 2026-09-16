#!/usr/bin/env python3
"""
Compute CLIP image embeddings for the camera_frames table and append them as
an `image_embedding` vector column (Lance schema evolution via add_columns).

Runs in the `clip` conda environment (torch CPU + transformers); the Lance
files stay fully compatible with the `waymo` environment for querying.

Pipeline:
  1. read camera_frames rows in physical order (episode, frame, camera)
  2. decode JPEG -> CLIP preprocess -> ViT-B/32 image embedding (512-d)
  3. L2-normalize, write back with lance.dataset.add_columns
  4. (default on) build an IVF_FLAT cosine vector index

Usage:
  anaconda3/envs/clip/bin/python embed_lancedb.py [--db data/lancedb] \
      [--limit N] [--batch 32] [--no-index] [--force]
"""

import argparse
import io
import time

import lance
import numpy as np
import pyarrow as pa
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

MODEL_ID = "openai/clip-vit-base-patch32"
EMBED_DIM = 512
NEW_SCHEMA = pa.schema([
    pa.field("image_embedding", pa.list_(pa.float32(), EMBED_DIM))
])


def compute_embeddings(dataset, batch_size, total, model, processor):
    """Precompute L2-normalized CLIP embeddings in scan order.

    Must finish reading before add_columns — LanceDataset cannot be
    scanned while a mutable add_columns borrow is held.
    """
    all_emb = np.empty((total, EMBED_DIM), dtype=np.float32)
    offset = 0
    while offset < total:
        n = min(batch_size, total - offset)
        t0 = time.time()
        tbl = dataset.scanner(columns=["image"], offset=offset, limit=n).to_table()
        images = [
            Image.open(io.BytesIO(j.as_py())).convert("RGB")
            for j in tbl.column("image")
        ]
        inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            emb = model(**inputs).image_embeds.numpy().astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)  # L2 -> cosine
        all_emb[offset:offset + n] = emb
        offset += n
        print(f"  embedded {offset:,}/{total:,} ({n / (time.time() - t0):.1f} img/s)",
              flush=True)
    return all_emb


def embeddings_to_reader(emb, chunk=1024):
    """Wrap a (N, EMBED_DIM) array into a RecordBatchReader for add_columns."""
    def gen():
        for i in range(0, len(emb), chunk):
            block = emb[i:i + chunk]
            yield pa.RecordBatch.from_arrays(
                [pa.FixedSizeListArray.from_arrays(
                    pa.array(block.reshape(-1), type=pa.float32()), EMBED_DIM)],
                schema=NEW_SCHEMA)
    return pa.RecordBatchReader.from_batches(NEW_SCHEMA, gen())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/lancedb")
    parser.add_argument("--table", default="camera_frames")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--no-index", action="store_true",
                        help="skip building the vector index")
    parser.add_argument("--force", action="store_true",
                        help="drop and recompute an existing embedding column")
    args = parser.parse_args()

    reader_path = f"{args.db}/{args.table}.lance"
    dataset = lance.dataset(reader_path)
    has_col = "image_embedding" in dataset.schema.names

    if has_col and not args.force:
        print("image_embedding already present; use --force to recompute")
        return
    if has_col and args.force:
        print("dropping existing image_embedding column...")
        dataset = dataset.drop_columns(["image_embedding"])

    print(f"loading {MODEL_ID} ...")
    processor = CLIPImageProcessor.from_pretrained(MODEL_ID)
    model = CLIPVisionModelWithProjection.from_pretrained(MODEL_ID)
    model.eval()

    total = dataset.count_rows()
    if args.limit:
        total = min(total, args.limit)
    print(f"rows: {total:,} | batch: {args.batch} | device: cpu")
    t0 = time.time()

    emb = compute_embeddings(dataset, args.batch, total, model, processor)
    print(f"compute done in {time.time() - t0:.0f}s; writing column...")
    dataset.add_columns(embeddings_to_reader(emb))
    print(f"embedding column written in {time.time() - t0:.0f}s total")

    if args.no_index:
        return

    import lancedb
    table = lancedb.connect(args.db).open_table(args.table)
    n = table.count_rows()
    print(f"building IVF_FLAT cosine index on image_embedding ({n:,} vectors)...")
    table.create_index(
        metric="cosine",
        vector_column_name="image_embedding",
        index_type="IVF_FLAT",
        num_partitions=max(2, int(n ** 0.5) // 4),
    )
    print("index built")


if __name__ == "__main__":
    main()
