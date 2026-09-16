#!/usr/bin/env python3
"""
Single-process E2E: LanceDB is the store, FiftyOne is a session lens.

FO's embedded Mongo does not persist across processes, so this script
rehydrates the FO view from Lance every run — the correct pattern when
Lance is the source of truth.

  1. register MCAP episodes (zero-copy filepath -> .mcap)
  2. materialize a small frame dataset from camera_frames
  3. attach Brain to the existing CLIP embeddings via a FO-shaped
     Lance table (no recompute)
  4. run similarity search through FO, prove the index is live
"""

import os
import sys

import lancedb
import numpy as np
import pyarrow as pa

import fiftyone as fo
import fiftyone.brain as fob

DB = "/mnt/d/src/waymo2mcap/data/lancedb"
EPISODES = "waymo-episodes"
FRAMES = "waymo-frames"
BRAIN_TABLE = "fo_frames_clip"
MEDIA = "/mnt/d/src/waymo2mcap/data/fiftyone_media"


def stable_key(ep, fi, cam):
    import hashlib
    return hashlib.sha1(f"{ep}|{fi}|{cam}".encode()).hexdigest()[:24]


def main():
    db = lancedb.connect(DB)

    # --- 1. episodes: zero-copy MCAP ---
    if EPISODES in fo.list_datasets():
        fo.delete_dataset(EPISODES)
    eps = fo.Dataset(EPISODES)
    rows = db.open_table("episodes").to_arrow().to_pylist()
    samples = []
    for r in rows:
        if not os.path.isfile(r["mcap_path"]):
            continue
        s = fo.Sample(filepath=r["mcap_path"])
        s["episode_id"] = r["episode_id"]
        s["frame_count"] = r["frame_count"]
        samples.append(s)
    eps.add_samples(samples)
    print(f"[1] episodes: {len(eps)} multimodal samples "
          f"(media_type={eps.media_type})")

    # --- 2. frames: small FRONT slice with real JPEGs ---
    if FRAMES in fo.list_datasets():
        fo.delete_dataset(FRAMES)
    frames = fo.Dataset(FRAMES)
    os.makedirs(MEDIA, exist_ok=True)
    cf = db.open_table("camera_frames")
    arrow = (cf.search()
             .where("camera = 'FRONT'")
             .select(["episode_id", "frame_index", "camera", "image"])
             .limit(80)
             .to_arrow())
    fsamples = []
    for r in arrow.to_pylist():
        rel = f"{r['episode_id']}/{r['frame_index']:04d}_FRONT.jpg"
        path = os.path.join(MEDIA, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.isfile(path):
            with open(path, "wb") as f:
                f.write(r["image"])
        s = fo.Sample(filepath=path)
        s["episode_id"] = r["episode_id"]
        s["frame_index"] = r["frame_index"]
        s["camera"] = r["camera"]
        s["lance_key"] = stable_key(r["episode_id"], r["frame_index"],
                                    r["camera"])
        fsamples.append(s)
    frames.add_samples(fsamples)
    print(f"[2] frames: {len(frames)} image samples (media exported)")

    # --- 3. Brain <- existing image_embedding ---
    key_to_id = dict(zip(frames.values("lance_key"), frames.values("id")))
    full = (cf.search()
            .where("camera = 'FRONT'")
            .select(["episode_id", "frame_index", "camera", "image_embedding"])
            .limit(80)
            .to_arrow()
            .to_pylist())
    ids, vecs = [], []
    for r in full:
        if r["image_embedding"] is None:
            continue
        fo_id = key_to_id.get(stable_key(r["episode_id"], r["frame_index"],
                                         r["camera"]))
        if fo_id is None:
            continue
        ids.append(fo_id)
        vecs.append(r["image_embedding"])
    dim = len(vecs[0])
    emb = np.asarray(vecs, dtype=np.float32)
    tbl = pa.table({
        "id": pa.array(ids, type=pa.string()),
        "sample_id": pa.array(ids, type=pa.string()),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(emb.reshape(-1), type=pa.float32()), dim),
    })
    names = db.list_tables()
    names = names.tables if hasattr(names, "tables") else names
    names = [n if isinstance(n, str) else n[0] for n in names]
    if BRAIN_TABLE in names:
        db.drop_table(BRAIN_TABLE)
    db.create_table(BRAIN_TABLE, tbl)
    print(f"[3] Lance brain table {BRAIN_TABLE}: {len(ids)} vecs dim={dim}")

    index = fob.compute_similarity(
        frames,
        embeddings=False,
        backend="lancedb",
        brain_key="clip_lancedb",
        table_name=BRAIN_TABLE,
        uri=DB,
        metric="cosine",
    )
    print(f"[3] brain attached via official LanceDB backend "
          f"(uri={DB}, table={BRAIN_TABLE})")

    # --- 4. similarity through FO ---
    qid = frames.first().id
    view = frames.sort_by_similarity(qid, k=5, brain_key="clip_lancedb")
    print(f"\n[4] sort_by_similarity(query=first sample) -> {len(view)} hits")
    for s in view:
        print(f"    ep={s['episode_id'][:44]} frame={s['frame_index']} "
              f"cam={s['camera']}")

    # nearest neighbour should be the query itself
    first_hit = view.first()
    assert first_hit.id == qid, "self should rank first"
    print("[4] OK: self-match rank-1, index live")

    # raw Lance table still queryable in parallel
    raw = db.open_table(BRAIN_TABLE)
    print(f"[4] raw Lance rows: {raw.count_rows()}  "
          f"(Doris/DuckDB can read this table directly)")

    print("\nE2E PASSED — Lance is the store, FO is the lens")
    print("datasets this session:", fo.list_datasets())


if __name__ == "__main__":
    main()
