#!/usr/bin/env python3
"""
Single-process FiftyOne x LanceDB demo.

The embedded mongod does not persist across python processes (each run gets a
fresh DB), so registration + brain attach + query + App launch must happen in
one process. For long-lived review sessions, keep this process running, or
point FIFTYONE_DATABASE_URI at a real mongod.

Run (viz env):
  python fo_lance_demo.py                 # register + frames + brain + query
  python fo_lance_demo.py --launch        # also open the App
  python fo_lance_demo.py --export-media  # write JPEGs for the image grid
"""

import argparse
import os

import lancedb
import numpy as np
import pyarrow as pa

DB = "/mnt/d/src/waymo2mcap/data/lancedb"
FRAMES = "waymo-frames"
EPISODES = "waymo-episodes"
BRAIN_TABLE = "fo_frames_clip"
MEDIA_ROOT = "/mnt/d/src/waymo2mcap/data/fiftyone_media"


def stable_id(episode_id, frame_index, camera):
    import hashlib
    return hashlib.sha1(f"{episode_id}|{frame_index}|{camera}".encode()).hexdigest()[:24]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--export-media", action="store_true")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--query", default="a car driving on the road")
    args = ap.parse_args()

    import fiftyone as fo
    import fiftyone.brain as fob

    db = lancedb.connect(args.db)

    # ---- 1. zero-copy episode registration (multimodal MCAP playback) ----
    for name in (EPISODES, FRAMES):
        if name in fo.list_datasets():
            fo.delete_dataset(name)

    eps = db.open_table("episodes").to_arrow().to_pylist()
    ep_ds = fo.Dataset(EPISODES)
    ep_samples = []
    for r in eps:
        if not os.path.isfile(r["mcap_path"]):
            continue
        s = fo.Sample(filepath=r["mcap_path"])
        s["episode_id"] = r["episode_id"]
        s["source_type"] = r["source_type"]
        s["frame_count"] = r["frame_count"]
        s["duration_s"] = r["duration_s"]
        ep_samples.append(s)
    ep_ds.add_samples(ep_samples)
    print(f"[1] episodes: {len(ep_ds)} multimodal MCAP samples "
          f"(media_type={ep_ds.media_type})")

    # ---- 2. frame grid from camera_frames (media exported on demand) ----
    cf = db.open_table("camera_frames")
    cols = ["episode_id", "frame_index", "camera", "timestamp_ns",
            "num_labels", "labels", "image"]
    rows = cf.search().select(cols).to_arrow().to_pylist()
    if args.camera:
        rows = [r for r in rows if r["camera"] == args.camera]
    if args.limit:
        rows = rows[: args.limit]

    if args.export_media:
        os.makedirs(MEDIA_ROOT, exist_ok=True)

    frame_ds = fo.Dataset(FRAMES)
    samples = []
    for r in rows:
        if args.export_media:
            rel = f"{r['episode_id']}/{r['frame_index']:04d}_{r['camera']}.jpg"
            path = os.path.join(MEDIA_ROOT, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.isfile(path):
                with open(path, "wb") as f:
                    f.write(r["image"])
        else:
            # FO image samples need a real file; use a tiny placeholder and
            # keep full JPEG in Lance. Prefer --export-media for grid review.
            path = os.path.join(MEDIA_ROOT, "_placeholder.jpg")
            os.makedirs(MEDIA_ROOT, exist_ok=True)
            if not os.path.isfile(path):
                from PIL import Image
                Image.new("RGB", (8, 8), (30, 30, 30)).save(path, "JPEG")
        s = fo.Sample(filepath=path)
        s["episode_id"] = r["episode_id"]
        s["frame_index"] = r["frame_index"]
        s["camera"] = r["camera"]
        s["timestamp_ns"] = r["timestamp_ns"]
        s["num_labels"] = r["num_labels"]
        s["lance_key"] = stable_id(r["episode_id"], r["frame_index"], r["camera"])
        samples.append(s)
    frame_ds.add_samples(samples)
    print(f"[2] frames: {len(frame_ds)} samples "
          f"(export_media={args.export_media})")

    # ---- 3. mount Brain on the EXISTING CLIP embeddings ----
    id_map = dict(zip(frame_ds.values("lance_key"), frame_ds.values("id")))
    cols = ["episode_id", "frame_index", "camera", "image_embedding"]
    emb_rows = cf.search().select(cols).to_arrow().to_pylist()
    ids, vecs = [], []
    for r in emb_rows:
        if r["image_embedding"] is None:
            continue
        fo_id = id_map.get(stable_id(r["episode_id"], r["frame_index"], r["camera"]))
        if fo_id is None:
            continue
        ids.append(fo_id)
        vecs.append(r["image_embedding"])
    if not vecs:
        raise SystemExit("no embeddings overlap")
    dim = len(vecs[0])
    emb = np.asarray(vecs, dtype=np.float32)
    brain_tbl = pa.table({
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
    db.create_table(BRAIN_TABLE, brain_tbl)

    index = fob.compute_similarity(
        frame_ds,
        embeddings=False,          # do NOT recompute — reuse Lance vectors
        backend="lancedb",
        brain_key="clip_lancedb",
        table_name=BRAIN_TABLE,
        uri=args.db,
        metric="cosine",
    )
    print(f"[3] brain mounted: {len(ids)} vectors dim={dim} "
          f"table={BRAIN_TABLE} uri={args.db}")

    # ---- 4. query through FiftyOne API (hits our Lance table) ----
    query = args.query
    try:
        view = frame_ds.sort_by_similarity(
            query, k=5, brain_key="clip_lancedb")
        print(f"[4] text query {query!r} -> {len(view)} samples")
        for s in view:
            print(f"    {s.episode_id} f={s.frame_index} {s.camera}")
    except Exception as e:
        print(f"[4] text query unavailable ({type(e).__name__}: {e}); "
              f"falling back to sample-id query")
        qid = frame_ds.first().id
        view = frame_ds.sort_by_similarity(
            qid, k=5, brain_key="clip_lancedb")
        print(f"    id query -> {len(view)} samples")
        for s in view:
            print(f"    {s.episode_id} f={s.frame_index} {s.camera}")

    print(f"\nFO datasets: {fo.list_datasets()}")
    print("Lance tables:", names + [BRAIN_TABLE])
    print("\nLanceDB remains the source of truth; FO is the edit surface.")
    print("Tag samples in the App, then sync back with fiftyone_bridge.py sync-labels")

    if args.launch:
        print("\nlaunching App for waymo-frames ...")
        fo.launch_app(frame_ds, auto=False)
        print("keep this process alive to serve the App")


if __name__ == "__main__":
    main()
