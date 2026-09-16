#!/usr/bin/env python3
"""
FiftyOne <-> LanceDB bridge.

Architecture: LanceDB is the single source of truth (media + structure +
vectors). FiftyOne is the interactive edit/review surface that reads through
two official hooks and writes edits back.

Hooks:
  1. Multimodal episodes — sample.filepath points at the original MCAP.
     FiftyOne reads via byte-range; nothing is copied.
  2. Brain similarity — backend="lancedb", uri=<our db>. A FO-shaped
     table `fo_frames_clip` (id, sample_id, vector) is materialized from
     camera_frames.image_embedding so no recompute is needed.

Usage (viz env):
  python fiftyone_bridge.py register-episodes
  python fiftyone_bridge.py build-frames [--limit N] [--export-media]
  python fiftyone_bridge.py attach-brain
  python fiftyone_bridge.py sync-labels        # FO tags -> Lance
  python fiftyone_bridge.py launch             # fo.launch_app()
"""

import argparse
import os
import sys

import lancedb
import numpy as np
import pyarrow as pa

DB_DEFAULT = "/mnt/d/src/waymo2mcap/data/lancedb"
FRAMES_DATASET = "waymo-frames"
EPISODES_DATASET = "waymo-episodes"
BRAIN_TABLE = "fo_frames_clip"
MEDIA_ROOT = "/mnt/d/src/waymo2mcap/data/fiftyone_media"


def connect(db_path):
    return lancedb.connect(db_path)


def table_names(db):
    names = db.list_tables()
    if hasattr(names, "tables"):
        names = names.tables
    return [n if isinstance(n, str) else n[0] for n in names]


def stable_frame_id(episode_id, frame_index, camera):
    """Deterministic 24-hex id so FO and Lance agree across rebuilds."""
    import hashlib
    key = f"{episode_id}|{frame_index}|{camera}".encode()
    return hashlib.sha1(key).hexdigest()[:24]


def register_episodes(db_path):
    """Zero-copy: point FO multimodal samples at the original MCAP files."""
    import fiftyone as fo

    db = connect(db_path)
    eps = db.open_table("episodes").to_arrow().to_pylist()
    if EPISODES_DATASET in fo.list_datasets():
        fo.delete_dataset(EPISODES_DATASET)
    dataset = fo.Dataset(EPISODES_DATASET)
    samples = []
    for r in eps:
        mcap = r["mcap_path"]
        if not os.path.isfile(mcap):
            print(f"  skip missing: {mcap}")
            continue
        s = fo.Sample(filepath=mcap)
        s["episode_id"] = r["episode_id"]
        s["source_type"] = r["source_type"]
        s["frame_count"] = r["frame_count"]
        s["duration_s"] = r["duration_s"]
        s["mcap_bytes"] = os.path.getsize(mcap)
        samples.append(s)
    ids = dataset.add_samples(samples)
    print(f"registered {len(ids)} MCAP episodes as '{EPISODES_DATASET}'")
    print(f"  media_type={dataset.media_type}  (multimodal playback)")
    return dataset


def build_frames(db_path, limit=None, export_media=False, camera=None):
    """
    Materialize a FO image dataset whose samples mirror camera_frames.

    Embeddings are NOT copied here — attach_brain() points Brain at the
    existing image_embedding column via a FO-shaped index table.
    """
    import fiftyone as fo

    db = connect(db_path)
    cf = db.open_table("camera_frames")
    cols = ["episode_id", "frame_index", "camera", "timestamp_ns",
            "num_labels", "labels", "image"]
    arrow = cf.search().select(cols).to_arrow()
    rows = arrow.to_pylist()
    if camera:
        rows = [r for r in rows if r["camera"] == camera]
    if limit:
        rows = rows[:limit]

    if FRAMES_DATASET in fo.list_datasets():
        fo.delete_dataset(FRAMES_DATASET)
    dataset = fo.Dataset(FRAMES_DATASET)

    if export_media:
        os.makedirs(MEDIA_ROOT, exist_ok=True)

    samples = []
    for i, r in enumerate(rows):
        fid = stable_frame_id(r["episode_id"], r["frame_index"], r["camera"])
        if export_media:
            rel = f"{r['episode_id']}/{r['frame_index']:04d}_{r['camera']}.jpg"
            path = os.path.join(MEDIA_ROOT, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.isfile(path):
                with open(path, "wb") as f:
                    f.write(r["image"])
            filepath = path
        else:
            # Placeholder path; App image tile needs a real file. Use a
            # 1x1 JPEG written once so FO accepts the sample, and rely on
            # episode MCAP + Lance for actual media. Prefer --export-media
            # for grid review.
            placeholder = os.path.join(MEDIA_ROOT, "_placeholder.jpg")
            os.makedirs(MEDIA_ROOT, exist_ok=True)
            if not os.path.isfile(placeholder):
                from PIL import Image
                Image.new("RGB", (8, 8), (32, 32, 32)).save(placeholder, "JPEG")
            filepath = placeholder

        s = fo.Sample(filepath=filepath)
        # keep FO id aligned with the Lance brain table
        s.id = None  # let FO assign; we remap in attach_brain
        s["episode_id"] = r["episode_id"]
        s["frame_index"] = r["frame_index"]
        s["camera"] = r["camera"]
        s["timestamp_ns"] = r["timestamp_ns"]
        s["num_labels"] = r["num_labels"]
        s["lance_key"] = fid
        if r["labels"]:
            s["labels_json"] = str(r["labels"])
        samples.append(s)

    dataset.add_samples(samples, progress=True)
    print(f"built '{FRAMES_DATASET}': {len(dataset)} samples "
          f"(export_media={export_media})")
    return dataset


def attach_brain(db_path, metric="cosine"):
    """
    Create FO-shaped Lance table from camera_frames.image_embedding and
    register it as the Brain similarity index. Zero embedding recompute.
    """
    import fiftyone as fo
    import fiftyone.brain as fob

    db = connect(db_path)
    if FRAMES_DATASET not in fo.list_datasets():
        raise SystemExit("run build-frames first")
    dataset = fo.load_dataset(FRAMES_DATASET)

    # map lance_key -> FO sample id
    keys = dataset.values("lance_key")
    fo_ids = dataset.values("id")
    id_map = dict(zip(keys, fo_ids))

    cf = db.open_table("camera_frames")
    cols = ["episode_id", "frame_index", "camera", "image_embedding"]
    # skip rows with null embeddings
    arrow = (cf.search()
             .select(cols)
             .to_arrow())
    rows = arrow.to_pylist()

    ids, sample_ids, vecs = [], [], []
    skipped = 0
    for r in rows:
        if r["image_embedding"] is None:
            skipped += 1
            continue
        key = stable_frame_id(r["episode_id"], r["frame_index"], r["camera"])
        fo_id = id_map.get(key)
        if fo_id is None:
            skipped += 1
            continue
        ids.append(fo_id)
        sample_ids.append(fo_id)
        vecs.append(r["image_embedding"])

    if not vecs:
        raise SystemExit("no overlapping embeddings between FO and Lance")

    dim = len(vecs[0])
    emb = np.asarray(vecs, dtype=np.float32)
    table = pa.table({
        "id": pa.array(ids, type=pa.string()),
        "sample_id": pa.array(sample_ids, type=pa.string()),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(emb.reshape(-1), type=pa.float32()), dim),
    })

    if BRAIN_TABLE in table_names(db):
        db.drop_table(BRAIN_TABLE)
    db.create_table(BRAIN_TABLE, table)
    print(f"wrote {BRAIN_TABLE}: {len(ids):,} vectors dim={dim} "
          f"(skipped {skipped})")

    # Register existing table; embeddings=False prevents recompute
    index = fob.compute_similarity(
        dataset,
        embeddings=False,
        backend="lancedb",
        brain_key="clip_lancedb",
        table_name=BRAIN_TABLE,
        uri=db_path,
        metric=metric,
    )
    print(f"brain attached: key=clip_lancedb uri={db_path} table={BRAIN_TABLE}")
    print("  try: dataset.sort_by_similarity('pedestrian', brain_key='clip_lancedb', k=5)")
    return index


def sync_labels(db_path, tag_field="tags", lance_column="fo_tags"):
    """
    Write FiftyOne tags back onto camera_frames so analytics see human edits.

    Uses Lance `update(where=...)` per tagged row on the natural key
    (episode_id, frame_index, camera). Adds a string-list column if missing.
    """
    import fiftyone as fo

    if FRAMES_DATASET not in fo.list_datasets():
        raise SystemExit("run build-frames first")
    dataset = fo.load_dataset(FRAMES_DATASET)

    ep = dataset.values("episode_id")
    fi = dataset.values("frame_index")
    cam = dataset.values("camera")
    tags = dataset.values(tag_field)

    rows = []
    for e, f, c, t in zip(ep, fi, cam, tags):
        if t:
            rows.append((e, f, c, list(t)))
    if not rows:
        print("no tagged samples in FO; nothing to sync")
        return 0

    db = connect(db_path)
    cf = db.open_table("camera_frames")
    if lance_column not in cf.schema.names:
        # nullable list<string> via merge of a null-filled column
        n = cf.count_rows()
        null_col = pa.table({
            "episode_id": cf.search().select(["episode_id"]).to_arrow()
                          .column("episode_id"),
            lance_column: pa.array([None] * n, type=pa.list_(pa.string())),
        })
        # add_columns needs a reader aligned in scan order; use a simpler path:
        # rewrite once with the new column, then update tagged rows.
        full = cf.to_arrow()
        full = full.append_column(
            lance_column, pa.array([None] * n, type=pa.list_(pa.string())))
        db.drop_table("camera_frames")
        cf = db.create_table("camera_frames", full)
        # rebuild episode_id index
        try:
            cf.create_scalar_index("episode_id", index_type="BTREE")
        except Exception as e:
            print(f"  (index rebuild skipped: {e})")
        print(f"added column {lance_column}")

    for e, f, c, t in rows:
        where = (f"episode_id = '{e}' AND frame_index = {f} "
                 f"AND camera = '{c}'")
        cf.update(where=where, values={lance_column: t})

    print(f"synced {len(rows)} tagged frames into camera_frames.{lance_column}")
    return len(rows)


def launch(db_path, frames=True):
    import fiftyone as fo
    name = FRAMES_DATASET if frames else EPISODES_DATASET
    if name not in fo.list_datasets():
        raise SystemExit(f"dataset {name} missing; run build first")
    dataset = fo.load_dataset(name)
    session = fo.launch_app(dataset, auto=False)
    print(f"App launched for {name} — open the printed URL")
    return session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=[
        "register-episodes", "build-frames", "attach-brain",
        "sync-labels", "launch", "status"])
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--export-media", action="store_true")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--episodes", action="store_true",
                    help="launch episodes dataset instead of frames")
    args = ap.parse_args()

    if args.cmd == "register-episodes":
        register_episodes(args.db)
    elif args.cmd == "build-frames":
        build_frames(args.db, limit=args.limit,
                     export_media=args.export_media, camera=args.camera)
    elif args.cmd == "attach-brain":
        attach_brain(args.db)
    elif args.cmd == "sync-labels":
        sync_labels(args.db)
    elif args.cmd == "launch":
        launch(args.db, frames=not args.episodes)
    elif args.cmd == "status":
        import fiftyone as fo
        print("FO datasets:", fo.list_datasets())
        db = connect(args.db)
        print("Lance tables:", table_names(db))


if __name__ == "__main__":
    main()
