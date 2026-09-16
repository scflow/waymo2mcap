#!/usr/bin/env python3
"""
Demo: import waymo camera frames into FiftyOne with Lance dataset storage.

This is the "Lance is the store, FO is the lens" path end-to-end:

  1. configure fo.config.dataset_storage = "lance"
  2. create a FO dataset (registry + sample table land in Lance)
  3. pull frames from the analytical LanceDB (data/lancedb)
  4. add them as FO samples with dynamic fields
  5. query via the public API: match / count / first / iteration
  6. prove a fresh process can load the same dataset

Run:
  python demo_fo_lance_import.py              # import + query
  python demo_fo_lance_import.py --verify     # second-process check only
  python demo_fo_lance_import.py --limit 200  # smaller import
"""

import argparse
import os
import sys
import time
from collections import Counter

# use the branch source, not any installed fiftyone
sys.path.insert(0, "/mnt/d/src/fiftyone")

SRC_DB = "/mnt/d/src/waymo2mcap/data/lancedb"
FO_DB = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"
DATASET = "waymo-curated"
MEDIA_ROOT = "/mnt/d/src/waymo2mcap/data/fiftyone_media/frames"


def configure():
    import fiftyone as fo

    # env first: the App server runs in a subprocess and only sees env vars
    os.environ["FIFTYONE_DATASET_STORAGE"] = "lance"
    os.environ["FIFTYONE_DATASET_STORAGE_URI"] = FO_DB
    fo.config.dataset_storage = "lance"
    fo.config.dataset_storage_uri = FO_DB
    return fo


def load_source_rows(limit=None, camera=None):
    """Read frame metadata (and JPEG bytes) from the analytical LanceDB."""
    import lancedb

    db = lancedb.connect(SRC_DB)
    cf = db.open_table("camera_frames")
    cols = ["episode_id", "frame_index", "camera", "timestamp_ns",
            "num_labels", "image"]
    rows = cf.search().select(cols).to_arrow().to_pylist()
    if camera:
        rows = [r for r in rows if r["camera"] == camera]
    if limit:
        rows = rows[:limit]
    return rows


def export_jpeg(image_bytes, episode_id, frame_index, camera):
    rel = "%s/%04d_%s.jpg" % (episode_id, frame_index, camera)
    path = os.path.join(MEDIA_ROOT, rel)
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(image_bytes)
    return path


def run_import(limit=None, camera=None):
    fo = configure()

    if fo.dataset_exists(DATASET):
        print("deleting existing dataset...")
        fo.delete_dataset(DATASET)

    print("creating FO dataset on Lance at", FO_DB)
    ds = fo.Dataset(DATASET)
    ds.media_type = "image"
    ds.persistent = True
    ds.save()

    print("reading source frames from", SRC_DB)
    rows = load_source_rows(limit=limit, camera=camera)
    print("  %d frames" % len(rows))

    samples = []
    t0 = time.time()
    for r in rows:
        # media lives on disk for the App grid; Lance holds the pointer
        path = export_jpeg(r["image"], r["episode_id"], r["frame_index"],
                           r["camera"])
        s = fo.Sample(filepath=path)
        s["episode_id"] = r["episode_id"]
        s["frame_index"] = r["frame_index"]
        s["camera"] = r["camera"]
        s["timestamp_ns"] = r["timestamp_ns"]
        s["num_labels"] = r["num_labels"]
        # hard-case tag: crowded frames
        if r["num_labels"] and r["num_labels"] >= 10:
            s.tags = ["crowded"]
        samples.append(s)

    print("exporting media + building samples took %.1fs" % (time.time() - t0))

    t0 = time.time()
    ids = ds.add_samples(samples, progress=True)
    dt = time.time() - t0
    print("added %d samples in %.1fs (%.0f/s)" % (len(ids), dt, len(ids) / dt))
    print(ds)

    # ---- queries through the public API ----
    print("\n--- queries ---")
    print("total:", ds.count())

    front = ds.match(fo.ViewField("camera") == "FRONT")
    print("camera==FRONT:", len(front))

    busy = ds.match(fo.ViewField("num_labels") >= 10)
    print("num_labels>=10:", len(busy))

    # combine with tag filter
    crowded = ds.match(
        (fo.ViewField("camera") == "FRONT")
        & (fo.ViewField("num_labels") >= 8)
    )
    print("FRONT and num_labels>=8:", len(crowded))

    # distribution
    cams = Counter(s.camera for s in ds)
    print("camera dist:", dict(cams))

    # sample peek
    s = crowded.first()
    if s is not None:
        print("example crowded FRONT frame:")
        print("  ", s.filepath)
        print("   episode=", s.episode_id, "frame=", s.frame_index,
              "labels=", s.num_labels, "tags=", s.tags)

    # episode breakdown
    eps = Counter(s.episode_id for s in ds)
    print("episodes:", len(eps))
    for ep, n in eps.most_common(3):
        print("  %s: %d frames" % (ep[:56], n))

    print("\nIMPORT OK — dataset persisted at", FO_DB)
    return ds


def run_verify():
    fo = configure()
    print("list_datasets:", fo.list_datasets())
    if not fo.dataset_exists(DATASET):
        raise SystemExit("FAIL: dataset missing")
    ds = fo.load_dataset(DATASET)
    print("loaded", ds)
    n = ds.count()
    print("count:", n)
    assert n > 0

    front = ds.match(fo.ViewField("camera") == "FRONT")
    print("camera==FRONT:", len(front))

    s = ds.first()
    print("first:", s.filepath, s.camera, s.frame_index, s.num_labels)
    assert os.path.isfile(s.filepath), "media file missing: %s" % s.filepath

    print("VERIFY OK — public fo.* API over Lance survives restart")


def run_app(dataset_name=DATASET, port=5151, remote=True):
    """Launch the FiftyOne App against the imported Lance-backed dataset."""
    fo = configure()

    if not fo.dataset_exists(dataset_name):
        raise SystemExit(
            "dataset %r not found — available datasets:\n  %s"
            % (dataset_name, fo.list_datasets())
        )

    ds = fo.load_dataset(dataset_name)
    print("loaded", ds)
    print("count:", ds.count())
    print("media type:", ds.media_type)
    print()

    # WSL: bind 0.0.0.0 so the Windows host browser can reach localhost:port
    address = "0.0.0.0" if remote else None
    session = fo.launch_app(ds, port=port, address=address, auto=False)
    print("App running at http://localhost:%d (dataset: %s)" % (port, dataset_name))
    print("Press Ctrl+C to stop.")
    try:
        session.wait()
    except KeyboardInterrupt:
        print("\nstopping App...")
        try:
            session.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="only load + query an existing import")
    ap.add_argument("--app", action="store_true",
                    help="launch the FiftyOne App on the imported dataset")
    ap.add_argument("--dataset", default="waymo-episodes-video",
                    help="dataset to open in App (default waymo-episodes-video)")
    ap.add_argument("--port", type=int, default=5151,
                    help="App port (default 5151)")
    ap.add_argument("--limit", type=int, default=400,
                    help="max frames to import (default 400)")
    ap.add_argument("--camera", default=None,
                    help="restrict to one camera, e.g. FRONT")
    ap.add_argument("--all", action="store_true",
                    help="import all 5950 frames (ignore --limit)")
    args = parser = ap.parse_args()

    limit = None if args.all else args.limit
    if args.app:
        run_app(dataset_name=args.dataset, port=args.port, remote=True)
    elif args.verify:
        run_verify()
    else:
        run_import(limit=limit, camera=args.camera)


if __name__ == "__main__":
    main()
