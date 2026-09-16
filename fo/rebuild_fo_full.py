#!/usr/bin/env python3
"""
Rebuild FO datasets from the full Lance store.

1. waymo-curated  — ALL 5950 camera frames from 6 perception episodes,
   with embedding / keypoints / detections_3d / ego pose already in Lance.
2. waymo-episodes — 7 MCAP files as FO multimodal samples (zero-copy
   filepath; the App plays camera+lidar+3D in lockstep).

Run (viz env):
  python rebuild_fo_full.py [--frames] [--episodes] [--limit N] [--camera CAM]
"""

import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, "/mnt/d/src/fiftyone")

import lancedb
import numpy as np

import fiftyone as fo

SRC_DB = "/mnt/d/src/waymo2mcap/data/lancedb"
FO_DB = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"
FRAMES_DS = "waymo-curated"
EPISODES_DS = "waymo-episodes"
MEDIA_ROOT = "/mnt/d/src/waymo2mcap/data/fiftyone_media/frames"
BRAIN_KEY = "clip_viz"


def configure():
    os.environ["FIFTYONE_DATASET_STORAGE"] = "lance"
    os.environ["FIFTYONE_DATASET_STORAGE_URI"] = FO_DB
    fo.config.dataset_storage = "lance"
    fo.config.dataset_storage_uri = FO_DB
    return fo


def export_jpeg(image_bytes, episode_id, frame_index, camera):
    rel = "%s/%04d_%s.jpg" % (episode_id, frame_index, camera)
    path = os.path.join(MEDIA_ROOT, rel)
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(image_bytes)
    return path


def load_side_tables(db):
    """3D boxes, ego poses, 2D labels keyed by (episode, frame[, camera])."""
    boxes_by_frame = defaultdict(list)
    for b in (db.open_table("objects_3d").search()
              .select(["episode_id", "frame_index", "label",
                       "pos_x", "pos_y", "pos_z",
                       "size_x", "size_y", "size_z",
                       "quat_x", "quat_y", "quat_z", "quat_w"])
              .to_arrow().to_pylist()):
        boxes_by_frame[(b["episode_id"], b["frame_index"])].append(b)

    pose_by_frame = {}
    for p in (db.open_table("ego_poses").search()
              .select(["episode_id", "frame_index",
                       "px", "py", "pz", "speed_mps"])
              .to_arrow().to_pylist()):
        pose_by_frame[(p["episode_id"], p["frame_index"])] = p

    return boxes_by_frame, pose_by_frame


def rebuild_frames(limit=None, camera=None):
    """All camera frames, with the multimodal side tables attached."""
    configure()
    db = lancedb.connect(SRC_DB)

    if fo.dataset_exists(FRAMES_DS):
        print("deleting existing", FRAMES_DS)
        fo.delete_dataset(FRAMES_DS)

    ds = fo.Dataset(FRAMES_DS)
    ds.media_type = "image"
    ds.persistent = True
    ds.save()
    print("created", FRAMES_DS)

    print("loading side tables...")
    boxes_by_frame, pose_by_frame = load_side_tables(db)
    print("  3d frames", len(boxes_by_frame),
          " boxes", sum(len(v) for v in boxes_by_frame.values()))
    print("  poses", len(pose_by_frame))

    cf = db.open_table("camera_frames")
    cols = ["episode_id", "frame_index", "camera", "timestamp_ns",
            "image", "num_labels", "labels", "image_embedding"]
    rows = cf.search().select(cols).to_arrow().to_pylist()
    if camera:
        rows = [r for r in rows if r["camera"] == camera]
    # keep timeline order inside each episode
    rows.sort(key=lambda r: (r["episode_id"], r["frame_index"], r["camera"]))
    if limit:
        rows = rows[:limit]
    print("source frames:", len(rows))

    t0 = time.time()
    samples = []
    for r in rows:
        path = export_jpeg(r["image"], r["episode_id"], r["frame_index"],
                           r["camera"])
        s = fo.Sample(filepath=path)
        s["episode_id"] = r["episode_id"]
        s["frame_index"] = r["frame_index"]
        s["camera"] = r["camera"]
        s["timestamp_ns"] = r["timestamp_ns"]
        s["num_labels"] = r["num_labels"] or 0

        if r["image_embedding"] is not None:
            s["embedding"] = list(r["image_embedding"])

        labels = r["labels"]
        if labels:
            s["keypoints"] = fo.Keypoints(keypoints=[fo.Keypoint(
                points=[[lb["x"], lb["y"]] for lb in labels],
                labels=[lb["text"] for lb in labels],
            )])

        b3 = boxes_by_frame.get((r["episode_id"], r["frame_index"]))
        if b3:
            s["detections_3d"] = fo.Detections(detections=[
                fo.Detection(
                    label=b["label"],
                    location=[b["pos_x"], b["pos_y"], b["pos_z"]],
                    dimensions=[b["size_x"], b["size_y"], b["size_z"]],
                    rotation=[b["quat_x"], b["quat_y"], b["quat_z"], b["quat_w"]],
                ) for b in b3
            ])

        pose = pose_by_frame.get((r["episode_id"], r["frame_index"]))
        if pose:
            s["ego_px"] = pose["px"]
            s["ego_py"] = pose["py"]
            s["ego_pz"] = pose["pz"]
            s["speed_mps"] = pose["speed_mps"]

        if (r["num_labels"] or 0) >= 10:
            s.tags = ["crowded"]

        samples.append(s)

    print("adding %d samples..." % len(samples))
    ds.add_samples(samples, progress=True)
    print("done in %.0fs" % (time.time() - t0))
    print(ds)

    # per-episode breakdown
    from collections import Counter
    eps = Counter(s["episode_id"] for s in ds.iter_samples())
    print("episodes in dataset:", len(eps))
    for ep, n in eps.most_common():
        print("  %s: %d frames" % (ep[:56], n))
    return ds


def rebuild_brain(ds):
    """PCA-2D visualization run over whatever embeddings are present."""
    import fiftyone.brain as fob

    ids, vecs = [], []
    for s in ds.iter_samples():
        try:
            e = s["embedding"]
        except Exception:
            e = None
        if e is not None:
            ids.append(s.id)
            vecs.append(e)
    if not ids:
        print("no embeddings — skip brain")
        return
    print("brain: %d embeddings" % len(ids))

    arr = np.asarray(vecs, dtype=np.float32)
    X = arr.astype(np.float64)
    Xc = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    pts = U[:, :2] * S[:2]

    if BRAIN_KEY in ds.list_brain_runs():
        ds.delete_brain_run(BRAIN_KEY)
    fob.compute_visualization(
        ds, points=pts, brain_key=BRAIN_KEY, num_dims=2, method="manual",
    )
    print("brain run", BRAIN_KEY, "saved")


def register_episodes():
    """7 MCAP files as multimodal FO samples — zero-copy playback."""
    configure()
    db = lancedb.connect(SRC_DB)
    eps = db.open_table("episodes").to_arrow().to_pylist()

    if fo.dataset_exists(EPISODES_DS):
        print("deleting existing", EPISODES_DS)
        fo.delete_dataset(EPISODES_DS)

    ds = fo.Dataset(EPISODES_DS)
    samples = []
    for r in eps:
        p = r["mcap_path"]
        if not os.path.isfile(p):
            print("  skip missing", p)
            continue
        s = fo.Sample(filepath=p)
        s["episode_id"] = r["episode_id"]
        s["source_type"] = r["source_type"]
        s["frame_count"] = r["frame_count"]
        s["duration_s"] = r["duration_s"]
        s["num_images"] = r["num_images"]
        s["num_lidar_points"] = r["num_lidar_points"]
        s["num_objects_3d"] = r["num_objects_3d"]
        samples.append(s)

    ds.add_samples(samples)
    print("registered", EPISODES_DS, len(ds), "media_type=", ds.media_type)
    for s in ds.iter_samples():
        print("  %s  frames=%s  %s" % (
            s["episode_id"][:48], s["frame_count"],
            os.path.basename(s.filepath)[:50]))
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", action="store_true")
    ap.add_argument("--episodes", action="store_true")
    ap.add_argument("--brain", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--camera", default=None)
    args = ap.parse_args()

    if not args.frames and not args.episodes and not args.brain:
        args.frames = args.episodes = args.brain = True

    if args.episodes:
        print("\n########## EPISODES ##########")
        register_episodes()

    ds = None
    if args.frames:
        print("\n########## FRAMES ##########")
        ds = rebuild_frames(limit=args.limit, camera=args.camera)

    if args.brain:
        print("\n########## BRAIN ##########")
        if ds is None:
            configure()
            ds = fo.load_dataset(FRAMES_DS)
        rebuild_brain(ds)

    print("\nALL DONE")


if __name__ == "__main__":
    main()
