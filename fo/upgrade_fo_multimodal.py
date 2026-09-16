#!/usr/bin/env python3
"""
Upgrade waymo-curated with the rest of the multimodal payload:

  1. image_embedding  → sample field (from Lance camera_frames)
  2. fo.Keypoint      → 2D annotation points (text + x/y)
  3. fo.Detection3D   → 3D boxes for that frame
  4. ego pose         → scalar fields (px/py/pz, speed_mps)
  5. Brain similarity → LanceDB backend, embeddings precomputed
                         (App Embeddings panel → 2D scatter)

Run (viz env):
  python upgrade_fo_multimodal.py
"""

import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, "/mnt/d/src/fiftyone")

import lancedb
import numpy as np

import fiftyone as fo
import fiftyone.brain as fob

SRC_DB = "/mnt/d/src/waymo2mcap/data/lancedb"
FO_DB = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"
DATASET = "waymo-curated"
BRAIN_KEY = "clip_lancedb"


def configure():
    os.environ["FIFTYONE_DATASET_STORAGE"] = "lance"
    os.environ["FIFTYONE_DATASET_STORAGE_URI"] = FO_DB
    fo.config.dataset_storage = "lance"
    fo.config.dataset_storage_uri = FO_DB
    return fo


def load_side_tables(episode_ids):
    db = lancedb.connect(SRC_DB)
    ep_list = ", ".join("'%s'" % e for e in episode_ids)

    o3 = db.open_table("objects_3d")
    boxes = (o3.search()
             .where("episode_id IN (%s)" % ep_list)
             .select(["episode_id", "frame_index", "label",
                      "pos_x", "pos_y", "pos_z",
                      "size_x", "size_y", "size_z",
                      "quat_x", "quat_y", "quat_z", "quat_w"])
             .to_arrow()
             .to_pylist())
    boxes_by_frame = defaultdict(list)
    for b in boxes:
        boxes_by_frame[(b["episode_id"], b["frame_index"])].append(b)

    ep = db.open_table("ego_poses")
    poses = (ep.search()
             .where("episode_id IN (%s)" % ep_list)
             .select(["episode_id", "frame_index",
                      "px", "py", "pz", "speed_mps"])
             .to_arrow()
             .to_pylist())
    pose_by_frame = {(p["episode_id"], p["frame_index"]): p for p in poses}

    cf = db.open_table("camera_frames")
    emb_rows = (cf.search()
                .where("episode_id IN (%s)" % ep_list)
                .select(["episode_id", "frame_index", "camera",
                         "image_embedding", "labels"])
                .to_arrow()
                .to_pylist())
    emb_map = {}
    lab_map = {}
    for r in emb_rows:
        key = (r["episode_id"], r["frame_index"], r["camera"])
        if r["image_embedding"] is not None:
            emb_map[key] = r["image_embedding"]
        if r["labels"]:
            lab_map[key] = r["labels"]

    return boxes_by_frame, pose_by_frame, emb_map, lab_map


def main():
    configure()
    ds = fo.load_dataset(DATASET)
    print("loaded", ds)

    # ds.values() on dynamic fields goes through aggregate and is unreliable
    # on the Lance backend — read attributes while iterating.
    eps = sorted({s.episode_id for s in ds if s.episode_id})
    print("episodes:", eps)
    if not eps:
        raise SystemExit("no episode_id on samples — re-run demo import")

    print("loading side tables from Lance...")
    boxes_by_frame, pose_by_frame, emb_map, lab_map = load_side_tables(eps)
    print("  3D box frames:", len(boxes_by_frame),
          " total boxes:", sum(len(v) for v in boxes_by_frame.values()))
    print("  ego poses:", len(pose_by_frame))
    print("  embeddings:", len(emb_map))
    print("  2D label frames:", len(lab_map))

    t0 = time.time()
    n_boxes = n_pts = n_emb = 0
    for i, s in enumerate(ds.iter_samples()):
        key = (s.episode_id, s.frame_index, s.camera)

        emb = emb_map.get(key)
        if emb is not None:
            s["embedding"] = list(emb)
            n_emb += 1

        labels = lab_map.get(key)
        if labels:
            pts = fo.Keypoint(
                points=[[lb["x"], lb["y"]] for lb in labels],
                labels=[lb["text"] for lb in labels],
            )
            s["keypoints"] = fo.Keypoints(keypoints=[pts])
            n_pts += len(labels)

        b3 = boxes_by_frame.get((s.episode_id, s.frame_index), [])
        if b3:
            # FO represents 3D cuboids as fo.Detection with
            # location / dimensions / rotation (quaternion)
            dets = [
                fo.Detection(
                    label=b["label"],
                    location=[b["pos_x"], b["pos_y"], b["pos_z"]],
                    dimensions=[b["size_x"], b["size_y"], b["size_z"]],
                    rotation=[b["quat_x"], b["quat_y"], b["quat_z"], b["quat_w"]],
                )
                for b in b3
            ]
            s["detections_3d"] = fo.Detections(detections=dets)
            n_boxes += len(dets)

        pose = pose_by_frame.get((s.episode_id, s.frame_index))
        if pose:
            s["ego_px"] = pose["px"]
            s["ego_py"] = pose["py"]
            s["ego_pz"] = pose["pz"]
            s["speed_mps"] = pose["speed_mps"]

        s.save()
        if (i + 1) % 100 == 0:
            print("  saved %d/%d (%.0f/s)" % (i + 1, len(ds),
                                              (i + 1) / (time.time() - t0)))

    print("wrote fields in %.1fs  emb=%d  2d_pts=%d  3d_boxes=%d"
          % (time.time() - t0, n_emb, n_pts, n_boxes))
    print(ds)

    # ---- Brain index over the precomputed vectors ----
    ids, vecs = [], []
    for s in ds.iter_samples():
        try:
            e = s["embedding"]
        except Exception:
            e = None
        if e is not None:
            ids.append(s.id)
            vecs.append(e)
    print("registering brain index for %d samples..." % len(ids))

    if BRAIN_KEY in ds.list_brain_runs():
        ds.delete_brain_run(BRAIN_KEY)

    arr = np.asarray(vecs, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)

    fob.compute_similarity(
        ds,
        embeddings=arr,
        backend="lancedb",
        brain_key=BRAIN_KEY,
        table_name="fo_frames_clip",
        uri=FO_DB,
        metric="cosine",
    )
    print("brain attached:", BRAIN_KEY, arr.shape)

    view = ds.sort_by_similarity(ids[0], k=3, brain_key=BRAIN_KEY)
    print("top-3 similar to first sample:")
    for s in view:
        print("  ", s.camera, s.frame_index, s.episode_id[:40])

    print("\nUPGRADE OK")
    print("brain runs:", ds.list_brain_runs())
    print("fields:", list(ds.get_field_schema().keys()))


if __name__ == "__main__":
    main()
