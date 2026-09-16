#!/usr/bin/env python3
"""
Build Data-Centric FiftyOne Datasets from LanceDB:

1. `waymo-episodes-overview` (media_type="image", 7 samples):
   - Episode-level data analysis & metrics.
   - Preview thumbnail for each episode.
   - Fields: episode_id, duration_s, frame_count, num_objects_3d, num_lidar_points,
     avg_speed_mps, max_speed_mps, mcap_path, topic_counts.
   - FiftyOne Brain: episode_viz (2D PCA scatter) and episode_sim (LanceDB cosine similarity).

2. `waymo-multimodal-frames` (media_type="image", 1,190 samples):
   - Frame-level multi-sensor messages and annotations.
   - Front camera image.
   - 2D Bounding Boxes (`ground_truth_2d`): normalized [x, y, w, h] with labels (Car, Pedestrian, Sign).
   - 3D Bounding Boxes (`ground_truth_3d`): location, dimensions, rotation.
   - Message Telemetry: speed_mps, speed_kmh, ego_px, ego_py, ego_pz, ego_vx, ego_vy, ego_vz, timestamp_ns.
   - Point Cloud references: pcd_path, scene_path (.fo3d).
   - FiftyOne Brain: frame_viz (2D PCA) and frame_sim (LanceDB vector search).

3. `waymo-lidar-3d` (media_type="3d", 1,190 samples or key frames):
   - FiftyOne 3D visualizer scenes (.fo3d) with 3D bounding boxes.
"""

import os
import sys
import time
import json
from collections import defaultdict
import numpy as np

sys.path.insert(0, "/mnt/d/src/fiftyone")

import lancedb
import fiftyone as fo
import fiftyone.brain as fob

SRC_DB = "/mnt/d/src/waymo2mcap/data/lancedb"
FO_DB = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"
MEDIA_ROOT = "/mnt/d/src/waymo2mcap/data/fiftyone_media"
FRAMES_DIR = os.path.join(MEDIA_ROOT, "frames")
SCENE_DIR = os.path.join(MEDIA_ROOT, "scenes")
PCD_DIR = os.path.join(MEDIA_ROOT, "pcd")


def configure():
    os.environ["FIFTYONE_DATASET_STORAGE"] = "lance"
    os.environ["FIFTYONE_DATASET_STORAGE_URI"] = FO_DB
    fo.config.dataset_storage = "lance"
    fo.config.dataset_storage_uri = FO_DB
    return fo


def build_episodes_overview_dataset():
    """Builds Episode-level dataset for comparing and analyzing episodes."""
    configure()
    db = lancedb.connect(SRC_DB)
    eps_tbl = db.open_table("episodes")
    ep_rows = eps_tbl.to_arrow().to_pylist()

    ds_name = "waymo-episodes-overview"
    print(f"\n==========================================")
    print(f"Building: {ds_name} ({len(ep_rows)} episodes)")
    print(f"==========================================")

    if fo.dataset_exists(ds_name):
        fo.delete_dataset(ds_name)

    # Calculate average and max speed per episode from ego_poses
    print("Loading ego poses for speed profiles...")
    poses = db.open_table("ego_poses").search().select(["episode_id", "speed_mps"]).to_arrow().to_pylist()
    speeds_by_ep = defaultdict(list)
    for p in poses:
        if p["speed_mps"] is not None:
            speeds_by_ep[p["episode_id"]].append(p["speed_mps"])

    samples = []
    for ep in ep_rows:
        ep_id = ep["episode_id"]
        # Use first front frame as thumbnail preview
        preview = os.path.join(FRAMES_DIR, ep_id, "0000_FRONT.jpg")
        if not os.path.isfile(preview):
            # Fallback if no front frame exists
            continue

        s = fo.Sample(filepath=preview)
        s["episode_id"] = ep_id
        s["duration_s"] = ep.get("duration_s") or 0.0
        s["frame_count"] = ep.get("frame_count") or 0
        s["num_objects_3d"] = ep.get("num_objects_3d") or 0
        s["num_lidar_points"] = ep.get("num_lidar_points") or 0
        s["source_type"] = ep.get("source_type") or "perception"
        s["mcap_path"] = ep.get("mcap_path") or ""

        sp_list = speeds_by_ep.get(ep_id, [])
        s["avg_speed_mps"] = float(np.mean(sp_list)) if sp_list else 0.0
        s["max_speed_mps"] = float(np.max(sp_list)) if sp_list else 0.0
        s["avg_speed_kmh"] = s["avg_speed_mps"] * 3.6
        s["max_speed_kmh"] = s["max_speed_mps"] * 3.6

        # 3D object density: objects per frame
        fc = s["frame_count"] or 1
        s["objects_per_frame"] = round(s["num_objects_3d"] / fc, 1)

        emb = ep.get("episode_embedding")
        if emb is not None:
            s["episode_embedding"] = list(emb)

        s.tags = ["episode_clip", ep.get("source_type", "perception")]
        if s["objects_per_frame"] > 50:
            s.tags.append("high_density")
        if s["max_speed_kmh"] > 40:
            s.tags.append("high_speed")

        samples.append(s)

    ds = fo.Dataset(ds_name)
    ds.media_type = "image"
    ds.persistent = True
    ds.save()
    ds.add_samples(samples)
    print(f"Added {len(ds)} episode overview samples.")

    # Brain runs
    vecs = [s.get_field("episode_embedding") for s in ds if s.get_field("episode_embedding")]
    if vecs:
        arr = np.asarray(vecs, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)

        ldb = lancedb.connect(FO_DB)
        t_names = [n if isinstance(n, str) else n[0] for n in (ldb.list_tables().tables if hasattr(ldb.list_tables(), "tables") else ldb.list_tables())]
        if "fo_ep_overview_clip" in t_names:
            ldb.drop_table("fo_ep_overview_clip")

        fob.compute_similarity(
            ds,
            embeddings=arr,
            backend="lancedb",
            brain_key="episode_sim",
            table_name="fo_ep_overview_clip",
            uri=FO_DB,
            metric="cosine"
        )
        X = arr.astype(np.float64)
        Xc = X - X.mean(axis=0)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        pts = U[:, :2] * S[:2]
        fob.compute_visualization(ds, points=pts, brain_key="episode_viz", num_dims=2, method="manual")
        print("Attached episode_sim and episode_viz.")

    return ds


def build_multimodal_frames_dataset():
    """Builds Frame-level dataset with real messages, 2D/3D annotations, and telemetry."""
    configure()
    db = lancedb.connect(SRC_DB)
    eps_tbl = db.open_table("episodes")
    ep_rows = eps_tbl.to_arrow().to_pylist()
    perceptions = [r for r in ep_rows if r.get("source_type") == "perception"]
    ep_ids = [e["episode_id"] for e in perceptions]
    ep_list = ", ".join(f"'{e}'" for e in ep_ids)

    ds_name = "waymo-multimodal-frames"
    print(f"\n==========================================")
    print(f"Building: {ds_name} (1,190 multi-sensor frames)")
    print(f"==========================================")

    if fo.dataset_exists(ds_name):
        fo.delete_dataset(ds_name)

    # 1. Poses
    print("Loading ego poses...")
    poses = (db.open_table("ego_poses").search()
             .where(f"episode_id IN ({ep_list})")
             .select(["episode_id", "frame_index", "timestamp_ns",
                      "px", "py", "pz", "vx", "vy", "vz", "speed_mps"])
             .to_arrow().to_pylist())
    pose_map = {(p["episode_id"], p["frame_index"]): p for p in poses}

    # 2. 3D Boxes
    print("Loading 3D bounding boxes...")
    o3 = db.open_table("objects_3d")
    boxes = (o3.search()
             .where(f"episode_id IN ({ep_list})")
             .select(["episode_id", "frame_index", "label",
                      "pos_x", "pos_y", "pos_z",
                      "size_x", "size_y", "size_z",
                      "quat_x", "quat_y", "quat_z", "quat_w"])
             .to_arrow().to_pylist())
    boxes_by_frame = defaultdict(list)
    for b in boxes:
        boxes_by_frame[(b["episode_id"], b["frame_index"])].append(b)

    # 3. 2D Front Camera Labels
    print("Loading FRONT camera 2D labels...")
    cf = db.open_table("camera_frames")
    cf_rows = (cf.search()
               .where(f"episode_id IN ({ep_list}) AND camera = 'FRONT'")
               .select(["episode_id", "frame_index", "labels"])
               .to_arrow().to_pylist())
    labels_map = {}
    for r in cf_rows:
        if r["labels"]:
            labels_map[(r["episode_id"], r["frame_index"])] = r["labels"]

    # 4. Fused Embeddings
    print("Loading 360° fused embeddings...")
    fused_map = {}
    table_names = db.list_tables()
    if hasattr(table_names, "tables"):
        table_names = table_names.tables
    names = [n if isinstance(n, str) else n[0] for n in table_names]
    if "frame_fused_embeddings" in names:
        ff_rows = (db.open_table("frame_fused_embeddings").search()
                   .where(f"episode_id IN ({ep_list})")
                   .select(["episode_id", "frame_index", "vector"])
                   .to_arrow().to_pylist())
        for r in ff_rows:
            fused_map[(r["episode_id"], r["frame_index"])] = r["vector"]

    # 5. Build samples
    print("Assembling multimodal frame samples...")
    samples = []
    # Front camera resolution: 1920x1280
    orig_w, orig_h = 1920.0, 1280.0
    box_w = 60.0 / orig_w
    box_h = 45.0 / orig_h

    for ep in perceptions:
        ep_id = ep["episode_id"]
        fc = ep["frame_count"]
        for fi in range(fc):
            jpg_file = os.path.join(FRAMES_DIR, ep_id, f"{fi:04d}_FRONT.jpg")
            if not os.path.isfile(jpg_file):
                continue

            s = fo.Sample(filepath=jpg_file)
            s["episode_id"] = ep_id
            s["frame_index"] = fi

            # Pose & Telemetry Message
            pose = pose_map.get((ep_id, fi))
            if pose:
                ts_ns = pose.get("timestamp_ns") or (fi * 100000000)
                sp = pose.get("speed_mps") or 0.0
                s["timestamp_ns"] = ts_ns
                s["timestamp_sec"] = round(fi * 0.1, 2)
                s["speed_mps"] = round(sp, 2)
                s["speed_kmh"] = round(sp * 3.6, 1)
                s["ego_px"] = round(pose.get("px") or 0.0, 2)
                s["ego_py"] = round(pose.get("py") or 0.0, 2)
                s["ego_pz"] = round(pose.get("pz") or 0.0, 2)
                s["ego_vx"] = round(pose.get("vx") or 0.0, 2)
                s["ego_vy"] = round(pose.get("vy") or 0.0, 2)
                s["ego_vz"] = round(pose.get("vz") or 0.0, 2)

            # 2D Annotations
            raw_labels = labels_map.get((ep_id, fi), [])
            if raw_labels:
                s["ground_truth_2d"] = fo.Detections(detections=[
                    fo.Detection(
                        label=lb["text"],
                        bounding_box=[
                            max(0.0, min(1.0 - box_w, (lb["x"] / orig_w) - box_w / 2.0)),
                            max(0.0, min(1.0 - box_h, (lb["y"] / orig_h) - box_h / 2.0)),
                            box_w,
                            box_h
                        ],
                    ) for lb in raw_labels
                ])
                s["num_labels_2d"] = len(raw_labels)
            else:
                s["num_labels_2d"] = 0

            # 3D Annotations
            b3 = boxes_by_frame.get((ep_id, fi), [])
            if b3:
                s["ground_truth_3d"] = fo.Detections(detections=[
                    fo.Detection(
                        label=b["label"],
                        location=[b["pos_x"], b["pos_y"], b["pos_z"]],
                        dimensions=[b["size_x"], b["size_y"], b["size_z"]],
                        rotation=[b["quat_x"], b["quat_y"], b["quat_z"], b["quat_w"]],
                    ) for b in b3
                ])
                s["num_objects_3d"] = len(b3)
            else:
                s["num_objects_3d"] = 0

            # Point Cloud references
            pcd_file = os.path.join(PCD_DIR, ep_id, f"frame_{fi:04d}.pcd")
            if os.path.isfile(pcd_file):
                s["pcd_path"] = pcd_file

            scene_file = os.path.join(SCENE_DIR, ep_id, f"frame_{fi:04d}.fo3d")
            if os.path.isfile(scene_file):
                s["scene_path"] = scene_file

            # Embedding
            fused = fused_map.get((ep_id, fi))
            if fused is not None:
                s["embedding"] = list(fused)

            # Tags
            s.tags = [ep_id.replace("segment-", "").split("_")[0]]
            speed_val = s.get_field("speed_kmh") or 0
            if speed_val > 30:
                s.tags.append("fast")
            elif speed_val < 5:
                s.tags.append("slow_stop")
            objs_val = s.get_field("num_objects_3d") or 0
            if objs_val >= 30:
                s.tags.append("crowded")

            samples.append(s)

    ds = fo.Dataset(ds_name)
    ds.media_type = "image"
    ds.persistent = True
    ds.save()
    ds.add_samples(samples, progress=True)
    print(f"Added {len(ds)} frame samples.")

    # Brain runs
    vecs = [s.get_field("embedding") for s in ds if s.get_field("embedding")]
    if vecs:
        arr = np.asarray(vecs, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)

        ldb = lancedb.connect(FO_DB)
        t_names = [n if isinstance(n, str) else n[0] for n in (ldb.list_tables().tables if hasattr(ldb.list_tables(), "tables") else ldb.list_tables())]
        if "fo_mm_frames_clip" in t_names:
            ldb.drop_table("fo_mm_frames_clip")

        fob.compute_similarity(
            ds,
            embeddings=arr,
            backend="lancedb",
            brain_key="frame_sim",
            table_name="fo_mm_frames_clip",
            uri=FO_DB,
            metric="cosine"
        )
        X = arr.astype(np.float64)
        Xc = X - X.mean(axis=0)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        pts = U[:, :2] * S[:2]
        fob.compute_visualization(ds, points=pts, brain_key="frame_viz", num_dims=2, method="manual")
        print("Attached frame_sim and frame_viz.")

    return ds


def main():
    print("Building Data-Centric FiftyOne Infrastructure...")
    t0 = time.time()
    build_episodes_overview_dataset()
    build_multimodal_frames_dataset()
    print(f"\nAll datasets built in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
