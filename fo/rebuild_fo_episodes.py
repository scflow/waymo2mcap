#!/usr/bin/env python3
"""
Rebuild FiftyOne Episode Sequence Dataset on LanceDB.

Builds:
1. `waymo-episodes-video` (media_type="video"):
   - 6-7 Episode Video Samples (Foxglove synchronized surround video)
   - Native FiftyOne Video Player playback with 10 FPS playback & timeline scrub
   - Frame-level sequence in `sample.frames`:
     - 3D Detections (`fo.Detections` with 3D boxes)
     - 2D Keypoints from FRONT camera
     - Ego speed, position, heading
     - Frame-level 5-camera spatial fusion embeddings
     - Point cloud (.pcd) paths
   - Dual FiftyOne Brain runs (backed by LanceDB):
     - `episode_sim`: Episode-level semantic similarity
     - `episode_viz`: Episode-level 2D PCA scatter
     - `frame_sim`: Frame-level 360° moment similarity
     - `frame_viz`: Frame-level 2D PCA scatter
2. `waymo-episodes-grouped` (media_type="group"):
   - Multi-modal group slices:
     - `video`: Foxglove surround video stream
     - `lidar_3d`: FiftyOne 3D visualizer (.fo3d / .pcd) with 3D bounding boxes

Run (viz env):
  python fo/rebuild_fo_episodes.py [--video] [--grouped] [--brain]
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
import fiftyone.brain as fob

SRC_DB = "/mnt/d/src/waymo2mcap/data/lancedb"
FO_DB = "/mnt/d/src/waymo2mcap/data/fo_lance_demo"
MEDIA_ROOT = "/mnt/d/src/waymo2mcap/data/fiftyone_media"
VIDEO_DIR = os.path.join(MEDIA_ROOT, "videos")
PCD_DIR = os.path.join(MEDIA_ROOT, "pcd")
SCENE_DIR = os.path.join(MEDIA_ROOT, "scenes")
FRAMES_MEDIA_ROOT = os.path.join(MEDIA_ROOT, "frames")

VIDEO_DS = "waymo-episodes-video"
GROUP_DS = "waymo-episodes-grouped"


def configure():
    os.environ["FIFTYONE_DATASET_STORAGE"] = "lance"
    os.environ["FIFTYONE_DATASET_STORAGE_URI"] = FO_DB
    fo.config.dataset_storage = "lance"
    fo.config.dataset_storage_uri = FO_DB
    return fo


def load_side_tables(db, episode_ids):
    """Load ego poses, 3d boxes, 2d labels, and fused frame embeddings."""
    ep_list = ", ".join(f"'{e}'" for e in episode_ids)

    print("Loading 3D bounding boxes from LanceDB...")
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

    print("Loading ego poses from LanceDB...")
    ep_tbl = db.open_table("ego_poses")
    poses = (ep_tbl.search()
             .where(f"episode_id IN ({ep_list})")
             .select(["episode_id", "frame_index", "px", "py", "pz", "speed_mps"])
             .to_arrow().to_pylist())
    pose_by_frame = {(p["episode_id"], p["frame_index"]): p for p in poses}

    print("Loading 5-camera 2D labels from LanceDB and normalizing to canvas...")
    cf = db.open_table("camera_frames")
    cf_rows = (cf.search()
               .where(f"episode_id IN ({ep_list})")
               .select(["episode_id", "frame_index", "camera", "labels"])
               .to_arrow().to_pylist())

    CAM_DIMS = {
        "FRONT": (1920, 1280),
        "FRONT_LEFT": (1920, 1280),
        "FRONT_RIGHT": (1920, 1280),
        "SIDE_LEFT": (1920, 886),
        "SIDE_RIGHT": (1920, 886),
    }
    CAM_POS = {
        "FRONT_LEFT": (0, 0),
        "FRONT": (480, 0),
        "FRONT_RIGHT": (960, 0),
        "SIDE_LEFT": (0, 320),
        "SIDE_RIGHT": (960, 320),
    }

    labels_by_frame = defaultdict(list)
    for r in cf_rows:
        cam = r.get("camera")
        raw_labels = r.get("labels")
        if not raw_labels or cam not in CAM_POS:
            continue
        x_off, y_off = CAM_POS[cam]
        orig_w, orig_h = CAM_DIMS[cam]
        for lb in raw_labels:
            norm_x = (x_off + (lb["x"] / orig_w) * 480.0) / 1440.0
            norm_y = (y_off + (lb["y"] / orig_h) * 320.0) / 640.0
            # Clamp to [0, 1]
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            labels_by_frame[(r["episode_id"], r["frame_index"])].append({
                "label": lb.get("text", "object"),
                "norm_x": norm_x,
                "norm_y": norm_y,
                "camera": cam
            })

    print("Loading fused frame embeddings from LanceDB...")
    fused_by_frame = {}
    table_names = db.list_tables()
    if hasattr(table_names, "tables"):
        table_names = table_names.tables
    names = [n if isinstance(n, str) else n[0] for n in table_names]

    if "frame_fused_embeddings" in names:
        ff = db.open_table("frame_fused_embeddings")
        fused_rows = (ff.search()
                      .where(f"episode_id IN ({ep_list})")
                      .select(["episode_id", "frame_index", "vector"])
                      .to_arrow().to_pylist())
        for r in fused_rows:
            fused_by_frame[(r["episode_id"], r["frame_index"])] = r["vector"]

    return boxes_by_frame, pose_by_frame, labels_by_frame, fused_by_frame


def build_video_dataset(limit_episodes=None):
    """Builds the main Episode Video Dataset with frame sequence attached."""
    configure()
    db = lancedb.connect(SRC_DB)
    eps_tbl = db.open_table("episodes")
    ep_rows = eps_tbl.to_arrow().to_pylist()
    perceptions = [r for r in ep_rows if r.get("source_type") == "perception"]
    if limit_episodes:
        perceptions = perceptions[:limit_episodes]

    print(f"\n==========================================")
    print(f"Building FiftyOne Video Dataset: {VIDEO_DS}")
    print(f"Episodes: {len(perceptions)}")
    print(f"==========================================")

    if fo.dataset_exists(VIDEO_DS):
        print(f"Deleting existing {VIDEO_DS}...")
        fo.delete_dataset(VIDEO_DS)

    ds = fo.Dataset(VIDEO_DS)
    ds.media_type = "video"
    ds.persistent = True
    ds.save()

    ep_ids = [e["episode_id"] for e in perceptions]
    boxes_by_frame, pose_by_frame, labels_by_frame, fused_by_frame = load_side_tables(db, ep_ids)

    samples = []
    for ep in perceptions:
        ep_id = ep["episode_id"]
        video_path = os.path.join(VIDEO_DIR, f"{ep_id}.mp4")
        if not os.path.isfile(video_path):
            print(f"Warning: video not found for {ep_id}: {video_path}")
            continue

        s = fo.Sample(filepath=video_path)
        s["episode_id"] = ep_id
        s["frame_count"] = ep["frame_count"]
        s["duration_s"] = ep["duration_s"]
        s["num_objects_3d"] = ep["num_objects_3d"]
        s["num_lidar_points"] = ep["num_lidar_points"]
        s["source_type"] = ep.get("source_type", "perception")

        ep_emb = ep.get("episode_embedding")
        if ep_emb is not None:
            s["episode_embedding"] = list(ep_emb)

        s.tags = ["waymo", "foxglove_surround", "perception"]
        samples.append((s, ep))

    # Add samples to dataset
    sample_objs = [s for s, _ in samples]
    ds.add_samples(sample_objs)
    print(f"Added {len(sample_objs)} video episode samples.")

    # Now populate sample.frames for each video sample
    print("\nPopulating frame-level sequences (3D boxes, HUD, 2D keypoints, embeddings)...")
    for s, ep in samples:
        ep_id = ep["episode_id"]
        fc = ep["frame_count"]
        t0 = time.time()
        speeds = []

        for fi in range(fc):
            fn = fi + 1  # 1-indexed for FiftyOne frames
            frame = s.frames[fn]

            # Pose & speed
            pose = pose_by_frame.get((ep_id, fi))
            if pose:
                sp = pose.get("speed_mps") or 0.0
                frame["speed_mps"] = sp
                frame["ego_px"] = pose.get("px") or 0.0
                frame["ego_py"] = pose.get("py") or 0.0
                frame["ego_pz"] = pose.get("pz") or 0.0
                speeds.append(sp)

            # 3D bounding boxes
            b3 = boxes_by_frame.get((ep_id, fi), [])
            if b3:
                frame["detections_3d"] = fo.Detections(detections=[
                    fo.Detection(
                        label=b["label"],
                        location=[b["pos_x"], b["pos_y"], b["pos_z"]],
                        dimensions=[b["size_x"], b["size_y"], b["size_z"]],
                        rotation=[b["quat_x"], b["quat_y"], b["quat_z"], b["quat_w"]],
                    ) for b in b3
                ])
                frame["num_objects"] = len(b3)
            else:
                frame["num_objects"] = 0

            # 2D Keypoints and Detections normalized on composite surround canvas
            labels = labels_by_frame.get((ep_id, fi))
            if labels:
                frame["keypoints"] = fo.Keypoints(keypoints=[fo.Keypoint(
                    points=[[lb["norm_x"], lb["norm_y"]] for lb in labels],
                    labels=[lb["label"] for lb in labels],
                )])
                box_w = 40.0 / 1440.0
                box_h = 30.0 / 640.0
                frame["detections_2d"] = fo.Detections(detections=[
                    fo.Detection(
                        label=lb["label"],
                        bounding_box=[
                            max(0.0, min(1.0 - box_w, lb["norm_x"] - box_w / 2.0)),
                            max(0.0, min(1.0 - box_h, lb["norm_y"] - box_h / 2.0)),
                            box_w,
                            box_h
                        ],
                    ) for lb in labels
                ])

            # Fused 360° embedding
            fused = fused_by_frame.get((ep_id, fi))
            if fused is not None:
                frame["embedding"] = list(fused)

            # PCD point cloud file path
            pcd_file = os.path.join(PCD_DIR, ep_id, f"frame_{fi:04d}.pcd")
            if os.path.isfile(pcd_file):
                frame["pcd_path"] = pcd_file

            # JPEG frame path for FiftyOne frame view thumbnail
            jpg_file = os.path.join(FRAMES_MEDIA_ROOT, ep_id, f"{fi:04d}_FRONT.jpg")
            if os.path.isfile(jpg_file):
                frame["filepath"] = jpg_file

        if speeds:
            s["avg_speed_mps"] = float(np.mean(speeds))
            s["max_speed_mps"] = float(np.max(speeds))

        s.save()
        print(f"  Saved {fc} frames for {ep_id[:36]}... in {time.time() - t0:.2f}s")

    print("\nComputing video metadata for all episode video samples...")
    try:
        ds.compute_metadata()
    except Exception as e:
        print(f"Warning computing metadata: {e}")

    print(f"\n{VIDEO_DS} built successfully!")
    print(ds)
    return ds


def build_grouped_dataset(limit_episodes=None):
    """Builds the Grouped Dataset linking Foxglove Video and 3D Visualizer Scene."""
    configure()
    db = lancedb.connect(SRC_DB)
    eps_tbl = db.open_table("episodes")
    ep_rows = eps_tbl.to_arrow().to_pylist()
    perceptions = [r for r in ep_rows if r.get("source_type") == "perception"]
    if limit_episodes:
        perceptions = perceptions[:limit_episodes]

    print(f"\n==========================================")
    print(f"Building FiftyOne Grouped Dataset: {GROUP_DS}")
    print(f"==========================================")

    if fo.dataset_exists(GROUP_DS):
        print(f"Deleting existing {GROUP_DS}...")
        fo.delete_dataset(GROUP_DS)

    ds = fo.Dataset(GROUP_DS)
    ds.add_group_field("group", default="video")
    ds.persistent = True
    ds.save()

    samples = []
    for ep in perceptions:
        ep_id = ep["episode_id"]
        video_path = os.path.join(VIDEO_DIR, f"{ep_id}.mp4")
        scene_path = os.path.join(SCENE_DIR, ep_id, "frame_0000.fo3d")
        pcd_path = os.path.join(PCD_DIR, ep_id, "frame_0000.pcd")

        if not os.path.isfile(video_path):
            continue

        group = fo.Group()

        # Video slice
        s_vid = fo.Sample(filepath=video_path, group=group.element("video"))
        s_vid["episode_id"] = ep_id
        s_vid["frame_count"] = ep["frame_count"]
        s_vid["duration_s"] = ep["duration_s"]
        s_vid["num_objects_3d"] = ep["num_objects_3d"]
        samples.append(s_vid)

        # 3D slice (prefer .fo3d if present, else .pcd)
        target_3d = scene_path if os.path.isfile(scene_path) else pcd_path
        if os.path.isfile(target_3d):
            s_3d = fo.Sample(filepath=target_3d, group=group.element("lidar_3d"))
            s_3d["episode_id"] = ep_id
            s_3d["frame_index"] = 0
            s_3d["num_lidar_points"] = ep["num_lidar_points"]
            samples.append(s_3d)

    ds.add_samples(samples)
    print(f"Built {GROUP_DS} with {len(ds)} samples across slices: {ds.group_slices}")
    return ds


def attach_brain_runs(ds):
    """Attach LanceDB-backed Brain Similarity and 2D PCA Visualization."""
    print(f"\n==========================================")
    print(f"Attaching FiftyOne Brain Runs to {ds.name}...")
    print(f"==========================================")

    # 1. Episode-level Similarity & 2D PCA
    ep_ids, ep_vecs = [], []
    for s in ds.iter_samples():
        emb = s.get_field("episode_embedding")
        if emb is not None:
            ep_ids.append(s.id)
            ep_vecs.append(emb)

    if ep_vecs:
        arr = np.asarray(ep_vecs, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)

        print(f"Registering episode_sim (LanceDB backend, {len(ep_vecs)} vectors)...")
        if "episode_sim" in ds.list_brain_runs():
            ds.delete_brain_run("episode_sim")

        ldb = lancedb.connect(FO_DB)
        t_names = ldb.table_names()
        if hasattr(t_names, "tables"):
            t_names = t_names.tables
        t_names = [n if isinstance(n, str) else n[0] for n in t_names]
        if "fo_episodes_clip" in t_names:
            ldb.drop_table("fo_episodes_clip")

        fob.compute_similarity(
            ds,
            embeddings=arr,
            backend="lancedb",
            brain_key="episode_sim",
            table_name="fo_episodes_clip",
            uri=FO_DB,
            metric="cosine"
        )
        print("  Attached episode_sim!")

        # 2D PCA for Embeddings Panel
        print("Computing 2D PCA for episode_viz...")
        X = arr.astype(np.float64)
        Xc = X - X.mean(axis=0)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        pts = U[:, :2] * S[:2]
        if "episode_viz" in ds.list_brain_runs():
            ds.delete_brain_run("episode_viz")
        fob.compute_visualization(
            ds, points=pts, brain_key="episode_viz", num_dims=2, method="manual"
        )
        print("  Attached episode_viz (PCA 2D)!")

FRAMES_DS = "waymo-fused-frames"


def build_fused_frames_dataset(limit_episodes=None):
    """Builds the 360° fused frame dataset (1,190 samples) with full Brain runs."""
    configure()
    db = lancedb.connect(SRC_DB)
    eps_tbl = db.open_table("episodes")
    ep_rows = eps_tbl.to_arrow().to_pylist()
    perceptions = [r for r in ep_rows if r.get("source_type") == "perception"]
    if limit_episodes:
        perceptions = perceptions[:limit_episodes]

    print(f"\n==========================================")
    print(f"Building FiftyOne Fused Frames Dataset: {FRAMES_DS}")
    print(f"==========================================")

    if fo.dataset_exists(FRAMES_DS):
        print(f"Deleting existing {FRAMES_DS}...")
        fo.delete_dataset(FRAMES_DS)

    ds = fo.Dataset(FRAMES_DS)
    ds.media_type = "image"
    ds.persistent = True
    ds.save()

    ep_ids = [e["episode_id"] for e in perceptions]
    boxes_by_frame, pose_by_frame, labels_by_frame, fused_by_frame = load_side_tables(db, ep_ids)

    samples = []
    for ep in perceptions:
        ep_id = ep["episode_id"]
        fc = ep["frame_count"]
        for fi in range(fc):
            jpg_file = os.path.join(FRAMES_MEDIA_ROOT, ep_id, f"{fi:04d}_FRONT.jpg")
            if not os.path.isfile(jpg_file):
                continue
            s = fo.Sample(filepath=jpg_file)
            s["episode_id"] = ep_id
            s["frame_index"] = fi
            s["timestamp_ns"] = fi * 100000000

            pose = pose_by_frame.get((ep_id, fi))
            if pose:
                s["speed_mps"] = pose.get("speed_mps") or 0.0
                s["ego_px"] = pose.get("px") or 0.0
                s["ego_py"] = pose.get("py") or 0.0
                s["ego_pz"] = pose.get("pz") or 0.0

            b3 = boxes_by_frame.get((ep_id, fi), [])
            if b3:
                s["detections_3d"] = fo.Detections(detections=[
                    fo.Detection(
                        label=b["label"],
                        location=[b["pos_x"], b["pos_y"], b["pos_z"]],
                        dimensions=[b["size_x"], b["size_y"], b["size_z"]],
                        rotation=[b["quat_x"], b["quat_y"], b["quat_z"], b["quat_w"]],
                    ) for b in b3
                ])
                s["num_objects"] = len(b3)

            labels = labels_by_frame.get((ep_id, fi))
            if labels:
                s["keypoints"] = fo.Keypoints(keypoints=[fo.Keypoint(
                    points=[[lb["x"], lb["y"]] for lb in labels],
                    labels=[lb["text"] for lb in labels],
                )])

            fused = fused_by_frame.get((ep_id, fi))
            if fused is not None:
                s["embedding"] = list(fused)

            pcd_file = os.path.join(PCD_DIR, ep_id, f"frame_{fi:04d}.pcd")
            if os.path.isfile(pcd_file):
                s["pcd_path"] = pcd_file

            samples.append(s)

    ds.add_samples(samples, progress=True)
    print(f"Added {len(ds)} samples to {FRAMES_DS}.")

    # Brain runs on waymo-fused-frames
    ids, vecs = [], []
    for s in ds.iter_samples():
        emb = s.get_field("embedding")
        if emb is not None:
            ids.append(s.id)
            vecs.append(emb)

    if vecs:
        arr = np.asarray(vecs, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)

        print(f"Attaching frame_sim ({len(vecs)} vectors) with LanceDB backend...")
        if "frame_sim" in ds.list_brain_runs():
            ds.delete_brain_run("frame_sim")

        fob.compute_similarity(
            ds,
            embeddings=arr,
            backend="lancedb",
            brain_key="frame_sim",
            table_name="fo_frames_fused_clip",
            uri=FO_DB,
            metric="cosine"
        )

        print("Computing 2D PCA for frame_viz...")
        X = arr.astype(np.float64)
        Xc = X - X.mean(axis=0)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        pts = U[:, :2] * S[:2]
        if "frame_viz" in ds.list_brain_runs():
            ds.delete_brain_run("frame_viz")
        fob.compute_visualization(
            ds, points=pts, brain_key="frame_viz", num_dims=2, method="manual"
        )
        print("Attached frame_sim and frame_viz to", FRAMES_DS)

    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="store_true", help="build video dataset")
    parser.add_argument("--grouped", action="store_true", help="build grouped dataset")
    parser.add_argument("--frames", action="store_true", help="build fused frames dataset")
    parser.add_argument("--brain", action="store_true", help="compute brain similarity/viz")
    parser.add_argument("--limit", type=int, default=None, help="limit episodes")
    args = parser.parse_args()

    if not args.video and not args.grouped and not args.frames and not args.brain:
        args.video = args.grouped = args.frames = args.brain = True

    configure()
    if fo.dataset_exists("waymo-curated"):
        print("Cleaning up legacy flat images dataset waymo-curated...")
        fo.delete_dataset("waymo-curated")

    ds_video = None
    if args.video:
        ds_video = build_video_dataset(limit_episodes=args.limit)

    if args.grouped:
        build_grouped_dataset(limit_episodes=args.limit)

    if args.frames:
        build_fused_frames_dataset(limit_episodes=args.limit)

    if args.brain:
        if ds_video is None:
            configure()
            ds_video = fo.load_dataset(VIDEO_DS)
        attach_brain_runs(ds_video)

    print("\nREBUILD COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
