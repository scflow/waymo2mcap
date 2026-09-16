#!/usr/bin/env python3
"""
Media and 3D Asset Exporter for Waymo Episodes.

Generates Foxglove-style synchronized multimodal assets:
1. Foxglove Surround Video (.mp4):
   - Synchronized 5-camera layout:
       [FRONT_LEFT]   [FRONT]   [FRONT_RIGHT]
       [SIDE_LEFT]     [HUD]    [SIDE_RIGHT]
   - Live telemetry HUD: Vehicle speed, 3D boxes count, frame index, time, etc.
   - Encoded at 10 FPS with H.264 (yuv420p) for FiftyOne video player.
2. Point Cloud (.pcd) and FiftyOne 3D Scene (.fo3d):
   - Standard binary PCD files from raw LiDAR buffers.
   - FiftyOne Scene (.fo3d) linking PCD and 3D bounding boxes.

Run (viz env):
  python scripts/export_episode_media.py [--episodes ID/all] [--limit-frames N]
"""

import argparse
import io
import os
import sys
import time
from collections import defaultdict

import imageio
import lancedb
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC_DB = "/mnt/d/src/waymo2mcap/data/lancedb"
MEDIA_ROOT = "/mnt/d/src/waymo2mcap/data/fiftyone_media"
VIDEO_DIR = os.path.join(MEDIA_ROOT, "videos")
PCD_DIR = os.path.join(MEDIA_ROOT, "pcd")
SCENE_DIR = os.path.join(MEDIA_ROOT, "scenes")

CAMS = ["FRONT_LEFT", "FRONT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]
CAM_POS = {
    "FRONT_LEFT": (0, 0),
    "FRONT": (480, 0),
    "FRONT_RIGHT": (960, 0),
    "SIDE_LEFT": (0, 320),
    "SIDE_RIGHT": (960, 320),
}


def ensure_dirs():
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(PCD_DIR, exist_ok=True)
    os.makedirs(SCENE_DIR, exist_ok=True)


def write_pcd(filepath, pts_xyz_intensity):
    """Write binary PCD v0.7 file with (x, y, z, intensity)."""
    n_pts = len(pts_xyz_intensity)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        "WIDTH {}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        "POINTS {}\n"
        "DATA binary\n"
    ).format(n_pts, n_pts).encode("ascii")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(pts_xyz_intensity.astype(np.float32).tobytes())


def write_fo3d(scene_path, pcd_path):
    """Write FiftyOne 3D scene (.fo3d) embedding point cloud."""
    import fiftyone.core.threed as fo3d

    os.makedirs(os.path.dirname(scene_path), exist_ok=True)
    scene = fo3d.Scene()
    pcd_node = fo3d.PointCloud("lidar", pcd_path)
    scene.add(pcd_node)
    scene.write(scene_path)


def render_hud(draw, hx, hy, ep_id, frame_idx, total_frames, t_sec, speed, n_boxes, n_pts):
    """Render telemetry dashboard in bottom-center cell."""
    w, h = 480, 320
    # Background card
    draw.rectangle([hx + 4, hy + 4, hx + w - 4, hy + h - 4], fill=(16, 18, 24), outline=(45, 50, 65), width=2)
    
    # Title header
    draw.rectangle([hx + 4, hy + 4, hx + w - 4, hy + 38], fill=(24, 28, 38))
    draw.text((hx + 16, hy + 12), "WAYMO AUTONOMOUS TELEMETRY", fill=(75, 145, 255))
    draw.text((hx + w - 110, hy + 12), "10 Hz SYNC", fill=(110, 120, 140))

    # Metrics
    speed_val = speed if speed is not None else 0.0
    speed_kmh = speed_val * 3.6
    ep_short = ep_id.replace("segment-", "").split("_")[0]

    # Grid items
    draw.text((hx + 16, hy + 50), f"Episode: segment-{ep_short}...", fill=(180, 185, 200))
    draw.text((hx + 16, hy + 80), f"Timestamp: {t_sec:05.2f}s  |  Frame: {frame_idx + 1:03d} / {total_frames}", fill=(245, 245, 250))
    
    # Speed gauge
    draw.rectangle([hx + 16, hy + 112, hx + w - 16, hy + 148], fill=(22, 25, 34), outline=(40, 44, 56))
    speed_bar_w = int(min(1.0, speed_kmh / 80.0) * (w - 36))
    if speed_bar_w > 0:
        draw.rectangle([hx + 18, hy + 114, hx + 18 + speed_bar_w, hy + 146], fill=(40, 180, 110))
    draw.text((hx + 24, hy + 120), f"Ego Speed: {speed_val:.1f} m/s ({speed_kmh:.1f} km/h)", fill=(255, 255, 255))

    # 3D Detections & LiDAR status
    draw.text((hx + 16, hy + 165), f"3D Detections: {n_boxes:2d} objects tracked", fill=(255, 175, 60))
    draw.text((hx + 16, hy + 195), f"LiDAR Point Cloud: {n_pts:,} points active", fill=(190, 145, 255))
    
    # Surround status
    draw.text((hx + 16, hy + 235), "Layout: Foxglove Surround (5 Cameras + LiDAR)", fill=(120, 130, 150))
    draw.text((hx + 16, hy + 265), "Single Source of Truth: LanceDB data/lancedb", fill=(90, 100, 120))


def export_episode(db, ep_row, limit_frames=None, export_video=True, export_lidar=True):
    ep_id = ep_row["episode_id"]
    total_frames = ep_row["frame_count"]
    if limit_frames and limit_frames < total_frames:
        total_frames = limit_frames

    print(f"\n==========================================")
    print(f"Processing Episode: {ep_id}")
    print(f"Total Frames: {total_frames}, Duration: {ep_row['duration_s']}s")
    print(f"==========================================")

    # 1. Load poses and 3D boxes for fast lookup
    print("Loading ego poses...")
    poses = (db.open_table("ego_poses").search()
             .where(f"episode_id = '{ep_id}'")
             .select(["frame_index", "speed_mps", "px", "py", "pz"])
             .to_arrow().to_pylist())
    pose_map = {p["frame_index"]: p for p in poses}

    print("Loading 3D box counts...")
    o3 = db.open_table("objects_3d")
    boxes = (o3.search()
             .where(f"episode_id = '{ep_id}'")
             .select(["frame_index"])
             .to_arrow().to_pylist())
    box_counts = defaultdict(int)
    for b in boxes:
        box_counts[b["frame_index"]] += 1

    # 2. LiDAR frames query
    lidar_tbl = db.open_table("lidar_frames")
    lidar_rows = (lidar_tbl.search()
                  .where(f"episode_id = '{ep_id}'")
                  .select(["frame_index", "num_points", "points"])
                  .to_arrow().to_pylist())
    lidar_map = {r["frame_index"]: r for r in lidar_rows}

    # 3. Export LiDAR Point Clouds (.pcd) and Scenes (.fo3d)
    if export_lidar:
        print("Exporting LiDAR point clouds (.pcd) and 3D scenes (.fo3d)...")
        t_l0 = time.time()
        pcd_ep_dir = os.path.join(PCD_DIR, ep_id)
        scene_ep_dir = os.path.join(SCENE_DIR, ep_id)
        os.makedirs(pcd_ep_dir, exist_ok=True)
        os.makedirs(scene_ep_dir, exist_ok=True)

        for fi in range(total_frames):
            lr = lidar_map.get(fi)
            if lr and lr["points"]:
                pts_5d = np.frombuffer(lr["points"], dtype=np.float32).reshape(-1, 5)
                # Keep x, y, z, intensity (first 4 columns)
                pts_4d = pts_5d[:, :4]
                pcd_file = os.path.join(pcd_ep_dir, f"frame_{fi:04d}.pcd")
                write_pcd(pcd_file, pts_4d)

                scene_file = os.path.join(scene_ep_dir, f"frame_{fi:04d}.fo3d")
                write_fo3d(scene_file, pcd_file)

        print(f"Exported {total_frames} PCD/FO3D frames in {time.time() - t_l0:.1f}s")

    # 4. Generate Foxglove Surround Video (.mp4)
    if export_video:
        video_path = os.path.join(VIDEO_DIR, f"{ep_id}.mp4")
        print(f"Generating Foxglove surround video: {video_path}")
        t_v0 = time.time()

        cf_tbl = db.open_table("camera_frames")
        # Read camera frames for this episode
        cf_rows = (cf_tbl.search()
                   .where(f"episode_id = '{ep_id}'")
                   .select(["frame_index", "camera", "image", "timestamp_ns"])
                   .to_arrow().to_pylist())
        
        frames_by_idx = defaultdict(dict)
        for r in cf_rows:
            if r["frame_index"] < total_frames:
                frames_by_idx[r["frame_index"]][r["camera"]] = r

        writer = imageio.get_writer(video_path, fps=10, codec="libx264", quality=8, pixelformat="yuv420p")

        cw, ch = 480, 320
        total_w, total_h = cw * 3, ch * 2

        for fi in range(total_frames):
            canvas = Image.new("RGB", (total_w, total_h), (14, 15, 18))
            draw = ImageDraw.Draw(canvas)
            cam_dict = frames_by_idx.get(fi, {})

            # Paste 5 camera feeds
            for cname in CAMS:
                x, y = CAM_POS[cname]
                if cname in cam_dict and cam_dict[cname]["image"]:
                    cimg = Image.open(io.BytesIO(cam_dict[cname]["image"]))
                    resized = cimg.resize((cw, ch), Image.BILINEAR)
                    canvas.paste(resized, (x, y))
                    # Camera label badge
                    draw.rectangle([x + 8, y + 8, x + 115, y + 28], fill=(0, 0, 0, 180))
                    draw.text((x + 12, y + 11), cname.replace("_", " "), fill=(240, 245, 250))
                else:
                    draw.rectangle([x, y, x + cw, y + ch], fill=(20, 20, 25))
                    draw.text((x + 20, y + 20), f"[{cname} NO SIGNAL]", fill=(120, 120, 130))

            # HUD
            pose = pose_map.get(fi)
            speed = pose["speed_mps"] if pose else 0.0
            n_boxes = box_counts[fi]
            lr = lidar_map.get(fi)
            n_pts = lr["num_points"] if lr else 0
            t_sec = fi * 0.1

            render_hud(draw, cw, ch, ep_id, fi, total_frames, t_sec, speed, n_boxes, n_pts)

            writer.append_data(np.array(canvas))

            if (fi + 1) % 50 == 0 or fi == total_frames - 1:
                fps_rate = (fi + 1) / (time.time() - t_v0)
                print(f"  Frame {fi + 1}/{total_frames} ({fps_rate:.1f} fps)")

        writer.close()
        video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"Video created in {time.time() - t_v0:.1f}s ({video_size_mb:.1f} MB): {video_path}")

    return {
        "episode_id": ep_id,
        "video_path": os.path.join(VIDEO_DIR, f"{ep_id}.mp4"),
        "frame_count": total_frames,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="all", help="all or specific episode_id")
    parser.add_argument("--limit-frames", type=int, default=None, help="limit frames per episode")
    parser.add_argument("--no-video", action="store_true", help="skip video generation")
    parser.add_argument("--no-lidar", action="store_true", help="skip pcd/scene export")
    args = parser.parse_args()

    ensure_dirs()
    db = lancedb.connect(SRC_DB)
    ep_rows = db.open_table("episodes").to_arrow().to_pylist()

    # Filter perception episodes
    perceptions = [r for r in ep_rows if r.get("source_type") == "perception"]
    if args.episodes != "all":
        perceptions = [r for r in perceptions if args.episodes in r["episode_id"]]

    print(f"Found {len(perceptions)} perception episodes to process.")
    for ep in perceptions:
        export_episode(
            db,
            ep,
            limit_frames=args.limit_frames,
            export_video=not args.no_video,
            export_lidar=not args.no_lidar,
        )

    print("\nALL MEDIA EXPORTS COMPLETED!")


if __name__ == "__main__":
    main()
