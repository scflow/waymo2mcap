#!/usr/bin/env python3
"""
LanceDB Annotation & Perception Sync Server
Serves multi-modal episode data from LanceDB to Foxglove Studio Annotation Workspace:
- Episodes metadata & stats
- Frame point clouds (downsampled for 60fps web rendering)
- Camera calibration (K, T_veh_to_cam)
- Front camera HD JPEG images
- 3D bounding boxes with persistence back to LanceDB objects_3d table
"""

import os
import sys
import json
import io
import time
import math
import numpy as np
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import lancedb

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "lancedb")
PORT = 8765

# Connect to LanceDB
print(f"[Server] Connecting to LanceDB at: {DB_PATH}")
db = lancedb.connect(DB_PATH)
episodes_table = db.open_table("episodes")
camera_table = db.open_table("camera_frames")
lidar_table = db.open_table("lidar_frames")
objects_table = db.open_table("objects_3d")

# Camera Calibration for FRONT camera (Waymo benchmark calibration)
DEFAULT_CALIB = {
    "K": [
        2059.61201155, 0.0, 952.41218988,
        0.0, 2059.61201155, 634.58720825,
        0.0, 0.0, 1.0
    ],
    "width": 1920,
    "height": 1280,
    # 4x4 matrix from camera frame to vehicle base_link
    "extrinsic_cam_to_veh": [
        0.99997851,  0.00314207,  0.00575409,  1.53914676,
        -0.0032445,   0.9998349,   0.01787846, -0.02402951,
        -0.00569697, -0.01789675,  0.99982361,  2.11577847,
        0.0,          0.0,          0.0,          1.0
    ],
    "translation": [1.53914676, -0.02402951, 2.11577847],
}

def quat_to_yaw(qx, qy, qz, qw):
    """Compute yaw angle (around vehicle Z up axis) from quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

def yaw_to_quat(yaw):
    """Convert yaw angle (around vehicle Z axis) to quaternion [qx, qy, qz, qw]."""
    half = yaw * 0.5
    return [0.0, 0.0, math.sin(half), math.cos(half)]

class AnnotationHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS for Foxglove dev server (localhost:8080) and desktop
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        query = parse_qs(parsed.query)

        # 1. Health check
        if path == "/api/health" or path == "":
            self.send_json({"status": "ok", "service": "LanceDB Annotation Server", "port": PORT})
            return

        # 2. List episodes
        if path == "/api/episodes":
            try:
                df = episodes_table.to_pandas()
                episodes = []
                for _, row in df.iterrows():
                    episodes.append({
                        "episode_id": str(row["episode_id"]),
                        "frame_count": int(row.get("frame_count", 0)),
                        "duration_s": float(row.get("duration_s", 0.0)),
                        "num_objects_3d": int(row.get("num_objects_3d", 0)),
                        "num_images": int(row.get("num_images", 0)),
                    })
                # Sort alphabetically
                episodes.sort(key=lambda x: x["episode_id"])
                self.send_json({"status": "success", "episodes": episodes})
            except Exception as e:
                self.send_error_json(500, f"Failed to list episodes: {e}")
            return

        # 3. Stream camera image: /api/episodes/{episode_id}/frame/{frame_idx}/image
        if path.endswith("/image") and "/frame/" in path:
            parts = path.split("/")
            # /api/episodes/{id}/frame/{frame_idx}/image
            try:
                ep_id = parts[3]
                frame_idx = int(parts[5])
                cam_name = query.get("camera", ["FRONT"])[0]

                rows = camera_table.search().where(
                    f"episode_id = '{ep_id}' AND frame_index = {frame_idx} AND camera = '{cam_name}'"
                ).limit(1).to_pandas()

                if len(rows) == 0:
                    self.send_error_json(404, "Frame image not found")
                    return

                img_bytes = rows["image"].iloc[0]
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(img_bytes)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self.end_headers()
                self.wfile.write(img_bytes)
            except Exception as e:
                self.send_error_json(500, f"Error streaming image: {e}")
            return

        # 4. Get frame full data: /api/episodes/{episode_id}/frame/{frame_idx}
        if "/frame/" in path and not path.endswith("/image"):
            parts = path.split("/")
            try:
                ep_id = parts[3]
                frame_idx = int(parts[5])

                # A. Query 3D objects
                obj_rows = objects_table.search().where(
                    f"episode_id = '{ep_id}' AND frame_index = {frame_idx}"
                ).to_pandas()

                objects = []
                for _, row in obj_rows.iterrows():
                    qx = float(row.get("quat_x", 0.0))
                    qy = float(row.get("quat_y", 0.0))
                    qz = float(row.get("quat_z", 0.0))
                    qw = float(row.get("quat_w", 1.0))
                    yaw = quat_to_yaw(qx, qy, qz, qw)

                    objects.append({
                        "id": str(row.get("object_id", "")),
                        "label": str(row.get("label", "VEHICLE")),
                        "center": [
                            float(row.get("pos_x", 0.0)),
                            float(row.get("pos_y", 0.0)),
                            float(row.get("pos_z", 0.0))
                        ],
                        "size": [
                            float(row.get("size_x", 4.0)),
                            float(row.get("size_y", 2.0)),
                            float(row.get("size_z", 1.6))
                        ],
                        "yaw": yaw,
                        "quaternion": [qx, qy, qz, qw],
                        "color": [
                            float(row.get("color_r", 0.2)),
                            float(row.get("color_g", 0.8)),
                            float(row.get("color_b", 0.3)),
                            float(row.get("color_a", 0.6))
                        ]
                    })

                # B. Query LiDAR point cloud
                lidar_rows = lidar_table.search().where(
                    f"episode_id = '{ep_id}' AND frame_index = {frame_idx}"
                ).limit(1).to_pandas()

                points_data = []
                num_points = 0
                if len(lidar_rows) > 0:
                    raw_bytes = lidar_rows["points"].iloc[0]
                    stride = int(lidar_rows["point_stride"].iloc[0]) if "point_stride" in lidar_rows else 20
                    # raw points: float32 [x, y, z, intensity, elongation]
                    pts_np = np.frombuffer(raw_bytes, dtype=np.float32).reshape(-1, stride // 4)
                    num_points = len(pts_np)

                    # Downsample to ~15,000 points for ultra-smooth 60 FPS Three.js rendering
                    step = max(1, len(pts_np) // 15000)
                    downsampled = pts_np[::step]

                    # Extract [x, y, z, intensity]
                    points_data = downsampled[:, :4].round(3).tolist()

                # C. Query 2D labels on front camera
                cam_rows = camera_table.search().where(
                    f"episode_id = '{ep_id}' AND frame_index = {frame_idx} AND camera = 'FRONT'"
                ).limit(1).to_pandas()
                labels_2d = []
                if len(cam_rows) > 0 and "labels" in cam_rows:
                    raw_lbls = cam_rows["labels"].iloc[0]
                    if raw_lbls is not None and len(raw_lbls) > 0:
                        labels_2d = [{"text": str(l.get("text", "")), "x": float(l.get("x", 0)), "y": float(l.get("y", 0))} for l in raw_lbls]

                response_data = {
                    "status": "success",
                    "episode_id": ep_id,
                    "frame_index": frame_idx,
                    "image_url": f"http://127.0.0.1:{PORT}/api/episodes/{ep_id}/frame/{frame_idx}/image",
                    "calibration": DEFAULT_CALIB,
                    "objects": objects,
                    "labels_2d": labels_2d,
                    "points": points_data,
                    "total_points": num_points
                }
                self.send_json(response_data)
            except Exception as e:
                self.send_error_json(500, f"Error getting frame data: {e}")
            return

        self.send_error_json(404, "Endpoint not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        # Update single 3D object annotation: /api/objects/update
        if path == "/api/objects/update":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                payload = json.loads(body)
                obj_id = payload.get("object_id")
                if not obj_id:
                    self.send_error_json(400, "Missing object_id")
                    return

                pos = payload.get("pos") or payload.get("center")
                size = payload.get("size")
                yaw = payload.get("yaw")
                quat = payload.get("quat") or payload.get("quaternion")
                if quat is None and yaw is not None:
                    quat = yaw_to_quat(float(yaw))
                label = payload.get("label")

                update_values = {}
                if label is not None:
                    update_values["label"] = str(label)
                    color_map = {
                        "VEHICLE": [0.2, 0.8, 0.3, 0.7],
                        "Car": [0.2, 0.8, 0.3, 0.7],
                        "PEDESTRIAN": [1.0, 0.6, 0.0, 0.7],
                        "Pedestrian": [1.0, 0.6, 0.0, 0.7],
                        "CYCLIST": [0.9, 0.2, 0.8, 0.7],
                        "Cyclist": [0.9, 0.2, 0.8, 0.7],
                        "SIGN": [0.6, 0.3, 1.0, 0.7],
                        "Sign": [0.6, 0.3, 1.0, 0.7],
                    }
                    col = color_map.get(label, [0.3, 0.7, 1.0, 0.7])
                    update_values["color_r"] = float(col[0])
                    update_values["color_g"] = float(col[1])
                    update_values["color_b"] = float(col[2])
                    update_values["color_a"] = float(col[3])

                if pos and len(pos) >= 3:
                    update_values["pos_x"] = float(pos[0])
                    update_values["pos_y"] = float(pos[1])
                    update_values["pos_z"] = float(pos[2])
                if size and len(size) >= 3:
                    update_values["size_x"] = float(size[0])
                    update_values["size_y"] = float(size[1])
                    update_values["size_z"] = float(size[2])
                if quat and len(quat) >= 4:
                    update_values["quat_x"] = float(quat[0])
                    update_values["quat_y"] = float(quat[1])
                    update_values["quat_z"] = float(quat[2])
                    update_values["quat_w"] = float(quat[3])

                print(f"[Server] Updating object {obj_id} in LanceDB: {update_values}")
                objects_table.update(where=f"object_id = '{obj_id}'", values=update_values)

                self.send_json({
                    "status": "success",
                    "message": f"Successfully updated object {obj_id} in LanceDB",
                    "object_id": obj_id,
                    "updated_fields": list(update_values.keys())
                })
            except Exception as e:
                print(f"[Server] Error updating object: {e}")
                self.send_error_json(500, f"Failed to update object: {e}")
            return

        # Save annotations: /api/episodes/{episode_id}/frame/{frame_idx}/save
        if path.endswith("/save") and "/frame/" in path:
            parts = path.split("/")
            try:
                ep_id = parts[3]
                frame_idx = int(parts[5])

                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length).decode("utf-8")
                payload = json.loads(body)
                updated_objects = payload.get("objects", [])

                # 1. Delete existing objects for this frame
                print(f"[Server] Deleting existing objects for {ep_id} frame {frame_idx}...")
                objects_table.delete(f"episode_id = '{ep_id}' AND frame_index = {frame_idx}")

                # 2. Insert new/updated objects
                new_records = []
                ts_ns = int(time.time() * 1e9)
                for i, obj in enumerate(updated_objects):
                    center = obj.get("center", [0.0, 0.0, 0.0])
                    size = obj.get("size", [4.0, 2.0, 1.6])
                    yaw = float(obj.get("yaw", 0.0))
                    quat = obj.get("quaternion", yaw_to_quat(yaw))
                    label = str(obj.get("label", "VEHICLE"))
                    obj_id = str(obj.get("id") or f"user_obj_{i}")

                    # Colors by label
                    color_map = {
                        "VEHICLE": [0.2, 0.8, 0.3, 0.7],
                        "PEDESTRIAN": [1.0, 0.6, 0.0, 0.7],
                        "CYCLIST": [0.9, 0.2, 0.8, 0.7],
                        "SIGN": [0.6, 0.3, 1.0, 0.7]
                    }
                    col = color_map.get(label, [0.3, 0.7, 1.0, 0.7])

                    new_records.append({
                        "episode_id": ep_id,
                        "frame_index": frame_idx,
                        "timestamp_ns": ts_ns,
                        "object_id": obj_id,
                        "label": label,
                        "pos_x": float(center[0]),
                        "pos_y": float(center[1]),
                        "pos_z": float(center[2]),
                        "size_x": float(size[0]),
                        "size_y": float(size[1]),
                        "size_z": float(size[2]),
                        "quat_x": float(quat[0]),
                        "quat_y": float(quat[1]),
                        "quat_z": float(quat[2]),
                        "quat_w": float(quat[3]),
                        "color_r": float(col[0]),
                        "color_g": float(col[1]),
                        "color_b": float(col[2]),
                        "color_a": float(col[3]),
                    })

                if new_records:
                    objects_table.add(new_records)

                print(f"[Server] Successfully saved {len(new_records)} objects to LanceDB for {ep_id} frame {frame_idx}.")
                self.send_json({
                    "status": "success",
                    "message": f"Successfully updated {len(new_records)} objects in LanceDB",
                    "saved_count": len(new_records)
                })
            except Exception as e:
                self.send_error_json(500, f"Failed to save annotations: {e}")
            return

        self.send_error_json(404, "POST endpoint not found")

    def send_json(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, code, message):
        self.send_json({"status": "error", "message": message}, code=code)

    def log_message(self, format, *args):
        # Concise logging
        if "GET /api/episodes/" in args[0] and "/image" in args[0]:
            return
        sys.stderr.write(f"[{time.strftime('%X')}] {args[0]} - {args[1]}\n")

def run():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), AnnotationHandler)
    print(f"\n=======================================================")
    print(f"  LanceDB Annotation Bridge Server running on:")
    print(f"  http://127.0.0.1:{PORT}")
    print(f"  Episodes ready: {episodes_table.count_rows()}")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()

if __name__ == "__main__":
    run()
