#!/usr/bin/env python3
"""
Waymo Open Dataset (TFRecord) to MCAP Converter.
Converts images, lidars, ego poses (TF), 3D/2D bounding boxes, and HD map features
into standard Foxglove-compatible MCAP files.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np

# Add src to sys.path for waymo_open_dataset protobuf imports
WORKSPACE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKSPACE_DIR / "src"))

import tfrecord
from mcap_protobuf.writer import Writer
from waymo_open_dataset import dataset_pb2

from converter.point_cloud_util import (
    decode_range_image,
    extract_point_cloud_from_range_image,
    create_foxglove_point_cloud
)
from converter.tf_util import create_frame_transforms
from converter.ego_util import (
    create_ego_pose,
    create_ego_odometry,
    create_ego_vehicle_marker,
    create_trajectory_scene_update,
    create_ego_poses_path
)
from converter.annotation_util import (
    create_3d_boxes_scene_update,
    create_camera_annotations
)
from converter.map_util import create_map_layer_updates
from converter.camera_util import (
    create_compressed_image,
    create_camera_calibration
)


def convert_tfrecord_to_mcap(
    input_path: str,
    output_path: str,
    max_frames: int | None = None,
    merge_lidar: bool = True,
    split_lidar: bool = False,
    include_second_return: bool = True,
    include_raw_frame: bool = False
):
    print(f"Opening TFRecord: {input_path}")
    print(f"Output MCAP:      {output_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Pre-scan poses for complete 3D trajectory ribbon
    print("Pre-scanning trajectory poses...")
    all_poses = []
    temp_reader = tfrecord.tfrecord_iterator(input_path)
    for record in temp_reader:
        f = dataset_pb2.Frame()
        f.ParseFromString(record)
        all_poses.append(np.array(f.pose.transform, dtype=np.float64).reshape(4, 4))
        if max_frames is not None and len(all_poses) >= max_frames:
            break
    print(f"Loaded {len(all_poses)} trajectory poses for 3D path visualization.")

    reader = tfrecord.tfrecord_iterator(input_path)
    start_time = time.time()
    frame_count = 0
    total_points = 0
    total_images = 0
    map_features_written = False

    with open(output_path, "wb") as f_out:
        writer = Writer(f_out)

        for record in reader:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(record)

            ts_micros = frame.timestamp_micros
            ts_ns = ts_micros * 1000

            # 1. Ego Pose and Static TF tree
            tf_msg = create_frame_transforms(frame, include_static=True)
            writer.write_message("/tf", tf_msg, log_time=ts_ns, publish_time=ts_ns)

            ego_pose_msg = create_ego_pose(frame)
            writer.write_message("/ego_pose", ego_pose_msg, log_time=ts_ns, publish_time=ts_ns)

            odom_msg = create_ego_odometry(frame)
            writer.write_message("/odometry", odom_msg, log_time=ts_ns, publish_time=ts_ns)

            ego_marker_msg = create_ego_vehicle_marker(frame)
            writer.write_message("/perception/ego_vehicle", ego_marker_msg, log_time=ts_ns, publish_time=ts_ns)

            # 3D Trajectory Ribbon (driven trail + upcoming guide in world frame)
            traj_msg = create_trajectory_scene_update(all_poses, frame_count, ts_micros)
            writer.write_message("/ego_trajectory", traj_msg, log_time=ts_ns, publish_time=ts_ns)

            path_msg = create_ego_poses_path(all_poses, ts_micros)
            writer.write_message("/ego_path", path_msg, log_time=ts_ns, publish_time=ts_ns)

            # 2. Camera Images & Calibrations
            camera_calibs = {c.name: c for c in frame.context.camera_calibrations}
            for img in frame.images:
                cam_name = dataset_pb2.CameraName.Name.Name(img.name)
                img_msg = create_compressed_image(img.image, f"camera_{cam_name}", ts_micros)
                writer.write_message(f"/camera/{cam_name}/compressed", img_msg, log_time=ts_ns, publish_time=ts_ns)
                total_images += 1

                # Write calibration info
                if img.name in camera_calibs:
                    calib_msg = create_camera_calibration(camera_calibs[img.name], f"camera_{cam_name}", ts_micros)
                    writer.write_message(f"/camera/{cam_name}/calibration", calib_msg, log_time=ts_ns, publish_time=ts_ns)

            # 3. 2D Camera Annotations
            for cam_label in frame.camera_labels:
                cam_name = dataset_pb2.CameraName.Name.Name(cam_label.name)
                ann_msg = create_camera_annotations(cam_label, ts_micros)
                writer.write_message(f"/camera/{cam_name}/annotations", ann_msg, log_time=ts_ns, publish_time=ts_ns)

            # 4. LiDAR Point Clouds
            laser_calibs = {c.name: c for c in frame.context.laser_calibrations}
            merged_pts_list = []

            for laser in frame.lasers:
                laser_name = dataset_pb2.LaserName.Name.Name(laser.name)
                calib = laser_calibs.get(laser.name)
                if not calib:
                    continue

                pts_laser_list = []
                # First return
                if len(laser.ri_return1.range_image_compressed) > 0:
                    ri1 = decode_range_image(laser.ri_return1.range_image_compressed)
                    pts1 = extract_point_cloud_from_range_image(
                        ri1, calib, transform_to_vehicle=merge_lidar
                    )
                    pts_laser_list.append(pts1)

                # Second return
                if include_second_return and len(laser.ri_return2.range_image_compressed) > 0:
                    ri2 = decode_range_image(laser.ri_return2.range_image_compressed)
                    pts2 = extract_point_cloud_from_range_image(
                        ri2, calib, transform_to_vehicle=merge_lidar
                    )
                    pts_laser_list.append(pts2)

                if pts_laser_list:
                    all_laser_pts = np.vstack(pts_laser_list) if len(pts_laser_list) > 1 else pts_laser_list[0]

                    if split_lidar:
                        if merge_lidar:
                            # If merged was requested, pts are in vehicle frame; convert back to sensor frame for split
                            # Or compute without vehicle transform
                            ri1_raw = decode_range_image(laser.ri_return1.range_image_compressed)
                            pts_raw = extract_point_cloud_from_range_image(ri1_raw, calib, transform_to_vehicle=False)
                            pc_msg = create_foxglove_point_cloud(pts_raw, f"laser_{laser_name}", ts_micros)
                        else:
                            pc_msg = create_foxglove_point_cloud(all_laser_pts, f"laser_{laser_name}", ts_micros)
                        writer.write_message(f"/lidar/{laser_name}", pc_msg, log_time=ts_ns, publish_time=ts_ns)

                    if merge_lidar:
                        merged_pts_list.append(all_laser_pts)

            if merge_lidar and merged_pts_list:
                merged_pts = np.vstack(merged_pts_list)
                total_points += len(merged_pts)
                merged_pc_msg = create_foxglove_point_cloud(merged_pts, "base_link", ts_micros)
                writer.write_message("/lidar/points", merged_pc_msg, log_time=ts_ns, publish_time=ts_ns)

            # 5. 3D Bounding Boxes (always send to clear/update scene entities)
            boxes_msg = create_3d_boxes_scene_update(frame)
            writer.write_message("/perception/boxes_3d", boxes_msg, log_time=ts_ns, publish_time=ts_ns)

            # 6. HD Map Features (split into sub-topics for layer toggling in 3D viewer)
            if frame.map_features and not map_features_written:
                map_layers = create_map_layer_updates(list(frame.map_features), ts_micros)
                for topic, layer_msg in map_layers.items():
                    writer.write_message(topic, layer_msg, log_time=ts_ns, publish_time=ts_ns)
                map_features_written = True
                print(f"  [Frame {frame_count:03d}] Written HD map with {len(frame.map_features)} features across {len(map_layers)} layer sub-topics:")
                for top in sorted(map_layers.keys()):
                    print(f"    - {top}")

            # 7. Optional Raw Frame
            if include_raw_frame:
                writer.write_message("/waymo/frame", frame, log_time=ts_ns, publish_time=ts_ns)

            frame_count += 1
            if frame_count % 10 == 0 or frame_count == 1:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"  Processed frame {frame_count:3d} ({fps:.1f} fps) | context: {frame.context.name}")

            if max_frames is not None and frame_count >= max_frames:
                print(f"Reached max frames limit ({max_frames}).")
                break

        writer.finish()

    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("\n" + "=" * 60)
    print(f"Conversion Complete!")
    print(f"  Total frames:    {frame_count}")
    print(f"  Total images:    {total_images}")
    print(f"  Total points:    {total_points:,}")
    print(f"  Time elapsed:    {elapsed:.2f}s ({frame_count / elapsed:.1f} fps)")
    print(f"  Output MCAP:     {output_path} ({file_size_mb:.2f} MB)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Convert Waymo Open Dataset TFRecord to MCAP")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/individual_files_training_segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord",
        help="Path to input TFRecord file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output MCAP file (default: input file name with .mcap extension)"
    )
    parser.add_argument(
        "--max-frames", "-n",
        type=int,
        default=None,
        help="Maximum number of frames to convert"
    )
    parser.add_argument(
        "--no-merge-lidar",
        action="store_true",
        help="Do not merge LiDAR point clouds into /lidar/points"
    )
    parser.add_argument(
        "--split-lidar",
        action="store_true",
        help="Publish individual LiDAR topics (/lidar/TOP, etc.)"
    )
    parser.add_argument(
        "--no-second-return",
        action="store_true",
        help="Exclude second return LiDAR points"
    )
    parser.add_argument(
        "--include-raw-frame",
        action="store_true",
        help="Include raw waymo.open_dataset.Frame proto in MCAP"
    )

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        p = Path(input_path)
        output_path = str(p.with_suffix(".mcap"))
    else:
        output_path = os.path.abspath(args.output)

    convert_tfrecord_to_mcap(
        input_path=input_path,
        output_path=output_path,
        max_frames=args.max_frames,
        merge_lidar=not args.no_merge_lidar,
        split_lidar=args.split_lidar,
        include_second_return=not args.no_second_return,
        include_raw_frame=args.include_raw_frame
    )


if __name__ == "__main__":
    main()
