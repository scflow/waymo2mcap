import os
import math
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

from foxglove_schemas_protobuf.FrameTransforms_pb2 import FrameTransforms
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.PoseInFrame_pb2 import PoseInFrame
from foxglove_schemas_protobuf.PosesInFrame_pb2 import PosesInFrame
from foxglove_schemas_protobuf.Odometry_pb2 import Odometry
from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
from foxglove_schemas_protobuf.SceneEntity_pb2 import SceneEntity
from foxglove_schemas_protobuf.SceneEntityDeletion_pb2 import SceneEntityDeletion
from foxglove_schemas_protobuf.CubePrimitive_pb2 import CubePrimitive
from foxglove_schemas_protobuf.LinePrimitive_pb2 import LinePrimitive
from foxglove_schemas_protobuf.CylinderPrimitive_pb2 import CylinderPrimitive
from foxglove_schemas_protobuf.TextPrimitive_pb2 import TextPrimitive
from foxglove_schemas_protobuf.Color_pb2 import Color

from mcap_protobuf.writer import Writer
from waymo_open_dataset.protos import scenario_pb2, map_pb2
from converter.map_util import create_map_layer_updates
from converter.annotation_util import _get_box_wireframe_corners


TRACK_THEME = {
    scenario_pb2.Track.ObjectType.TYPE_VEHICLE:    {"name": "Car",  "rgb": (0.15, 0.65, 1.00), "fill_a": 0.25, "wire_a": 0.90},
    scenario_pb2.Track.ObjectType.TYPE_PEDESTRIAN: {"name": "Ped",  "rgb": (1.00, 0.35, 0.15), "fill_a": 0.30, "wire_a": 0.95},
    scenario_pb2.Track.ObjectType.TYPE_CYCLIST:    {"name": "Cycl", "rgb": (1.00, 0.80, 0.05), "fill_a": 0.30, "wire_a": 0.95},
    scenario_pb2.Track.ObjectType.TYPE_OTHER:      {"name": "Obj",  "rgb": (0.70, 0.70, 0.70), "fill_a": 0.25, "wire_a": 0.85},
}


def convert_scenario_to_mcap(scenario: scenario_pb2.Scenario, output_path: str):
    """Convert a single waymo.open_dataset.Scenario proto message into MCAP."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    predict_track_indices = {p.track_index for p in scenario.tracks_to_predict}
    sdc_index = scenario.sdc_track_index
    num_steps = len(scenario.timestamps_seconds)
    t0_micros = int(scenario.timestamps_seconds[0] * 1e6)
    t0_ns = int(scenario.timestamps_seconds[0] * 1e9)

    print(f"  Scenario ID:        {scenario.scenario_id}")
    print(f"  Timesteps:          {num_steps} ({scenario.timestamps_seconds[-1] - scenario.timestamps_seconds[0]:.2f}s)")
    print(f"  SDC Track Index:    {sdc_index}")
    print(f"  Total Tracks:       {len(scenario.tracks)}")
    print(f"  Tracks to Predict:  {len(predict_track_indices)}")
    print(f"  Map Features:       {len(scenario.map_features)}")

    # Pre-extract SDC all poses for trajectory ribbons
    sdc_track = scenario.tracks[sdc_index] if 0 <= sdc_index < len(scenario.tracks) else None
    sdc_points = []
    if sdc_track:
        for s in sdc_track.states:
            if s.valid:
                sdc_points.append([s.center_x, s.center_y, s.center_z])

    with open(output_path, "wb") as f_out:
        writer = Writer(f_out)

        # 1. Native Scenario Protocol Buffer Message
        writer.write_message("/waymo/scenario", scenario, log_time=t0_ns, publish_time=t0_ns)

        # 2. HD Map Features (split into sub-topics)
        if scenario.map_features:
            map_layers = create_map_layer_updates(list(scenario.map_features), t0_micros)
            for topic, layer_msg in map_layers.items():
                writer.write_message(topic, layer_msg, log_time=t0_ns, publish_time=t0_ns)

        # 3. Step through all timesteps
        for step_idx, ts_sec in enumerate(scenario.timestamps_seconds):
            ts_micros = int(ts_sec * 1e6)
            ts_ns = int(ts_sec * 1e9)

            # --- A. SDC Pose and TF ---
            if sdc_track and step_idx < len(sdc_track.states):
                sdc_state = sdc_track.states[step_idx]
                if sdc_state.valid:
                    cx, cy, cz = sdc_state.center_x, sdc_state.center_y, sdc_state.center_z
                    heading = sdc_state.heading
                    half_h = heading * 0.5
                    qx, qy, qz, qw = 0.0, 0.0, math.sin(half_h), math.cos(half_h)

                    # /tf (world -> base_link)
                    tf_msg = FrameTransforms()
                    tf = tf_msg.transforms.add()
                    tf.timestamp.FromMicroseconds(ts_micros)
                    tf.parent_frame_id = "world"
                    tf.child_frame_id = "base_link"
                    tf.translation.x = cx
                    tf.translation.y = cy
                    tf.translation.z = cz
                    tf.rotation.x = qx
                    tf.rotation.y = qy
                    tf.rotation.z = qz
                    tf.rotation.w = qw
                    writer.write_message("/tf", tf_msg, log_time=ts_ns, publish_time=ts_ns)

                    # /ego_pose
                    ego_pose = PoseInFrame()
                    ego_pose.timestamp.FromMicroseconds(ts_micros)
                    ego_pose.frame_id = "world"
                    ego_pose.pose.position.x = cx
                    ego_pose.pose.position.y = cy
                    ego_pose.pose.position.z = cz
                    ego_pose.pose.orientation.x = qx
                    ego_pose.pose.orientation.y = qy
                    ego_pose.pose.orientation.z = qz
                    ego_pose.pose.orientation.w = qw
                    writer.write_message("/ego_pose", ego_pose, log_time=ts_ns, publish_time=ts_ns)

                    # /odometry
                    odom = Odometry()
                    odom.timestamp.FromMicroseconds(ts_micros)
                    odom.frame_id = "world"
                    odom.body_frame_id = "base_link"
                    odom.pose.position.x = cx
                    odom.pose.position.y = cy
                    odom.pose.position.z = cz
                    odom.pose.orientation.x = qx
                    odom.pose.orientation.y = qy
                    odom.pose.orientation.z = qz
                    odom.pose.orientation.w = qw
                    odom.linear_velocity.x = sdc_state.velocity_x
                    odom.linear_velocity.y = sdc_state.velocity_y
                    writer.write_message("/odometry", odom, log_time=ts_ns, publish_time=ts_ns)

            # --- B. 3D Bounding Boxes for All Active Agents ---
            boxes_msg = SceneUpdate()
            del_all = boxes_msg.deletions.add()
            del_all.type = SceneEntityDeletion.Type.ALL
            del_all.timestamp.FromMicroseconds(ts_micros)

            for track_idx, track in enumerate(scenario.tracks):
                if step_idx >= len(track.states):
                    continue
                st = track.states[step_idx]
                if not st.valid:
                    continue

                is_sdc = (track_idx == sdc_index)
                is_target = (track_idx in predict_track_indices)

                theme = TRACK_THEME.get(track.object_type, TRACK_THEME[scenario_pb2.Track.ObjectType.TYPE_OTHER])
                if is_sdc:
                    r, g, b = 0.0, 1.0, 0.45  # Bright green for SDC
                    fill_a, wire_a = 0.35, 1.0
                    label_text = "SDC (Ego)"
                elif is_target:
                    r, g, b = 1.0, 0.15, 0.70  # Vibrant magenta for prediction targets
                    fill_a, wire_a = 0.35, 1.0
                    label_text = f"TARGET #{track.id}"
                else:
                    r, g, b = theme["rgb"]
                    fill_a, wire_a = theme["fill_a"], theme["wire_a"]
                    label_text = f"{theme['name']} #{track.id}"

                ent = boxes_msg.entities.add()
                ent.id = f"agent_{track.id}"
                ent.frame_id = "world"
                ent.timestamp.FromMicroseconds(ts_micros)
                ent.frame_locked = True

                # 1. Cube
                cube = ent.cubes.add()
                cube.size.x = float(st.length)
                cube.size.y = float(st.width)
                cube.size.z = float(st.height)
                cube.pose.position.x = float(st.center_x)
                cube.pose.position.y = float(st.center_y)
                cube.pose.position.z = float(st.center_z)
                hh = st.heading * 0.5
                cube.pose.orientation.z = math.sin(hh)
                cube.pose.orientation.w = math.cos(hh)
                cube.color.r = r
                cube.color.g = g
                cube.color.b = b
                cube.color.a = fill_a

                # 2. Wireframe lines + front cross
                corners = _get_box_wireframe_corners(
                    st.length, st.width, st.height, st.heading,
                    st.center_x, st.center_y, st.center_z
                )
                wire = ent.lines.add()
                wire.type = LinePrimitive.Type.LINE_LIST
                wire.thickness = 0.04
                wire.color.r = r
                wire.color.g = g
                wire.color.b = b
                wire.color.a = wire_a
                edge_indices = [
                    0, 1,  1, 2,  2, 3,  3, 0,
                    4, 5,  5, 6,  6, 7,  7, 4,
                    0, 4,  1, 5,  2, 6,  3, 7,
                    1, 6,  2, 5
                ]
                for idx in edge_indices:
                    p = wire.points.add()
                    p.x = float(corners[idx, 0])
                    p.y = float(corners[idx, 1])
                    p.z = float(corners[idx, 2])

                # 3. Label text
                txt = ent.texts.add()
                txt.text = label_text
                txt.billboard = True
                txt.scale_invariant = True
                txt.font_size = 11.0
                txt.pose.position.x = float(st.center_x)
                txt.pose.position.y = float(st.center_y)
                txt.pose.position.z = float(st.center_z + st.height * 0.5 + 0.25)
                txt.color.r = 1.0
                txt.color.g = 1.0
                txt.color.b = 1.0
                txt.color.a = 0.95

            writer.write_message("/perception/boxes_3d", boxes_msg, log_time=ts_ns, publish_time=ts_ns)

            # --- C. Agent Ground Truth Trajectories (History + Future) ---
            traj_msg = SceneUpdate()
            del_traj = traj_msg.deletions.add()
            del_traj.type = SceneEntityDeletion.Type.ALL
            del_traj.timestamp.FromMicroseconds(ts_micros)

            for track_idx, track in enumerate(scenario.tracks):
                is_sdc = (track_idx == sdc_index)
                is_target = (track_idx in predict_track_indices)
                # Only show trajectories for SDC, prediction targets, or moving tracks
                if not (is_sdc or is_target):
                    continue

                valid_pts = [(s.center_x, s.center_y, s.center_z) for s in track.states if s.valid]
                if len(valid_pts) < 2:
                    continue

                hist_pts = [(track.states[k].center_x, track.states[k].center_y, track.states[k].center_z)
                            for k in range(step_idx + 1) if k < len(track.states) and track.states[k].valid]
                fut_pts = [(track.states[k].center_x, track.states[k].center_y, track.states[k].center_z)
                           for k in range(step_idx, len(track.states)) if track.states[k].valid]

                ent = traj_msg.entities.add()
                ent.id = f"traj_{track.id}"
                ent.frame_id = "world"
                ent.timestamp.FromMicroseconds(ts_micros)
                ent.frame_locked = True

                # History ribbon
                if len(hist_pts) >= 2:
                    line_h = ent.lines.add()
                    line_h.type = LinePrimitive.Type.LINE_STRIP
                    line_h.thickness = 0.25 if is_sdc else 0.18
                    if is_sdc:
                        line_h.color.r, line_h.color.g, line_h.color.b, line_h.color.a = 0.0, 1.0, 0.4, 0.95
                    else:
                        line_h.color.r, line_h.color.g, line_h.color.b, line_h.color.a = 1.0, 0.2, 0.7, 0.90
                    for pt in hist_pts:
                        p = line_h.points.add()
                        p.x, p.y, p.z = float(pt[0]), float(pt[1]), float(pt[2])

                # Future ribbon (ground truth forecast)
                if len(fut_pts) >= 2:
                    line_f = ent.lines.add()
                    line_f.type = LinePrimitive.Type.LINE_STRIP
                    line_f.thickness = 0.15
                    if is_sdc:
                        line_f.color.r, line_f.color.g, line_f.color.b, line_f.color.a = 0.1, 0.8, 1.0, 0.60
                    else:
                        line_f.color.r, line_f.color.g, line_f.color.b, line_f.color.a = 1.0, 0.7, 0.1, 0.60
                    for pt in fut_pts:
                        p = line_f.points.add()
                        p.x, p.y, p.z = float(pt[0]), float(pt[1]), float(pt[2])

            writer.write_message("/tracks/trajectories", traj_msg, log_time=ts_ns, publish_time=ts_ns)

            # --- D. Dynamic Traffic Signals ---
            if step_idx < len(scenario.dynamic_map_states):
                dyn = scenario.dynamic_map_states[step_idx]
                if len(dyn.lane_states) > 0:
                    sig_msg = SceneUpdate()
                    del_sig = sig_msg.deletions.add()
                    del_sig.type = SceneEntityDeletion.Type.ALL
                    del_sig.timestamp.FromMicroseconds(ts_micros)

                    sig_ent = sig_msg.entities.add()
                    sig_ent.id = "traffic_signals"
                    sig_ent.frame_id = "world"
                    sig_ent.timestamp.FromMicroseconds(ts_micros)
                    sig_ent.frame_locked = True

                    for ls in dyn.lane_states:
                        if ls.stop_point and (ls.stop_point.x != 0.0 or ls.stop_point.y != 0.0):
                            cyl = sig_ent.cylinders.add()
                            cyl.size.x = 0.8
                            cyl.size.y = 0.8
                            cyl.size.z = 0.4
                            cyl.pose.position.x = ls.stop_point.x
                            cyl.pose.position.y = ls.stop_point.y
                            cyl.pose.position.z = ls.stop_point.z + 0.2

                            state_str = map_pb2.TrafficSignalLaneState.State.Name(ls.state)
                            if "STOP" in state_str:
                                cyl.color.r, cyl.color.g, cyl.color.b, cyl.color.a = 1.0, 0.1, 0.1, 0.95
                            elif "CAUTION" in state_str:
                                cyl.color.r, cyl.color.g, cyl.color.b, cyl.color.a = 1.0, 0.7, 0.0, 0.95
                            elif "GO" in state_str:
                                cyl.color.r, cyl.color.g, cyl.color.b, cyl.color.a = 0.0, 1.0, 0.2, 0.95
                            else:
                                cyl.color.r, cyl.color.g, cyl.color.b, cyl.color.a = 0.5, 0.5, 0.5, 0.6

                    writer.write_message("/map/traffic_signals", sig_msg, log_time=ts_ns, publish_time=ts_ns)

        writer.finish()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  --> Converted scenario to MCAP: {output_path} ({size_mb:.2f} MB)")
