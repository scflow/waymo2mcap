import numpy as np
from scipy.spatial.transform import Rotation
from foxglove_schemas_protobuf.PoseInFrame_pb2 import PoseInFrame
from foxglove_schemas_protobuf.PosesInFrame_pb2 import PosesInFrame
from foxglove_schemas_protobuf.Odometry_pb2 import Odometry
from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
from foxglove_schemas_protobuf.CubePrimitive_pb2 import CubePrimitive
from foxglove_schemas_protobuf.LinePrimitive_pb2 import LinePrimitive
from foxglove_schemas_protobuf.Color_pb2 import Color
from foxglove_schemas_protobuf.Point3_pb2 import Point3
from waymo_open_dataset import dataset_pb2


def create_ego_pose(frame: dataset_pb2.Frame) -> PoseInFrame:
    """Create foxglove.PoseInFrame message on /ego_pose."""
    msg = PoseInFrame()
    msg.timestamp.FromMicroseconds(frame.timestamp_micros)
    msg.frame_id = "world"

    if len(frame.pose.transform) == 16:
        mat = np.array(frame.pose.transform, dtype=np.float64).reshape(4, 4)
        t = mat[:3, 3]
        q = Rotation.from_matrix(mat[:3, :3]).as_quat()

        msg.pose.position.x = float(t[0])
        msg.pose.position.y = float(t[1])
        msg.pose.position.z = float(t[2])

        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])

    return msg


def create_ego_odometry(frame: dataset_pb2.Frame) -> Odometry:
    """Create foxglove.Odometry message on /odometry containing pose and velocities."""
    msg = Odometry()
    msg.timestamp.FromMicroseconds(frame.timestamp_micros)
    msg.frame_id = "world"
    msg.body_frame_id = "base_link"

    # 1. Pose
    if len(frame.pose.transform) == 16:
        mat = np.array(frame.pose.transform, dtype=np.float64).reshape(4, 4)
        t = mat[:3, 3]
        q = Rotation.from_matrix(mat[:3, :3]).as_quat()

        msg.pose.position.x = float(t[0])
        msg.pose.position.y = float(t[1])
        msg.pose.position.z = float(t[2])

        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])

    # 2. Velocity from FRONT camera (or first available image)
    for img in frame.images:
        if img.name == dataset_pb2.CameraName.FRONT or len(frame.images) == 1:
            vel = img.velocity
            msg.linear_velocity.x = float(vel.v_x)
            msg.linear_velocity.y = float(vel.v_y)
            msg.linear_velocity.z = float(vel.v_z)
            msg.angular_velocity.x = float(vel.w_x)
            msg.angular_velocity.y = float(vel.w_y)
            msg.angular_velocity.z = float(vel.w_z)
            break

    return msg


def create_ego_vehicle_marker(frame: dataset_pb2.Frame) -> SceneUpdate:
    """Create a 3D bounding box / model representation of the ego vehicle in base_link frame."""
    update = SceneUpdate()
    entity = update.entities.add()
    entity.id = "ego_vehicle_body"
    entity.frame_id = "base_link"
    entity.timestamp.FromMicroseconds(frame.timestamp_micros)
    entity.frame_locked = True
    entity.lifetime.seconds = 0
    entity.lifetime.nanos = 0

    # Waymo Chrysler Pacifica minivan approximate size: length 5.17m, width 2.02m, height 1.77m
    # Vehicle coordinate system center: base_link is typically centered on rear axle at ground level
    cube = entity.cubes.add()
    cube.size.x = 5.0
    cube.size.y = 2.0
    cube.size.z = 1.8
    # Center is approximately 1.4m forward of rear axle, 0.9m above ground
    cube.pose.position.x = 1.4
    cube.pose.position.y = 0.0
    cube.pose.position.z = 0.9
    cube.pose.orientation.w = 1.0

    # Translucent green/cyan car body
    cube.color.r = 0.1
    cube.color.g = 0.8
    cube.color.b = 0.6
    cube.color.a = 0.35

    return update


def create_trajectory_scene_update(
    all_poses: list[np.ndarray],
    current_index: int,
    timestamp_micros: int
) -> SceneUpdate:
    """Create 3D trajectory lines on /ego_trajectory in world frame.
    
    Renders:
    1. Historical driven path: bright, thick vibrant green ribbon (0.3m width) on the pavement.
    2. Future planned path: semi-transparent cyan line showing upcoming path.
    """
    update = SceneUpdate()

    # 1. Driven trajectory up to current frame
    if current_index >= 1:
        hist_entity = update.entities.add()
        hist_entity.id = "ego_trajectory_driven"
        hist_entity.frame_id = "world"
        hist_entity.timestamp.FromMicroseconds(timestamp_micros)
        hist_entity.frame_locked = True

        line = hist_entity.lines.add()
        line.type = LinePrimitive.Type.LINE_STRIP
        line.thickness = 0.28
        line.scale_invariant = False
        # Bright vibrant lime-green for driven trail
        line.color.r = 0.0
        line.color.g = 1.0
        line.color.b = 0.4
        line.color.a = 0.95
        for k in range(current_index + 1):
            p = line.points.add()
            pt = all_poses[k][:3, 3]
            p.x = float(pt[0])
            p.y = float(pt[1])
            p.z = float(pt[2])

    # 2. Future path ahead of current frame
    if current_index < len(all_poses) - 1:
        fut_entity = update.entities.add()
        fut_entity.id = "ego_trajectory_future"
        fut_entity.frame_id = "world"
        fut_entity.timestamp.FromMicroseconds(timestamp_micros)
        fut_entity.frame_locked = True

        line = fut_entity.lines.add()
        line.type = LinePrimitive.Type.LINE_STRIP
        line.thickness = 0.16
        line.scale_invariant = False
        # Clean semi-transparent cyan for upcoming path
        line.color.r = 0.1
        line.color.g = 0.75
        line.color.b = 1.0
        line.color.a = 0.55
        for k in range(current_index, len(all_poses)):
            p = line.points.add()
            pt = all_poses[k][:3, 3]
            p.x = float(pt[0])
            p.y = float(pt[1])
            p.z = float(pt[2])

    return update


def create_ego_poses_path(
    all_poses: list[np.ndarray],
    timestamp_micros: int
) -> PosesInFrame:
    """Create foxglove.PosesInFrame message on /ego_path in world frame."""
    msg = PosesInFrame()
    msg.timestamp.FromMicroseconds(timestamp_micros)
    msg.frame_id = "world"
    for mat in all_poses:
        p = msg.poses.add()
        t = mat[:3, 3]
        q = Rotation.from_matrix(mat[:3, :3]).as_quat()
        p.position.x = float(t[0])
        p.position.y = float(t[1])
        p.position.z = float(t[2])
        p.orientation.x = float(q[0])
        p.orientation.y = float(q[1])
        p.orientation.z = float(q[2])
        p.orientation.w = float(q[3])
    return msg
