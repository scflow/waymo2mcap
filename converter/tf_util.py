import numpy as np
from scipy.spatial.transform import Rotation
from foxglove_schemas_protobuf.FrameTransforms_pb2 import FrameTransforms
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from waymo_open_dataset import dataset_pb2


def matrix_to_transform(
    matrix_16: list[float],
    parent_frame_id: str,
    child_frame_id: str,
    timestamp_micros: int
) -> FrameTransform:
    """Convert 16-element row-major 4x4 matrix to Foxglove FrameTransform."""
    mat = np.array(matrix_16, dtype=np.float64).reshape(4, 4)
    translation = mat[:3, 3]
    rot_matrix = mat[:3, :3]

    quat = Rotation.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]

    tf = FrameTransform()
    tf.timestamp.FromMicroseconds(timestamp_micros)
    tf.parent_frame_id = parent_frame_id
    tf.child_frame_id = child_frame_id
    tf.translation.x = float(translation[0])
    tf.translation.y = float(translation[1])
    tf.translation.z = float(translation[2])
    tf.rotation.x = float(quat[0])
    tf.rotation.y = float(quat[1])
    tf.rotation.z = float(quat[2])
    tf.rotation.w = float(quat[3])
    return tf


def create_frame_transforms(
    frame: dataset_pb2.Frame,
    include_static: bool = True
) -> FrameTransforms:
    """Create FrameTransforms message containing ego pose and optional static sensor transforms."""
    msg = FrameTransforms()

    # Dynamic transform: world -> base_link
    if len(frame.pose.transform) == 16:
        tf_ego = matrix_to_transform(
            frame.pose.transform,
            parent_frame_id="world",
            child_frame_id="base_link",
            timestamp_micros=frame.timestamp_micros
        )
        msg.transforms.append(tf_ego)

    if include_static and frame.context:
        # Static transforms: base_link -> laser_*
        for calib in frame.context.laser_calibrations:
            laser_name = dataset_pb2.LaserName.Name.Name(calib.name)
            if len(calib.extrinsic.transform) == 16:
                tf_laser = matrix_to_transform(
                    calib.extrinsic.transform,
                    parent_frame_id="base_link",
                    child_frame_id=f"laser_{laser_name}",
                    timestamp_micros=frame.timestamp_micros
                )
                msg.transforms.append(tf_laser)

        # Static transforms: base_link -> camera_*
        for calib in frame.context.camera_calibrations:
            cam_name = dataset_pb2.CameraName.Name.Name(calib.name)
            if len(calib.extrinsic.transform) == 16:
                tf_cam = matrix_to_transform(
                    calib.extrinsic.transform,
                    parent_frame_id="base_link",
                    child_frame_id=f"camera_{cam_name}",
                    timestamp_micros=frame.timestamp_micros
                )
                msg.transforms.append(tf_cam)

    return msg
