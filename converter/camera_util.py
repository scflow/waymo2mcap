from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.CameraCalibration_pb2 import CameraCalibration
from waymo_open_dataset import dataset_pb2


def create_compressed_image(
    image_bytes: bytes,
    frame_id: str,
    timestamp_micros: int
) -> CompressedImage:
    """Create foxglove.CompressedImage from JPEG image bytes."""
    msg = CompressedImage()
    msg.timestamp.FromMicroseconds(timestamp_micros)
    msg.frame_id = frame_id
    msg.format = "jpeg"
    msg.data = image_bytes
    return msg


def create_camera_calibration(
    calib: dataset_pb2.CameraCalibration,
    frame_id: str,
    timestamp_micros: int
) -> CameraCalibration:
    """Create foxglove.CameraCalibration from Waymo CameraCalibration."""
    msg = CameraCalibration()
    msg.timestamp.FromMicroseconds(timestamp_micros)
    msg.frame_id = frame_id
    msg.width = calib.width
    msg.height = calib.height
    msg.distortion_model = "plumb_bob"

    if len(calib.intrinsic) >= 4:
        fx = calib.intrinsic[0]
        fy = calib.intrinsic[1]
        cx = calib.intrinsic[2]
        cy = calib.intrinsic[3]
        # K: 3x3 row major
        msg.K.extend([
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0
        ])
        # R: 3x3 identity
        msg.R.extend([
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ])
        # P: 3x4 projection
        msg.P.extend([
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ])

    if len(calib.intrinsic) > 4:
        # k1, k2, p1, p2, k3
        msg.D.extend(calib.intrinsic[4:])

    return msg
