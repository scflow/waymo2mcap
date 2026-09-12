import zlib
import numpy as np
from google.protobuf.timestamp_pb2 import Timestamp
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud
from foxglove_schemas_protobuf.PackedElementField_pb2 import PackedElementField
from waymo_open_dataset import dataset_pb2


def decode_range_image(compressed_bytes: bytes) -> np.ndarray:
    """Decompress ZLIB range image to numpy array [H, W, C]."""
    decompressed = zlib.decompress(compressed_bytes)
    matrix = dataset_pb2.MatrixFloat()
    matrix.ParseFromString(decompressed)
    shape = list(matrix.shape.dims)
    return np.array(matrix.data, dtype=np.float32).reshape(shape)


def extract_point_cloud_from_range_image(
    data: np.ndarray,
    calib: dataset_pb2.LaserCalibration,
    transform_to_vehicle: bool = True
) -> np.ndarray:
    """Extract Cartesian point cloud [N, 5] (x, y, z, intensity, elongation) from decoded range image."""
    H, W, C = data.shape
    extrinsic = np.array(calib.extrinsic.transform, dtype=np.float64).reshape(4, 4)

    # 1. Beam inclination
    if len(calib.beam_inclinations) > 0:
        inclination = np.array(calib.beam_inclinations, dtype=np.float32)
    else:
        diff = calib.beam_inclination_max - calib.beam_inclination_min
        inclination = (0.5 + np.arange(0, H, dtype=np.float32)) / H * diff + calib.beam_inclination_min

    # 2. Azimuth angles
    az_correction = np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
    ratios = (np.arange(W, 0, -1, dtype=np.float32) - 0.5) / W
    azimuth = (ratios * 2.0 - 1.0) * np.pi - az_correction

    # 3. Polar to Cartesian in sensor frame
    r = data[:, :, 0]
    valid = r > 0
    if not np.any(valid):
        return np.zeros((0, 5), dtype=np.float32)

    cos_incl = np.cos(inclination)[:, None]
    sin_incl = np.sin(inclination)[:, None]
    cos_az = np.cos(azimuth)[None, :]
    sin_az = np.sin(azimuth)[None, :]

    x = cos_az * cos_incl * r
    y = sin_az * cos_incl * r
    z = sin_incl * r

    x_valid = x[valid]
    y_valid = y[valid]
    z_valid = z[valid]
    intensity_valid = data[:, :, 1][valid]
    elongation_valid = data[:, :, 2][valid] if C > 2 else np.zeros_like(intensity_valid)

    # 4. Transform to vehicle frame if requested
    if transform_to_vehicle:
        pts_sensor = np.stack([x_valid, y_valid, z_valid, np.ones(len(x_valid), dtype=np.float32)], axis=-1)
        pts_vehicle = (extrinsic @ pts_sensor.T).T[:, :3].astype(np.float32)
        pts_out = np.column_stack([pts_vehicle, intensity_valid, elongation_valid])
    else:
        pts_out = np.column_stack([x_valid, y_valid, z_valid, intensity_valid, elongation_valid])

    return pts_out.astype(np.float32)


def create_foxglove_point_cloud(
    points: np.ndarray,
    frame_id: str,
    timestamp_micros: int
) -> PointCloud:
    """Create a foxglove.PointCloud protobuf message from [N, 5] points."""
    pc = PointCloud()
    pc.frame_id = frame_id
    pc.timestamp.FromMicroseconds(timestamp_micros)
    pc.point_stride = 20  # 5 * 4 bytes

    fields = [
        ("x", 0, PackedElementField.NumericType.FLOAT32),
        ("y", 4, PackedElementField.NumericType.FLOAT32),
        ("z", 8, PackedElementField.NumericType.FLOAT32),
        ("intensity", 12, PackedElementField.NumericType.FLOAT32),
        ("elongation", 16, PackedElementField.NumericType.FLOAT32),
    ]
    for name, offset, type_ in fields:
        f = pc.fields.add()
        f.name = name
        f.offset = offset
        f.type = type_

    if len(points) > 0:
        pc.data = np.ascontiguousarray(points, dtype=np.float32).tobytes()
    else:
        pc.data = b""

    return pc
