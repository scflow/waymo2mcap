import math
import numpy as np
from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
from foxglove_schemas_protobuf.SceneEntity_pb2 import SceneEntity
from foxglove_schemas_protobuf.SceneEntityDeletion_pb2 import SceneEntityDeletion
from foxglove_schemas_protobuf.CubePrimitive_pb2 import CubePrimitive
from foxglove_schemas_protobuf.LinePrimitive_pb2 import LinePrimitive
from foxglove_schemas_protobuf.TextPrimitive_pb2 import TextPrimitive
from foxglove_schemas_protobuf.Point3_pb2 import Point3
from foxglove_schemas_protobuf.ImageAnnotations_pb2 import ImageAnnotations
from foxglove_schemas_protobuf.PointsAnnotation_pb2 import PointsAnnotation
from foxglove_schemas_protobuf.TextAnnotation_pb2 import TextAnnotation
from foxglove_schemas_protobuf.Point2_pb2 import Point2
from waymo_open_dataset import dataset_pb2, label_pb2


# Color theme for 3D boxes: (R, G, B, Fill_Alpha, Wire_Alpha)
BOX_THEME = {
    label_pb2.Label.Type.TYPE_VEHICLE:    {"name": "Car",   "rgb": (0.15, 0.65, 1.00), "fill_a": 0.25, "wire_a": 0.95},  # Blue
    label_pb2.Label.Type.TYPE_PEDESTRIAN: {"name": "Ped",   "rgb": (1.00, 0.35, 0.15), "fill_a": 0.30, "wire_a": 0.95},  # Red/Orange
    label_pb2.Label.Type.TYPE_CYCLIST:    {"name": "Cycl",  "rgb": (1.00, 0.80, 0.05), "fill_a": 0.30, "wire_a": 0.95},  # Yellow
    label_pb2.Label.Type.TYPE_SIGN:       {"name": "Sign",  "rgb": (0.75, 0.25, 0.85), "fill_a": 0.25, "wire_a": 0.90},  # Purple
}


def _get_box_wireframe_corners(length: float, width: float, height: float, heading: float, cx: float, cy: float, cz: float):
    """Compute the 8 corners of an oriented 3D bounding box."""
    # 8 local corners: x is length/2, y is width/2, z is height/2
    dx = length * 0.5
    dy = width * 0.5
    dz = height * 0.5

    local_corners = np.array([
        [-dx, -dy, -dz],  # 0: bottom-rear-left
        [ dx, -dy, -dz],  # 1: bottom-front-left
        [ dx,  dy, -dz],  # 2: bottom-front-right
        [-dx,  dy, -dz],  # 3: bottom-rear-right
        [-dx, -dy,  dz],  # 4: top-rear-left
        [ dx, -dy,  dz],  # 5: top-front-left
        [ dx,  dy,  dz],  # 6: top-front-right
        [-dx,  dy,  dz],  # 7: top-rear-right
    ], dtype=np.float32)

    # Rotate by heading around Z
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    rot_z = np.array([
        [cos_h, -sin_h, 0.0],
        [sin_h,  cos_h, 0.0],
        [  0.0,    0.0, 1.0]
    ], dtype=np.float32)

    world_corners = (rot_z @ local_corners.T).T + np.array([cx, cy, cz], dtype=np.float32)
    return world_corners


def create_3d_boxes_scene_update(
    frame: dataset_pb2.Frame,
    show_wireframe: bool = True,
    show_labels: bool = True
) -> SceneUpdate:
    """Convert frame.laser_labels to foxglove.SceneUpdate in base_link frame.
    
    CRITICAL: Adds SceneEntityDeletion.Type.ALL so previous frames' boxes
    do not accumulate across frames.
    """
    update = SceneUpdate()
    timestamp_micros = frame.timestamp_micros

    # 1. Clear previous frame's entities to prevent accumulation
    del_all = update.deletions.add()
    del_all.type = SceneEntityDeletion.Type.ALL
    del_all.timestamp.FromMicroseconds(timestamp_micros)

    for i, label in enumerate(frame.laser_labels):
        box = label.box
        entity = update.entities.add()
        entity.id = f"box3d_{label.id}_{i}"
        entity.frame_id = "base_link"
        entity.timestamp.FromMicroseconds(timestamp_micros)
        entity.frame_locked = True
        entity.lifetime.seconds = 0
        entity.lifetime.nanos = 0

        theme = BOX_THEME.get(label.type, {"name": "Obj", "rgb": (0.7, 0.7, 0.7), "fill_a": 0.25, "wire_a": 0.9})
        r, g, b = theme["rgb"]

        # 1. Semi-transparent 3D Cube
        cube = entity.cubes.add()
        cube.size.x = float(box.length)
        cube.size.y = float(box.width)
        cube.size.z = float(box.height)
        cube.pose.position.x = float(box.center_x)
        cube.pose.position.y = float(box.center_y)
        cube.pose.position.z = float(box.center_z)

        half_heading = box.heading * 0.5
        cube.pose.orientation.x = 0.0
        cube.pose.orientation.y = 0.0
        cube.pose.orientation.z = math.sin(half_heading)
        cube.pose.orientation.w = math.cos(half_heading)

        cube.color.r = r
        cube.color.g = g
        cube.color.b = b
        cube.color.a = theme["fill_a"]

        # 2. Sharp 12-edge wireframe lines + front heading cross
        if show_wireframe:
            corners = _get_box_wireframe_corners(
                box.length, box.width, box.height, box.heading,
                box.center_x, box.center_y, box.center_z
            )
            wire = entity.lines.add()
            wire.type = LinePrimitive.Type.LINE_LIST
            wire.thickness = 0.04
            wire.scale_invariant = False
            wire.color.r = r
            wire.color.g = g
            wire.color.b = b
            wire.color.a = theme["wire_a"]

            # 12 edges: (0-1, 1-2, 2-3, 3-0), (4-5, 5-6, 6-7, 7-4), (0-4, 1-5, 2-6, 3-7)
            edge_indices = [
                # Bottom face
                0, 1,  1, 2,  2, 3,  3, 0,
                # Top face
                4, 5,  5, 6,  6, 7,  7, 4,
                # Vertical pillars
                0, 4,  1, 5,  2, 6,  3, 7,
                # Front face X indicator (1-2 and 5-6 is front: 1 to 6, 2 to 5)
                1, 6,  2, 5
            ]
            for idx in edge_indices:
                p = wire.points.add()
                p.x = float(corners[idx, 0])
                p.y = float(corners[idx, 1])
                p.z = float(corners[idx, 2])

        # 3. Clean billboarded screen-space text (pixels, not meters!)
        if show_labels:
            text = entity.texts.add()
            text.text = f"{theme['name']}"
            text.billboard = True
            text.scale_invariant = True  # Screen-space pixels, not world meters!
            text.font_size = 11.0        # 11 pixels high
            text.pose.position.x = float(box.center_x)
            text.pose.position.y = float(box.center_y)
            text.pose.position.z = float(box.center_z + box.height * 0.5 + 0.25)
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 0.95

    return update


def create_camera_annotations(
    camera_labels: dataset_pb2.CameraLabels,
    timestamp_micros: int
) -> ImageAnnotations:
    """Convert 2D camera_labels to foxglove.ImageAnnotations."""
    ann = ImageAnnotations()
    for label in camera_labels.labels:
        box = label.box
        x0 = float(box.center_x - box.length * 0.5)
        x1 = float(box.center_x + box.length * 0.5)
        y0 = float(box.center_y - box.width * 0.5)
        y1 = float(box.center_y + box.width * 0.5)

        poly = ann.points.add()
        poly.timestamp.FromMicroseconds(timestamp_micros)
        poly.type = PointsAnnotation.Type.LINE_LOOP
        poly.thickness = 2.0

        theme = BOX_THEME.get(label.type, {"name": "Obj", "rgb": (0.8, 0.8, 0.8)})
        r, g, b = theme["rgb"]
        poly.outline_color.r = r
        poly.outline_color.g = g
        poly.outline_color.b = b
        poly.outline_color.a = 1.0

        for px, py in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
            pt = poly.points.add()
            pt.x = px
            pt.y = py

        # Text label
        txt = ann.texts.add()
        txt.timestamp.FromMicroseconds(timestamp_micros)
        txt.text = f"{theme['name']}"
        txt.position.x = x0
        txt.position.y = max(0.0, y0 - 4.0)
        txt.font_size = 11.0
        txt.text_color.r = r
        txt.text_color.g = g
        txt.text_color.b = b
        txt.text_color.a = 1.0

    return ann
