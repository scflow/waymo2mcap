from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
from foxglove_schemas_protobuf.SceneEntity_pb2 import SceneEntity
from foxglove_schemas_protobuf.LinePrimitive_pb2 import LinePrimitive
from foxglove_schemas_protobuf.TextPrimitive_pb2 import TextPrimitive
from foxglove_schemas_protobuf.CylinderPrimitive_pb2 import CylinderPrimitive
from foxglove_schemas_protobuf.Color_pb2 import Color
from foxglove_schemas_protobuf.Point3_pb2 import Point3
from waymo_open_dataset.protos import map_pb2


def create_map_layer_updates(
    map_features: list[map_pb2.MapFeature],
    timestamp_micros: int
) -> dict[str, SceneUpdate]:
    """Convert Waymo MapFeature list into separate SceneUpdates per layer topic.
    
    Returns dict of {topic_name: SceneUpdate}:
      /map/lanes
      /map/road_lines
      /map/road_edges
      /map/crosswalks
      /map/speed_bumps
      /map/driveways
      /map/stop_signs
    """
    layers = {
        "lanes": SceneUpdate(),
        "road_lines": SceneUpdate(),
        "road_edges": SceneUpdate(),
        "crosswalks": SceneUpdate(),
        "speed_bumps": SceneUpdate(),
        "driveways": SceneUpdate(),
        "stop_signs": SceneUpdate(),
    }

    # Initialize each layer with an entity
    entities = {}
    for name, update in layers.items():
        ent = update.entities.add()
        ent.id = f"map_{name}"
        ent.frame_id = "world"
        ent.timestamp.FromMicroseconds(timestamp_micros)
        ent.frame_locked = True
        ent.lifetime.seconds = 0
        ent.lifetime.nanos = 0
        entities[name] = ent

    for feat in map_features:
        feat_type = feat.WhichOneof("feature_data")
        if not feat_type:
            continue

        if feat_type == "lane":
            lane = feat.lane
            if len(lane.polyline) >= 2:
                line = entities["lanes"].lines.add()
                line.type = LinePrimitive.Type.LINE_STRIP
                line.thickness = 0.12
                line.scale_invariant = False
                line.color.r = 0.15
                line.color.g = 0.75
                line.color.b = 0.70
                line.color.a = 0.65
                for pt in lane.polyline:
                    p = line.points.add()
                    p.x = pt.x
                    p.y = pt.y
                    p.z = pt.z

        elif feat_type == "road_line":
            road_line = feat.road_line
            if len(road_line.polyline) >= 2:
                line = entities["road_lines"].lines.add()
                line.type = LinePrimitive.Type.LINE_STRIP
                line.thickness = 0.16
                line.scale_invariant = False
                type_name = map_pb2.RoadLine.RoadLineType.Name(road_line.type)
                if "YELLOW" in type_name:
                    line.color.r = 1.0
                    line.color.g = 0.84
                    line.color.b = 0.0
                    line.color.a = 1.0
                else:
                    line.color.r = 1.0
                    line.color.g = 1.0
                    line.color.b = 1.0
                    line.color.a = 0.95

                for pt in road_line.polyline:
                    p = line.points.add()
                    p.x = pt.x
                    p.y = pt.y
                    p.z = pt.z

        elif feat_type == "road_edge":
            road_edge = feat.road_edge
            if len(road_edge.polyline) >= 2:
                line = entities["road_edges"].lines.add()
                line.type = LinePrimitive.Type.LINE_STRIP
                line.thickness = 0.15
                line.scale_invariant = False
                line.color.r = 0.70
                line.color.g = 0.72
                line.color.b = 0.78
                line.color.a = 0.85
                for pt in road_edge.polyline:
                    p = line.points.add()
                    p.x = pt.x
                    p.y = pt.y
                    p.z = pt.z

        elif feat_type == "crosswalk":
            crosswalk = feat.crosswalk
            if len(crosswalk.polygon) >= 3:
                line = entities["crosswalks"].lines.add()
                line.type = LinePrimitive.Type.LINE_LOOP
                line.thickness = 0.20
                line.scale_invariant = False
                line.color.r = 0.0
                line.color.g = 0.90
                line.color.b = 0.90
                line.color.a = 0.90
                for pt in crosswalk.polygon:
                    p = line.points.add()
                    p.x = pt.x
                    p.y = pt.y
                    p.z = pt.z

        elif feat_type == "speed_bump":
            speed_bump = feat.speed_bump
            if len(speed_bump.polygon) >= 3:
                line = entities["speed_bumps"].lines.add()
                line.type = LinePrimitive.Type.LINE_LOOP
                line.thickness = 0.20
                line.scale_invariant = False
                line.color.r = 1.0
                line.color.g = 0.55
                line.color.b = 0.0
                line.color.a = 0.90
                for pt in speed_bump.polygon:
                    p = line.points.add()
                    p.x = pt.x
                    p.y = pt.y
                    p.z = pt.z

        elif feat_type == "driveway":
            driveway = feat.driveway
            if len(driveway.polygon) >= 3:
                line = entities["driveways"].lines.add()
                line.type = LinePrimitive.Type.LINE_LOOP
                line.thickness = 0.08
                line.scale_invariant = False
                line.color.r = 0.50
                line.color.g = 0.52
                line.color.b = 0.55
                line.color.a = 0.40
                for pt in driveway.polygon:
                    p = line.points.add()
                    p.x = pt.x
                    p.y = pt.y
                    p.z = pt.z

        elif feat_type == "stop_sign":
            stop_sign = feat.stop_sign
            cyl = entities["stop_signs"].cylinders.add()
            cyl.size.x = 0.5
            cyl.size.y = 0.5
            cyl.size.z = 0.3
            cyl.pose.position.x = stop_sign.position.x
            cyl.pose.position.y = stop_sign.position.y
            cyl.pose.position.z = stop_sign.position.z + 0.15
            cyl.color.r = 0.9
            cyl.color.g = 0.1
            cyl.color.b = 0.1
            cyl.color.a = 0.9

            text = entities["stop_signs"].texts.add()
            text.text = "STOP"
            text.billboard = True
            text.scale_invariant = True
            text.font_size = 11.0
            text.pose.position.x = stop_sign.position.x
            text.pose.position.y = stop_sign.position.y
            text.pose.position.z = stop_sign.position.z + 0.6
            text.color.r = 1.0
            text.color.g = 0.2
            text.color.b = 0.2
            text.color.a = 1.0

    # Only return layers that have actual features
    result = {}
    for name, update in layers.items():
        ent = entities[name]
        has_content = (len(ent.lines) > 0 or len(ent.cylinders) > 0 or len(ent.texts) > 0)
        if has_content:
            result[f"/map/{name}"] = update

    return result


def create_map_scene_update(
    map_features: list[map_pb2.MapFeature],
    timestamp_micros: int
) -> SceneUpdate:
    """Backward-compatible full map merged update."""
    layer_updates = create_map_layer_updates(map_features, timestamp_micros)
    merged = SceneUpdate()
    ent = merged.entities.add()
    ent.id = "hd_map_features"
    ent.frame_id = "world"
    ent.timestamp.FromMicroseconds(timestamp_micros)
    ent.frame_locked = True
    ent.lifetime.seconds = 0
    ent.lifetime.nanos = 0

    for upd in layer_updates.values():
        for e in upd.entities:
            ent.lines.extend(e.lines)
            ent.cylinders.extend(e.cylinders)
            ent.texts.extend(e.texts)

    return merged
