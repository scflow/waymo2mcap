use std::collections::HashMap;
use std::io::Cursor;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{
        ws::{Message, WebSocket},
        Path, Query, State, WebSocketUpgrade,
    },
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::prelude::*;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use futures::{SinkExt, StreamExt};
use arrow_array::cast::AsArray;
use arrow_array::{Array, BinaryArray, ListArray, StructArray, StringArray};
use arrow_array::types::{Float64Type, Float32Type, Int64Type};
use lancedb::connect;
use lancedb::query::{ExecutableQuery, QueryBase};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, RwLock};
use tower_http::cors::{Any, CorsLayer};

const DB_PATH: &str = "/mnt/d/src/waymo2mcap/data/lancedb";
const SERVER_PORT: u16 = 8765;

const POINT_CLOUD_SCHEMA: &str = r#"{"type":"object","properties":{"timestamp":{"type":"object","properties":{"sec":{"type":"integer"},"nsec":{"type":"integer"}}},"frame_id":{"type":"string"},"pose":{"type":"object","properties":{"position":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"}}},"orientation":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"},"w":{"type":"number"}}}}},"point_stride":{"type":"integer"},"fields":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"offset":{"type":"integer"},"type":{"type":"integer"}}}},"data":{"type":"string","contentEncoding":"base64"}}}"#;

const COMPRESSED_IMAGE_SCHEMA: &str = r#"{"type":"object","properties":{"timestamp":{"type":"object","properties":{"sec":{"type":"integer"},"nsec":{"type":"integer"}}},"frame_id":{"type":"string"},"data":{"type":"string","contentEncoding":"base64"},"format":{"type":"string"}}}"#;

const TF_SCHEMA: &str = r#"{"type":"object","properties":{"transforms":{"type":"array","items":{"type":"object","properties":{"timestamp":{"type":"object","properties":{"sec":{"type":"integer"},"nsec":{"type":"integer"}}},"parent_frame_id":{"type":"string"},"child_frame_id":{"type":"string"},"translation":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"}}},"rotation":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"},"w":{"type":"number"}}}}}}}}"#;

const SCENE_UPDATE_SCHEMA: &str = r#"{"type":"object","properties":{"deletions":{"type":"array","items":{"type":"object","properties":{"type":{"type":"integer"},"id":{"type":"string"}}}},"entities":{"type":"array","items":{"type":"object","properties":{"id":{"type":"string"},"timestamp":{"type":"object","properties":{"sec":{"type":"integer"},"nsec":{"type":"integer"}}},"frame_id":{"type":"string"},"lifetime":{"type":"object","properties":{"sec":{"type":"integer"},"nsec":{"type":"integer"}}},"frame_locked":{"type":"boolean"},"cubes":{"type":"array","items":{"type":"object","properties":{"pose":{"type":"object","properties":{"position":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"}}},"orientation":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"},"w":{"type":"number"}}}}},"size":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"}}},"color":{"type":"object","properties":{"r":{"type":"number"},"g":{"type":"number"},"b":{"type":"number"},"a":{"type":"number"}}}}}},"texts":{"type":"array","items":{"type":"object","properties":{"pose":{"type":"object","properties":{"position":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"}}},"orientation":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"},"w":{"type":"number"}}}}},"billboard":{"type":"boolean"},"font_size":{"type":"number"},"scale_invariant":{"type":"boolean"},"color":{"type":"object","properties":{"r":{"type":"number"},"g":{"type":"number"},"b":{"type":"number"},"a":{"type":"number"}}},"text":{"type":"string"}}}},"lines":{"type":"array","items":{"type":"object","properties":{"type":{"type":"integer"},"pose":{"type":"object","properties":{"position":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"}}},"orientation":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"},"w":{"type":"number"}}}}},"thickness":{"type":"number"},"scale_invariant":{"type":"boolean"},"points":{"type":"array","items":{"type":"object","properties":{"x":{"type":"number"},"y":{"type":"number"},"z":{"type":"number"}}}},"color":{"type":"object","properties":{"r":{"type":"number"},"g":{"type":"number"},"b":{"type":"number"},"a":{"type":"number"}}}}}}}}}}}"#;

#[derive(Clone, Serialize, Deserialize)]
pub struct EpisodeSummary {
    pub episode_id: String,
    pub frame_count: i32,
    pub duration_s: f64,
    pub num_objects_3d: i64,
    pub num_images: i64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ChannelDef {
    pub id: u32,
    pub topic: String,
    pub encoding: String,
    #[serde(rename = "schemaName")]
    pub schema_name: String,
    pub schema: String,
    #[serde(rename = "schemaEncoding")]
    pub schema_encoding: String,
}

#[derive(Serialize, Deserialize)]
pub struct ServerInfoMsg {
    pub op: String,
    pub name: String,
    pub capabilities: Vec<String>,
    #[serde(rename = "supportedEncodings")]
    pub supported_encodings: Vec<String>,
    pub metadata: HashMap<String, String>,
    #[serde(rename = "sessionId")]
    pub session_id: String,
    #[serde(rename = "dataStartTime")]
    pub data_start_time: TimeDef,
    #[serde(rename = "dataEndTime")]
    pub data_end_time: TimeDef,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct TimeDef {
    pub sec: u32,
    pub nsec: u32,
}

impl TimeDef {
    pub fn from_ns(ns: i64) -> Self {
        let sec = (ns / 1_000_000_000).max(0) as u32;
        let nsec = (ns % 1_000_000_000).max(0) as u32;
        Self { sec, nsec }
    }
}

#[derive(Serialize)]
pub struct AdvertiseMsg {
    pub op: String,
    pub channels: Vec<ChannelDef>,
}

#[derive(Deserialize)]
pub struct ClientSubscribeMsg {
    pub op: String,
    pub subscriptions: Option<Vec<SubscriptionItem>>,
    #[serde(rename = "subscriptionIds")]
    pub subscription_ids: Option<Vec<u32>>,
}

#[derive(Deserialize)]
pub struct SubscriptionItem {
    pub id: u32,
    #[serde(rename = "channelId")]
    pub channel_id: u32,
}

#[derive(Deserialize)]
pub struct SelectEpisodeReq {
    pub episode_id: String,
}

#[derive(Deserialize)]
pub struct UpdateObjectReq {
    pub object_id: Option<String>,
    pub id: Option<String>,
    pub label: Option<String>,
    pub pos: Option<Vec<f64>>,
    pub center: Option<Vec<f64>>,
    pub size: Option<Vec<f64>>,
    pub yaw: Option<f64>,
    pub quat: Option<Vec<f64>>,
    pub quaternion: Option<Vec<f64>>,
}

#[derive(Deserialize)]
pub struct PlaybackSeekReq {
    pub frame_index: Option<usize>,
    pub timestamp_ns: Option<i64>,
}

#[derive(Deserialize)]
pub struct PlaybackSpeedReq {
    pub speed: f32,
}

pub struct PreloadedEpisode {
    pub episode_id: String,
    pub timestamps: Vec<i64>,
    pub tf_msgs: HashMap<i64, Arc<Vec<u8>>>,
    pub box_msgs: RwLock<HashMap<i64, Arc<Vec<u8>>>>,
    pub lidar_msgs: RwLock<HashMap<i64, Arc<Vec<u8>>>>,
    pub cam_msgs: RwLock<HashMap<(String, i64), Arc<Vec<u8>>>>,
    pub map_msg: Option<Arc<Vec<u8>>>,
}

pub struct AppState {
    pub db: Arc<lancedb::Connection>,
    pub current_episode: RwLock<String>,
    pub start_time_ns: AtomicI64,
    pub end_time_ns: AtomicI64,
    pub current_time_ns: AtomicI64,
    pub current_frame_idx: AtomicUsize,
    pub is_playing: AtomicBool,
    pub playback_speed: RwLock<f32>,
    pub channels: RwLock<Vec<ChannelDef>>,
    pub camera_names: RwLock<Vec<String>>,
    pub episode_cache: RwLock<Option<Arc<PreloadedEpisode>>>,
    pub notify_tx: broadcast::Sender<i64>,
    pub client_reset_tx: broadcast::Sender<()>,
}

impl AppState {
    pub async fn reload_episode(&self, ep_id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        *self.current_episode.write().await = ep_id.to_string();

        // 1. Load ego poses & extract sorted timestamps
        let ego_table = self.db.open_table("ego_poses").execute().await?;
        let filter = format!("episode_id = '{}'", ep_id);
        let mut stream = ego_table.query().only_if(&filter).execute().await?;
        let mut poses: HashMap<i64, (f64, f64, f64, f64, f64, f64, f64)> = HashMap::new();
        let mut timestamps: Vec<i64> = Vec::new();

        while let Some(batch_res) = stream.next().await {
            let batch = batch_res?;
            let ts_col = batch.column_by_name("timestamp_ns").map(|c| c.as_primitive::<Int64Type>());
            let px_col = batch.column_by_name("px").map(|c| c.as_primitive::<Float64Type>());
            let py_col = batch.column_by_name("py").map(|c| c.as_primitive::<Float64Type>());
            let pz_col = batch.column_by_name("pz").map(|c| c.as_primitive::<Float64Type>());
            let qx_col = batch.column_by_name("qx").map(|c| c.as_primitive::<Float64Type>());
            let qy_col = batch.column_by_name("qy").map(|c| c.as_primitive::<Float64Type>());
            let qz_col = batch.column_by_name("qz").map(|c| c.as_primitive::<Float64Type>());
            let qw_col = batch.column_by_name("qw").map(|c| c.as_primitive::<Float64Type>());

            for i in 0..batch.num_rows() {
                if let Some(ts_arr) = ts_col {
                    let ts = ts_arr.value(i);
                    timestamps.push(ts);
                    let px = px_col.map(|a| a.value(i)).unwrap_or(0.0);
                    let py = py_col.map(|a| a.value(i)).unwrap_or(0.0);
                    let pz = pz_col.map(|a| a.value(i)).unwrap_or(0.0);
                    let qx = qx_col.map(|a| a.value(i)).unwrap_or(0.0);
                    let qy = qy_col.map(|a| a.value(i)).unwrap_or(0.0);
                    let qz = qz_col.map(|a| a.value(i)).unwrap_or(0.0);
                    let qw = qw_col.map(|a| a.value(i)).unwrap_or(1.0);
                    poses.insert(ts, (px, py, pz, qx, qy, qz, qw));
                }
            }
        }
        timestamps.sort_unstable();
        timestamps.dedup();

        let min_ts = timestamps.first().copied().unwrap_or(1599876543000000000);
        let max_ts = timestamps.last().copied().unwrap_or(min_ts + 20_000_000_000);

        self.start_time_ns.store(min_ts, Ordering::SeqCst);
        self.end_time_ns.store(max_ts, Ordering::SeqCst);
        self.current_time_ns.store(min_ts, Ordering::SeqCst);
        self.current_frame_idx.store(0, Ordering::SeqCst);

        // Pre-serialize TF transforms
        let mut tf_msgs: HashMap<i64, Arc<Vec<u8>>> = HashMap::with_capacity(timestamps.len());
        for &ts in &timestamps {
            let time_def = TimeDef::from_ns(ts);
            let (px, py, pz, qx, qy, qz, qw) = poses.get(&ts).copied().unwrap_or((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0));
            let tf_json = serde_json::json!({
                "transforms": [
                    {
                        "timestamp": { "sec": time_def.sec, "nsec": time_def.nsec },
                        "parent_frame_id": "map",
                        "child_frame_id": "vehicle",
                        "translation": { "x": px, "y": py, "z": pz },
                        "rotation": { "x": qx, "y": qy, "z": qz, "w": qw }
                    },
                    {
                        "timestamp": { "sec": time_def.sec, "nsec": time_def.nsec },
                        "parent_frame_id": "vehicle",
                        "child_frame_id": "FRONT",
                        "translation": { "x": 1.54, "y": 0.0, "z": 1.5 },
                        "rotation": { "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0 }
                    },
                    {
                        "timestamp": { "sec": time_def.sec, "nsec": time_def.nsec },
                        "parent_frame_id": "vehicle",
                        "child_frame_id": "FRONT_LEFT",
                        "translation": { "x": 1.45, "y": 0.45, "z": 1.5 },
                        "rotation": { "x": 0.0, "y": 0.0, "z": 0.38268, "w": 0.92388 }
                    },
                    {
                        "timestamp": { "sec": time_def.sec, "nsec": time_def.nsec },
                        "parent_frame_id": "vehicle",
                        "child_frame_id": "FRONT_RIGHT",
                        "translation": { "x": 1.45, "y": -0.45, "z": 1.5 },
                        "rotation": { "x": 0.0, "y": 0.0, "z": -0.38268, "w": 0.92388 }
                    },
                    {
                        "timestamp": { "sec": time_def.sec, "nsec": time_def.nsec },
                        "parent_frame_id": "vehicle",
                        "child_frame_id": "SIDE_LEFT",
                        "translation": { "x": 0.25, "y": 0.85, "z": 1.5 },
                        "rotation": { "x": 0.0, "y": 0.0, "z": 0.70711, "w": 0.70711 }
                    },
                    {
                        "timestamp": { "sec": time_def.sec, "nsec": time_def.nsec },
                        "parent_frame_id": "vehicle",
                        "child_frame_id": "SIDE_RIGHT",
                        "translation": { "x": 0.25, "y": -0.85, "z": 1.5 },
                        "rotation": { "x": 0.0, "y": 0.0, "z": -0.70711, "w": 0.70711 }
                    }
                ]
            });
            tf_msgs.insert(ts, Arc::new(tf_json.to_string().into_bytes()));
        }

        // 2. Pre-serialize 3D objects
        let box_msgs = load_and_build_boxes(&self.db, ep_id, &timestamps).await;

        // 3. Pre-build HD Map
        let map_msg = build_map_scene_update(&self.db, ep_id).await.map(|s| Arc::new(s.into_bytes()));

        // 4. Detect cameras
        let cam_table = self.db.open_table("camera_frames").execute().await?;
        let mut cam_stream = cam_table.query().only_if(&filter).limit(20).execute().await?;
        let mut detected_cams = Vec::new();
        while let Some(batch_res) = cam_stream.next().await {
            let batch = batch_res?;
            if let Some(col) = batch.column_by_name("camera") {
                let s_arr = col.as_string::<i32>();
                for i in 0..s_arr.len() {
                    let c = s_arr.value(i).to_string();
                    if !detected_cams.contains(&c) {
                        detected_cams.push(c);
                    }
                }
            }
        }
        if detected_cams.is_empty() {
            detected_cams.push("FRONT".to_string());
        }

        // 5. Build channel definitions
        let mut chs = Vec::new();
        let mut cid = 1;
        for c in &detected_cams {
            chs.push(ChannelDef {
                id: cid,
                topic: format!("/camera/{}", c),
                encoding: "json".to_string(),
                schema_name: "foxglove.CompressedImage".to_string(),
                schema: COMPRESSED_IMAGE_SCHEMA.to_string(),
                schema_encoding: "jsonschema".to_string(),
            });
            cid += 1;
        }

        chs.push(ChannelDef {
            id: cid,
            topic: "/lidar/top".to_string(),
            encoding: "json".to_string(),
            schema_name: "foxglove.PointCloud".to_string(),
            schema: POINT_CLOUD_SCHEMA.to_string(),
            schema_encoding: "jsonschema".to_string(),
        });
        cid += 1;

        chs.push(ChannelDef {
            id: cid,
            topic: "/perception/boxes_3d".to_string(),
            encoding: "json".to_string(),
            schema_name: "foxglove.SceneUpdate".to_string(),
            schema: SCENE_UPDATE_SCHEMA.to_string(),
            schema_encoding: "jsonschema".to_string(),
        });
        cid += 1;

        chs.push(ChannelDef {
            id: cid,
            topic: "/map".to_string(),
            encoding: "json".to_string(),
            schema_name: "foxglove.SceneUpdate".to_string(),
            schema: SCENE_UPDATE_SCHEMA.to_string(),
            schema_encoding: "jsonschema".to_string(),
        });
        cid += 1;

        chs.push(ChannelDef {
            id: cid,
            topic: "/tf".to_string(),
            encoding: "json".to_string(),
            schema_name: "foxglove.FrameTransforms".to_string(),
            schema: TF_SCHEMA.to_string(),
            schema_encoding: "jsonschema".to_string(),
        });

        *self.camera_names.write().await = detected_cams;
        *self.channels.write().await = chs;

        let preloaded = Arc::new(PreloadedEpisode {
            episode_id: ep_id.to_string(),
            timestamps,
            tf_msgs,
            box_msgs: RwLock::new(box_msgs),
            lidar_msgs: RwLock::new(HashMap::new()),
            cam_msgs: RwLock::new(HashMap::new()),
            map_msg,
        });

        *self.episode_cache.write().await = Some(preloaded.clone());

        println!("[Rust Cache] Episode '{}' base indexed ({} frames). Warming LiDAR & Cameras in background...", ep_id, preloaded.timestamps.len());

        // Spawn background worker to asynchronously warm LiDAR and Camera frame caches
        let db_clone = self.db.clone();
        let preloaded_clone = preloaded.clone();
        let ep_str = ep_id.to_string();
        tokio::spawn(async move {
            prefetch_lidar_and_cameras(&db_clone, &ep_str, &preloaded_clone).await;
        });

        // Notify all active WebSocket connections to reset cleanly for the new episode timeline bounds
        let _ = self.client_reset_tx.send(());

        Ok(())
    }
}

async fn load_and_build_boxes(db: &lancedb::Connection, ep_id: &str, timestamps: &[i64]) -> HashMap<i64, Arc<Vec<u8>>> {
    let mut frame_boxes: HashMap<i64, Vec<serde_json::Value>> = HashMap::new();
    if let Ok(obj_tbl) = db.open_table("objects_3d").execute().await {
        let filter = format!("episode_id = '{}'", ep_id);
        if let Ok(mut stream) = obj_tbl.query().only_if(&filter).execute().await {
            while let Some(Ok(batch)) = stream.next().await {
                let ts_col = batch.column_by_name("timestamp_ns").map(|c| c.as_primitive::<Int64Type>());
                let id_col = batch.column_by_name("object_id").map(|c| c.as_string::<i32>());
                let lbl_col = batch.column_by_name("label").map(|c| c.as_string::<i32>());
                let px = batch.column_by_name("pos_x").map(|c| c.as_primitive::<Float64Type>());
                let py = batch.column_by_name("pos_y").map(|c| c.as_primitive::<Float64Type>());
                let pz = batch.column_by_name("pos_z").map(|c| c.as_primitive::<Float64Type>());
                let sx = batch.column_by_name("size_x").map(|c| c.as_primitive::<Float64Type>());
                let sy = batch.column_by_name("size_y").map(|c| c.as_primitive::<Float64Type>());
                let sz = batch.column_by_name("size_z").map(|c| c.as_primitive::<Float64Type>());
                let qx = batch.column_by_name("quat_x").map(|c| c.as_primitive::<Float64Type>());
                let qy = batch.column_by_name("quat_y").map(|c| c.as_primitive::<Float64Type>());
                let qz = batch.column_by_name("quat_z").map(|c| c.as_primitive::<Float64Type>());
                let qw = batch.column_by_name("quat_w").map(|c| c.as_primitive::<Float64Type>());
                let cr = batch.column_by_name("color_r").map(|c| c.as_primitive::<Float32Type>());
                let cg = batch.column_by_name("color_g").map(|c| c.as_primitive::<Float32Type>());
                let cb = batch.column_by_name("color_b").map(|c| c.as_primitive::<Float32Type>());
                let ca = batch.column_by_name("color_a").map(|c| c.as_primitive::<Float32Type>());

                for i in 0..batch.num_rows() {
                    if let Some(ts_arr) = ts_col {
                        let ts = ts_arr.value(i);
                        let oid = id_col.map(|a| a.value(i)).unwrap_or("obj");
                        let lbl = lbl_col.map(|a| a.value(i)).unwrap_or("OBJECT");
                        let pos_x = px.map(|a| a.value(i)).unwrap_or(0.0);
                        let pos_y = py.map(|a| a.value(i)).unwrap_or(0.0);
                        let pos_z = pz.map(|a| a.value(i)).unwrap_or(0.0);
                        let size_x = sx.map(|a| a.value(i)).unwrap_or(4.0);
                        let size_y = sy.map(|a| a.value(i)).unwrap_or(2.0);
                        let size_z = sz.map(|a| a.value(i)).unwrap_or(1.6);
                        let rot_x = qx.map(|a| a.value(i)).unwrap_or(0.0);
                        let rot_y = qy.map(|a| a.value(i)).unwrap_or(0.0);
                        let rot_z = qz.map(|a| a.value(i)).unwrap_or(0.0);
                        let rot_w = qw.map(|a| a.value(i)).unwrap_or(1.0);

                        let time_def = TimeDef::from_ns(ts);

                        frame_boxes.entry(ts).or_default().push(serde_json::json!({
                            "id": oid,
                            "timestamp": { "sec": time_def.sec, "nsec": time_def.nsec },
                            "frame_id": "vehicle",
                            "lifetime": { "sec": 0, "nsec": 150_000_000 },
                            "frame_locked": true,
                            "cubes": [{
                                "pose": {
                                    "position": { "x": pos_x, "y": pos_y, "z": pos_z },
                                    "orientation": { "x": rot_x, "y": rot_y, "z": rot_z, "w": rot_w }
                                },
                                "size": {
                                    "x": size_x,
                                    "y": size_y,
                                    "z": size_z,
                                },
                                "color": {
                                    "r": cr.map(|a| a.value(i)).unwrap_or(0.15),
                                    "g": cg.map(|a| a.value(i)).unwrap_or(0.65),
                                    "b": cb.map(|a| a.value(i)).unwrap_or(1.0),
                                    "a": ca.map(|a| a.value(i)).unwrap_or(0.35),
                                }
                            }],
                            "texts": [{
                                "pose": {
                                    "position": { "x": pos_x, "y": pos_y, "z": pos_z + size_z * 0.5 + 0.3 },
                                    "orientation": { "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0 }
                                },
                                "billboard": true,
                                "font_size": 0.6,
                                "scale_invariant": false,
                                "color": { "r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0 },
                                "text": lbl
                            }]
                        }));
                    }
                }
            }
        }
    }

    let mut box_msgs = HashMap::with_capacity(timestamps.len());
    for &ts in timestamps {
        let mut entities = frame_boxes.remove(&ts).unwrap_or_default();
        let time_def = TimeDef::from_ns(ts);

        // Inject Ego Vehicle bounding box, heading arrow and billboard label in "vehicle" frame
        let ego_entity = serde_json::json!({
            "id": "ego_vehicle",
            "timestamp": { "sec": time_def.sec, "nsec": time_def.nsec },
            "frame_id": "vehicle",
            "lifetime": { "sec": 0, "nsec": 250_000_000 },
            "frame_locked": true,
            "cubes": [{
                "pose": {
                    "position": { "x": 1.45, "y": 0.0, "z": 0.75 },
                    "orientation": { "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0 }
                },
                "size": {
                    "x": 4.95,
                    "y": 2.05,
                    "z": 1.75
                },
                "color": {
                    "r": 0.05,
                    "g": 0.82,
                    "b": 1.0,
                    "a": 0.40
                }
            }],
            "texts": [{
                "pose": {
                    "position": { "x": 1.45, "y": 0.0, "z": 1.85 },
                    "orientation": { "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0 }
                },
                "billboard": true,
                "font_size": 0.55,
                "scale_invariant": false,
                "color": { "r": 0.1, "g": 0.9, "b": 1.0, "a": 1.0 },
                "text": "🚗 自车 (EGO VEHICLE)"
            }],
            "lines": [{
                "type": 1,
                "thickness": 0.08,
                "scale_invariant": false,
                "color": { "r": 0.0, "g": 1.0, "b": 0.6, "a": 0.95 },
                "points": [
                    { "x": 1.45, "y": 0.0, "z": 0.75 },
                    { "x": 4.10, "y": 0.0, "z": 0.75 }
                ]
            }]
        });
        entities.insert(0, ego_entity);

        let box_json = serde_json::json!({
            "deletions": [{ "type": 1, "id": "" }],
            "entities": entities
        });
        box_msgs.insert(ts, Arc::new(box_json.to_string().into_bytes()));
    }
    box_msgs
}

async fn prefetch_lidar_and_cameras(db: &lancedb::Connection, ep_id: &str, cache: &PreloadedEpisode) {
    let t0 = std::time::Instant::now();

    // 1. Prefetch LiDAR
    if let Ok(lidar_tbl) = db.open_table("lidar_frames").execute().await {
        let filter = format!("episode_id = '{}'", ep_id);
        if let Ok(mut stream) = lidar_tbl.query().only_if(&filter).execute().await {
            while let Some(Ok(batch)) = stream.next().await {
                let ts_col = batch.column_by_name("timestamp_ns").map(|c| c.as_primitive::<Int64Type>());
                let pts_col = batch.column_by_name("points").and_then(|c| c.as_any().downcast_ref::<BinaryArray>());

                if let (Some(ts_arr), Some(bin_arr)) = (ts_col, pts_col) {
                    for i in 0..batch.num_rows() {
                        let ts = ts_arr.value(i);
                        let pts_bytes = bin_arr.value(i);
                        let total_pts = pts_bytes.len() / 20;
                        let step = (total_pts / 8000).max(1);
                        let mut downsampled = Vec::with_capacity((total_pts / step + 1) * 16);
                        for k in (0..total_pts).step_by(step) {
                            let offset = k * 20;
                            if offset + 16 <= pts_bytes.len() {
                                downsampled.extend_from_slice(&pts_bytes[offset..offset + 16]);
                            }
                        }
                        let b64 = BASE64_STANDARD.encode(&downsampled);
                        let time_def = TimeDef::from_ns(ts);
                        let pc_msg = serde_json::json!({
                            "timestamp": { "sec": time_def.sec, "nsec": time_def.nsec },
                            "frame_id": "vehicle",
                            "pose": {
                                "position": { "x": 0.0, "y": 0.0, "z": 0.0 },
                                "orientation": { "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0 }
                            },
                            "point_stride": 16,
                            "fields": [
                                { "name": "x", "offset": 0, "type": 7 },
                                { "name": "y", "offset": 4, "type": 7 },
                                { "name": "z", "offset": 8, "type": 7 },
                                { "name": "intensity", "offset": 12, "type": 7 }
                            ],
                            "data": b64
                        });
                        cache.lidar_msgs.write().await.insert(ts, Arc::new(pc_msg.to_string().into_bytes()));
                    }
                }
            }
        }
    }

    // 2. Prefetch Cameras
    if let Ok(cam_tbl) = db.open_table("camera_frames").execute().await {
        let filter = format!("episode_id = '{}'", ep_id);
        if let Ok(mut stream) = cam_tbl.query().only_if(&filter).execute().await {
            while let Some(Ok(batch)) = stream.next().await {
                let cam_col = batch.column_by_name("camera").map(|c| c.as_string::<i32>());
                let ts_col = batch.column_by_name("timestamp_ns").map(|c| c.as_primitive::<Int64Type>());
                let img_col = batch.column_by_name("image").and_then(|c| c.as_any().downcast_ref::<BinaryArray>());

                if let (Some(cams), Some(ts_arr), Some(imgs)) = (cam_col, ts_col, img_col) {
                    for i in 0..batch.num_rows() {
                        let cam_name = cams.value(i);
                        let ts = ts_arr.value(i);
                        let img_bytes = imgs.value(i);
                        let b64 = BASE64_STANDARD.encode(img_bytes);
                        let time_def = TimeDef::from_ns(ts);
                        let img_msg = serde_json::json!({
                            "timestamp": { "sec": time_def.sec, "nsec": time_def.nsec },
                            "frame_id": cam_name,
                            "data": b64,
                            "format": "jpeg"
                        });
                        cache.cam_msgs.write().await.insert((cam_name.to_string(), ts), Arc::new(img_msg.to_string().into_bytes()));
                    }
                }
            }
        }
    }

    println!("[Rust Cache] 🎉 Episode '{}' fully pre-cached in RAM ({:?})! Playback is now 100% in-memory.", ep_id, t0.elapsed());
}

async fn build_map_scene_update(db: &lancedb::Connection, ep_id: &str) -> Option<String> {
    let map_table = match db.open_table("map_features").execute().await {
        Ok(t) => t,
        Err(_) => return None,
    };
    let filter = format!("episode_id = '{}'", ep_id);
    let mut stream = match map_table.query().only_if(&filter).execute().await {
        Ok(s) => s,
        Err(_) => return None,
    };

    let mut layer_lines: HashMap<String, Vec<serde_json::Value>> = HashMap::new();

    while let Some(Ok(batch)) = stream.next().await {
        let layer_col = batch.column_by_name("layer").and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let geom_col = batch.column_by_name("geometry_type").and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let points_col = batch.column_by_name("points").and_then(|c| c.as_any().downcast_ref::<ListArray>());

        if let (Some(layers), Some(geoms), Some(pts_list)) = (layer_col, geom_col, points_col) {
            for i in 0..batch.num_rows() {
                let layer = layers.value(i);
                let geom = geoms.value(i);
                let struct_val = pts_list.value(i);
                let struct_arr = match struct_val.as_any().downcast_ref::<StructArray>() {
                    Some(s) => s,
                    None => continue,
                };
                let x_col = match struct_arr.column_by_name("x").map(|c| c.as_primitive::<Float64Type>()) {
                    Some(c) => c,
                    None => continue,
                };
                let y_col = match struct_arr.column_by_name("y").map(|c| c.as_primitive::<Float64Type>()) {
                    Some(c) => c,
                    None => continue,
                };
                let z_col = match struct_arr.column_by_name("z").map(|c| c.as_primitive::<Float64Type>()) {
                    Some(c) => c,
                    None => continue,
                };

                let num_pts = struct_arr.len();
                if num_pts < 2 {
                    continue;
                }

                let mut pts_json = Vec::with_capacity(num_pts);
                for k in 0..num_pts {
                    let px = x_col.value(k);
                    let py = y_col.value(k);
                    let pz = z_col.value(k);
                    if !px.is_finite() || !py.is_finite() || !pz.is_finite() {
                        continue;
                    }
                    pts_json.push(serde_json::json!({
                        "x": px,
                        "y": py,
                        "z": pz,
                    }));
                }
                if pts_json.len() < 2 {
                    continue;
                }

                let line_type = if geom == "line_loop" { 2 } else { 1 };
                let (thickness, color) = match layer {
                    "lanes" => (0.12, serde_json::json!({"r": 0.15, "g": 0.75, "b": 0.70, "a": 0.65})),
                    "road_lines" => (0.16, serde_json::json!({"r": 1.0, "g": 0.84, "b": 0.0, "a": 0.95})),
                    "road_edges" => (0.15, serde_json::json!({"r": 0.75, "g": 0.75, "b": 0.80, "a": 0.85})),
                    "crosswalks" => (0.20, serde_json::json!({"r": 0.0, "g": 0.90, "b": 0.90, "a": 0.90})),
                    "speed_bumps" => (0.20, serde_json::json!({"r": 1.0, "g": 0.55, "b": 0.0, "a": 0.90})),
                    "driveways" => (0.15, serde_json::json!({"r": 0.45, "g": 0.75, "b": 0.45, "a": 0.70})),
                    "stop_signs" => (0.25, serde_json::json!({"r": 0.9, "g": 0.1, "b": 0.1, "a": 1.0})),
                    _ => (0.15, serde_json::json!({"r": 0.6, "g": 0.6, "b": 0.6, "a": 0.7})),
                };

                layer_lines.entry(layer.to_string()).or_default().push(serde_json::json!({
                    "type": line_type,
                    "pose": {
                        "position": { "x": 0.0, "y": 0.0, "z": 0.0 },
                        "orientation": { "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0 }
                    },
                    "thickness": thickness,
                    "scale_invariant": false,
                    "points": pts_json,
                    "color": color
                }));
            }
        }
    }

    let mut entities = Vec::new();
    for (layer_name, lines) in layer_lines {
        entities.push(serde_json::json!({
            "id": format!("map_{}", layer_name),
            "timestamp": { "sec": 0, "nsec": 0 },
            "frame_id": "map",
            "lifetime": { "sec": 0, "nsec": 0 },
            "frame_locked": true,
            "lines": lines
        }));
    }

    let scene_msg = serde_json::json!({
        "deletions": [],
        "entities": entities
    });
    Some(scene_msg.to_string())
}

// Helpers for Foxglove binary protocol
fn encode_message_data(sub_id: u32, timestamp_ns: u64, payload: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(13 + payload.len());
    buf.push(1); // Opcode 1: MESSAGE_DATA
    buf.write_u32::<LittleEndian>(sub_id).unwrap();
    buf.write_u64::<LittleEndian>(timestamp_ns).unwrap();
    buf.extend_from_slice(payload);
    buf
}

fn encode_time(timestamp_ns: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(9);
    buf.push(2); // Opcode 2: TIME
    buf.write_u64::<LittleEndian>(timestamp_ns).unwrap();
    buf
}

fn encode_playback_state(status: u8, current_time: u64, speed: f32, did_seek: u8, req_id: &str) -> Vec<u8> {
    let mut buf = Vec::with_capacity(19 + req_id.len());
    buf.push(5); // Opcode 5: PLAYBACK_STATE
    buf.push(status);
    buf.write_u64::<LittleEndian>(current_time).unwrap();
    buf.write_f32::<LittleEndian>(speed).unwrap();
    buf.push(did_seek);
    buf.write_u32::<LittleEndian>(req_id.len() as u32).unwrap();
    buf.extend_from_slice(req_id.as_bytes());
    buf
}

fn decode_playback_control_request(data: &[u8]) -> Option<(u8, f32, Option<u64>, String)> {
    if data.len() < 19 || data[0] != 3 {
        return None;
    }
    let mut rdr = Cursor::new(&data[1..]);
    let command = rdr.read_u8().ok()?;
    let speed = rdr.read_f32::<LittleEndian>().ok()?;
    let has_seek = rdr.read_u8().ok()?;
    let seek_time = rdr.read_u64::<LittleEndian>().ok()?;
    let req_id_len = rdr.read_u32::<LittleEndian>().ok()? as usize;
    let pos = 1 + (rdr.position() as usize);
    if data.len() < pos + req_id_len {
        return None;
    }
    let req_id = String::from_utf8_lossy(&data[pos..pos + req_id_len]).to_string();
    Some((command, speed, if has_seek != 0 { Some(seek_time) } else { None }, req_id))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==================================================");
    println!("   LanceDB Native Rust Streamer (Foxglove v1)     ");
    println!("==================================================");

    let db = Arc::new(connect(DB_PATH).execute().await?);
    let (notify_tx, _) = broadcast::channel(128);
    let (client_reset_tx, _) = broadcast::channel(16);

    let state = Arc::new(AppState {
        db: db.clone(),
        current_episode: RwLock::new(String::new()),
        start_time_ns: AtomicI64::new(0),
        end_time_ns: AtomicI64::new(0),
        current_time_ns: AtomicI64::new(0),
        current_frame_idx: AtomicUsize::new(0),
        is_playing: AtomicBool::new(true),
        playback_speed: RwLock::new(1.0),
        channels: RwLock::new(Vec::new()),
        camera_names: RwLock::new(Vec::new()),
        episode_cache: RwLock::new(None),
        notify_tx,
        client_reset_tx,
    });

    // Load initial episode from episodes table
    let ep_table = db.open_table("episodes").execute().await?;
    let mut ep_stream = ep_table.query().limit(1).execute().await?;
    let mut first_ep = String::new();
    if let Some(batch_res) = ep_stream.next().await {
        let batch = batch_res?;
        if let Some(col) = batch.column_by_name("episode_id") {
            let s_arr = col.as_string::<i32>();
            if s_arr.len() > 0 {
                first_ep = s_arr.value(0).to_string();
            }
        }
    }

    if !first_ep.is_empty() {
        state.reload_episode(&first_ep).await.ok();
    }

    // High-precision frame-by-frame playback ticker
    let ticker_state = state.clone();
    tokio::spawn(async move {
        loop {
            let speed = *ticker_state.playback_speed.read().await;
            let ms = ((100.0 / speed.max(0.05)).clamp(10.0, 2000.0)) as u64;
            tokio::time::sleep(Duration::from_millis(ms)).await;

            if ticker_state.is_playing.load(Ordering::Relaxed) {
                if let Some(cache) = ticker_state.episode_cache.read().await.as_ref() {
                    if !cache.timestamps.is_empty() {
                        let cur_idx = ticker_state.current_frame_idx.load(Ordering::Relaxed);
                        let next_idx = (cur_idx + 1) % cache.timestamps.len();
                        ticker_state.current_frame_idx.store(next_idx, Ordering::Relaxed);
                        let next_ts = cache.timestamps[next_idx];
                        ticker_state.current_time_ns.store(next_ts, Ordering::Relaxed);
                        let _ = ticker_state.notify_tx.send(next_ts);
                    }
                }
            }
        }
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/", get(ws_handler))
        .route("/ws", get(ws_handler))
        .route("/api/health", get(health_handler))
        .route("/api/episodes", get(episodes_handler))
        .route("/api/select_episode", post(select_episode_handler))
        .route("/api/episodes/select", post(select_episode_handler))
        .route("/api/update_object", post(update_object_handler))
        .route("/api/objects/update", post(update_object_handler))
        .route("/api/frame_image/{episode_id}/{frame_idx}", get(frame_image_handler))
        .route("/api/playback/play", post(playback_play_handler))
        .route("/api/playback/pause", post(playback_pause_handler))
        .route("/api/playback/seek", post(playback_seek_handler))
        .route("/api/playback/speed", post(playback_speed_handler))
        .route("/api/playback/status", get(playback_status_handler))
        .route("/api/objects/current", get(current_objects_handler))
        .layer(cors)
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], SERVER_PORT));
    println!("[Rust Engine] HTTP & WebSocket Server running at http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health_handler() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok", "engine": "rust", "version": "0.2.0-cached" }))
}

async fn episodes_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let ep_table = match state.db.open_table("episodes").execute().await {
        Ok(t) => t,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))),
    };

    let mut stream = match ep_table.query().execute().await {
        Ok(s) => s,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))),
    };

    let mut episodes = Vec::new();
    while let Some(Ok(batch)) = stream.next().await {
        let ep_ids = batch.column_by_name("episode_id").and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
        let frame_counts = batch.column_by_name("frame_count").and_then(|c| c.as_any().downcast_ref::<arrow_array::Int32Array>());
        let durations = batch.column_by_name("duration_s").and_then(|c| c.as_any().downcast_ref::<arrow_array::Float64Array>());
        let num_objs = batch.column_by_name("num_objects_3d").and_then(|c| c.as_any().downcast_ref::<arrow_array::Int64Array>());
        let num_imgs = batch.column_by_name("num_images").and_then(|c| c.as_any().downcast_ref::<arrow_array::Int64Array>());

        for i in 0..batch.num_rows() {
            let ep_id = ep_ids.map(|a| a.value(i).to_string()).unwrap_or_default();
            let fc = frame_counts.map(|a| a.value(i)).unwrap_or(0);
            let dur = durations.map(|a| a.value(i)).unwrap_or(0.0);
            let no = num_objs.map(|a| a.value(i)).unwrap_or(0);
            let ni = num_imgs.map(|a| a.value(i)).unwrap_or(0);

            episodes.push(EpisodeSummary {
                episode_id: ep_id,
                frame_count: fc,
                duration_s: dur,
                num_objects_3d: no,
                num_images: ni,
            });
        }
    }
    episodes.sort_by(|a, b| a.episode_id.cmp(&b.episode_id));
    (StatusCode::OK, Json(serde_json::json!({ "status": "success", "episodes": episodes })))
}

async fn select_episode_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SelectEpisodeReq>,
) -> impl IntoResponse {
    let cur = state.current_episode.read().await.clone();
    if cur == payload.episode_id {
        println!("[Rust Engine] Episode '{}' is already active, skipping reload", cur);
        return Json(serde_json::json!({ "status": "success", "episode_id": payload.episode_id, "already_loaded": true }));
    }
    match state.reload_episode(&payload.episode_id).await {
        Ok(_) => Json(serde_json::json!({ "status": "success", "episode_id": payload.episode_id })),
        Err(e) => Json(serde_json::json!({ "status": "error", "message": e.to_string() })),
    }
}

async fn update_object_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<UpdateObjectReq>,
) -> impl IntoResponse {
    let obj_id = payload.object_id.or(payload.id).unwrap_or_default();
    if obj_id.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": "missing object_id" })));
    }

    let obj_table = match state.db.open_table("objects_3d").execute().await {
        Ok(t) => t,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))),
    };

    let mut update_builder = obj_table.update().only_if(format!("object_id = '{}'", obj_id));

    if let Some(lbl) = &payload.label {
        update_builder = update_builder.column("label", format!("'{}'", lbl));
    }
    let pos = payload.pos.or(payload.center);
    if let Some(p) = pos {
        if p.len() >= 3 {
            update_builder = update_builder
                .column("pos_x", p[0].to_string())
                .column("pos_y", p[1].to_string())
                .column("pos_z", p[2].to_string());
        }
    }
    if let Some(s) = payload.size {
        if s.len() >= 3 {
            update_builder = update_builder
                .column("size_x", s[0].to_string())
                .column("size_y", s[1].to_string())
                .column("size_z", s[2].to_string());
        }
    }
    let quat = payload.quat.or(payload.quaternion);
    if let Some(q) = quat {
        if q.len() >= 4 {
            update_builder = update_builder
                .column("quat_x", q[0].to_string())
                .column("quat_y", q[1].to_string())
                .column("quat_z", q[2].to_string())
                .column("quat_w", q[3].to_string());
        }
    } else if let Some(yaw) = payload.yaw {
        let half = yaw * 0.5;
        let qz = half.sin();
        let qw = half.cos();
        update_builder = update_builder
            .column("quat_x", "0.0")
            .column("quat_y", "0.0")
            .column("quat_z", qz.to_string())
            .column("quat_w", qw.to_string());
    }

    println!("[Rust Engine] Updating object '{}' in LanceDB objects_3d", obj_id);
    match update_builder.execute().await {
        Ok(_) => {
            // Also refresh in-memory 3D boxes cache
            if let Some(cache) = state.episode_cache.read().await.as_ref() {
                let db_clone = state.db.clone();
                let ep_clone = cache.episode_id.clone();
                let timestamps_clone = cache.timestamps.clone();
                let cache_clone = cache.clone();
                tokio::spawn(async move {
                    let new_box_msgs = load_and_build_boxes(&db_clone, &ep_clone, &timestamps_clone).await;
                    *cache_clone.box_msgs.write().await = new_box_msgs;
                });
            }
            (StatusCode::OK, Json(serde_json::json!({ "status": "success", "object_id": obj_id })))
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))),
    }
}

async fn frame_image_handler(
    State(state): State<Arc<AppState>>,
    Path((ep_id, frame_idx)): Path<(String, i32)>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let cam = params.get("camera").cloned().unwrap_or_else(|| "FRONT".to_string());
    let cam_table = match state.db.open_table("camera_frames").execute().await {
        Ok(t) => t,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    let filter = format!("episode_id = '{}' AND frame_index = {} AND camera = '{}'", ep_id, frame_idx, cam);
    let mut stream = match cam_table.query().only_if(&filter).limit(1).execute().await {
        Ok(s) => s,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };

    if let Some(Ok(batch)) = stream.next().await {
        if let Some(col) = batch.column_by_name("image") {
            let bin_arr = col.as_binary::<i32>();
            if bin_arr.len() > 0 {
                let bytes = bin_arr.value(0).to_vec();
                let mut headers = HeaderMap::new();
                headers.insert(header::CONTENT_TYPE, "image/jpeg".parse().unwrap());
                headers.insert(header::CACHE_CONTROL, "public, max-age=3600".parse().unwrap());
                return (StatusCode::OK, headers, bytes).into_response();
            }
        }
    }
    StatusCode::NOT_FOUND.into_response()
}

async fn playback_play_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    state.is_playing.store(true, Ordering::SeqCst);
    Json(serde_json::json!({ "status": "success", "is_playing": true }))
}

async fn playback_pause_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    state.is_playing.store(false, Ordering::SeqCst);
    Json(serde_json::json!({ "status": "success", "is_playing": false }))
}

async fn playback_seek_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<PlaybackSeekReq>,
) -> impl IntoResponse {
    if let Some(cache) = state.episode_cache.read().await.as_ref() {
        if let Some(idx) = payload.frame_index {
            let clamped = idx.min(cache.timestamps.len().saturating_sub(1));
            state.current_frame_idx.store(clamped, Ordering::SeqCst);
            if clamped < cache.timestamps.len() {
                let ts = cache.timestamps[clamped];
                state.current_time_ns.store(ts, Ordering::SeqCst);
                let _ = state.notify_tx.send(ts);
            }
            return Json(serde_json::json!({ "status": "success", "frame_index": clamped }));
        } else if let Some(ts) = payload.timestamp_ns {
            let idx = match cache.timestamps.binary_search(&ts) {
                Ok(i) => i,
                Err(i) => i.min(cache.timestamps.len().saturating_sub(1)),
            };
            state.current_frame_idx.store(idx, Ordering::SeqCst);
            state.current_time_ns.store(ts, Ordering::SeqCst);
            let _ = state.notify_tx.send(ts);
            return Json(serde_json::json!({ "status": "success", "frame_index": idx, "timestamp_ns": ts }));
        }
    }
    Json(serde_json::json!({ "status": "error", "message": "No episode loaded or missing seek target" }))
}

async fn playback_speed_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<PlaybackSpeedReq>,
) -> impl IntoResponse {
    *state.playback_speed.write().await = payload.speed;
    Json(serde_json::json!({ "status": "success", "speed": payload.speed }))
}

async fn playback_status_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let is_playing = state.is_playing.load(Ordering::Relaxed);
    let cur_idx = state.current_frame_idx.load(Ordering::Relaxed);
    let cur_ts = state.current_time_ns.load(Ordering::Relaxed);
    let speed = *state.playback_speed.read().await;
    let total_frames = state.episode_cache.read().await.as_ref().map(|c| c.timestamps.len()).unwrap_or(0);
    let ep_id = state.current_episode.read().await.clone();

    Json(serde_json::json!({
        "status": "success",
        "episode_id": ep_id,
        "is_playing": is_playing,
        "frame_index": cur_idx,
        "total_frames": total_frames,
        "current_time_ns": cur_ts,
        "speed": speed,
    }))
}

async fn current_objects_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let cur_ts = state.current_time_ns.load(Ordering::Relaxed);
    if let Some(cache) = state.episode_cache.read().await.as_ref() {
        if let Some(bytes) = cache.box_msgs.read().await.get(&cur_ts) {
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
            return (StatusCode::OK, headers, bytes.as_ref().clone()).into_response();
        }
    }
    (StatusCode::OK, Json(serde_json::json!({ "deletions": [], "entities": [] }))).into_response()
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.protocols(["foxglove.websocket.v1"]).on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();
    let subscriptions = Arc::new(RwLock::new(HashMap::<u32, u32>::new())); // sub_id -> channel_id

    // 1. Send ServerInfo
    let start_ts = state.start_time_ns.load(Ordering::Relaxed);
    let end_ts = state.end_time_ns.load(Ordering::Relaxed);
    let server_info = ServerInfoMsg {
        op: "serverInfo".to_string(),
        name: "LanceDB Native Rust Streamer".to_string(),
        capabilities: vec![
            "playbackControl".to_string(),
            "time".to_string(),
            "clientPublish".to_string(),
        ],
        supported_encodings: vec!["json".to_string()],
        metadata: [
            ("engine".to_string(), "Rust 1.98.1 (In-Memory Preload)".to_string()),
            ("source".to_string(), "LanceDB / Apache Arrow".to_string()),
        ].into_iter().collect(),
        session_id: "lancedb-rust-session-1".to_string(),
        data_start_time: TimeDef::from_ns(start_ts),
        data_end_time: TimeDef::from_ns(end_ts),
    };

    if let Ok(info_json) = serde_json::to_string(&server_info) {
        let _ = sender.send(Message::Text(info_json.into())).await;
    }

    // 2. Send Advertise
    let chs = state.channels.read().await.clone();
    let adv = AdvertiseMsg {
        op: "advertise".to_string(),
        channels: chs,
    };
    if let Ok(adv_json) = serde_json::to_string(&adv) {
        let _ = sender.send(Message::Text(adv_json.into())).await;
    }

    // Channel for pushing frames to this client
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Message>(128);

    // Send task
    let send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if sender.send(msg).await.is_err() {
                break;
            }
        }
    });

    // Listen to global timeline ticker
    let mut notify_rx = state.notify_tx.subscribe();
    let state_stream = state.clone();
    let subs_stream = subscriptions.clone();
    let tx_stream = tx.clone();
    let stream_task = tokio::spawn(async move {
        while let Ok(ts) = notify_rx.recv().await {
            stream_frame(&state_stream, &subs_stream, &tx_stream, ts, false).await;
        }
    });

    let mut reset_rx = state.client_reset_tx.subscribe();

    // Receive loop
    loop {
        tokio::select! {
            Ok(()) = reset_rx.recv() => {
                println!("[Rust WS] Episode switched: notifying client with updated ServerInfo & Advertise");
                // Keep subscriptions intact so client panels immediately receive frames for the new episode

                // 2. Send updated ServerInfo with new session_id and updated timeline bounds
                let start_ts = state.start_time_ns.load(Ordering::Relaxed);
                let end_ts = state.end_time_ns.load(Ordering::Relaxed);
                let new_session_id = format!("lancedb-rust-session-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis());
                let server_info = ServerInfoMsg {
                    op: "serverInfo".to_string(),
                    name: "LanceDB Native Rust Streamer".to_string(),
                    capabilities: vec![
                        "playbackControl".to_string(),
                        "time".to_string(),
                        "clientPublish".to_string(),
                    ],
                    supported_encodings: vec!["json".to_string()],
                    metadata: [
                        ("engine".to_string(), "Rust 1.98.1 (In-Memory Preload)".to_string()),
                        ("source".to_string(), "LanceDB / Apache Arrow".to_string()),
                    ].into_iter().collect(),
                    session_id: new_session_id,
                    data_start_time: TimeDef::from_ns(start_ts),
                    data_end_time: TimeDef::from_ns(end_ts),
                };
                if let Ok(info_json) = serde_json::to_string(&server_info) {
                    let _ = tx.send(Message::Text(info_json.into())).await;
                }

                // 3. Send Advertise with channels for the new episode
                let chs = state.channels.read().await.clone();
                let adv = AdvertiseMsg {
                    op: "advertise".to_string(),
                    channels: chs,
                };
                if let Ok(adv_json) = serde_json::to_string(&adv) {
                    let _ = tx.send(Message::Text(adv_json.into())).await;
                }

                // 4. Immediately stream initial frame of the new episode
                let cur_ts = state.current_time_ns.load(Ordering::Relaxed);
                stream_frame(&state, &subscriptions, &tx, cur_ts, true).await;
            }
            msg_opt = receiver.next() => {
                let msg = match msg_opt {
                    Some(Ok(m)) => m,
                    _ => break,
                };
                match msg {
                    Message::Text(text) => {
                        if let Ok(sub_req) = serde_json::from_str::<ClientSubscribeMsg>(&text) {
                            if sub_req.op == "subscribe" {
                                if let Some(items) = sub_req.subscriptions {
                                    let mut map = subscriptions.write().await;
                                    for item in items {
                                        map.insert(item.id, item.channel_id);
                                        println!("[Rust WS] Client subscribed sub_id: {} -> chan_id: {}", item.id, item.channel_id);
                                    }
                                }
                                // Immediately stream the initial frame & map to the client upon subscribe
                                let cur_ts = state.current_time_ns.load(Ordering::Relaxed);
                                stream_frame(&state, &subscriptions, &tx, cur_ts, true).await;
                            } else if sub_req.op == "unsubscribe" {
                                if let Some(sub_ids) = sub_req.subscription_ids {
                                    let mut map = subscriptions.write().await;
                                    for sid in sub_ids {
                                        map.remove(&sid);
                                    }
                                }
                            }
                        }
                    }
                    Message::Binary(bin) => {
                        if let Some((command, speed, maybe_seek, req_id)) = decode_playback_control_request(&bin) {
                            let did_seek = maybe_seek.is_some();
                            if let Some(seek_ts) = maybe_seek {
                                state.current_time_ns.store(seek_ts as i64, Ordering::SeqCst);
                                if let Some(cache) = state.episode_cache.read().await.as_ref() {
                                    let idx = match cache.timestamps.binary_search(&(seek_ts as i64)) {
                                        Ok(i) => i,
                                        Err(i) => i.min(cache.timestamps.len().saturating_sub(1)),
                                    };
                                    state.current_frame_idx.store(idx, Ordering::Relaxed);
                                }
                            }
                            *state.playback_speed.write().await = speed;
                            state.is_playing.store(command == 0, Ordering::SeqCst);

                            let cur_ts = state.current_time_ns.load(Ordering::Relaxed) as u64;
                            let resp = encode_playback_state(command, cur_ts, speed, if did_seek { 1 } else { 0 }, &req_id);
                            let _ = tx.send(Message::Binary(resp.into())).await;

                            if did_seek {
                                stream_frame(&state, &subscriptions, &tx, cur_ts as i64, true).await;
                            }
                        }
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
        }
    }

    send_task.abort();
    stream_task.abort();
}

async fn stream_frame(
    state: &Arc<AppState>,
    subscriptions: &Arc<RwLock<HashMap<u32, u32>>>,
    tx: &tokio::sync::mpsc::Sender<Message>,
    target_ts: i64,
    force_map: bool,
) {
    let subs = subscriptions.read().await.clone();
    if subs.is_empty() {
        return;
    }

    let channels = state.channels.read().await.clone();
    let cache_opt = state.episode_cache.read().await.clone();
    let cache = match cache_opt {
        Some(c) => c,
        None => return,
    };

    // Fast binary search for closest frame timestamp in memory
    let frame_ts = match cache.timestamps.binary_search(&target_ts) {
        Ok(i) => cache.timestamps[i],
        Err(i) => {
            if i >= cache.timestamps.len() {
                *cache.timestamps.last().unwrap_or(&target_ts)
            } else if i == 0 {
                cache.timestamps[0]
            } else {
                let prev = cache.timestamps[i - 1];
                let next = cache.timestamps[i];
                if (target_ts - prev).abs() <= (target_ts - next).abs() {
                    prev
                } else {
                    next
                }
            }
        }
    };

    // Foxglove binary time sync (Opcode 2: TIME)
    let time_msg = encode_time(frame_ts as u64);
    let _ = tx.send(Message::Binary(time_msg.into())).await;

    for (sub_id, chan_id) in subs {
        let ch = match channels.iter().find(|c| c.id == chan_id) {
            Some(c) => c,
            None => continue,
        };

        // 1. Camera image (from RAM cache)
        if ch.topic.starts_with("/camera/") {
            let cam_name = ch.topic.trim_start_matches("/camera/");
            if let Some(payload) = cache.cam_msgs.read().await.get(&(cam_name.to_string(), frame_ts)) {
                let data = encode_message_data(sub_id, frame_ts as u64, payload);
                let _ = tx.send(Message::Binary(data.into())).await;
            } else {
                // If not yet warmed in cache, query on demand & memoize
                if let Ok(cam_table) = state.db.open_table("camera_frames").execute().await {
                    let filter = format!("episode_id = '{}' AND camera = '{}' AND timestamp_ns = {}", cache.episode_id, cam_name, frame_ts);
                    if let Ok(mut stream) = cam_table.query().only_if(&filter).limit(1).execute().await {
                        if let Some(Ok(batch)) = stream.next().await {
                            if let Some(col) = batch.column_by_name("image") {
                                let bin_arr = col.as_binary::<i32>();
                                if bin_arr.len() > 0 {
                                    let img_bytes = bin_arr.value(0);
                                    let b64 = BASE64_STANDARD.encode(img_bytes);
                                    let time_def = TimeDef::from_ns(frame_ts);
                                    let msg_json = serde_json::json!({
                                        "timestamp": { "sec": time_def.sec, "nsec": time_def.nsec },
                                        "frame_id": cam_name,
                                        "data": b64,
                                        "format": "jpeg"
                                    });
                                    let bytes = Arc::new(msg_json.to_string().into_bytes());
                                    cache.cam_msgs.write().await.insert((cam_name.to_string(), frame_ts), bytes.clone());
                                    let data = encode_message_data(sub_id, frame_ts as u64, &bytes);
                                    let _ = tx.send(Message::Binary(data.into())).await;
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. 3D bounding boxes (100% in RAM!)
        if ch.topic == "/perception/boxes_3d" {
            if let Some(payload) = cache.box_msgs.read().await.get(&frame_ts) {
                let data = encode_message_data(sub_id, frame_ts as u64, payload);
                let _ = tx.send(Message::Binary(data.into())).await;
            }
        }

        // 3. PointCloud (from RAM cache with lightweight 8,000 pts downsampling)
        if ch.topic == "/lidar/top" {
            if let Some(payload) = cache.lidar_msgs.read().await.get(&frame_ts) {
                let data = encode_message_data(sub_id, frame_ts as u64, payload);
                let _ = tx.send(Message::Binary(data.into())).await;
            } else {
                // If not yet warmed in cache, query on demand & memoize
                if let Ok(lidar_table) = state.db.open_table("lidar_frames").execute().await {
                    let filter = format!("episode_id = '{}' AND timestamp_ns = {}", cache.episode_id, frame_ts);
                    if let Ok(mut stream) = lidar_table.query().only_if(&filter).limit(1).execute().await {
                        if let Some(Ok(batch)) = stream.next().await {
                            if let Some(col) = batch.column_by_name("points") {
                                let bin_arr = col.as_binary::<i32>();
                                if bin_arr.len() > 0 {
                                    let pts_bytes = bin_arr.value(0);
                                    let total_pts = pts_bytes.len() / 20;
                                    let step = (total_pts / 8000).max(1);
                                    let mut downsampled = Vec::with_capacity((total_pts / step + 1) * 16);
                                    for i in (0..total_pts).step_by(step) {
                                        let offset = i * 20;
                                        if offset + 16 <= pts_bytes.len() {
                                            downsampled.extend_from_slice(&pts_bytes[offset..offset + 16]);
                                        }
                                    }
                                    let b64 = BASE64_STANDARD.encode(&downsampled);
                                    let time_def = TimeDef::from_ns(frame_ts);
                                    let pc_msg = serde_json::json!({
                                        "timestamp": { "sec": time_def.sec, "nsec": time_def.nsec },
                                        "frame_id": "vehicle",
                                        "pose": {
                                            "position": { "x": 0.0, "y": 0.0, "z": 0.0 },
                                            "orientation": { "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0 }
                                        },
                                        "point_stride": 16,
                                        "fields": [
                                            { "name": "x", "offset": 0, "type": 7 },
                                            { "name": "y", "offset": 4, "type": 7 },
                                            { "name": "z", "offset": 8, "type": 7 },
                                            { "name": "intensity", "offset": 12, "type": 7 }
                                        ],
                                        "data": b64
                                    });
                                    let bytes = Arc::new(pc_msg.to_string().into_bytes());
                                    cache.lidar_msgs.write().await.insert(frame_ts, bytes.clone());
                                    let data = encode_message_data(sub_id, frame_ts as u64, &bytes);
                                    let _ = tx.send(Message::Binary(data.into())).await;
                                }
                            }
                        }
                    }
                }
            }
        }

        // 4. HD Map features (100% in RAM!)
        if ch.topic == "/map" {
            if force_map || (frame_ts % 3_000_000_000 < 100_000_000) {
                if let Some(payload) = cache.map_msg.as_ref() {
                    let data = encode_message_data(sub_id, frame_ts as u64, payload);
                    let _ = tx.send(Message::Binary(data.into())).await;
                }
            }
        }

        // 5. TF FrameTransforms (100% in RAM!)
        if ch.topic == "/tf" {
            if let Some(payload) = cache.tf_msgs.get(&frame_ts) {
                let data = encode_message_data(sub_id, frame_ts as u64, payload);
                let _ = tx.send(Message::Binary(data.into())).await;
            }
        }
    }
}
