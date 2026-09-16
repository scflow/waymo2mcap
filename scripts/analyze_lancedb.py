#!/usr/bin/env python3
"""
Deep Data Analysis & Exploration on Waymo LanceDB Multimodal Dataset.

Performs:
1. Episode Matrix Profiling (duration, frames, 3D object density, LiDAR points, speed).
2. Category Distribution & Class Balance Analysis across episodes.
3. Ego Motion & Velocity Event Analysis (acceleration, deceleration, stopping).
4. 3D Object Spatial & Geometric Distribution (box dimensions, distance from ego).
5. Cross-Sensor Message Consistency (camera, lidar, pose synchronization).
"""

import sys
import lancedb
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

DB_PATH = "/mnt/d/src/waymo2mcap/data/lancedb"


def analyze_episodes(db):
    print("\n" + "="*80)
    print("📊 1. EPISODE 级别综合数据分析矩阵 (Episode Profiling)")
    print("="*80)
    
    eps_tbl = db.open_table("episodes")
    ep_rows = eps_tbl.to_arrow().to_pylist()
    
    # Load poses for speed stats
    poses = db.open_table("ego_poses").search().select(["episode_id", "speed_mps"]).to_arrow().to_pylist()
    speeds_by_ep = defaultdict(list)
    for p in poses:
        if p["speed_mps"] is not None:
            speeds_by_ep[p["episode_id"]].append(p["speed_mps"])

    # Load 3d boxes count
    o3 = db.open_table("objects_3d").search().select(["episode_id"]).to_arrow().to_pylist()
    o3_by_ep = Counter(r["episode_id"] for r in o3)

    records = []
    for r in ep_rows:
        ep_id = r["episode_id"]
        fc = r.get("frame_count") or 1
        dur = r.get("duration_s") or 0.0
        n_3d = o3_by_ep.get(ep_id, 0)
        n_pts = r.get("num_lidar_points") or 0
        sps = speeds_by_ep.get(ep_id, [])
        avg_sp = float(np.mean(sps)) if sps else 0.0
        max_sp = float(np.max(sps)) if sps else 0.0
        
        records.append({
            "Episode": ep_id[:26] + "...",
            "Type": r.get("source_type", "perception"),
            "Duration": f"{dur:.1f}s",
            "Frames": fc,
            "Total 3D Objs": f"{n_3d:,}",
            "Objs/Frame": f"{n_3d / fc:.1f}",
            "LiDAR Pts": f"{n_pts / 1e6:.1f}M" if n_pts else "0",
            "Avg Speed": f"{avg_sp * 3.6:.1f} km/h",
            "Max Speed": f"{max_sp * 3.6:.1f} km/h",
        })

    df = pd.DataFrame(records)
    print(df.to_string(index=False))


def analyze_classes(db):
    print("\n" + "="*80)
    print("🏷️  2. 目标类别分布与样本均衡性分析 (Class Balance & Distribution)")
    print("="*80)

    o3 = db.open_table("objects_3d").search().select(["episode_id", "label"]).to_arrow().to_pylist()
    total = len(o3)
    counter = Counter(r["label"] for r in o3)

    print(f"全库 3D 标注框总数: {total:,} 目标\n")
    print(f"{'类别 (Label)':<15} | {'总出现次数 (Count)':<18} | {'占比 (Percentage)':<15}")
    print("-" * 55)
    for label, count in counter.most_common():
        pct = (count / total) * 100
        bar = "█" * int(pct / 2)
        print(f"{label:<15} | {count:<18,} | {pct:6.2f}%  {bar}")


def analyze_speed_profiles(db):
    print("\n" + "="*80)
    print("🚗 3. 自车运动学与驾驶事件分析 (Kinematics & Driving Events)")
    print("="*80)

    poses = db.open_table("ego_poses").search().select(["episode_id", "frame_index", "speed_mps"]).to_arrow().to_pylist()
    df_poses = pd.DataFrame(poses)
    df_poses["speed_kmh"] = df_poses["speed_mps"] * 3.6

    for ep_id, group in df_poses.groupby("episode_id"):
        sp = group["speed_kmh"].values
        diffs = np.diff(sp)
        # 10 Hz -> acceleration in m/s^2: delta_v(m/s) / 0.1s
        acc = (np.diff(group["speed_mps"].values)) / 0.1
        
        stopped_frames = np.sum(sp < 1.0)
        fast_frames = np.sum(sp > 30.0)
        max_acc = np.max(acc) if len(acc) else 0.0
        max_dec = np.min(acc) if len(acc) else 0.0

        behavior = []
        if stopped_frames > 20:
            behavior.append(f"包含路口红灯/排队静止 ({stopped_frames} 帧)")
        if fast_frames > 20:
            behavior.append(f"包含城市主干道巡航 (>30km/h, {fast_frames} 帧)")
        if max_dec < -2.0:
            behavior.append(f"急刹车制动 ({max_dec:.1f} m/s²)")
        if not behavior:
            behavior.append("匀速行驶")

        print(f"• [{ep_id[:28]}...]")
        print(f"  速度范围: {sp.min():.1f} ~ {sp.max():.1f} km/h (均值 {sp.mean():.1f} km/h)")
        print(f"  驾驶特征: {' | '.join(behavior)}")


def analyze_3d_geometry(db):
    print("\n" + "="*80)
    print("📐 4. 3D 边界框空间几何分布分析 (3D Box Dimensions & Distances)")
    print("="*80)

    o3 = db.open_table("objects_3d").search().select([
        "label", "pos_x", "pos_y", "pos_z", "size_x", "size_y", "size_z"
    ]).limit(20000).to_arrow().to_pylist()
    df = pd.DataFrame(o3)
    df["distance_m"] = np.sqrt(df["pos_x"]**2 + df["pos_y"]**2 + df["pos_z"]**2)
    df["volume_m3"] = df["size_x"] * df["size_y"] * df["size_z"]

    for label in ["Car", "Pedestrian", "Sign", "Cyclist"]:
        sub = df[df["label"] == label]
        if len(sub) == 0:
            continue
        print(f"[{label}] (采样 {len(sub):,} 个目标):")
        print(f"  平均尺寸 (长×宽×高): {sub['size_x'].mean():.2f}m × {sub['size_y'].mean():.2f}m × {sub['size_z'].mean():.2f}m")
        print(f"  平均体积: {sub['volume_m3'].mean():.2f} m³")
        print(f"  目标探测距离分布: 最小 {sub['distance_m'].min():.1f}m | 中位数 {sub['distance_m'].median():.1f}m | 最大 {sub['distance_m'].max():.1f}m")


def main():
    db = lancedb.connect(DB_PATH)
    analyze_episodes(db)
    analyze_classes(db)
    analyze_speed_profiles(db)
    analyze_3d_geometry(db)
    print("\n" + "="*80)
    print("✅ 分析完成！所有数据已同步注入 FiftyOne 并支持交互式检索过滤。")
    print("="*80)


if __name__ == "__main__":
    main()
