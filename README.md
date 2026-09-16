# waymo2mcap

Waymo Open Dataset → MCAP → LanceDB → 可视化/检索/策展。

详细工作记录见 [WORKLOG.md](WORKLOG.md)。

## 目录结构

```
waymo2mcap/
├── 管线（根目录，主要入口）
│   ├── waymo_to_mcap.py          TFRecord → MCAP
│   ├── scenario_to_mcap.py       scenario TFRecord → MCAP
│   ├── mcap_to_lancedb.py        MCAP → LanceDB 导入
│   ├── embed_lancedb.py          单图 CLIP 嵌入（512-d）
│   ├── embed_episodes.py         双层多模态 Embedding 引擎（环视融合 + 时序事件）
│   ├── query_lancedb.py          分析查询演示
│   ├── vector_search_demo.py     向量检索演示
│   ├── plot_embeddings.py        PCA 散点 HTML
│   └── episode_player.py         LanceDB 时序播放器（:8765）
│
├── fo/                            FiftyOne 集成
│   ├── rebuild_fo_episodes.py    【核心】构建 Episode 序列视频/3D/时序多模态数据集
│   ├── rebuild_fo_full.py        全量重建 FO 单帧数据集 (waymo-curated)
│   ├── demo_fo_lance_import.py   导入 + App 启动
│   ├── upgrade_fo_multimodal.py  补 embedding/2D/3D/ego
│   ├── fiftyone_bridge.py        FO↔Lance 桥接 CLI
│   ├── fo_lance_e2e.py           端到端验证
│   └── fo_lance_demo.py          演示脚本
│
├── scripts/                       运维与资产脚本
│   ├── export_episode_media.py   Foxglove 环视视频合成 + PCD/FO3D 3D 资产导出
│   ├── rebuild_episodes.sh       一键构建多模态 Episode 全量管线
│   ├── start_fo_app.sh           启动 FO App (:5151)
│   ├── restart_fo_app.sh         重启 FO App
│   ├── start_episode_player.sh   启动播放器 (:8765)
│   ├── rebuild_full.sh           从 Lance 全量重建 FO 单帧
│   ├── install_fiftyone.sh       安装 FiftyOne
│   ├── copy_fo_static.sh         拷贝 FO 静态资源
│   ├── inventory_episodes.py     episode 覆盖率盘点
│   └── check_lancedb.py          LanceDB 健康检查
│
├── archive/                       早期杂项脚本（可删）
├── data/                          数据（gitignore）
│   ├── lancedb/                  分析库（源）
│   ├── fo_lance_demo/            FO 会话库（Lance 后端）
│   ├── fiftyone_media/
│   │   ├── videos/               Foxglove 环视视频 (.mp4)
│   │   ├── pcd/                  点云文件 (.pcd)
│   │   ├── scenes/               FiftyOne 3D 场景 (.fo3d)
│   │   └── frames/               导出 JPEG
│   └── mcap/                     原始 MCAP
├── converter/                     TFRecord 转换器
├── src/                           waymo_open_dataset 本地拷贝
└── WORKLOG.md                     完整工作记录
```

## 环境

| conda env | 用途 |
|---|---|
| `waymo` | TFRecord→MCAP、分析查询 |
| `viz` | FiftyOne + 播放器 |
| `clip` | CLIP 嵌入 |

所有脚本在 WSL 下运行，项目路径 `/mnt/d/src/waymo2mcap`。

## 快速开始

### 方式 A：多模态 Episode 全量管线（Foxglove 环视视频 + 3D 点云 + 双层 Embedding）

```bash
# 一键运行全部多模态 Episode 构建
bash scripts/rebuild_episodes.sh

# 或者分步执行：
# 1. 合成 Foxglove 环视视频与导出 PCD/FO3D 点云资产
conda activate viz
python scripts/export_episode_media.py

# 2. 计算双层多模态 Embeddings（环视融合 + 时序事件）
python embed_episodes.py --compute

# 3. 构建 FiftyOne Episode 序列数据集与 LanceDB Brain 索引
python fo/rebuild_fo_episodes.py --video --grouped --frames --brain

# 4. 语义自然语言检索 Episode 或指定驾驶瞬间
conda activate clip
python embed_episodes.py --query "busy urban street with pedestrians" --target episode
python embed_episodes.py --query "stopping at red traffic light" --target frame

# 5. 启动 FiftyOne App 回放与探索 (:5151)
bash scripts/start_fo_app.sh waymo-episodes-video
```

### 方式 B：早期单帧探索与回放

```bash
# 1. 导入 MCAP → LanceDB
conda activate waymo
python mcap_to_lancedb.py

# 2. 计算单帧 CLIP 嵌入
conda activate clip
python embed_lancedb.py

# 3. 启动独立轻量播放器
bash scripts/start_episode_player.sh
# → http://localhost:8765

# 4. 启动 FiftyOne App
bash scripts/start_fo_app.sh waymo-curated
# → http://localhost:5151
```

## FO App 使用 Lance 后端

```bash
export FIFTYONE_DATASET_STORAGE=lance
export FIFTYONE_DATASET_STORAGE_URI=/mnt/d/src/waymo2mcap/data/fo_lance_demo
export PYTHONPATH=/mnt/d/src/fiftyone
```

FiftyOne 源码在 `D:\src\fiftyone`，分支 `feat/lance-dataset-backend`。

## 服务端口

| 服务 | 端口 | 数据源 |
|---|---|---|
| FiftyOne App | 5151 | data/fo_lance_demo |
| Episode 播放器 | 8765 | data/lancedb |
