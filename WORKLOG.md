# Waymo → LanceDB → 可视化：完整工作记录

> 会话范围：从 MCAP 解码入库，到 CLIP 向量检索，到把 FiftyOne 的存储后端
> 改成 Lance，再到基于 LanceDB 的 episode 时序播放器。
>
> 项目根目录：`D:\src\waymo2mcap`（WSL: `/mnt/d/src/waymo2mcap`）
> FiftyOne 分支：`D:\src\fiftyone` · `feat/lance-dataset-backend`

---

## 1. 总览

### 1.1 最终架构

```
TFRecord / Waymo Open Dataset
        │  waymo_to_mcap.py / scenario_to_mcap.py
        ▼
     MCAP 文件                    data/mcap/  (~800MB × 7)
        │  mcap_to_lancedb.py
        ▼
┌──────────────────────────────────────────────────────────┐
│  LanceDB  data/lancedb   ←—— 唯一事实源（可编辑）          │
│  episodes / camera_frames / lidar_frames                 │
│  ego_poses / objects_3d / map_features                   │
│  JPEG blob + CLIP 512-d 向量 + 2D/3D 标注                 │
└──────────────────────────────────────────────────────────┘
        │
        ├─► query_lancedb.py / vector_search_demo.py   分析检索
        ├─► episode_player.py  :8765                   时序回放
        ├─► FiftyOne App       :5151                   策展透镜
        └─► plot_embeddings.py                         向量散点
```

核心原则：**Lance 是库，其他都是透镜。** FiftyOne 的嵌入式 Mongo 跨进程即丢（已实测），不能当库用。

### 1.2 环境

| conda env | 用途 | 关键包 |
|---|---|---|
| `waymo` | TFRecord→MCAP 转换、分析查询 | waymo-open-dataset, lancedb, protobuf |
| `viz` | FiftyOne + 播放器 | fiftyone 1.22, fiftyone-brain 0.25, lancedb 0.38 |
| `clip` | CLIP 嵌入 | torch 2.14 CPU, transformers 5.17, pylance 0.38 |

WSL 访问 Windows 路径：`/mnt/d/src/waymo2mcap`。PowerShell 里调 WSL 必须用脚本文件，内联 heredoc 会被拆碎。

---

## 2. 数据层：LanceDB

### 2.1 表结构（`data/lancedb`，~4.7 GB）

| 表 | 行数 | 内容 |
|---|---|---|
| `episodes` | 7 | episode 注册表：mcap 路径、时间范围、统计 |
| `camera_frames` | 5,950 | 每帧×每相机：JPEG blob + 2D labels + `image_embedding`(512-d) |
| `lidar_frames` | 1,190 | 点云原始 buffer（20 字节步长：x,y,z,intensity,elongation） |
| `ego_poses` | 1,281 | 位姿 px/py/pz + 四元数 + 速度 |
| `objects_3d` | 59,374 | 每个 3D 框一行：label/location/size/quat/color |
| `map_features` | 2,657 | HD 地图要素（line_strip / line_loop / cylinder） |

行键：`(episode_id, frame_index, timestamp_ns)`。`episode_id` 建了 BTree 标量索引。

### 2.2 Episode 清单

6 段 perception + 1 段 scenario：

| episode | 帧数 | 图像 | 点云 | 3D 框 |
|---|---|---|---|---|
| segment-10017090168044687777_6380 | 198 | 990 | 34.7M | 7,013 |
| segment-10023947602400723454_1120 | 199 | 995 | 36.1M | 18,633 |
| segment-1005081002024129653_5313 | 199 | 995 | 34.4M | 3,075 |
| segment-10061305430875486848_1080 | 198 | 990 | 36.1M | 6,969 |
| segment-10072140764565668044_4060 | 198 | 990 | 30.0M | 19,558 |
| segment-10072231702153043603_5725 | 198 | 990 | 35.2M | 3,028 |
| scenario_28fe360951cf98d6 | 91 | – | – | 1,098 |

每段 perception 有 5 路相机：FRONT / FRONT_LEFT / FRONT_RIGHT / SIDE_LEFT / SIDE_RIGHT。

### 2.3 相关脚本

- `mcap_to_lancedb.py` — MCAP → Lance 导入器（`--max-frames` / `--rebuild` / `--force` / `--no-lidar`）
- `query_lancedb.py` — 5 种分析模式演示（过滤/聚合/跨表关联/媒体检索）
- `scripts/inventory_episodes.py` — episode 覆盖率盘点

---

## 3. 向量层：CLIP 嵌入

### 3.1 嵌入

- 模型：`openai/clip-vit-base-patch32`，512 维，L2 归一化
- 全量 5,950 张相机图，CPU 约 31 img/s，总耗时 185s
- 写入 `camera_frames.image_embedding`（FixedSizeList float32）
- 脚本：`embed_lancedb.py`（先预计算再 `add_columns`，避开 Lance 的 mutable borrow）

### 3.2 向量检索

`vector_search_demo.py` 演示三种模式：

1. **text→image**：CLIP 文本塔编码，Lance cosine 检索
2. **image→image**：以某帧为查询，自匹配 rank-1，相邻帧紧随
3. **混合过滤**：向量召回 + `camera='FRONT'` 结构化收窄

IVF_FLAT cosine 索引已建在 `image_embedding` 上。

### 3.3 2D 散点

`plot_embeddings.py` → PCA 降维（26% + 12.3% 方差）→ 独立 HTML
（`data/lancedb_exports/embedding_scatter.html`），按相机着色，悬停显示 episode/frame。

---

## 4. FiftyOne：把存储后端换成 Lance

这是本场工作量最大的一块。FiftyOne OSS 的 dataset 本体焊死在 MongoDB 上，我们把它改成了可选 Lance 后端。

### 4.1 为什么不能只当「导出工具」用

- FO 自带嵌入式 Mongo，**跨进程即丢**（实测：process-1 写入，process-2 `list_datasets()` 为空）
- FO 官方把 LanceDB 定为 Brain 的一等 similarity 后端（`backend="lancedb"`）
- FO Enterprise 的 MCAP indexing 写 Parquet/Iceberg——我们在 Lance 上做的正是它的开源等价物

所以正确分工：**Lance 是库，FO 是可丢弃的会话透镜，每次从 Lance 重挂。**

### 4.2 分支与提交

分支：`feat/lance-dataset-backend`（`D:\src\fiftyone`）

| commit | 内容 |
|---|---|
| `44c46e13fc` | LanceStore / LanceDataset 原型（独立于 FO API） |
| `7c36ef2ae5` | 公开 `fo.*` API 接入 Lance：config 路由、create/list/load/delete、add_samples、match |
| `743f8272d7` | App GraphQL 层：paginator、estimated count、samples 聚合、ObjectId 自动生成 |
| `c59cbf32b6` | Connection `endCursor` 修复（Relay 在 edges 非空但 endCursor=null 时不提交） |
| `8964493aeb` | `sample.save()` / brain runs / view pipeline 支持 |
| `93155c9d5a` | `to_dict` 兼容内联 dict run docs |
| `d9d0c5e6a7` | visualization brain run 可用（object 数组序列化、GridFS delete 跳过） |
| `cbcd29c022` | `results_json` 判定为 ready（App 不再显示 Pending） |
| `88737e93df` | `/embeddings/v2/*` 接口走 Lance |
| `7b7b905d2a` | `delete_dataset` 兼容 dict run docs |

### 4.3 配置方式

```bash
export FIFTYONE_DATASET_STORAGE=lance
export FIFTYONE_DATASET_STORAGE_URI=/mnt/d/src/waymo2mcap/data/fo_lance_demo
export PYTHONPATH=/mnt/d/src/fiftyone
```

或 Python 内：

```python
import fiftyone as fo
fo.config.dataset_storage = "lance"
fo.config.dataset_storage_uri = "/path/to/lance_db"
```

**必须用环境变量**：App 服务端是子进程，收不到内存里的 `fo.config`。

### 4.4 存储布局（`data/fo_lance_demo`）

```
_fo_registry/              # dataset 注册表（BSON extended JSON）
samples_<oid>/             # 每 dataset 一张样本表
runs/                      # brain/eval run 文档
fo_frames_clip/            # FO 形状的向量表 (id, sample_id, vector)
```

样本表三列：`_id` (str ObjectId)、`filepath`、`_doc_json`（BSON JSON，含全部动态字段）。

### 4.5 关键路由点

| 文件 | 改动 |
|---|---|
| `fiftyone/core/config.py` | 新增 `dataset_storage` / `dataset_storage_uri` |
| `fiftyone/core/dataset.py` | create/list/load/delete、`_get_sample_collection` 按 flag 分发 |
| `fiftyone/core/odm/dataset.py` | `DatasetDocument.save/reload/delete` 走 Lance；run 字段改 `DictField` |
| `fiftyone/core/odm/mixins.py` | `DatasetMixin._get_collection` → LanceCollection（**否则 `sample.save()` 绕过 Lance 直连 Mongo**） |
| `fiftyone/core/odm/runs.py` | `RunDocument._get_collection` / `save` / `delete`；新增 `results_json` 内联字段 |
| `fiftyone/core/runs.py` | `save_run_results` 写 `results_json`；`load_run_results` 读它；run doc 反序列化为 RunDocument |
| `fiftyone/core/lance/collection.py` | pymongo 形状 Collection shim |
| `fiftyone/core/lance/router.py` | 注册表 + BSON JSON 往返 |
| `fiftyone/server/paginator.py` | datasets 列表从 Lance 注册表读 |
| `fiftyone/server/query.py` | estimated count、brain ready 判定 |
| `fiftyone/server/samples.py` | `paginate_samples` 走 LanceCollection.aggregate |
| `fiftyone/server/routes/embeddings_v2.py` | `runs-status` 从 Lance runs 读 |
| `fiftyone/core/utils.py` | `serialize_numpy_array` 处理 object 数组 |

### 4.6 Collection shim 踩过的坑

| 症状 | 根因 | 修复 |
|---|---|---|
| `sample with id not found` | `{'_id': {'$in': [ObjectId]}}` 中 str vs ObjectId | `$in/$eq/$ne/$nin` 对 `_id` 走 `_oid()` 归一化 |
| `values()` 全 None | `$project` 不认 `{'$ifNull': ['$f', None]}` | `_project_doc` 求值操作符表达式 |
| view 字段全空 | `$project: {'_select_order': False}` 是排除语义 | 区分 inclusion / exclusion |
| `datetime` 校验失败 | 普通 `json.dumps` 把 datetime 变字符串 | BSON extended JSON (`json_util`) |
| `sample.save()` 直连 Mongo | mongoengine `Document._get_collection()` | `DatasetMixin` 覆写路由 |
| `updatedExisting` 判定失败 | `update_one` 不返回 pymongo 形状 | `_UpdateResult.raw_result` |
| `bytes.fromhex` 崩溃 | `sample_ids` 被存成字符串 `"None"` | 修 `values("id")` 后重建 viz run |
| Embeddings 面板显示 Enterprise 墙 | `isVisualizationConfig` 只认 `brain.visualization.*` | 用 `fob.compute_visualization(method="manual")` 建真 viz run |
| 面板 Pending | `ready` 只看 GridFS `results` | 同时认 `results_json` |
| `delete_dataset` 崩溃 | 对 dict run doc 调 `.results` | `_delete_dataset_extras` 兼容 dict |

### 4.7 App 静态资源

源码树的 `fiftyone/server/static/` 是空的，SPA 资产在 pip 包里。**symlink 跨 `/mnt/d`↔`/home` 读大 JS 会超时**，必须物理拷贝：

```bash
cp -a /home/zr/anaconda3/envs/viz/lib/python3.10/site-packages/fiftyone/server/static/. \
      /home/zr/src/fiftyone/fiftyone/server/static/
```

### 4.8 FO 数据集现状

`waymo-curated`（`data/fo_lance_demo`）：

- **5,950 samples**，来自 6 段 perception 全量
- 字段：`episode_id` / `frame_index` / `camera` / `timestamp_ns` / `num_labels` / `embedding`(512) / `keypoints`(2D) / `detections_3d`(3D框) / `ego_px/py/pz` / `speed_mps`
- Brain run `clip_viz`：PCA-2D，5950 点，`ManualVisualizationConfig`

导入脚本：`rebuild_fo_full.py`、`demo_fo_lance_import.py`、`upgrade_fo_multimodal.py`

---

## 5. Episode 播放器（LanceDB 直读）

FiftyOne 的网格是静态的，MCAP 回放又被否掉（绕开 Lance）。所以做了一个**直接读 LanceDB 的时序播放器**。

### 5.1 技术

- 标准库 `http.server`，零新增依赖
- 端口 **8765**
- 后端：`episode_player.py`
- 启动：`scripts/start_episode_player.sh`

### 5.2 数据流

```
GET /api/episodes                    ← episodes 表
GET /api/timeline?episode=X          ← 每帧 num_labels/num_boxes/speed
GET /api/frame?episode=X&frame=N     ← 五路 JPEG(base64) + 2D labels + 3D boxes + ego
```

### 5.3 界面

- 顶部切换 6 个 episode
- 播放/暂停（空格）、前后帧（←→）、拖动进度条
- 五路相机同步布局（FRONT 居中，四路环绕）
- 红色圆点 = 2D 关键点，带类别文字
- HUD：帧号 / 标签数 / 3D 框数 / 车速 m/s
- 底部 timeline 柱高 = 该帧 3D 框数量，点柱跳帧

### 5.4 常用命令

```bash
bash scripts/start_episode_player.sh    # 启动
kill $(cat data/episode_player.pid)     # 停止
tail -f data/episode_player.log
```

---

## 6. 服务与端口

| 服务 | 端口 | 启动 | 数据源 |
|---|---|---|---|
| FiftyOne App | 5151 | `scripts/start_fo_app.sh` | `data/fo_lance_demo` |
| Episode 播放器 | 8765 | `scripts/start_episode_player.sh` | `data/lancedb` |

两个都是 WSL 内监听 `0.0.0.0`，Windows 浏览器走 `localhost` 即可。

---

## 7. 目录结构

```
D:\src\waymo2mcap\
├── waymo_to_mcap.py / scenario_to_mcap.py     TFRecord → MCAP
├── mcap_to_lancedb.py                         MCAP → Lance 导入器
├── embed_lancedb.py                           CLIP 嵌入
├── query_lancedb.py                           分析查询演示
├── vector_search_demo.py                      向量检索演示
├── plot_embeddings.py                         PCA 散点 HTML
├── episode_player.py                          LanceDB 时序播放器
│
├── fo/                                         FiftyOne 集成
│   ├── rebuild_fo_full.py                    全量 5950 帧重建
│   ├── demo_fo_lance_import.py               导入 + App 启动
│   ├── upgrade_fo_multimodal.py              补 embedding/2D/3D/ego
│   ├── fiftyone_bridge.py                    FO↔Lance 桥接 CLI
│   ├── fo_lance_e2e.py                       端到端验证
│   └── fo_lance_demo.py                      演示脚本
│
├── scripts/                                    运维脚本
│   ├── start_fo_app.sh / restart_fo_app.sh   FO App (:5151)
│   ├── start_episode_player.sh               播放器 (:8765)
│   ├── rebuild_full.sh                       从 Lance 重建 FO
│   ├── install_fiftyone.sh / copy_fo_static.sh
│   └── inventory_episodes.py / check_lancedb.py
│
├── archive/                                    早期杂项（可删）
├── README.md / WORKLOG.md
└── data/
    ├── lancedb/                               分析库（源）
    ├── fo_lance_demo/                         FO 会话库
    ├── fiftyone_media/frames/                 导出的 JPEG
    ├── mcap/                                  原始 MCAP
    └── lancedb_exports/                       散点图等导出

D:\src\fiftyone\                               FO 分支 feat/lance-dataset-backend
└── fiftyone/core/lance/                       collection.py + router.py
```

---

## 8. 已验证的能力

| 能力 | 状态 | 入口 |
|---|---|---|
| MCAP → Lance 全量入库 | ✅ 7 段，4.7GB | `mcap_to_lancedb.py` |
| CLIP 向量 + IVF 索引 | ✅ 5950×512 | `embed_lancedb.py` |
| text→image / image→image 检索 | ✅ | `vector_search_demo.py` |
| 向量 2D 散点 | ✅ 独立 HTML | `plot_embeddings.py` |
| FO 公开 API 落 Lance | ✅ 跨进程持久 | `dataset_storage=lance` |
| FO App 网格 + 侧栏过滤 | ✅ 5950 帧 | `:5151` |
| FO Embeddings 面板 | ✅ clip_viz 散点 | `:5151` |
| Episode 时序播放 | ✅ 6 段，五路同步 | `:8765` |
| 标签叠图（App 网格上画框） | ⚠️ 字段在，渲染未完成 | — |
| 点云 3D 可视化 | ⚠️ 数据在 Lance，未接查看器 | — |
| Doris Catalog 直查 Lance | 未做 | `CREATE CATALOG ... "type"="lance"` |

---

## 9. 经验教训

1. **不要把 FO 当库。** 嵌入式 Mongo 跨进程即丢；每次会话从 Lance 重挂。
2. **App 服务端是子进程**，只认环境变量 `FIFTYONE_DATASET_STORAGE`，不认内存里的 `fo.config`。
3. **`add_columns` 期间不能扫同一 dataset**（mutable borrow）——先预计算再写回。
4. **PowerShell + WSL heredoc 必炸**，一律写脚本文件再 `wsl -e bash xxx.sh`。
5. **symlink 跨 `/mnt/d`↔`/home` 读大文件会超时**，静态资源要物理拷贝。
6. **FO 的 Embeddings 面板只认 `brain.visualization.*`**，similarity run 会掉进 Enterprise 墙。
7. **`sample.save()` 走 mongoengine `Document._get_collection()`**，不在 Dataset 层路由就会绕过 Lance。
8. **Lance 是列存**，Mongo 的点更新语义要靠 delete+insert 或 `update()` 模拟，大批量逐条 `save()` 只有 ~7 samples/s；批量 `add_samples` 能到 200+/s。

---

---

## 10. 已完成的新能力（Episode 多模态索引、播放与双层 Embedding）

本阶段根据用户要求（方案 B + Foxglove 范式 + 3D 点云与 3D 框 + 帧/Episode 双层 Embedding），完整构建了基于 FiftyOne 与 LanceDB 的多模态 Episode 基础设施：

### 10.1 Foxglove 环视视频合成与 3D 点云导出 (`scripts/export_episode_media.py`)
- **Foxglove 环视画布 (1440×640 @ 10 FPS, H.264)**:
  - 布局：
    - `(0, 0)`: `FRONT_LEFT` (480×320)
    - `(1, 0)`: `FRONT` (480×320)
    - `(2, 0)`: `FRONT_RIGHT` (480×320)
    - `(0, 1)`: `SIDE_LEFT` (480×320)
    - `(1, 1)`: `HUD Dashboard` (自车瞬时速度进度条、3D 目标数量、LiDAR 激活点数、帧索引与时间戳)
    - `(2, 1)`: `SIDE_RIGHT` (480×320)
  - 使用 `imageio-ffmpeg` 压制为 H.264 MP4（~18-37 MB/段），10 FPS 丝滑播放，存储于 `data/fiftyone_media/videos/<episode_id>.mp4`。
- **LiDAR 点云与 3D 场景导出**:
  - `lidar_frames` 原始 20-byte buffer 导出为标准二进制 `.pcd` (183,680 pts/帧)；
  - 同步生成 FiftyOne 3D 场景文件 `.fo3d` (`data/fiftyone_media/scenes/`)。

### 10.2 双层多模态 Embedding 引擎 (`embed_episodes.py`)
- **帧级 5 相机空间加权融合向量 (`frame_fused_embeddings`)**:
  - 方向权重：`FRONT` (0.36), `FRONT_LEFT` (0.16), `FRONT_RIGHT` (0.16), `SIDE_LEFT` (0.16), `SIDE_RIGHT` (0.16)。
  - 空间加权归一化生成 512-d 环视场景向量，全量 1,190 帧存入 LanceDB 并建立 IVF_FLAT 向量索引。
- **Episode 级时序事件向量 (`episodes.episode_embedding`)**:
  - 对整段时序帧向量做全局 Temporal Pooling 并 L2 归一化；
  - 写入 `episodes` 表的 `episode_embedding` 列，建立基于 LanceDB 的 Episode 相似度索引。
- **文本检索 CLI**:
  - `python embed_episodes.py --query "busy urban street with pedestrians" --target episode`（精准召回 3D 目标密度最高的两段 episode）；
  - `python embed_episodes.py --query "stopping at red traffic light" --target frame`（精准召回正在等红灯的特定秒级帧 22-42）。

### 10.3 FiftyOne 数据集三维矩阵重构 (`fo/rebuild_fo_episodes.py`)
在 Lance 存储后端上构建了三个互补的数据集：
1. **`waymo-episodes-video` (media_type="video", 6 samples)**:
   - 每个样本对应 1 个 Episode，支持 FiftyOne 原生播放器以 10 FPS 播放 Foxglove 环视视频与拖拽时间轴；
   - `sample.frames` 挂载逐帧时序指标：`speed_mps`, `ego_px/py/pz`, `detections_3d`, `keypoints`, `embedding`, `pcd_path`；
   - 接入 FiftyOne Brain runs：`episode_sim` (LanceDB 向量检索) 与 `episode_viz` (2D PCA 散点)。
2. **`waymo-episodes-grouped` (media_type="group", 6 samples)**:
   - 包含多模态切片：`video`（环视回放）与 `lidar_3d`（FiftyOne 3D Visualizer 点云与 3D 框）。
3. **`waymo-fused-frames` (media_type="image", 1,190 samples)**:
   - 逐帧 360° 融合特征集，接入 `frame_sim` (LanceDB 相似度) 与 `frame_viz` (2D PCA 面板)。

---

## 11. 后续可做

- **App 3D 实时流式切片播放**：在 3D 查看器中动态随帧同步切点云 PCD。
- **Doris Catalog**：`CREATE CATALOG lance TYPE=lance` 直接查分析库。
- **扩大导入**：`data/mcap` 下还有 ~260 个未入库的 MCAP 文件（约 800MB/个）。
