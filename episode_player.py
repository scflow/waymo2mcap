#!/usr/bin/env python3
"""
Episode player backed directly by LanceDB.

No MCAP, no FiftyOne — reads camera_frames / objects_3d / ego_poses
from data/lancedb and serves a synchronized multi-camera player.

Usage:
  python episode_player.py [--port 8765] [--db data/lancedb]

Then open http://localhost:8765
"""

import argparse
import base64
import json
import os
import sys
import threading
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import lancedb

DEFAULT_DB = "/mnt/d/src/waymo2mcap/data/lancedb"
CAMERAS = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]

# ---------------------------------------------------------------------------
# Lance access
# ---------------------------------------------------------------------------

class Store:
    def __init__(self, uri):
        self.uri = uri
        self._db = lancedb.connect(uri)
        self._lock = threading.Lock()
        self._cache = {}

    def _table(self, name):
        return self._db.open_table(name)

    def episodes(self):
        rows = self._table("episodes").to_arrow().to_pylist()
        out = []
        for r in rows:
            if r.get("source_type") != "perception":
                continue
            out.append({
                "episode_id": r["episode_id"],
                "frame_count": r["frame_count"],
                "duration_s": r["duration_s"],
                "num_images": r["num_images"],
                "num_objects_3d": r["num_objects_3d"],
                "num_lidar_points": r["num_lidar_points"],
            })
        out.sort(key=lambda x: x["episode_id"])
        return out

    def frame_index(self, episode_id):
        """All frame_index values present for this episode, sorted."""
        key = ("idx", episode_id)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        rows = (self._table("camera_frames").search()
                .where("episode_id = '%s'" % episode_id)
                .select(["frame_index"])
                .to_arrow().column("frame_index").to_pylist())
        frames = sorted(set(rows))
        with self._lock:
            self._cache[key] = frames
        return frames

    def frame_payload(self, episode_id, frame_index):
        """One timestep: 5 cameras + boxes + ego."""
        cf = self._table("camera_frames")
        cams = (cf.search()
                .where("episode_id = '%s' AND frame_index = %d"
                       % (episode_id, frame_index))
                .select(["camera", "image", "num_labels", "labels", "timestamp_ns"])
                .to_arrow().to_pylist())
        by_cam = {c["camera"]: c for c in cams}

        images = {}
        labels = {}
        ts = None
        for cam, row in by_cam.items():
            img = row.get("image")
            if img:
                images[cam] = base64.b64encode(img).decode("ascii")
            labels[cam] = row.get("labels") or []
            if ts is None:
                ts = row.get("timestamp_ns")

        boxes = (self._table("objects_3d").search()
                 .where("episode_id = '%s' AND frame_index = %d"
                        % (episode_id, frame_index))
                 .select(["label", "pos_x", "pos_y", "pos_z",
                          "size_x", "size_y", "size_z"])
                 .to_arrow().to_pylist())

        ego = (self._table("ego_poses").search()
               .where("episode_id = '%s' AND frame_index = %d"
                      % (episode_id, frame_index))
               .select(["px", "py", "pz", "speed_mps"])
               .to_arrow().to_pylist())

        return {
            "episode_id": episode_id,
            "frame_index": frame_index,
            "timestamp_ns": ts,
            "cameras": CAMERAS,
            "images": images,
            "labels": labels,
            "boxes": boxes,
            "ego": ego[0] if ego else None,
        }

    def frame_meta(self, episode_id, frame_index):
        """Lightweight per-frame stats for the timeline (no image bytes)."""
        cf = self._table("camera_frames")
        cams = (cf.search()
                .where("episode_id = '%s' AND frame_index = %d"
                       % (episode_id, frame_index))
                .select(["camera", "num_labels"])
                .to_arrow().to_pylist())
        n_labels = sum(c.get("num_labels") or 0 for c in cams)

        n_boxes = self._table("objects_3d").count_rows(
            "episode_id = '%s' AND frame_index = %d" % (episode_id, frame_index)
        )
        ego = (self._table("ego_poses").search()
               .where("episode_id = '%s' AND frame_index = %d"
                      % (episode_id, frame_index))
               .select(["speed_mps"])
               .to_arrow().to_pylist())
        speed = ego[0]["speed_mps"] if ego else None
        return {
            "frame_index": frame_index,
            "num_labels": n_labels,
            "num_boxes": n_boxes,
            "speed_mps": speed,
        }

    def episode_timeline(self, episode_id):
        frames = self.frame_index(episode_id)
        return [self.frame_meta(episode_id, f) for f in frames]


STORE: Store = None


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<title>Waymo Episode Player · LanceDB</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: #0e0e10; color: #e8e8ea;
    font: 13px/1.45 -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  }
  header {
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    padding: 10px 16px; background: #16161a; border-bottom: 1px solid #2a2a30;
    position: sticky; top: 0; z-index: 10;
  }
  header h1 { font-size: 14px; font-weight: 600; margin: 0; }
  header .src { color: #7a7a85; font-size: 11px; }
  select, button {
    background: #22222a; color: #e8e8ea; border: 1px solid #3a3a44;
    border-radius: 6px; padding: 6px 10px; font: inherit; cursor: pointer;
  }
  button:hover { background: #2c2c36; }
  button.primary { background: #2f6fed; border-color: #2f6fed; }
  button.primary:hover { background: #4580f5; }
  input[type=range] { flex: 1; min-width: 160px; accent-color: #2f6fed; }
  .hud { display: flex; gap: 14px; color: #a0a0aa; font-variant-numeric: tabular-nums; }
  .hud b { color: #e8e8ea; font-weight: 600; }

  .cams {
    display: grid; gap: 6px; padding: 10px 12px;
    grid-template-columns: repeat(3, 1fr);
    grid-template-areas: "fl fr fr" "sl fr fr" "sr rr rr";
  }
  @media (max-width: 900px) { .cams { grid-template-columns: 1fr 1fr; grid-template-areas: none; } .cams > div { grid-area: auto !important; } }
  .cam { position: relative; background: #000; border-radius: 8px; overflow: hidden; aspect-ratio: 3/2; }
  .cam img { width: 100%; height: 100%; object-fit: contain; display: block; }
  .cam .tag {
    position: absolute; left: 8px; top: 6px; padding: 2px 7px;
    background: rgba(0,0,0,.65); border-radius: 4px; font-size: 11px; letter-spacing: .04em;
  }
  .cam canvas { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
  .cam[data-cam=FRONT]      { grid-area: fr; }
  .cam[data-cam=FRONT_LEFT] { grid-area: fl; }
  .cam[data-cam=FRONT_RIGHT]{ grid-area: rr; }
  .cam[data-cam=SIDE_LEFT]  { grid-area: sl; }
  .cam[data-cam=SIDE_RIGHT] { grid-area: sr; }

  footer { padding: 8px 16px 18px; color: #6a6a75; font-size: 11px; }
  .err { color: #ff6b6b; padding: 12px 16px; }
  .timeline {
    display: flex; gap: 2px; height: 28px; align-items: flex-end;
    padding: 0 16px 6px;
  }
  .timeline .bar { flex: 1; background: #2a2a34; border-radius: 2px 2px 0 0; min-height: 2px; }
  .timeline .bar.hot { background: #2f6fed; }
  .timeline .cursor { position: relative; }
</style>
</head>
<body>
<header>
  <h1>Episode Player</h1>
  <span class="src">LanceDB · data/lancedb</span>
  <select id="ep"></select>
  <button id="prev" title="上一帧">‹</button>
  <button id="play" class="primary">▶ 播放</button>
  <button id="next" title="下一帧">›</button>
  <input id="scrub" type="range" min="0" max="0" value="0"/>
  <div class="hud">
    <span>帧 <b id="hFrame">–</b></span>
    <span>标签 <b id="hLabels">–</b></span>
    <span>3D框 <b id="hBoxes">–</b></span>
    <span>速度 <b id="hSpeed">–</b> m/s</span>
  </div>
</header>
<div class="timeline" id="timeline"></div>
<div id="err" class="err" hidden></div>
<div class="cams" id="cams"></div>
<footer id="foot">选择 episode 开始播放 · 五路相机同步 · 叠加 2D 关键点 · 数据来自 LanceDB</footer>

<script>
const CAMS = ["FRONT","FRONT_LEFT","FRONT_RIGHT","SIDE_LEFT","SIDE_RIGHT"];
const TAGS = {FRONT:"FRONT", FRONT_LEFT:"FRONT LEFT", FRONT_RIGHT:"FRONT RIGHT", SIDE_LEFT:"SIDE LEFT", SIDE_RIGHT:"SIDE RIGHT"};
let episodes = [], timeline = [], frames = [];
let curEp = null, cur = 0, playing = false, timer = null;
const FPS = 10; // waymo camera ~10Hz

const $ = id => document.getElementById(id);

function buildCams() {
  const root = $("cams");
  root.innerHTML = "";
  for (const cam of CAMS) {
    const d = document.createElement("div");
    d.className = "cam"; d.dataset.cam = cam;
    d.innerHTML = `<img id="img-${cam}" alt="${cam}"/><canvas id="cv-${cam}"></canvas><div class="tag">${TAGS[cam]}</div>`;
    root.appendChild(d);
  }
}

function showErr(msg) { const e = $("err"); e.hidden = false; e.textContent = msg; }
function clearErr() { $("err").hidden = true; }

async function loadEpisodes() {
  const r = await fetch("/api/episodes");
  episodes = await r.json();
  const sel = $("ep");
  sel.innerHTML = "";
  for (const ep of episodes) {
    const o = document.createElement("option");
    o.value = ep.episode_id;
    o.textContent = `${ep.episode_id.slice(8, 40)} · ${ep.frame_count}帧 · ${ep.num_objects_3d}框`;
    sel.appendChild(o);
  }
  if (episodes.length) { sel.value = episodes[0].episode_id; await selectEpisode(sel.value); }
  sel.onchange = () => selectEpisode(sel.value);
}

async function selectEpisode(id) {
  clearErr();
  curEp = id; cur = 0; stop();
  const r = await fetch(`/api/timeline?episode=${encodeURIComponent(id)}`);
  timeline = await r.json();
  frames = timeline.map(t => t.frame_index);
  $("scrub").max = Math.max(0, frames.length - 1);
  $("scrub").value = 0;
  drawTimeline();
  await showFrame(0);
}

function drawTimeline() {
  const el = $("timeline"); el.innerHTML = "";
  const max = Math.max(1, ...timeline.map(t => t.num_boxes || 0));
  for (let i = 0; i < timeline.length; i++) {
    const t = timeline[i];
    const b = document.createElement("div");
    b.className = "bar" + (i === cur ? " hot" : "");
    b.style.height = (6 + 22 * (t.num_boxes || 0) / max) + "px";
    b.title = `帧 ${t.frame_index} · 3D框 ${t.num_boxes} · 速度 ${(t.speed_mps||0).toFixed(1)}`;
    b.onclick = () => { stop(); showFrame(i); };
    el.appendChild(b);
  }
}

function paintLabels(cam, labels, imgEl, cv) {
  const ctx = cv.getContext("2d");
  const W = imgEl.clientWidth, H = imgEl.clientHeight;
  cv.width = W; cv.height = H;
  ctx.clearRect(0, 0, W, H);
  if (!labels || !labels.length || !imgEl.naturalWidth) return;
  // original JPEG size from naturalWidth; labels are in original pixel space
  const sx = W / imgEl.naturalWidth, sy = H / imgEl.naturalHeight;
  ctx.fillStyle = "#ff4d4f";
  ctx.strokeStyle = "rgba(255,77,79,.85)";
  ctx.lineWidth = 1.5;
  for (const lb of labels) {
    const x = lb.x * sx, y = lb.y * sy;
    ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fill();
    if (lb.text) {
      ctx.font = "11px sans-serif";
      ctx.fillStyle = "rgba(0,0,0,.6)";
      const tw = ctx.measureText(lb.text).width;
      ctx.fillRect(x + 5, y - 12, tw + 6, 14);
      ctx.fillStyle = "#fff";
      ctx.fillText(lb.text, x + 8, y - 2);
      ctx.fillStyle = "#ff4d4f";
    }
  }
}

async function showFrame(i) {
  if (i < 0 || i >= frames.length) return;
  cur = i;
  $("scrub").value = i;
  const fi = frames[i];
  const r = await fetch(`/api/frame?episode=${encodeURIComponent(curEp)}&frame=${fi}`);
  if (!r.ok) { showErr("加载帧失败: " + r.status); return; }
  const d = await r.json();
  clearErr();

  for (const cam of CAMS) {
    const img = $(`img-${cam}`);
    const b64 = d.images[cam];
    if (!b64) { img.removeAttribute("src"); continue; }
    await new Promise(res => { img.onload = res; img.onerror = res; img.src = "data:image/jpeg;base64," + b64; });
    paintLabels(cam, d.labels[cam], img, $(`cv-${cam}`));
  }

  const t = timeline[i] || {};
  $("hFrame").textContent = `${fi} / ${frames[frames.length-1]}`;
  $("hLabels").textContent = t.num_labels ?? "–";
  $("hBoxes").textContent = d.boxes ? d.boxes.length : (t.num_boxes ?? "–");
  const sp = d.ego ? d.ego.speed_mps : t.speed_mps;
  $("hSpeed").textContent = sp != null ? sp.toFixed(1) : "–";

  // highlight timeline
  const bars = $("timeline").children;
  for (let k = 0; k < bars.length; k++) bars[k].classList.toggle("hot", k === i);
}

function tick() { if (playing) showFrame(cur + 1 >= frames.length ? 0 : cur + 1); }
function play() { playing = true; $("play").textContent = "⏸ 暂停"; timer = setInterval(tick, 1000 / FPS); }
function stop() { playing = false; $("play").textContent = "▶ 播放"; if (timer) clearInterval(timer); timer = null; }

$("play").onclick = () => playing ? stop() : play();
$("prev").onclick = () => { stop(); showFrame(Math.max(0, cur - 1)); };
$("next").onclick = () => { stop(); showFrame(Math.min(frames.length - 1, cur + 1)); };
$("scrub").oninput = e => { stop(); showFrame(+e.target.value); };
document.addEventListener("keydown", e => {
  if (e.code === "Space") { e.preventDefault(); playing ? stop() : play(); }
  if (e.code === "ArrowLeft")  { stop(); showFrame(Math.max(0, cur - 1)); }
  if (e.code === "ArrowRight") { stop(); showFrame(Math.min(frames.length - 1, cur + 1)); }
});

buildCams();
loadEpisodes().catch(e => showErr(e.message));
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # quiet

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, text):
        body = text.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            u = urlparse(self.path)
            q = parse_qs(u.query)
            path = u.path

            if path in ("/", "/index.html"):
                return self._html(PAGE)

            if path == "/api/episodes":
                return self._json(STORE.episodes())

            if path == "/api/timeline":
                ep = q.get("episode", [None])[0]
                if not ep:
                    return self._json({"error": "episode required"}, 400)
                return self._json(STORE.episode_timeline(ep))

            if path == "/api/frame":
                ep = q.get("episode", [None])[0]
                fi = q.get("frame", [None])[0]
                if not ep or fi is None:
                    return self._json({"error": "episode+frame required"}, 400)
                return self._json(STORE.frame_payload(ep, int(fi)))

            self._json({"error": "not found"}, 404)
        except BrokenPipeError:
            pass
        except Exception as e:
            try:
                self._json({"error": str(e)}, 500)
            except Exception:
                pass


def main():
    global STORE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--db", default=DEFAULT_DB)
    args = ap.parse_args()

    STORE = Store(args.db)
    eps = STORE.episodes()
    print("LanceDB:", args.db)
    print("episodes:", len(eps))
    for e in eps:
        print("  %s  frames=%s boxes=%s" % (
            e["episode_id"][:52], e["frame_count"], e["num_objects_3d"]))

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print("\nEpisode player → http://localhost:%d" % args.port)
    print("Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
