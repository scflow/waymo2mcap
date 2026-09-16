#!/usr/bin/env bash
source /home/zr/anaconda3/etc/profile.d/conda.sh
conda activate viz

PIDF=/mnt/d/src/waymo2mcap/data/episode_player.pid
LOG=/mnt/d/src/waymo2mcap/data/episode_player.log

if [ -f "$PIDF" ]; then
  old=$(cat "$PIDF" 2>/dev/null || true)
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
    kill "$old" 2>/dev/null || true
    sleep 0.5
  fi
fi
pkill -f "episode_player.py" 2>/dev/null || true
sleep 0.3

cd /mnt/d/src/waymo2mcap
: > "$LOG"
nohup python episode_player.py --port 8765 --db data/lancedb >>"$LOG" 2>&1 &
echo $! > "$PIDF"
echo "PID=$(cat "$PIDF")"

for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 1
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/episodes', timeout=2)" 2>/dev/null; then
    echo "READY after ${i}s"
    echo "URL http://localhost:8765"
    exit 0
  fi
  if ! kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    echo "DIED"; cat "$LOG"; exit 1
  fi
done
echo "TIMEOUT"; tail -30 "$LOG"; exit 1
