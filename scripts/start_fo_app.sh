#!/usr/bin/env bash
source /home/zr/anaconda3/etc/profile.d/conda.sh
conda activate viz
cd /mnt/d/src/waymo2mcap

# App server is a subprocess — it only sees env vars, not fo.config
export FIFTYONE_DATASET_STORAGE=lance
export FIFTYONE_DATASET_STORAGE_URI=/mnt/d/src/waymo2mcap/data/fo_lance_demo
export PYTHONPATH=/mnt/d/src/fiftyone:${PYTHONPATH:-}

LOG=/mnt/d/src/waymo2mcap/data/fo_app.log
PIDF=/mnt/d/src/waymo2mcap/data/fo_app.pid

# stop previous
if [ -f "$PIDF" ]; then
  old=$(cat "$PIDF" 2>/dev/null || true)
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
    kill "$old" 2>/dev/null || true
    sleep 1
  fi
fi
pkill -f "fo/demo_fo_lance_import.py --app" 2>/dev/null || true
pkill -f "fiftyone.*5151" 2>/dev/null || true
sleep 0.5

DS=${1:-waymo-episodes-video}
: > "$LOG"
nohup python fo/demo_fo_lance_import.py --app --dataset "$DS" --port 5151 >>"$LOG" 2>&1 &
echo $! > "$PIDF"
echo "PID=$(cat "$PIDF") (dataset: $DS)"

# wait for HTTP
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  sleep 1
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5151', timeout=2)" 2>/dev/null; then
    echo "READY after ${i}s"
    echo "URL http://localhost:5151"
    exit 0
  fi
  # bail if process died
  if ! kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    echo "PROCESS DIED"
    echo "---log---"
    cat "$LOG"
    exit 1
  fi
done

echo "TIMEOUT waiting for port"
echo "---log---"
tail -40 "$LOG"
exit 1
