#!/usr/bin/env bash
source /home/zr/anaconda3/etc/profile.d/conda.sh
conda activate viz

# App server is a subprocess — it only sees env vars, not fo.config
export FIFTYONE_DATASET_STORAGE=lance
export FIFTYONE_DATASET_STORAGE_URI=/mnt/d/src/waymo2mcap/data/fo_lance_demo
export PYTHONPATH=/mnt/d/src/fiftyone:${PYTHONPATH:-}

# stop
pkill -f "fo/demo_fo_lance_import.py --app" 2>/dev/null || true
sleep 1

cd /mnt/d/src/waymo2mcap
LOG=data/fo_app.log
PIDF=data/fo_app.pid

: > "$LOG"
nohup python fo/demo_fo_lance_import.py --app --port 5151 >>"$LOG" 2>&1 &
echo $! > "$PIDF"
echo "PID=$(cat "$PIDF")"
echo "env FIFTYONE_DATASET_STORAGE=$FIFTYONE_DATASET_STORAGE"
echo "env FIFTYONE_DATASET_STORAGE_URI=$FIFTYONE_DATASET_STORAGE_URI"

for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  sleep 1
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5151', timeout=2)" 2>/dev/null; then
    echo "READY after ${i}s"
    exit 0
  fi
  if ! kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    echo "DIED"
    cat "$LOG"
    exit 1
  fi
done
echo "TIMEOUT"
tail -40 "$LOG"
exit 1
