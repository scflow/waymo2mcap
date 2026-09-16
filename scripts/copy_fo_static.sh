#!/usr/bin/env bash
set -euo pipefail

SRC_STATIC=/home/zr/src/fiftyone/fiftyone/server/static
PKG_STATIC=/home/zr/anaconda3/envs/viz/lib/python3.10/site-packages/fiftyone/server/static

# replace symlink with a real copy — WSL reads across the
# /mnt/d <-> /home boundary are slow and can time out on big chunks
if [ -L "$SRC_STATIC" ]; then
  rm "$SRC_STATIC"
fi
if [ -d "$SRC_STATIC" ] && [ -n "$(ls -A "$SRC_STATIC" 2>/dev/null)" ]; then
  rm -rf "${SRC_STATIC}.old"
  mv "$SRC_STATIC" "${SRC_STATIC}.old"
fi

mkdir -p "$SRC_STATIC"
cp -a "$PKG_STATIC/." "$SRC_STATIC/"
echo "copied $(find "$SRC_STATIC" -type f | wc -l) files"
ls "$SRC_STATIC"
