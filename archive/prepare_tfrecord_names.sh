#!/bin/bash
set -e
SRC=/mnt/d/src/waymo2mcap/data/tfrecord
DST=/mnt/d/src/waymo2mcap/data/tfrecord_clean
mkdir -p "$DST"
cd "$SRC"
for f in *.tfrecord; do
  new=$(echo "$f" | sed -E 's/^individual_files_(training|validation)_//')
  if [ ! -e "$DST/$new" ]; then
    ln "$f" "$DST/$new"
    echo "link: $f -> $new"
  else
    echo "exists: $new"
  fi
done
ls -la "$DST"
