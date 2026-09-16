#!/usr/bin/env bash
export PROTOC=/home/zr/anaconda3/envs/waymo/bin/protoc
export CARGO_TARGET_DIR=/home/zr/.cargo_targets/lancedb_streame
export PATH="$HOME/.cargo/bin:/home/zr/anaconda3/envs/waymo/bin:$PATH"
cd "$(dirname "$0")"
exec cargo "$@"
