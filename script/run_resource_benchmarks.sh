#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <stage-output-directory>" >&2
  exit 2
fi

OUT_DIR="$1"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SMALL_SCENE="/Users/lolney/Downloads/in_the_wild_1.ply"
LARGE_SCENE="$ROOT_DIR/BenchData/bicycle_point_cloud.ply"

mkdir -p "$OUT_DIR"

swift run splatbench \
  --input "$SMALL_SCENE" \
  --frames 12 \
  --warmup 4 \
  --width 640 \
  --height 360 \
  --sort unsorted \
  --output "$OUT_DIR/wild_unsorted_all.json"

swift run splatbench \
  --input "$SMALL_SCENE" \
  --frames 10 \
  --warmup 4 \
  --width 640 \
  --height 360 \
  --sort gpu \
  --max-splats 100000 \
  --output "$OUT_DIR/wild_gpu_100k.json"

swift run splatbench \
  --input "$SMALL_SCENE" \
  --frames 10 \
  --warmup 4 \
  --width 640 \
  --height 360 \
  --sort cpu \
  --max-splats 100000 \
  --output "$OUT_DIR/wild_cpu_100k.json"

swift run splatbench \
  --input "$LARGE_SCENE" \
  --frames 8 \
  --warmup 4 \
  --width 640 \
  --height 360 \
  --sort unsorted \
  --max-splats 500000 \
  --output "$OUT_DIR/bicycle_unsorted_500k.json"

swift run splatbench \
  --input "$LARGE_SCENE" \
  --frames 7 \
  --warmup 4 \
  --width 640 \
  --height 360 \
  --sort gpu \
  --max-splats 250000 \
  --output "$OUT_DIR/bicycle_gpu_250k.json"
