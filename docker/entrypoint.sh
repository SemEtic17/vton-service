#!/usr/bin/env bash
set -euo pipefail

: "${VTON_WEIGHTS_DIR:=/app/weights}"
: "${PORT:=8000}"

mkdir -p "$VTON_WEIGHTS_DIR"

if [ ! -f "$VTON_WEIGHTS_DIR/model.safetensors" ] || [ ! -f "$VTON_WEIGHTS_DIR/dwpose/yolox_l.onnx" ]; then
  echo "Weights not found in $VTON_WEIGHTS_DIR — downloading now (this may take a while)..."
  python scripts/download_weights.py --weights-dir "$VTON_WEIGHTS_DIR"
else
  echo "Weights found in $VTON_WEIGHTS_DIR"
fi

exec uvicorn server:app --host 0.0.0.0 --port "$PORT" --workers 1
