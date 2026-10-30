#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
CARGO_BIN="$(command -v cargo || true)"
CARGO_BIN="${CARGO_BIN:-$HOME/.cargo/bin/cargo}"
DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
MLX_RS_METAL_PATH="${MLX_RS_METAL_PATH:-$HOME/Library/Caches/bsearch/build/mlx-rs-0.33}"
export DEVELOPER_DIR MLX_RS_METAL_PATH
MODEL_DIR="${BSEARCH_GEMMA2_MODEL_DIR:-$HOME/Library/Caches/bsearch/models/embeddinggemma-2-bf16-text}"
"$CARGO_BIN" test --release --features test-backend,gemma2-mlx --manifest-path "$DIR/Cargo.toml" --locked
"$CARGO_BIN" fmt --manifest-path "$DIR/Cargo.toml" -- --check
if [ -f "$MODEL_DIR/model.safetensors" ]; then
  "$CARGO_BIN" run --release --locked --features gemma2-mlx --manifest-path "$DIR/Cargo.toml" --example gemma2_validate -- "$MODEL_DIR" "$DIR/tests/fixtures/gemma2_reference.json"
else
  printf 'SKIP real-model parity: model not found at %s\n' "$MODEL_DIR" >&2
fi
