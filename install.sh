#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
CARGO_BIN="${CARGO_BIN:-$(command -v cargo 2>/dev/null || echo "$HOME/.cargo/bin/cargo")}"
DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
MLX_RS_METAL_PATH="${MLX_RS_METAL_PATH:-$HOME/Library/Caches/bsearch/build/mlx-rs-0.33}"
export DEVELOPER_DIR MLX_RS_METAL_PATH
mkdir -p "$MLX_RS_METAL_PATH"
"$CARGO_BIN" build --release --locked --features gemma2-mlx --manifest-path "$DIR/Cargo.toml"
BUILD_DIR="$(otool -l "$DIR/target/release/bsearch" | awk '/cmd LC_RPATH/ {getline; getline; sub(/^[[:space:]]*path /, ""); sub(/ \(offset [0-9]+\)$/, ""); if ($0 ~ /crispembed-build$/) {print; exit}}')"
[ -n "$BUILD_DIR" ] || { echo 'Native libraries not found' >&2; exit 1; }
METALLIB="$MLX_RS_METAL_PATH/mlx.metallib"
if [ ! -f "$METALLIB" ]; then
  CANDIDATE="$(find "$DIR/target/release/build" -name "mlx.metallib" -print -quit 2>/dev/null || true)"
  if [ -n "$CANDIDATE" ] && [ -f "$CANDIDATE" ]; then
    METALLIB="$CANDIDATE"
  fi
fi
if [ ! -f "$METALLIB" ] && [ -f "$HOME/.local/lib/bsearch/mlx-rs-0.33.metallib" ]; then
  METALLIB="$HOME/.local/lib/bsearch/mlx-rs-0.33.metallib"
fi
[ -f "$METALLIB" ] || { echo 'MLX Metal library not found' >&2; exit 1; }
STAGE_ROOT="${BSEARCH_INSTALL_TMPDIR:-$HOME/scratch/bsearch/install}"
mkdir -p "$STAGE_ROOT"
STAGE="$(mktemp -d "$STAGE_ROOT/stage.XXXXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/bin" "$STAGE/lib/bsearch"
cp "$DIR/target/release/bsearch" "$STAGE/bin/bsearch"
find "$BUILD_DIR" -name '*.0.dylib' -exec cp -L {} "$STAGE/lib/bsearch/" \;
cp "$METALLIB" "$STAGE/lib/bsearch/mlx-rs-0.33.metallib"
for item in "$STAGE/bin/bsearch" "$STAGE/lib/bsearch/"*.dylib; do
  while IFS= read -r rpath_entry; do
    case "$rpath_entry" in "$DIR"/target/*) install_name_tool -delete_rpath "$rpath_entry" "$item" ;; esac
  done < <(otool -l "$item" | awk '/cmd LC_RPATH/ {getline; getline; sub(/^[[:space:]]*path /, ""); sub(/ \(offset [0-9]+\)$/, ""); print}')
done
"$STAGE/bin/bsearch" --help >/dev/null
mkdir -p "$HOME/.local/bin" "$HOME/.local/lib/bsearch"
# Existing ggml libraries serve the rollback binary too; do not replace a different build.
for item in "$STAGE/lib/bsearch/"*.dylib; do
  dest="$HOME/.local/lib/bsearch/${item##*/}"
  if [ -e "$dest" ]; then
    cmp -s "$item" "$dest" || { echo "Existing library differs: $dest; retain it for rollback" >&2; exit 1; }
  else
    install -m 755 "$item" "$dest"
  fi
done
install -m 644 "$STAGE/lib/bsearch/mlx-rs-0.33.metallib" "$HOME/.local/lib/bsearch/mlx-rs-0.33.metallib.new"
mv -f "$HOME/.local/lib/bsearch/mlx-rs-0.33.metallib.new" "$HOME/.local/lib/bsearch/mlx-rs-0.33.metallib"
if [ -f "$HOME/.local/bin/bsearch" ]; then
  mkdir -p "$HOME/.local/lib/bsearch-backups"
  cp -p "$HOME/.local/bin/bsearch" "$HOME/.local/lib/bsearch-backups/bsearch-$(date +%Y%m%d-%H%M%S)-$$"
fi
install -m 755 "$STAGE/bin/bsearch" "$HOME/.local/bin/bsearch.new"
mv -f "$HOME/.local/bin/bsearch.new" "$HOME/.local/bin/bsearch"
mkdir -p "$HOME/.config/bsearch"
if [ -f "$DIR/config.json" ] && [ ! -e "$HOME/.config/bsearch/config.json" ] && [ ! -L "$HOME/.config/bsearch/config.json" ]; then
  ln -s "$DIR/config.json" "$HOME/.config/bsearch/config.json"
fi
printf 'Installed %s\n' "$HOME/.local/bin/bsearch"
